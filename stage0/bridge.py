"""Measured CubeCL pointer ABI on PyTorch storage and the current stream."""
import ctypes as C
import json
import os
from pathlib import Path
import re
import time
import threading

import torch

ROOT = Path(__file__).resolve().parent


def native_path(name):
    profile = os.environ.get("STAGE0_CARGO_PROFILE", "debug")
    path = ROOT / "target" / profile / name
    if not path.is_file():
        raise RuntimeError(f"missing {path}; build that Cargo profile or set STAGE0_CARGO_PROFILE")
    return path


def nvrtc_compile(source, artifact_dir, device):
    """Compile the unmodified CubeCL-emitted source with native NVRTC."""
    lib_path = os.environ.get("STAGE0_NVRTC")
    if not lib_path:
        if torch.version.cuda is None:
            raise RuntimeError("Stage 0 requires a CUDA-enabled PyTorch installation")
        major = torch.version.cuda.split(".")[0]
        candidates = sorted({p.resolve() for p in Path(torch.__file__).parent.parent.glob(f"nvidia/**/libnvrtc.so.{major}")})
        if not candidates:
            raise RuntimeError("set STAGE0_NVRTC to libnvrtc.so")
        if len(candidates) != 1:
            raise RuntimeError("multiple NVRTC libraries found; select one with STAGE0_NVRTC")
        lib_path = str(candidates[0])
    lib = C.CDLL(lib_path)
    program = C.c_void_p()
    lib.nvrtcCreateProgram.argtypes = [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p, C.c_int, C.c_void_p, C.c_void_p]
    lib.nvrtcCompileProgram.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)]
    lib.nvrtcGetProgramLogSize.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
    lib.nvrtcGetProgramLog.argtypes = [C.c_void_p, C.c_void_p]
    lib.nvrtcGetPTXSize.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
    lib.nvrtcGetPTX.argtypes = [C.c_void_p, C.c_void_p]
    lib.nvrtcDestroyProgram.argtypes = [C.POINTER(C.c_void_p)]
    def check(rc):
        if rc:
            raise RuntimeError(f"NVRTC error {rc}")
    check(lib.nvrtcCreateProgram(C.byref(program), source.encode(), b"cubecl.cu", 0, None, None))
    major, minor = torch.cuda.get_device_capability(device)
    options = (C.c_char_p * 3)(f"--gpu-architecture=compute_{major}{minor}".encode(), b"--std=c++17", b"--fmad=false")
    started = time.perf_counter()
    try:
        rc = lib.nvrtcCompileProgram(program, len(options), options)
        size = C.c_size_t()
        check(lib.nvrtcGetProgramLogSize(program, C.byref(size)))
        log = C.create_string_buffer(size.value)
        check(lib.nvrtcGetProgramLog(program, log))
        (artifact_dir / "nvrtc.log").write_bytes(log.value)
        if rc:
            raise RuntimeError(f"NVRTC error {rc}: {log.value.decode()}")
        check(lib.nvrtcGetPTXSize(program, C.byref(size)))
        ptx = C.create_string_buffer(size.value)
        check(lib.nvrtcGetPTX(program, ptx))
        (artifact_dir / "kernel.ptx").write_bytes(ptx.value)
        timing = {"nvrtc_compile_ms": (time.perf_counter() - started) * 1000, "options": [o.decode() for o in options], "library": lib_path}
        (artifact_dir / "nvrtc.json").write_text(json.dumps(timing, indent=2))
        return ptx.value
    finally:
        lib.nvrtcDestroyProgram(C.byref(program))


class Kernel:
    """Prepare outside capture; instances/modules remain loaded for the process.

    Cross-stream inputs require caller waits for readiness. record_stream below
    prevents allocation reuse; it does not add execution dependencies.
    Captured graphs require caller-retained input/output/metadata across replays.
    """
    def __init__(self, artifact_dir, device=0, compile_ptx=False):
        self.artifact_dir = Path(artifact_dir)
        self.manifest = json.loads((self.artifact_dir / "manifest.json").read_text())
        self.buffer_count = len(self.manifest["arguments"]) - 1
        if (self.manifest["schema"] != 1 or self.manifest["address_bits"] != 64
                or self.manifest["vector_width"] != 1
                or self.buffer_count not in (2, 3)
                or self.manifest["metadata"]["bytes"] != 8 * self.buffer_count
                or self.manifest["block"] != [256, 1, 1]
                or self.manifest["dynamic_shared_bytes"] != 0
                or self.manifest["grid_constants"]):
            raise ValueError("launcher supports only the measured scalar-pointer ABI")
        self.device = torch.device("cuda", device)
        # Stage 0 intentionally retains metadata for every prepared N.
        self._metadata_cache = {}
        self._cache_lock = threading.RLock()
        self._metadata_complete = set()
        self.native = C.CDLL(str(ROOT / "libstage0_launcher.so"))
        self.native.stage0_load.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.c_void_p, C.c_size_t]
        self.native.stage0_load.restype = C.c_int
        self.native.stage0_launch.argtypes = [C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_uint64, C.c_void_p, C.c_void_p, C.c_size_t]
        self.native.stage0_launch.restype = C.c_int
        self.native.stage1_launch.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.c_size_t, C.c_uint64, C.c_void_p, C.c_void_p, C.c_size_t]
        self.native.stage1_launch.restype = C.c_int
        self.metadata_lib = C.CDLL(str(native_path("libcubecl_stage0.so")))
        self.metadata_lib.stage0_metadata.argtypes = [C.c_uint64, C.POINTER(C.c_uint64), C.c_size_t]
        self.metadata_lib.stage0_metadata.restype = C.c_size_t
        self.metadata_lib.stage1_metadata.argtypes = [C.c_uint64, C.c_size_t, C.POINTER(C.c_uint64), C.c_size_t]
        self.metadata_lib.stage1_metadata.restype = C.c_size_t
        self.handle = C.c_void_p()
        with torch.cuda.device(self.device):
            torch.cuda.init()
            # Establish PyTorch's primary context using its native allocation path.
            torch.empty(1, device=self.device)
            if compile_ptx:
                ptx = nvrtc_compile((self.artifact_dir / "kernel.cu").read_text(), self.artifact_dir, self.device)
            else:
                ptx = (self.artifact_dir / "kernel.ptx").read_bytes()
            # Fail before launching if the actual artifact disagrees with the ABI.
            signature = re.search(rb"\.entry\s+" + self.manifest["entrypoint"].encode() + rb"\s*\((.*?)\)", ptx, re.S)
            if not signature or len(re.findall(rb"\.param\s+\.u64", signature[1])) != self.buffer_count + 1 or len(re.findall(rb"\.param", signature[1])) != self.buffer_count + 1:
                raise ValueError("PTX does not match the measured pointer ABI")
            (self.artifact_dir / "ptx-signature.txt").write_bytes(signature[0])
            error = C.create_string_buffer(2048)
            started = time.perf_counter()
            rc = self.native.stage0_load(ptx, self.manifest["entrypoint"].encode(), C.byref(self.handle), error, len(error))
            if rc:
                raise RuntimeError(error.value.decode())
            self.load_ms = (time.perf_counter() - started) * 1000

    def prepare(self, n):
        """Build metadata with CubeCL InfoBuilder; one small setup upload per N."""
        with torch.cuda.device(self.device), self._cache_lock:
            self._prepare(n)

    def _prepare(self, n):
        if not isinstance(n, int) or not 0 <= n <= 2147483647 * 256:
            raise ValueError("N must fit the fixed Stage 0 CUDA grid")
        if n not in self._metadata_cache:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare metadata before CUDA Graph capture")
            if self.buffer_count == 2:
                count = self.metadata_lib.stage0_metadata(n, None, 0)
                words = (C.c_uint64 * count)()
                self.metadata_lib.stage0_metadata(n, words, count)
            else:
                count = self.metadata_lib.stage1_metadata(n, self.buffer_count, None, 0)
                words = (C.c_uint64 * count)()
                self.metadata_lib.stage1_metadata(n, self.buffer_count, words, count)
            if count != self.buffer_count:
                raise RuntimeError("unexpected CubeCL metadata size")
            with torch.cuda.device(self.device):
                stream = torch.cuda.current_stream(self.device)
                tensor = torch.tensor(list(words), dtype=torch.uint64, device=self.device)
                ready = torch.cuda.Event()
                ready.record(stream)
                self._metadata_cache[n] = (tensor, ready)
        return self._metadata_cache[n][0]

    def prepare_capture(self, n):
        """Complete immutable metadata setup outside capture (event only)."""
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare_capture must run before capture")
            with self._cache_lock:
                self._prepare(n)
                ready = self._metadata_cache[n][1]
            ready.synchronize()
            with self._cache_lock:
                self._metadata_complete.add(n)

    def __call__(self, *tensors):
        if len(tensors) != self.buffer_count:
            raise ValueError("tensor argument count differs from manifest")
        *inputs, y = tensors
        x = inputs[0]
        if any(t.device != self.device for t in tensors):
            raise ValueError("input/output device must equal loaded kernel device")
        if any(t.dtype != torch.float32 or not t.is_contiguous() for t in tensors):
            raise ValueError("only contiguous CUDA float32 tensors are supported")
        if any(t.shape != y.shape for t in inputs):
            raise ValueError("input/output shapes must match")
        n = x.numel()
        if any(n and t.data_ptr() < y.data_ptr() + n * 4 and y.data_ptr() < t.data_ptr() + n * 4 for t in inputs):
            raise ValueError("input/output must not alias")
        with torch.cuda.device(self.device):
            with self._cache_lock:
                info = self._prepare(n)
            stream = torch.cuda.current_stream(self.device)
            # Completion is a property of immutable metadata, not a stream ID.
            # CUDA event queries themselves are prohibited during capture.
            with self._cache_lock:
                capturing = torch.cuda.is_current_stream_capturing()
                ready = self._metadata_cache[n][1]
                if n not in self._metadata_complete:
                    if capturing:
                        raise RuntimeError("call prepare_capture(N) before CUDA Graph capture")
                    if ready.query():
                        self._metadata_complete.add(n)
                    else:
                        stream.wait_event(ready)
            for tensor in tensors:
                tensor.record_stream(stream)
            info.record_stream(stream)
            error = C.create_string_buffer(2048)
            if self.buffer_count == 2:
                rc = self.native.stage0_launch(self.handle, x.data_ptr(), y.data_ptr(), info.data_ptr(), n, stream.cuda_stream, error, len(error))
            else:
                pointers = (C.c_uint64 * (self.buffer_count + 1))(*(t.data_ptr() for t in tensors), info.data_ptr())
                rc = self.native.stage1_launch(self.handle, pointers, len(pointers), n, stream.cuda_stream, error, len(error))
            if rc:
                raise RuntimeError(error.value.decode())
        return y
