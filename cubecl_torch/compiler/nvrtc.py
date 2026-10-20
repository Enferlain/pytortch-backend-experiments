"""NVRTC compilation of the unmodified CubeCL-emitted CUDA source."""

import ctypes as C
import json
import os
import time
from functools import cache
from pathlib import Path

import torch

OPTIONS = ("--std=c++17", "--fmad=false")


@cache
def library_path():
    """The NVRTC library: ``CUBECL_TORCH_NVRTC``/``STAGE0_NVRTC`` or PyTorch's own."""
    path = os.environ.get("CUBECL_TORCH_NVRTC") or os.environ.get("STAGE0_NVRTC")
    if path:
        return path
    if torch.version.cuda is None:
        raise RuntimeError("CubeCL compilation requires a CUDA-enabled PyTorch")
    major = torch.version.cuda.split(".")[0]
    candidates = sorted(
        {
            str(p.resolve())
            for p in Path(torch.__file__).parent.parent.glob(
                f"nvidia/**/libnvrtc.so.{major}"
            )
        }
    )
    if len(candidates) != 1:
        raise RuntimeError("select the exact NVRTC library with CUBECL_TORCH_NVRTC")
    return candidates[0]


def include_dirs():
    return [Path(torch.__file__).parent.parent / "nvidia/cu13/include"]


@cache
def _library():
    lib = C.CDLL(library_path())
    lib.nvrtcCreateProgram.argtypes = [
        C.POINTER(C.c_void_p),
        C.c_char_p,
        C.c_char_p,
        C.c_int,
        C.c_void_p,
        C.c_void_p,
    ]
    lib.nvrtcCompileProgram.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)]
    lib.nvrtcGetProgramLogSize.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
    lib.nvrtcGetProgramLog.argtypes = [C.c_void_p, C.c_void_p]
    lib.nvrtcGetPTXSize.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
    lib.nvrtcGetPTX.argtypes = [C.c_void_p, C.c_void_p]
    lib.nvrtcDestroyProgram.argtypes = [C.POINTER(C.c_void_p)]
    return lib


def compile_ptx(source, directory, capability):
    """Compile ``source`` for ``capability``; writes nvrtc.log and nvrtc.json."""
    lib = _library()

    def check(rc):
        if rc:
            raise RuntimeError(f"NVRTC error {rc}")

    program = C.c_void_p()
    check(
        lib.nvrtcCreateProgram(
            C.byref(program), source.encode(), b"cubecl.cu", 0, None, None
        )
    )
    major, minor = capability
    values = [
        f"--gpu-architecture=compute_{major}{minor}".encode(),
        *(o.encode() for o in OPTIONS),
        *(f"--include-path={p}".encode() for p in include_dirs()),
    ]
    options = (C.c_char_p * len(values))(*values)
    started = time.perf_counter()
    try:
        rc = lib.nvrtcCompileProgram(program, len(options), options)
        size = C.c_size_t()
        check(lib.nvrtcGetProgramLogSize(program, C.byref(size)))
        log = C.create_string_buffer(size.value)
        check(lib.nvrtcGetProgramLog(program, log))
        (directory / "nvrtc.log").write_bytes(log.value)
        if rc:
            raise RuntimeError(f"NVRTC error {rc}: {log.value.decode()}")
        check(lib.nvrtcGetPTXSize(program, C.byref(size)))
        ptx = C.create_string_buffer(size.value)
        check(lib.nvrtcGetPTX(program, ptx))
        (directory / "nvrtc.json").write_text(
            json.dumps(
                {
                    "nvrtc_compile_ms": (time.perf_counter() - started) * 1000,
                    "options": [v.decode() for v in values],
                    "library": library_path(),
                },
                indent=2,
            )
        )
        return ptx.value
    finally:
        lib.nvrtcDestroyProgram(C.byref(program))
