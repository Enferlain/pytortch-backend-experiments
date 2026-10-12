"""Schema-2 metadata adapter; reuse Stage 0 compilation and native enqueue."""

import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import torch
from stage0.bridge import Kernel as BaselineKernel, native_path
from stage1.plan import PhysicalKernelPlan
from stage2.indexing import evaluate, LIMIT

ROOT = Path(__file__).resolve().parent
CACHE = {}
EVENTS = []
LOCK = threading.RLock()
COMPILER_INVOCATIONS = 0
KERNEL_LAUNCHES = 0
DTYPES = {
    str(t).removeprefix("torch."): t
    for t in (torch.float32, torch.float16, torch.bfloat16, torch.bool)
}


def bridge_description(semantic, plan):
    PhysicalKernelPlan.from_dict(plan)
    d = {**semantic, "physical_plan": plan}
    d["entrypoint"] = (
        "cubecl2_"
        + hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
    )
    return d


class Kernel(BaselineKernel):
    def __init__(self, directory, semantic, device):
        self.semantic = semantic
        headers = Path(torch.__file__).parent.parent / "nvidia/cu13/include"
        super().__init__(directory, device, compile_ptx=True, include_dirs=[headers])
        self.native.stage2_launch.argtypes = self.native.stage1_launch.argtypes
        self.native.stage2_launch.restype = C.c_int
        self.metadata_lib.stage2_metadata.argtypes = [
            C.POINTER(C.c_uint64),
            C.c_size_t,
            C.POINTER(C.c_int64),
            C.c_size_t,
            C.POINTER(C.c_uint64),
            C.c_size_t,
        ]
        self.metadata_lib.stage2_metadata.restype = C.c_size_t

    def prepare(self, n):
        raise NotImplementedError(
            "Stage 2 metadata requires tensor layouts and scalar arguments"
        )

    def prepare_capture(self, n):
        raise NotImplementedError("Stage 2 CUDA graph capture is unsupported")

    def _prepare(self, key):
        lengths, scalars = key
        if key not in self._metadata_cache:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare metadata before capture")
            ns = (C.c_uint64 * len(lengths))(*lengths)
            ss = (C.c_int64 * len(scalars))(*scalars)
            count = self.metadata_lib.stage2_metadata(ns, len(ns), ss, len(ss), None, 0)
            if count * 8 != self.manifest["metadata"]["bytes"]:
                raise RuntimeError("InfoBuilder metadata ABI mismatch")
            words = (C.c_uint64 * count)()
            if (
                self.metadata_lib.stage2_metadata(
                    ns, len(ns), ss, len(ss), words, count
                )
                != count
            ):
                raise RuntimeError("metadata serialization failed")
            tensor = torch.tensor(list(words), dtype=torch.uint64, device=self.device)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            self._metadata_cache[key] = (tensor, ready)
        return self._metadata_cache[key][0]

    def __call__(self, scalars, *tensors):
        if len(tensors) != len(self.semantic["buffers"]) or len(scalars) != len(
            self.semantic["scalars"]
        ):
            raise ValueError("runtime argument count")
        scalars = tuple(int(x) for x in scalars)
        if any(not -LIMIT <= s <= LIMIT for s in scalars):
            raise ValueError("runtime scalar outside i64")
        if any(
            declaration.get("positive", False) and value <= 0
            for declaration, value in zip(self.semantic["scalars"], scalars)
        ):
            raise ValueError("runtime scalar violates positive index domain")
        n = evaluate(self.semantic["numel"], scalars)
        if not 0 <= n <= 2147483647 * 256:
            raise ValueError("runtime numel exceeds physical grid")
        lengths = []
        for t, b in zip(tensors, self.semantic["buffers"]):
            if t.device != self.device or t.dtype != DTYPES[b["dtype"]]:
                raise ValueError("runtime tensor device/dtype mismatch")
            expected_size = tuple(evaluate(v, scalars) for v in b["size"])
            expected_stride = tuple(evaluate(v, scalars) for v in b["stride"])
            if tuple(t.shape) != expected_size or tuple(t.stride()) != expected_stride:
                raise ValueError(
                    "runtime tensor layout differs from guarded descriptor"
                )
            # Wrapper data_ptr is already offset to the passed view. Inductor offsets
            # refer to that pointer; never add Tensor.storage_offset twice.
            offset = evaluate(b["offset"], scalars)
            span = (
                offset
                + 1
                + sum(
                    (size - 1) * stride
                    for size, stride in zip(expected_size, expected_stride)
                )
                if t.numel()
                else 0
            )
            available = (
                t.untyped_storage().nbytes() // t.element_size() - t.storage_offset()
            )
            if (
                offset < 0
                or any(s < 0 for s in expected_stride)
                or not 0 <= span <= available
            ):
                raise ValueError("direct layout exceeds original storage")
            lengths.append(span)
        y = tensors[-1]
        if y.numel() != n:
            raise ValueError("pointwise output extent mismatch")
        if n and any(
            x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
            for x in tensors[:-1]
        ):
            raise ValueError("output must be disjoint")
        key = (tuple(lengths), (n, *scalars))
        return self._enqueue(tensors, key, n)


class CompiledKernel:
    def __init__(self, semantic, device, plan):
        global COMPILER_INVOCATIONS
        d = bridge_description(semantic, plan)
        raw = json.dumps(d, sort_keys=True)
        directory = (
            ROOT / "artifacts/kernels" / hashlib.sha256(raw.encode()).hexdigest()[:16]
        )
        directory.mkdir(parents=True, exist_ok=True)
        for name, v in [
            ("descriptor", d),
            ("semantic", semantic),
            ("physical-plan", plan),
        ]:
            (directory / (name + ".json")).write_text(
                json.dumps(v, indent=2, sort_keys=True)
            )
        proc = subprocess.run(
            [
                str(native_path("cubecl-stage0")),
                str(directory),
                "cpp",
                str(directory / "descriptor.json"),
            ],
            capture_output=True,
            text=True,
        )
        (directory / "compiler.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            raise RuntimeError("CubeCL compiler failure: " + proc.stderr)
        COMPILER_INVOCATIONS += 1
        self.kernel = Kernel(directory, semantic, device)
        self.directory = directory
        self.entrypoint = d["entrypoint"]
        EVENTS.append(
            {
                "event": "compile",
                "directory": str(directory),
                "entrypoint": self.entrypoint,
                "ptx_sha256": hashlib.sha256(
                    (directory / "kernel.ptx").read_bytes()
                ).hexdigest(),
            }
        )

    def __call__(self, scalars, *tensors):
        global KERNEL_LAUNCHES
        with torch.profiler.record_function("stage2::CubeCL.launch"):
            result = self.kernel(scalars, *tensors)
        n = evaluate(self.kernel.semantic["numel"], scalars)
        KERNEL_LAUNCHES += int(n != 0)
        EVENTS.append(
            {
                "event": "launch",
                "entrypoint": self.entrypoint,
                "scalars": list(scalars),
                "numel": n,
                "pointers": [t.data_ptr() for t in tensors],
                "storage_offsets": [t.storage_offset() for t in tensors],
                "strides": [list(t.stride()) for t in tensors],
                "shapes": [list(t.shape) for t in tensors],
                "metadata_scalars": [n, *map(int, scalars)],
                "grid": [(n + 255) // 256, 1, 1],
                "block": [256, 1, 1],
                "stream": torch.cuda.current_stream(self.kernel.device).cuda_stream,
            }
        )
        return result


def compile_kernel(semantic, device, plan):
    key = (
        json.dumps(semantic, sort_keys=True),
        device,
        json.dumps(plan, sort_keys=True),
    )
    with LOCK:
        if key not in CACHE:
            CACHE[key] = CompiledKernel(semantic, device, plan)
        return CACHE[key]
