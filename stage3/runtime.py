"""Reduction plan adapter on the established compiler/metadata/enqueue bridge."""

import hashlib
import ctypes as C
import json
import subprocess
import threading
from pathlib import Path
import torch
from stage0.bridge import native_path
from stage1.plan import PhysicalKernelPlan
from stage2.indexing import evaluate
from stage2.runtime import Kernel as PointwiseKernel
from stage3.plan import ReductionPlan

ROOT = Path(__file__).resolve().parent
CACHE = {}
EVENTS = []
LOCK = threading.RLock()
COMPILER_INVOCATIONS = 0
KERNEL_LAUNCHES = 0


class Kernel(PointwiseKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        query = self.native.stage0_kernel_attributes
        query.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_void_p, C.c_size_t]
        query.restype = C.c_int
        values = (C.c_int * 3)()
        error = C.create_string_buffer(2048)
        with torch.cuda.device(self.device):
            if query(self.handle, values, error, len(error)):
                raise RuntimeError(error.value.decode())
        self.attributes = dict(
            zip(
                (
                    "registers_per_thread",
                    "static_shared_bytes",
                    "local_bytes_per_thread",
                ),
                values,
            )
        )

    def launch_extent(self, scalars, tensors):
        if self.semantic["schema"] != 3:
            return super().launch_extent(scalars, tensors)
        outer = evaluate(self.semantic["outer"], scalars)
        reduction = evaluate(self.semantic["reduction_extent"], scalars)
        if not 0 <= outer <= 2147483647 or not 0 <= reduction <= 1048576:
            raise ValueError(
                "unsupported physical reduction extent (outer grid or reduction > 1048576)"
            )
        scopes = {
            op["iteration_scope"] for op in self.semantic["ops"] if op["op"] == "store"
        }
        expected = outer * reduction if scopes == {"reduction"} else outer
        if scopes not in ({"outer"}, {"reduction"}) or tensors[-1].numel() != expected:
            raise ValueError("reduction output logical domain mismatch")
        return outer * 256


class CompiledKernel:
    def __init__(
        self, semantic, device, plan, *, root=ROOT, kernel_type=Kernel, events=EVENTS
    ):
        global COMPILER_INVOCATIONS
        (ReductionPlan if semantic["schema"] == 3 else PhysicalKernelPlan).from_dict(
            plan
        )
        self.semantic = semantic
        self.events = events
        descriptor = {**semantic, "physical_plan": plan}
        prefix = f"cubecl{semantic['schema']}_"
        descriptor["entrypoint"] = (
            prefix
            + hashlib.sha256(
                json.dumps(descriptor, sort_keys=True).encode()
            ).hexdigest()[:16]
        )
        raw = json.dumps(descriptor, sort_keys=True)
        self.directory = (
            root / "artifacts/kernels" / hashlib.sha256(raw.encode()).hexdigest()[:16]
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        for name, value in (
            ("descriptor", descriptor),
            ("semantic", semantic),
            ("physical-plan", plan),
        ):
            (self.directory / (name + ".json")).write_text(
                json.dumps(value, indent=2, sort_keys=True)
            )
        process = subprocess.run(
            [
                str(native_path("cubecl-stage0")),
                str(self.directory),
                "cpp",
                str(self.directory / "descriptor.json"),
            ],
            capture_output=True,
            text=True,
        )
        (self.directory / "compiler.log").write_text(process.stdout + process.stderr)
        if process.returncode:
            raise RuntimeError("CubeCL compiler failure: " + process.stderr)
        COMPILER_INVOCATIONS += 1
        self.kernel = kernel_type(self.directory, semantic, device)
        self.entrypoint = descriptor["entrypoint"]
        manifest = json.loads((self.directory / "manifest.json").read_text())
        manifest["driver_function_attributes"] = self.kernel.attributes
        manifest["cubecl_patch_sha256"] = hashlib.sha256(
            (ROOT.parent / "fp_hardening/cubecl-fp.patch").read_bytes()
        ).hexdigest()
        manifest["descriptor_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
        manifest["compiler_binary_sha256"] = hashlib.sha256(
            native_path("cubecl-stage0").read_bytes()
        ).hexdigest()
        (self.directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self.events.append(
            {
                "event": "compile",
                "directory": str(self.directory),
                "entrypoint": self.entrypoint,
                "ptx_sha256": hashlib.sha256(
                    (self.directory / "kernel.ptx").read_bytes()
                ).hexdigest(),
            }
        )

    def __call__(self, scalars, *tensors):
        global KERNEL_LAUNCHES
        with torch.profiler.record_function("stage3::CubeCL.launch"):
            result = self.kernel(scalars, *tensors)
        extent = self.kernel.launch_extent(scalars, tensors)
        KERNEL_LAUNCHES += int(extent != 0)
        self.events.append(
            {
                "event": "launch",
                "entrypoint": self.entrypoint,
                "scalars": list(map(int, scalars)),
                "pointers": [t.data_ptr() for t in tensors],
                "shapes": [list(t.shape) for t in tensors],
                "strides": [list(t.stride()) for t in tensors],
                "storage_offsets": [t.storage_offset() for t in tensors],
                "grid": [(extent + 255) // 256, 1, 1],
                "metadata_scalars": [extent, *map(int, scalars)],
                "block": [256, 1, 1],
                "shared_bytes": self.kernel.manifest["dynamic_shared_bytes"],
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
