"""Validated physical adapter on the existing compiler/current-stream bridge."""

import ctypes as C
from contextlib import contextmanager
import fcntl
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import threading

import torch
from stage0.bridge import native_path
from stage2.indexing import evaluate
from stage4.runtime import Kernel as ExistingKernel
from stage5.plan import PhysicalCandidate

ROOT = Path(__file__).resolve().parent
COMPILER_INVOCATIONS = 0
EVENTS = []


def digest(data):
    return hashlib.sha256(data).hexdigest()


@cache
def compiler_identity(device):
    from stage0.bridge import ROOT as BRIDGE_ROOT

    # Resolved library and CUDA header provenance are part of the cache, not
    # just the toolkit version string. The conservative options are fixed here.
    lib = os.environ.get("STAGE0_NVRTC")
    if not lib:
        candidates = sorted(
            {
                str(p.resolve())
                for p in Path(torch.__file__).parent.parent.glob(
                    "nvidia/**/libnvrtc.so.13"
                )
            }
        )
        if len(candidates) != 1:
            raise RuntimeError("select the exact NVRTC library with STAGE0_NVRTC")
        lib = candidates[0]
    props = torch.cuda.get_device_properties(device)
    headers = Path(torch.__file__).parent.parent / "nvidia/cu13/include"
    header_names = (
        "cuda_fp16.h",
        "cuda_fp16.hpp",
        "cuda_bf16.h",
        "cuda_bf16.hpp",
        "vector_types.h",
        "vector_functions.h",
    )
    header_hashes = {
        name: digest((headers / name).read_bytes()) for name in header_names
    }
    builtins = {
        str(p.resolve()): digest(p.read_bytes())
        for p in Path(lib).parent.glob("libnvrtc-builtins.so*")
        if p.is_file()
    }
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    return {
        "compiler": digest(native_path("cubecl-stage0").read_bytes()),
        "metadata_library": digest(native_path("libcubecl_stage0.so").read_bytes()),
        "launcher": digest((BRIDGE_ROOT / "libstage0_launcher.so").read_bytes()),
        "fp_patch": digest((ROOT.parent / "fp_hardening/cubecl-fp.patch").read_bytes()),
        "nvrtc": {"path": lib, "sha256": digest(Path(lib).read_bytes())},
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "cuda_header_sha256": header_hashes,
        "nvrtc_builtins": builtins,
        "driver_snapshot": driver,
        "torch_commit": torch.version.git_version,
        "device": str(props.uuid),
        "capability": [props.major, props.minor],
        "math": ["--fmad=false", "precision_cast_barriers", "conservative_fp_patch"],
        "adapter": digest(Path(__file__).read_bytes()),
    }


class Kernel(ExistingKernel):
    def __init__(self, directory, semantic, device, plan, *, compile_ptx=True):
        self.plan = PhysicalCandidate.from_dict(plan)
        super().__init__(directory, semantic, device, compile_ptx=compile_ptx)
        if (
            self.manifest["descriptor_schema"] != 5
            or self.manifest["physical_plan"] != self.plan.to_dict()
        ):
            raise ValueError("physical manifest mismatch")
        launch = self.native.stage5_launch
        launch.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_uint64),
            C.c_size_t,
            C.c_uint64,
            C.POINTER(C.c_uint),
            C.c_uint,
            C.c_uint,
            C.c_void_p,
            C.c_void_p,
            C.c_size_t,
        ]
        launch.restype = C.c_int
        query = self.native.stage5_resources
        query.argtypes = [
            C.c_void_p,
            C.c_uint,
            C.c_uint,
            C.POINTER(C.c_int),
            C.c_void_p,
            C.c_size_t,
        ]
        query.restype = C.c_int
        error = C.create_string_buffer(2048)
        values = (C.c_int * 6)()
        with torch.cuda.device(self.device):
            rc = query(
                self.handle,
                self.plan.units,
                self.manifest["dynamic_shared_bytes"],
                values,
                error,
                len(error),
            )
        if rc:
            raise RuntimeError(error.value.decode())
        self.resources = dict(
            zip(
                (
                    "device_max_threads",
                    "plane_width",
                    "multiprocessors",
                    "threads_per_sm",
                    "function_max_threads",
                    "active_blocks_per_sm",
                ),
                values,
            )
        )
        if self.plan.units > min(values[0], values[4]) or (
            self.plan.plane_width and self.plan.plane_width != values[1]
        ):
            raise ValueError("unsupported target physical plan")
        self.resources["theoretical_occupancy"] = (
            values[5] * self.plan.units / values[3]
        )
        self._prepared = None
        self._for_batch = False
        self._batch_lock = threading.RLock()

    def __call__(self, scalars, *tensors):
        with self._batch_lock:
            return super().__call__(scalars, *tensors)

    def geometry(self, scalars):
        if self.semantic["schema"] == 3:
            n = evaluate(self.semantic["outer"], scalars)
            span = self.plan.outputs_per_workgroup
        else:
            n = evaluate(self.semantic["numel"], scalars)
            span = (
                self.plan.units * self.plan.vector_width * self.plan.elements_per_unit
            )
        return (n + span - 1) // span

    def _launch_native(self, tensors, info, n, stream, error, metadata_key):
        scalars = metadata_key[1][1:]
        pointers = (C.c_uint64 * (len(tensors) + 1))(
            *(t.data_ptr() for t in tensors), info.data_ptr()
        )
        dims = (C.c_uint * 3)(*self.plan.workgroup)
        grid = self.geometry(scalars)
        # Kept alive explicitly for prepared timing; ordinary launches still use
        # the common readiness waits and allocator record_stream contract.
        if self._for_batch:
            self._prepared = (tensors, info, pointers, dims, grid, stream)
        return self.native.stage5_launch(
            self.handle,
            pointers,
            len(pointers),
            grid,
            dims,
            self.manifest["dynamic_shared_bytes"],
            1,
            stream.cuda_stream,
            error,
            len(error),
        )

    def _enqueue(self, tensors, metadata_key, n):
        width = self.plan.vector_width
        if width > 1:
            from stage5.legality import reason

            why = reason(
                self.plan, self.semantic, metadata_key[1][1:], tensors, self.resources
            )
            if why:
                raise ValueError("unsupported vector physical contract: " + why)
        return super()._enqueue(tensors, metadata_key, n)

    @contextmanager
    def prepared(self, scalars, *tensors):
        """Offline timing scope; ordinary launches retain no user tensor refs."""
        if self.semantic.get("effects", {}).get("rng") or self.semantic.get(
            "effects", {}
        ).get("mutation"):
            raise ValueError("prepared repeated launch only permits pure kernels")
        with self._batch_lock, torch.cuda.device(self.device):
            if self._for_batch:
                raise ValueError("nested prepared timing scope")
            self._for_batch = True
            try:
                self(scalars, *tensors)
                yield self
            finally:
                self._for_batch = False
                self._prepared = None

    def batch(self, repeat=1):
        if self.semantic.get("effects", {}).get("rng") or self.semantic.get(
            "effects", {}
        ).get("mutation"):
            raise ValueError("prepared repeated launch only permits pure kernels")
        if not isinstance(repeat, int) or not 1 <= repeat <= 10000:
            raise ValueError("invalid prepared batch repeat")
        if self._prepared is None:
            raise ValueError("prepare with a validated launch first")
        tensors, info, pointers, dims, grid, stream = self._prepared
        if torch.cuda.current_stream(self.device).cuda_stream != stream.cuda_stream:
            raise ValueError("prepare again after changing current stream")
        error = C.create_string_buffer(2048)
        rc = self.native.stage5_launch(
            self.handle,
            pointers,
            len(pointers),
            grid,
            dims,
            self.manifest["dynamic_shared_bytes"],
            repeat,
            stream.cuda_stream,
            error,
            len(error),
        )
        if rc:
            raise RuntimeError(error.value.decode())


class CompiledKernel:
    def __init__(self, semantic, device, plan, identity=None):
        global COMPILER_INVOCATIONS
        plan = PhysicalCandidate.from_dict(plan).to_dict()
        self.semantic = semantic
        identity = identity or compiler_identity(device)
        packet = {"schema": 5, "semantic": semantic, "physical_plan": plan}
        raw = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
        self.entrypoint = "cubecl5_" + digest(raw)[:16]
        packet["entrypoint"] = self.entrypoint
        raw = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
        key = digest(raw + json.dumps(identity, sort_keys=True).encode())
        self.directory = ROOT / "cache/artifacts" / key
        self.directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        self.cache_hit = False
        names = (
            "descriptor.json",
            "semantic.json",
            "physical-plan.json",
            "manifest.json",
            "kernel.cu",
            "kernel.ptx",
            "cubecl.ir",
            "nvrtc.json",
        )
        with (self.directory / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            integrity = self.directory / "integrity.json"
            if integrity.exists():
                saved = json.loads(integrity.read_text())
                self.cache_hit = (
                    saved["identity"] == identity
                    and saved["descriptor"] == digest(raw)
                    and set(saved["files"]) == set(names)
                    and all(
                        (self.directory / name).is_file()
                        and digest((self.directory / name).read_bytes()) == value
                        for name, value in saved["files"].items()
                    )
                )
            if not self.cache_hit:
                for name, value in (
                    ("descriptor", packet),
                    ("semantic", semantic),
                    ("physical-plan", plan),
                ):
                    (self.directory / (name + ".json")).write_text(
                        json.dumps(value, indent=2, sort_keys=True)
                    )
                proc = subprocess.run(
                    [
                        str(native_path("cubecl-stage0")),
                        str(self.directory),
                        "cpp",
                        str(self.directory / "descriptor.json"),
                    ],
                    capture_output=True,
                    text=True,
                )
                (self.directory / "compiler.log").write_text(proc.stdout + proc.stderr)
                if proc.returncode:
                    raise RuntimeError(
                        "CubeCL physical compiler failure: " + proc.stderr
                    )
                COMPILER_INVOCATIONS += 1
            self.kernel = Kernel(
                self.directory, semantic, device, plan, compile_ptx=not self.cache_hit
            )
            if not self.cache_hit:
                payload = {
                    "identity": identity,
                    "descriptor": digest(raw),
                    "files": {
                        n: digest((self.directory / n).read_bytes()) for n in names
                    },
                }
                temp = integrity.with_suffix(".tmp")
                temp.write_text(json.dumps(payload, indent=2))
                temp.replace(integrity)
        self.load_compile_ms = (time.perf_counter() - started) * 1000
        (self.directory / "resources.json").write_text(
            json.dumps(
                {
                    "attributes": self.kernel.attributes,
                    "resources": self.kernel.resources,
                    "module_load_ms": self.kernel.load_ms,
                    "cache_hit": self.cache_hit,
                    "classification": "actual production CUDA driver JIT resource queries",
                    "ptx_sha256": digest((self.directory / "kernel.ptx").read_bytes()),
                    "identity": identity,
                },
                indent=2,
            )
        )
        EVENTS.append(
            {
                "event": "artifact",
                "entrypoint": self.entrypoint,
                "directory": str(self.directory),
                "cache_hit": self.cache_hit,
                "elapsed_ms": self.load_compile_ms,
                "attributes": self.kernel.attributes,
                "resources": self.kernel.resources,
            }
        )

    def __call__(self, scalars, *tensors):
        result = self.kernel(scalars, *tensors)
        EVENTS.append(
            {
                "event": "launch",
                "entrypoint": self.entrypoint,
                "grid": [self.kernel.geometry(scalars), 1, 1],
                "block": list(self.kernel.plan.workgroup),
                "stream": torch.cuda.current_stream(self.kernel.device).cuda_stream,
                "pointers": [t.data_ptr() for t in tensors],
            }
        )
        return result
