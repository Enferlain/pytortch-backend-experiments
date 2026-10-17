"""Validated physical adapter on the existing compiler/current-stream bridge."""

import ctypes as C
from collections import OrderedDict
import copy
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
from stage2.indexing import LIMIT
from stage4.runtime import Kernel as ExistingKernel
from stage5.plan import PhysicalCandidate
from stage5.final_plan import STRATEGY as FINAL9, semantic_reason
from stage5.diagnostics import detailed_trace as detailed_trace, tracing

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
        "final_contract": digest((ROOT / "final_plan.py").read_bytes()),
    }


class Kernel(ExistingKernel):
    def __init__(self, directory, semantic, device, plan, *, compile_ptx=True):
        self.plan = PhysicalCandidate.from_dict(plan)
        if self.plan.strategy == FINAL9 and (why := semantic_reason(semantic)):
            raise ValueError("unsupported final physical contract: " + why)
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
        self._pointer_type = C.c_uint64 * (len(semantic["buffers"]) + 1)
        self._dimensions = (C.c_uint * 3)(*self.plan.workgroup)
        self._layouts = OrderedDict()
        self._call_pointers = None
        self._launch_count = 0
        self._buffers = tuple(semantic["buffers"])
        self._store_scopes = frozenset(
            op["iteration_scope"]
            for op in semantic["ops"]
            if semantic["schema"] == 3 and op["op"] == "store"
        )

    @property
    def launch_count(self):
        with self._batch_lock:
            return self._launch_count

    def _layout(self, scalars):
        """Bounded immutable descriptor results, never tensor/storage facts.

        Executed under _batch_lock. Runtime extents remain scalar metadata;
        changing scalars recomputes all original checked index expressions.
        """
        if scalars in self._layouts:
            self._layouts.move_to_end(scalars)
            return self._layouts[scalars]
        result = []
        for b in self._buffers:
            size = tuple(evaluate(v, scalars) for v in b["size"])
            stride = tuple(evaluate(v, scalars) for v in b["stride"])
            offset = evaluate(b["offset"], scalars)
            span = (
                offset + 1 + sum((s - 1) * t for s, t in zip(size, stride))
                if all(size)
                else 0
            )
            result.append((size, stride, offset, span, self.storage_dtypes[b["dtype"]]))
        if self.semantic["schema"] == 3:
            outer = evaluate(self.semantic["outer"], scalars)
            reduction = evaluate(self.semantic["reduction_extent"], scalars)
            if not 0 <= outer <= 2147483647 or not 0 <= reduction <= 1048576:
                raise ValueError(
                    "unsupported physical reduction extent (outer grid or reduction > 1048576)"
                )
            if self._store_scopes not in ({"outer"}, {"reduction"}):
                raise ValueError("reduction output logical domain mismatch")
            expected = (
                outer * reduction if self._store_scopes == {"reduction"} else outer
            )
            n = outer * 256
        else:
            n = evaluate(self.semantic["numel"], scalars)
            expected = None if self.semantic.get("effects", {}).get("mutation") else n
        effects = self.semantic.get("effects", {})
        if effects.get("rng") and not effects.get("mutation") and n > (1 << 31) - 1:
            raise ValueError(
                "unsupported RNG physical extent: signed-i32 logical counter domain"
            )
        if not 0 <= n <= 2147483647 * 256:
            raise ValueError("runtime numel exceeds physical grid")
        value = (tuple(result), n, expected)
        self._layouts[scalars] = value
        if len(self._layouts) > 64:
            self._layouts.popitem(last=False)
        return value

    def _validate(self, scalars, tensors):
        if len(tensors) != len(self._buffers) or len(scalars) != len(
            self.semantic["scalars"]
        ):
            raise ValueError("runtime argument count")
        scalars = tuple(int(s) for s in scalars)
        if any(not -LIMIT <= s <= LIMIT for s in scalars):
            raise ValueError("runtime scalar outside i64")
        if any(
            d.get("positive", False) and s <= 0
            for d, s in zip(self.semantic["scalars"], scalars)
        ):
            raise ValueError("runtime scalar violates positive index domain")
        layouts, n, expected = self._layout(scalars)
        if expected is not None and tensors[-1].numel() != expected:
            raise ValueError(
                "reduction output logical domain mismatch"
                if self.semantic["schema"] == 3
                else "pointwise output extent mismatch"
            )
        storage_pointers, pointers, lengths = [], [], []
        for tensor, (size, stride, offset, span, dtype) in zip(tensors, layouts):
            if tensor.device != self.device or tensor.dtype != dtype:
                raise ValueError("runtime tensor device/dtype mismatch")
            if tuple(tensor.shape) != size or tuple(tensor.stride()) != stride:
                raise ValueError(
                    "runtime tensor layout differs from guarded descriptor"
                )
            storage = tensor.untyped_storage()
            available = (
                storage.nbytes() // tensor.element_size() - tensor.storage_offset()
            )
            if offset < 0 or any(s < 0 for s in stride) or not 0 <= span <= available:
                raise ValueError("direct layout exceeds original storage")
            storage_pointers.append(storage.data_ptr())
            pointers.append(tensor.data_ptr())
            lengths.append(span)
        if n and storage_pointers[-1] in storage_pointers[:-1]:
            raise ValueError("output must be disjoint")
        return (tuple(lengths), (n, *scalars)), n, tuple(pointers)

    def __call__(self, scalars, *tensors):
        with self._batch_lock:
            # Same Stage4 capture veto and Stage2/3 runtime contracts. Only
            # descriptor-derived values are cached; tensors are revalidated.
            if torch.cuda.is_current_stream_capturing():
                raise NotImplementedError("Stage4 CUDA graph capture unsupported")
            key, n, pointers = self._validate(scalars, tensors)
            previous = self._call_pointers
            self._call_pointers = pointers
            try:
                result = self._enqueue(tensors, key, n)
                if n:
                    self._launch_count += 1
                return result
            finally:
                self._call_pointers = previous

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
        if self._call_pointers is None:
            raise RuntimeError("native enqueue requires a validated invocation")
        scalars = metadata_key[1][1:]
        pointers = self._pointer_type(*self._call_pointers, info.data_ptr())
        dims = self._dimensions
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
        if width > 1 or self.plan.strategy == FINAL9:
            from stage5.legality import reason

            why = reason(
                self.plan, self.semantic, metadata_key[1][1:], tensors, self.resources
            )
            if why:
                kind = "vector" if width > 1 else "final"
                raise ValueError(f"unsupported {kind} physical contract: " + why)
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
        semantic = copy.deepcopy(semantic)
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
        if tracing():
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
