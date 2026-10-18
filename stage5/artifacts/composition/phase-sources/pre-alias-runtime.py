"""Frozen-PTX composition, following the existing rms_packed_frozen loader."""

import ast
import copy
from functools import cache
from pathlib import Path
import shutil
import threading

from stage5.artifact import Kernel as PhysicalKernel
from stage5.column_layout_runtime import Kernel as ColumnKernel
from stage5.rms_packed_trial_runtime import Kernel as PackedKernel, PLAN, cpu_gate
from stage5.composition_trial import ROOT, OUT, read, sha, role

CACHE = {}
LOCK = threading.RLock()
MODES = ("ordinary", "rms", "column", "both")


def gate():
    cpu_gate()
    cpu = read(OUT / "cpu.json")
    assert cpu["complete"] and not cpu["loader_existed"]
    assert len(cpu["rms"]) == 36 and len(cpu["column"]) == 21
    assert cpu["source_sha256"] == sha(OUT / "cpu-source.py")
    def function(path):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body
                             if isinstance(n, ast.FunctionDef) and n.name == "cpu"))
    assert function(OUT / "cpu-source.py") == function(ROOT / "stage5/composition_trial.py")
    assert cpu["reference_sha256"] == sha(ROOT / "stage5/artifacts/reuse-first-rms-vector4/reference.pt")


class FrozenPacked(PackedKernel):
    def __init__(self, directory, semantic, device):
        # Translate the existing frozen-wqc loader pattern to packed RMS; retain
        # PackedKernel._validate, PhysicalKernel enqueue and the exact old PTX.
        from stage5.rms_packed_trial_runtime import admit, MECHANISM

        admit(semantic)
        PhysicalKernel.__init__(self, directory, semantic, device, PLAN, compile_ptx=False)
        if (self.manifest.get("mechanism") != MECHANISM
            or self.manifest.get("operand_vector_width") != 4
            or self.manifest.get("storage_vector_width") != 4
            or self.manifest.get("metadata") != {"scalar_count": 3, "bytes": 40}):
            raise ValueError("frozen packed RMS manifest mismatch")


@cache
def source(kind, dtype):
    if kind == "rms":
        packet = read(ROOT / "stage5/artifacts/reuse-first-rms-packed-trial/gpu.json")
        return Path(packet["kernels"][dtype]["directory"])
    packet = read(ROOT / "stage5/artifacts/column-layout/post-review-gpu.json")
    return Path(next(k["directory"] for k in packet["kernels"]
                     if k["semantic"]["buffers"][0]["dtype"] == dtype))


class Dispatch:
    def __init__(self, semantic, device, kind):
        self.semantic = semantic
        self.kind = kind
        original = source(kind, semantic["buffers"][0]["dtype"])
        assert read(original / "semantic.json") == semantic
        self.directory = OUT / "frozen" / (kind + "-" + semantic["buffers"][0]["dtype"])
        self.directory.mkdir(parents=True, exist_ok=True)
        for name in ("kernel.cu", "kernel.ptx", "cubecl.ir", "semantic.json",
                     "physical-plan.json", "manifest.json", "nvrtc.json"):
            target = self.directory / name
            if target.exists():
                assert sha(target) == sha(original / name), name
            else:
                shutil.copyfile(original / name, target)
        if kind == "rms":
            self.kernel = FrozenPacked(self.directory, semantic, device)
        else:
            self.kernel = ColumnKernel(self.directory, semantic, device, compile_ptx=False)
        self.entrypoint = self.kernel.manifest["entrypoint"]

    def __call__(self, scalars, *tensors):
        return self.kernel(scalars, *tensors)


def compile_kernel(semantic, device, mode):
    semantic = copy.deepcopy(semantic)
    if mode not in MODES:
        raise ValueError("unknown bounded composition mode")
    kind = role(semantic)
    if kind == "unadmitted":
        raise ValueError("composition requires existing exact admitted components")
    if mode not in (kind, "both"):
        from stage5.runtime import compile_kernel as ordinary

        return ordinary(semantic, device, policy="first_use")
    with LOCK:
        import json

        key = (json.dumps(semantic, sort_keys=True), device)
        if key not in CACHE:
            gate()
            CACHE[key] = Dispatch(semantic, device, kind)
        return CACHE[key]
