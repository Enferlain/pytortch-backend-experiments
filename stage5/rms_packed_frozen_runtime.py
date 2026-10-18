"""Load copied frozen wqc PTX; never regenerate or write historical artifacts."""

import copy
import json
import shutil
import threading

from stage5.artifact import Kernel as PhysicalKernel
from stage5.rms_vector_trial_runtime import Kernel as WqcKernel, PLAN, admit, cpu_gate
from stage5.rms_packed_trial import OLD, OUT, sha

CACHE = {}
LOCK = threading.RLock()


class Kernel(WqcKernel):
    def __init__(self, directory, semantic, device):
        admit(semantic)
        PhysicalKernel.__init__(
            self, directory, semantic, device, PLAN, compile_ptx=False
        )
        if (
            self.manifest.get("mechanism") != "actual-cubek-rms-plane-vector4-v1"
            or self.manifest.get("operand_vector_width") != 4
        ):
            raise ValueError("frozen wqc manifest mismatch")


class CompiledKernel:
    def __init__(self, semantic, device):
        cpu_gate()
        self.semantic = copy.deepcopy(semantic)
        dtype = semantic["buffers"][0]["dtype"]
        metadata = json.loads((OLD / "gpu.json").read_text())
        assert metadata["complete"]
        from pathlib import Path

        original = Path(metadata["kernels"][dtype]["directory"])
        assert json.loads((original / "semantic.json").read_text()) == semantic
        self.directory = OUT / "frozen-wqc" / dtype
        self.directory.mkdir(parents=True, exist_ok=True)
        for name in (
            "kernel.cu",
            "kernel.ptx",
            "cubecl.ir",
            "semantic.json",
            "physical-plan.json",
            "manifest.json",
            "nvrtc.json",
        ):
            destination = self.directory / name
            if destination.exists():
                assert sha(destination) == sha(original / name)
            else:
                shutil.copyfile(original / name, destination)
        self.kernel = Kernel(self.directory, semantic, device)
        self.entrypoint = self.kernel.manifest["entrypoint"]

    def __call__(self, scalars, *tensors):
        return self.kernel(scalars, *tensors)


def compile_kernel(semantic, device):
    admit(semantic)
    key = (json.dumps(semantic, sort_keys=True), device)
    with LOCK:
        if key not in CACHE:
            CACHE[key] = CompiledKernel(semantic, device)
        return CACHE[key]
