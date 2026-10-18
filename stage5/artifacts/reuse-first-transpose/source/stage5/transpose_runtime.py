"""Explicit-only perpendicular component; existing validation and enqueue ownership."""

import copy
import ast
import json
import subprocess
import threading

from stage5.artifact import Kernel as ExistingKernel, compiler_identity
from stage5.plan import PhysicalCandidate
from stage5.transpose_trial import OUT, ROOT, sha
from stage5.transpose_model import SHAPES, width

PLAN = PhysicalCandidate("scalar_pointwise_baseline").to_dict()
MECHANISM = "cubecl-perpendicular-virtual-v1"
BINARY = ROOT / "stage5/cache/reuse-first-rms/target/release/stage5-transpose-probe"
TEMPLATES = ROOT / "stage5/transpose_probe/src/templates.json"
CACHE = {}
LOCK = threading.RLock()


def admit(semantic):
    if json.dumps(semantic, sort_keys=True) not in {
        json.dumps(t, sort_keys=True) for t in json.loads(TEMPLATES.read_text())
    }:
        raise ValueError("transpose requires an exact captured contiguous template")


class Kernel(ExistingKernel):
    def __init__(self, directory, semantic, device, vector):
        admit(semantic)
        self.vector = vector
        super().__init__(directory, semantic, device, PLAN, compile_ptx=True)
        if (
            self.manifest.get("mechanism") != MECHANISM
            or self.manifest.get("storage_vector_width") != vector
        ):
            raise ValueError("transpose manifest mechanism/width mismatch")
        if self.manifest.get("metadata") != {"scalar_count": 3, "bytes": 40}:
            raise ValueError("transpose metadata ABI mismatch")

    def _validate(self, scalars, tensors):
        if len(scalars) != 2 or (int(scalars[1]), int(scalars[0])) not in SHAPES:
            raise ValueError("transpose extent outside bounded trial")
        n, m = map(int, scalars)
        if width(m, n, self.semantic["buffers"][0]["dtype"]) != self.vector:
            raise ValueError("transpose width/divisibility mismatch")
        result = super()._validate(scalars, tensors)
        if any(t.data_ptr() % (self.vector * t.element_size()) for t in tensors):
            raise ValueError("transpose vector alignment")
        return result

    def geometry(self, scalars):
        # Same upstream number of workers, one worker per vector group of rows.
        return (int(scalars[0]) // self.vector + 255) // 256


class Compiled:
    def __init__(self, semantic, device, vector):
        admit(semantic)
        gate = json.loads((OUT / "cpu.json").read_text())
        if not gate["complete"] or gate["gpu_source_existed"]:
            raise ValueError("CPU address gate incomplete")
        snapshot = OUT / "phase-sources/transpose_model.py"
        if gate["model_sha256"] != sha(snapshot) or ast.dump(
            ast.parse(snapshot.read_text())
        ) != ast.dump(ast.parse((ROOT / "stage5/transpose_model.py").read_text())):
            raise ValueError("CPU model changed after qualification")
        self.semantic = copy.deepcopy(semantic)
        self.vector = vector
        identity = dict(
            compiler_identity(device),
            diagnostic_binary=sha(BINARY),
            diagnostic_adapter=sha(__file__),
            cpu_gate=sha(OUT / "cpu.json"),
        )
        packet = dict(
            schema=5,
            semantic=semantic,
            physical_plan=PLAN,
            mechanism=MECHANISM,
            storage_vector_width=vector,
        )
        import hashlib

        key = hashlib.sha256(
            json.dumps([packet, identity], sort_keys=True).encode()
        ).hexdigest()
        self.entrypoint = "cubecl5_transpose_" + key[:16]
        packet["entrypoint"] = self.entrypoint
        self.directory = OUT / "candidate" / key
        self.directory.mkdir(parents=True, exist_ok=True)
        for name, value in (
            ("descriptor", packet),
            ("semantic", semantic),
            ("physical-plan", PLAN),
            ("identity", identity),
        ):
            (self.directory / (name + ".json")).write_text(
                json.dumps(value, indent=2) + "\n"
            )
        p = subprocess.run(
            [str(BINARY), str(self.directory), str(self.directory / "descriptor.json")],
            capture_output=True,
            text=True,
        )
        (self.directory / "compiler.log").write_text(p.stdout + p.stderr)
        if p.returncode:
            raise RuntimeError("transpose compile failed: " + p.stderr)
        self.kernel = Kernel(self.directory, semantic, device, vector)
        (self.directory / "resources.json").write_text(
            json.dumps(
                dict(
                    attributes=self.kernel.attributes,
                    resources=self.kernel.resources,
                    ptx_sha256=sha(self.directory / "kernel.ptx"),
                ),
                indent=2,
            )
            + "\n"
        )

    def __call__(self, scalars, *tensors):
        return self.kernel(scalars, *tensors)


class Dispatch:
    def __init__(self, semantic, device):
        admit(semantic)
        self.semantic, self.device = copy.deepcopy(semantic), device
        self.last = None

    def __call__(self, scalars, *tensors):
        if len(scalars) != 2 or (int(scalars[1]), int(scalars[0])) not in SHAPES:
            raise ValueError("transpose extent outside bounded trial")
        n, m = map(int, scalars)
        vector = width(m, n, self.semantic["buffers"][0]["dtype"])
        key = (json.dumps(self.semantic, sort_keys=True), self.device, vector)
        with LOCK:
            if key not in CACHE:
                CACHE[key] = Compiled(self.semantic, self.device, vector)
            self.last = CACHE[key]
        return self.last(scalars, *tensors)


def compile_kernel(semantic, device):
    return Dispatch(semantic, device)
