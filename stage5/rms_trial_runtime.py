"""Explicit-only CubeK RMS adapter; ordinary catalog/compiler remain untouched."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import threading

from stage5.artifact import Kernel as ExistingKernel
from stage5.plan import PhysicalCandidate
from stage5.rms_trial_data import OUT, ROOT, WIDTHS

MECHANISM = "actual-cubek-rms-plane-v1"
PLAN = PhysicalCandidate(
    "independent_planes", (32, 8, 1), outputs_per_workgroup=8, plane_width=32
).to_dict()
BINARY = ROOT / "cache/reuse-first-rms/target/release/stage5-rms-probe"
CACHE = {}
LOCK = threading.RLock()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def templates():
    packet = json.loads(
        (ROOT / "artifacts/reuse-first-selection/baseline-capture.json").read_text()
    )
    result = {}
    for row in packet["records"]:
        path = (
            ROOT
            / "artifacts/reuse-first-selection"
            / row["artifacts"][0]["directory"]
            / "semantic.json"
        )
        result[row["dtype"]] = json.loads(path.read_text())
    return result


def admit(semantic):
    if not isinstance(semantic, dict) or json.dumps(semantic, sort_keys=True) not in {
        json.dumps(t, sort_keys=True) for t in templates().values()
    }:
        raise ValueError("RMS diagnostic requires an exact qualified semantic template")


def cpu_gate():
    cpu = json.loads((OUT / "cpu.json").read_text())
    assert (
        cpu["complete"]
        and len(cpu["rows"]) == 66
        and all(r["passes"] for r in cpu["rows"])
    )
    assert cpu["model_sha256"] == sha(OUT / "cpu-model.pt")
    assert cpu["reference_sha256"] == sha(OUT / "reference.json")
    assert cpu["source_sha256"]["rms_trial_data.py"] == sha(ROOT / "rms_trial_data.py")
    # The phase driver grew after CPU qualification; bind the retained producer.
    assert cpu["source_sha256"]["rms_trial.py"] == sha(
        OUT / "phase-sources/cpu-rms_trial.py"
    )


class Kernel(ExistingKernel):
    def __init__(self, directory, semantic, device):
        admit(semantic)
        super().__init__(directory, semantic, device, PLAN, compile_ptx=True)
        if self.manifest.get("mechanism") != MECHANISM:
            raise ValueError("RMS diagnostic manifest mismatch")
        scalar_count = len(semantic["scalars"]) + 1
        if self.manifest.get("metadata") != {
            "scalar_count": scalar_count,
            "bytes": 8 * (len(semantic["buffers"]) + scalar_count),
        }:
            raise ValueError("RMS diagnostic metadata ABI mismatch")

    def _validate(self, scalars, tensors):
        scalars = tuple(int(s) for s in scalars)
        if len(scalars) != 2 or scalars[0] not in WIDTHS or not 1 <= scalars[1] <= 65:
            raise ValueError("RMS diagnostic extent outside qualified trial")
        return super()._validate(scalars, tensors)


class CompiledKernel:
    def __init__(self, semantic, device):
        cpu_gate()
        admit(semantic)
        self.semantic, self.device = copy.deepcopy(semantic), device
        packet = {
            "schema": 5,
            "semantic": self.semantic,
            "physical_plan": PLAN,
            "mechanism": MECHANISM,
        }
        identity = {
            "compiler_binary": sha(BINARY),
            "adapter": sha(__file__),
            "semantic_emitter": sha(ROOT.parent / "stage0/src/descriptor2.rs"),
            "cpu_gate": sha(OUT / "cpu.json"),
        }
        raw = json.dumps(packet, sort_keys=True)
        key = hashlib.sha256(
            (raw + json.dumps(identity, sort_keys=True)).encode()
        ).hexdigest()
        packet["entrypoint"] = "cubecl5_rms_cubek_" + key[:16]
        self.directory = OUT / "candidate" / key
        self.directory.mkdir(parents=True, exist_ok=True)
        for name, data in (
            ("descriptor", packet),
            ("semantic", self.semantic),
            ("physical-plan", PLAN),
            ("identity", identity),
        ):
            (self.directory / (name + ".json")).write_text(json.dumps(data, indent=2))
        p = subprocess.run(
            [str(BINARY), str(self.directory), str(self.directory / "descriptor.json")],
            capture_output=True,
            text=True,
        )
        (self.directory / "compiler.txt").write_text(p.stdout + p.stderr)
        if p.returncode:
            raise RuntimeError("actual CubeK RMS compilation failed: " + p.stderr)
        self.kernel = Kernel(self.directory, self.semantic, device)
        self.entrypoint = packet["entrypoint"]
        (self.directory / "resources.json").write_text(
            json.dumps(
                {
                    "attributes": self.kernel.attributes,
                    "resources": self.kernel.resources,
                    "ptx_sha256": sha(self.directory / "kernel.ptx"),
                },
                indent=2,
            )
        )

    def __call__(self, scalars, *tensors):
        return self.kernel(scalars, *tensors)


def compile_kernel(semantic, device):
    admit(semantic)
    key = (json.dumps(semantic, sort_keys=True), device)
    with LOCK:
        if key not in CACHE:
            CACHE[key] = CompiledKernel(semantic, device)
        return CACHE[key]
