"""Explicit component adapter; storage, readiness and launches stay in Stage5."""

import copy
from functools import cache
import json
import subprocess
import threading

from stage5.artifact import Kernel as ExistingKernel, compiler_identity
from stage5.plan import PhysicalCandidate
from stage5.layout_trial import OUT, ROOT, sha
from stage5.layout_model import maps, params, signed_bits

PLAN = PhysicalCandidate("scalar_pointwise_baseline").to_dict()
MECHANISM = "cubecl-linear-permuted-fast64-v1"
BINARY = ROOT / "stage5/cache/reuse-first-rms/target/release/stage5-layout-probe"
CACHE = {}
LOCK = threading.RLock()


def canonical(value):
    return json.dumps(value, sort_keys=True)


@cache
def templates():
    cap = json.loads((OUT / "capture.json").read_text())
    result = {}
    for row in cap["rows"]:
        sem = json.loads((OUT / row["baseline"] / "semantic.json").read_text())
        result[canonical(sem)] = (row["case"]["family"], row["case"]["dtype"], sem)
    return result


def admit(semantic):
    if canonical(semantic) not in templates():
        raise ValueError("layout requires an exact captured pointwise template")
    return templates()[canonical(semantic)][:2]


@cache
def allowed(family, dtype):
    rows = json.loads((OUT / "qualification-plan.json").read_text())["cases"]
    return frozenset(
        tuple(r["scalars"])
        for r in rows
        if r["family"] == family and r["dtype"] == dtype
    )


class Kernel(ExistingKernel):
    def __init__(self, directory, semantic, device, *, compile_ptx=True):
        self.family, self.dtype = admit(semantic)
        super().__init__(directory, semantic, device, PLAN, compile_ptx=compile_ptx)
        self.rank = len(semantic["buffers"][-1]["size"])
        count = 1 + len(semantic["scalars"]) + 2 * self.rank
        if (
            self.manifest.get("mechanism") != MECHANISM
            or self.manifest.get("layout_rank") != self.rank
        ):
            raise ValueError("layout mechanism/rank mismatch")
        if self.manifest.get("metadata") != {
            "scalar_count": count,
            "bytes": 8 * (len(semantic["buffers"]) + count),
        }:
            raise ValueError("layout metadata ABI mismatch")

    def _validate(self, scalars, tensors):
        if len(scalars) != len(self.semantic["scalars"]) or tuple(
            int(v) for v in scalars
        ) not in allowed(self.family, self.dtype):
            raise ValueError("layout scalar values outside bounded qualification plan")
        key, n, pointers = super()._validate(scalars, tensors)
        shape, _ = maps(self.semantic, scalars)
        extras = tuple(
            v for d in shape for v in (signed_bits(params(d)[0]), params(d)[1])
        )
        # Existing InfoBuilder accepts arbitrary i64 scalar lists. Extra values
        # become part of the same immutable metadata/readiness/lifetime key.
        return (key[0], key[1] + extras), n, pointers


class Compiled:
    def __init__(self, semantic, device):
        admit(semantic)
        cpu = json.loads((OUT / "cpu.json").read_text())
        neighbors = json.loads((OUT / "cpu-neighbors.json").read_text())
        assert (
            cpu["complete"] and not cpu["gpu_source_existed"] and len(cpu["rows"]) == 33
        )
        assert neighbors["complete"] and len(neighbors["rows"]) == 66
        assert (
            cpu["source_sha256"]
            == sha(ROOT / "stage5/layout_model.py")
            == neighbors["model_sha256"]
        )
        assert neighbors["plan_sha256"] == sha(OUT / "qualification-plan.json")
        assert cpu["capture_sha256"] == sha(OUT / "capture.json")
        self.semantic = copy.deepcopy(semantic)
        identity = dict(
            compiler_identity(device),
            diagnostic_binary=sha(BINARY),
            layout_adapter=sha(__file__),
            semantic_emitter=sha(ROOT / "stage0/src/descriptor2.rs"),
            cpu_gate=sha(OUT / "cpu.json"),
            neighbors=sha(OUT / "cpu-neighbors.json"),
        )
        packet = dict(
            schema=5, semantic=self.semantic, physical_plan=PLAN, mechanism=MECHANISM
        )
        import hashlib

        key = hashlib.sha256(canonical([packet, identity]).encode()).hexdigest()
        self.entrypoint = "cubecl5_layout_" + key[:16]
        packet["entrypoint"] = self.entrypoint
        self.directory = OUT / "candidate" / key
        self.directory.mkdir(parents=True, exist_ok=True)
        for name, value in (
            ("descriptor", packet),
            ("semantic", self.semantic),
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
            raise RuntimeError("layout compile failed: " + p.stderr)
        self.kernel = Kernel(self.directory, self.semantic, device)
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


def compile_kernel(semantic, device):
    admit(semantic)
    key = (canonical(semantic), device)
    with LOCK:
        if key not in CACHE:
            CACHE[key] = Compiled(semantic, device)
        return CACHE[key]
