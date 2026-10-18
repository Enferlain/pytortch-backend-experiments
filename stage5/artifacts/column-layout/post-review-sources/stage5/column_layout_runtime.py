"""One explicit column map; frozen generic PTX is loaded without recompilation."""

import ast
import copy
from functools import cache
import json
import subprocess
import threading

from stage5.artifact import Kernel as ExistingKernel, compiler_identity
from stage5.layout_runtime import Kernel as GenericKernel, PLAN, admit as generic_admit
from stage5.column_layout_trial import OUT, ROOT, FROZEN, read, sha, fingerprint

MECHANISM = "cubecl-column-plain-fast64-v1"
BINARY = ROOT / "stage5/cache/reuse-first-rms/target/release/stage5-column-layout-probe"
CACHE = {}
LOCK = threading.RLock()


def admit(semantic):
    family, dtype = generic_admit(semantic)
    if family != "bias_column":
        raise ValueError(
            "column specialization requires exact captured column template"
        )
    return family, dtype


def verify_cpu():
    gate = read(OUT / "cpu.json")
    assert (
        gate["complete"] and not gate["gpu_source_existed"] and len(gate["rows"]) == 66
    )
    assert gate["source_sha256"] == sha(OUT / "cpu-source.py")

    # The only post-gate model-source edit marks public imports as re-exports.
    def normalized(path):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.alias) and node.name == node.asname:
                node.asname = None
        return ast.dump(tree)

    assert normalized(OUT / "cpu-source.py") == normalized(
        ROOT / "stage5/column_layout_trial.py"
    )
    assert gate["frozen_model_sha256"] == sha(ROOT / "stage5/layout_model.py")
    assert gate["frozen_integrity_sha256"] == sha(FROZEN / "integrity.json")
    assert gate["qualification_plan_sha256"] == sha(OUT / "qualification-plan.json")
    assert gate["timing_plan_sha256"] == sha(OUT / "timing-plan.json")
    api = read(OUT / "cpu-component-api.json")
    assert api["complete"] and len(api["rows"]) == 21
    assert api["plain_layout_sha256"] == sha(
        ROOT / ".stage0-sources/cubecl/crates/cubecl-std/src/tensor/layout/plain.rs"
    )


class Kernel(GenericKernel):
    def __init__(self, directory, semantic, device, *, compile_ptx):
        self.family, self.dtype = admit(semantic)
        # Reuse GenericKernel._validate unchanged, including original layouts,
        # bounds, metadata extra values, immutable readiness and stream ownership.
        ExistingKernel.__init__(
            self, directory, semantic, device, PLAN, compile_ptx=compile_ptx
        )
        self.rank = 2
        if (
            self.manifest.get("mechanism") != MECHANISM
            or self.manifest.get("layout_rank") != 2
        ):
            raise ValueError("column mechanism/rank mismatch")
        if self.manifest.get("metadata") != {"scalar_count": 7, "bytes": 80}:
            raise ValueError("column metadata ABI mismatch")


class Dispatch:
    def __init__(self, directory, semantic, kernel):
        self.directory, self.semantic, self.kernel = directory, semantic, kernel
        self.entrypoint = kernel.manifest["entrypoint"]

    def __call__(self, scalars, *tensors):
        return self.kernel(scalars, *tensors)


@cache
def retained_records():
    return read(FROZEN / "post-review-gpu.json")["kernels"]


def generic(semantic, device):
    semantic = copy.deepcopy(semantic)
    record = next(r for r in retained_records() if r["semantic"] == semantic)
    directory = FROZEN / record["directory"]
    assert sha(directory / "kernel.ptx") == record["ptx_sha256"]
    return Dispatch(
        directory,
        semantic,
        GenericKernel(directory, semantic, device, compile_ptx=False),
    )


def compile_kernel(semantic, device):
    semantic = copy.deepcopy(semantic)
    admit(semantic)
    with LOCK:
        cache_key = (fingerprint(semantic), device)
        if cache_key in CACHE:
            return CACHE[cache_key]
        verify_cpu()
        identity = dict(
            compiler_identity(device),
            column_adapter=sha(__file__),
            diagnostic_binary=sha(BINARY),
            native_emitter=sha(ROOT / "stage5/column_layout_probe/src/main.rs"),
            generic_adapter=sha(ROOT / "stage5/layout_runtime.py"),
            semantic_emitter=sha(ROOT / "stage0/src/descriptor2.rs"),
            cpu_gate=sha(OUT / "cpu.json"),
            component_gate=sha(OUT / "cpu-component-api.json"),
        )
        packet = dict(
            schema=5, semantic=semantic, physical_plan=PLAN, mechanism=MECHANISM
        )
        key = fingerprint([packet, identity])
        packet["entrypoint"] = "cubecl5_column_" + key[:16]
        directory = OUT / "candidate" / key
        integrity = directory / "integrity.json"
        if integrity.exists():
            for name, digest in read(integrity).items():
                if sha(directory / name) != digest:
                    raise ValueError("column cached artifact changed: " + name)
            if (
                read(directory / "descriptor.json") != packet
                or read(directory / "identity.json") != identity
            ):
                raise ValueError("column artifact identity mismatch")
            kernel = Kernel(directory, semantic, device, compile_ptx=False)
        else:
            if directory.exists():
                raise ValueError(
                    "incomplete column artifact retained: " + str(directory)
                )
            directory.mkdir(parents=True)
            for name, value in (
                ("descriptor", packet),
                ("semantic", semantic),
                ("physical-plan", PLAN),
                ("identity", identity),
            ):
                (directory / (name + ".json")).write_text(
                    json.dumps(value, indent=2) + "\n"
                )
            p = subprocess.run(
                [str(BINARY), str(directory), str(directory / "descriptor.json")],
                capture_output=True,
                text=True,
            )
            (directory / "compiler.log").write_text(p.stdout + p.stderr)
            if p.returncode:
                raise RuntimeError("column component compile failed: " + p.stderr)
            kernel = Kernel(directory, semantic, device, compile_ptx=True)
            (directory / "resources.json").write_text(
                json.dumps(
                    dict(
                        attributes=kernel.attributes,
                        resources=kernel.resources,
                        ptx_sha256=sha(directory / "kernel.ptx"),
                    ),
                    indent=2,
                )
                + "\n"
            )
            integrity.write_text(
                json.dumps(
                    {p.name: sha(p) for p in directory.iterdir() if p.is_file()},
                    indent=2,
                )
                + "\n"
            )
        result = Dispatch(directory, semantic, kernel)
        CACHE[cache_key] = result
        return result


def qualified_dispatch(semantic, device):
    if generic_admit(semantic)[0] == "bias_column":
        return compile_kernel(semantic, device)
    return generic(semantic, device)


def generic_compile(semantic, device):
    # Explicit generated-wrapper route to frozen PTX; never calls its compiler.
    return generic(semantic, device)
