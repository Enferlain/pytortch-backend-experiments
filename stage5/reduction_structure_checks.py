"""Neighboring topology, numerical eligibility and evidence-integrity controls."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from stage5.plan import PhysicalCandidate
from stage5.reduction_structure_analyze import fingerprint, verify_environment

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/reduction-structure"


def rejects(fn, error=ValueError):
    try:
        fn()
    except error:
        return
    raise AssertionError("invalid neighboring case accepted")


def main():
    from stage5.legality import numerical_reason
    from stage5.reduction_policy import qualified_for_runtime

    plan = PhysicalCandidate(
        "plane_shared_load_batch", plane_width=32, private_accumulators=4
    )
    assert PhysicalCandidate.from_dict(plan.to_dict()) == plan
    for change in (
        {"private_accumulators": 1},
        {"private_accumulators": 2},
        {"private_accumulators": True},
        {"workgroup": [128, 1, 1]},
        {"vector_width": 2},
        {"plane_width": 64},
        {"outputs_per_workgroup": 2},
    ):
        rejects(
            lambda change=change: PhysicalCandidate.from_dict(
                {**plan.to_dict(), **change}
            )
        )
    for n in (9, 7283):
        semantic = {"schema": 3, "reduction_extent": {"op": "constant", "value": n}}
        assert numerical_reason(plan, semantic, ())
        assert not qualified_for_runtime(semantic, plan)
    capture = json.loads((OUTPUT / "cubecl-r0.json").read_text())
    env = capture["environment"]
    verify_environment(env, True)
    for field, value in (
        ("gil", True),
        ("PYTHON_GIL", "0"),
        ("fp_patch_sha256", "wrong"),
    ):
        rejects(
            lambda field=field, value=value: verify_environment(
                {**env, field: value}, True
            ),
            AssertionError,
        )
    bad = copy.deepcopy(env)
    bad["independence"]["native_mappings"] = ["fake/native/triton.so"]
    rejects(lambda: verify_environment(bad, True), AssertionError)
    k = capture["timings"][0]["selected_kernels"][-1]
    changed = {**k, "plan": {**k["plan"], "workgroup": [128, 1, 1]}}
    assert fingerprint(k, True) != fingerprint(changed, True)
    # Send the same invalid neighbors to the independent Rust validator.
    components = json.loads((OUTPUT / "candidate-components.json").read_text())
    directory = Path(components["kernels"]["batch"]["directory"])
    packet = json.loads((directory / "descriptor.json").read_text())
    cache = ROOT / "cache/reduction-structure/negative"
    cache.mkdir(parents=True, exist_ok=True)
    compiler = ROOT.parent / "stage0/target/release/cubecl-stage0"
    for i, change in enumerate(
        (
            {"private_accumulators": 1},
            {"private_accumulators": 2},
            {"private_accumulators": True},
            {"workgroup": [128, 1, 1]},
            {"vector_width": 2},
            {"plane_width": 64},
            {"outputs_per_workgroup": 2},
        )
    ):
        candidate = {**packet, "physical_plan": {**packet["physical_plan"], **change}}
        path = cache / f"{i}.json"
        path.write_text(json.dumps(candidate))
        p = subprocess.run(
            [str(compiler), str(cache / str(i)), "cpp", str(path)],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "STAGE0_CARGO_PROFILE": "release"},
        )
        assert p.returncode and ("Error" in p.stderr or "unsupported" in p.stderr), (
            change,
            p.stderr,
        )
    # These argument failures must happen before importing torch or writing a
    # baseline capture. Default first_use behavior remains unchanged.
    for flags in (
        ["--impl", "cubecl"],
        ["--impl", "cubecl", "--output-dir", str(OUTPUT)],
        [
            "--impl",
            "cubecl",
            "--output-dir",
            str(ROOT / "artifacts/long-row"),
            "--cache-dir",
            str(ROOT / "cache/reduction-structure/norm"),
        ],
        [
            "--impl",
            "triton",
            "--output-dir",
            str(OUTPUT),
            "--cache-dir",
            str(ROOT / "cache/reduction-structure/norm"),
        ],
    ):
        p = subprocess.run(
            [
                sys.executable,
                "-m",
                "stage5.long_row_normalization",
                *flags,
                "--cubecl-policy",
                "load_batch_diagnostic",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert p.returncode == 2 and "error:" in p.stderr, p.stderr
    result = {
        "complete": True,
        "normalization_cli_rejections": 4,
        "python_topology_neighbors": 7,
        "rust_topology_neighbors": 7,
        "runtime_eligibility_rejections": 2,
        "environment_tampering": 4,
        "final_plan_identity_tampering": 1,
    }
    (OUTPUT / "checks.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
