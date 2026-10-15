"""Neighboring ABI/eligibility checks for the sole private-merge experiment."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from stage5.plan import PhysicalCandidate
from stage5.reduction_rounding_probe import OUTPUT, SOURCE


def runner_controls():
    from stage5 import reduction_merge_gates as gates

    # Execute the real runner with CPU-only stand-ins for its two GPU commands.
    # Protect an untracked previous packet and exercise success/failure cleanup.
    for fail in (False, True):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            source, output = project / "prior", project / "prior/rounding"
            output.mkdir(parents=True)
            sentinel = source / "untracked.json"
            sentinel.write_text('{"historical": true}')
            commands = [
                {"name": name, "command": ["probe", str(source / (name + ".json"))]}
                for name in ("corpus", "precision-regression")
            ]
            (source / "gates.json").write_text(json.dumps(commands))

            def run(command, *, output=output, sentinel=sentinel, fail=fail, **_kwargs):
                target = Path(command[1])
                assert target.parent == output
                target.write_text('{"fresh": true}')
                sentinel.write_text('{"changed": true}')
                return SimpleNamespace(
                    returncode=int(fail and target.stem == "precision-regression")
                )

            with (
                patch.object(gates, "SOURCE", source),
                patch.object(gates, "OUTPUT", output),
                patch.object(gates, "__file__", str(project / "stage5/gates.py")),
                patch.object(gates.subprocess, "check_output", return_value=b""),
                patch.object(gates.subprocess, "run", side_effect=run),
            ):
                try:
                    gates.main()
                except SystemExit as error:
                    assert fail and error.code == 1
                else:
                    assert not fail
            assert sentinel.read_text() == '{"historical": true}'
            status = json.loads((output / "gate-status.json").read_text())
            assert status == {"expected": 2, "finished": 2, "complete": not fail}
            assert (output / "precision-regression.json").exists()


def main():
    from stage5.legality import numerical_reason
    from stage5.reduction_policy import qualified_for_runtime

    candidate = PhysicalCandidate(
        "plane_shared_load_batch",
        plane_width=32,
        private_accumulators=4,
        private_merge="adjacent",
    )
    baseline = PhysicalCandidate(
        "plane_shared_load_batch", plane_width=32, private_accumulators=4
    )
    assert "private_merge" not in baseline.to_dict()
    assert candidate.identity != baseline.identity
    assert PhysicalCandidate.from_dict(candidate.to_dict()) == candidate
    semantic = {"schema": 3, "reduction_extent": {"op": "constant", "value": 7283}}
    assert numerical_reason(candidate, semantic, ())
    assert not qualified_for_runtime(semantic, candidate)
    packet = json.loads(
        (
            Path(
                json.loads((OUTPUT / "merge-qualification.json").read_text())[
                    "kernels"
                ]["float32"]["adjacent"]
            )
            / "descriptor.json"
        ).read_text()
    )
    cache = SOURCE.parents[1] / "cache/reduction-rounding/negative"
    cache.mkdir(parents=True, exist_ok=True)
    compiler = (
        Path(__file__).resolve().parents[1] / "stage0/target/release/cubecl-stage0"
    )
    changes = (
        {"private_merge": "unknown"},
        {"private_merge": None},
        {"private_merge": True},
        {"private_merge": 1},
        {"strategy": "plane_shared"},
        {"strategy": "shared_tree_multi_accumulator", "plane_width": 0},
        {"private_accumulators": 2},
        {"workgroup": [128, 1, 1]},
        {"vector_width": 2},
        {"plane_width": 64},
        {"outputs_per_workgroup": 2},
    )
    for i, change in enumerate(changes):
        plan = {**candidate.to_dict(), **change}
        try:
            PhysicalCandidate.from_dict(plan)
        except ValueError:
            pass
        else:
            raise AssertionError(("Python accepted invalid plan", change))
        path = cache / f"{i}.json"
        path.write_text(json.dumps({**packet, "physical_plan": plan}))
        result = subprocess.run(
            [str(compiler), str(cache / str(i)), "cpp", str(path)],
            capture_output=True,
            text=True,
            env={**os.environ, "STAGE0_CARGO_PROFILE": "release"},
            check=False,
        )
        assert result.returncode and "Error" in result.stderr, (change, result.stderr)
    # A balanced merge order is not universally more accurate: preserve this
    # counterexample instead of presenting the reference order as an oracle.
    import torch

    from stage5.reduction_rounding_probe import model

    # Directly demonstrate the FP32 arithmetic used by the two private merges.
    a0, a1, a2, a3 = [
        torch.tensor(v, dtype=torch.float32) for v in (2**24, 1, -(2**24), 0)
    ]
    assert ((a0 + a2) + (a1 + a3)).item() == 1
    assert ((a0 + a1) + (a2 + a3)).item() == 0
    assert model(torch.zeros((1, 1)))["output"].item() == 0
    runner_controls()
    result = {
        "complete": True,
        "python_invalid_neighbors": len(changes),
        "rust_invalid_neighbors": len(changes),
        "old_serialized_identities_preserved": True,
        "candidate_still_diagnostic": True,
        "merge_order_has_no_universal_accuracy_dominance": True,
        "runner_success_failure_and_untracked_preservation": True,
    }
    (OUTPUT / "checks.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
