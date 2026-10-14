"""Failed strict diagnostics preserve the last completed qualification packet."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from stage5 import accumulation_probe as probe
from stage5.plan import PhysicalCandidate


with TemporaryDirectory() as temporary:
    root = Path(temporary)
    canonical = root / "artifacts/accumulation/comparison.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("previous completed qualification\n")

    def fail(args):
        assert not args.qualify_reference, "strict is the default"
        progress = root / "cache/accumulation/comparison.progress.json"
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text(
            json.dumps(
                {
                    "qualification_complete": False,
                    "rows": [{"eager_gate_passes": False}],
                }
            )
        )
        raise AssertionError("controlled eager-relative failure")

    with (
        patch.object(probe, "ROOT", root),
        patch.object(probe, "run", side_effect=fail),
        patch("sys.argv", ["probe", "--output", str(canonical)]),
    ):
        try:
            probe.main()
        except AssertionError as error:
            assert str(error) == "controlled eager-relative failure"
        else:
            raise AssertionError("strict numerical failure accepted")
    failed = json.loads(canonical.with_suffix(".failed.json").read_text())
    assert (
        not failed["qualification_complete"]
        and not failed["rows"][0]["eager_gate_passes"]
    )
    assert "controlled eager-relative failure" in failed["traceback"]
    assert canonical.read_text() == "previous completed qualification\n"

plan = PhysicalCandidate("shared_tree_compensated").to_dict()
for value in (True, False, 2.0, "2", None):
    try:
        PhysicalCandidate.from_dict({**plan, "private_accumulators": value})
    except ValueError as error:
        assert "integer" in str(error)
    else:
        raise AssertionError("non-integer private count accepted")

output = Path(__file__).parent / "artifacts/accumulation/harness-controls.json"
output.write_text(
    json.dumps(
        {
            "strict_failure_propagates": True,
            "canonical_capture_preserved": True,
            "failed_rows_retained": True,
            "integer_policy_checked_before_compilation": True,
        },
        indent=2,
    )
)
print("Accumulation harness controls PASS", flush=True)
