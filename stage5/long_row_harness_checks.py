"""Reject incomplete/mislabelled evidence without touching completed captures."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from stage5 import long_row_analyze as analysis


def main():
    output = analysis.OUTPUT
    baseline = (output / "analysis.json").read_bytes()
    data = {
        name: json.loads((output / (name + ".json")).read_text())
        for name in analysis.NAMES
    }
    cases = []

    def bad_hash(d):
        d["rows"][0]["input_sha256"] = "wrong input"

    def missing_round(d):
        d["timings"][0]["profiles"].pop()

    def wrong_role(d):
        d["history"] = "short_first"

    def native_triton(d):
        d["environment"]["independence"]["native_mappings"] = [
            "forbidden native module"
        ]

    def incomplete(d):
        d["qualification_complete"] = False

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name, capture in data.items():
            (root / (name + ".json")).write_text(json.dumps(capture))
        for name, change in (
            ("mismatched_input", bad_hash),
            ("missing_round", missing_round),
            ("wrong_reference_role", wrong_role),
            ("native_triton", native_triton),
            ("incomplete_capture", incomplete),
        ):
            changed = copy.deepcopy(data["combined-fixed"])
            change(changed)
            path = root / "combined-fixed.json"
            path.write_text(json.dumps(changed))
            with patch.object(analysis, "OUTPUT", root):
                try:
                    analysis.analyze()
                except (AssertionError, ValueError):
                    cases.append({"case": name, "rejected": True})
                else:
                    raise AssertionError("invalid evidence accepted: " + name)
            path.write_text(json.dumps(data["combined-fixed"]))
    assert (output / "analysis.json").read_bytes() == baseline
    (output / "harness-controls.json").write_text(
        json.dumps(
            {
                "checks": cases,
                "canonical_analysis_preserved": True,
            },
            indent=2,
        )
    )
    print("Long-row harness controls PASS", len(cases), flush=True)


if __name__ == "__main__":
    main()
