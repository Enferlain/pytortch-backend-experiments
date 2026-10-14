"""All established gates plus guarded-path checks; preserve historical bytes."""

import sys
from stage5 import precision_gates


if __name__ == "__main__":
    precision_gates.OUTPUT = precision_gates.ROOT / "artifacts/long-row"
    precision_gates.CACHE = precision_gates.ROOT / "cache/long-row/gates"
    precision_gates.main(
        [
            (
                "precision_matrix",
                [
                    sys.executable,
                    "-m",
                    "stage5.precision_checks",
                    "--output",
                    str(precision_gates.OUTPUT / "precision-regression.json"),
                ],
            ),
            (
                "precision_harness",
                [sys.executable, "-m", "stage5.precision_harness_checks"],
            ),
            (
                "frozen_order_qualification",
                [sys.executable, "-m", "stage5.reduction_order_checks"],
            ),
            (
                "accumulation_checks",
                [sys.executable, "-m", "stage5.accumulation_checks"],
            ),
            (
                "accumulation_harness",
                [sys.executable, "-m", "stage5.accumulation_harness_checks"],
            ),
            ("long_row_checks", [sys.executable, "-m", "stage5.long_row_checks"]),
        ]
    )
