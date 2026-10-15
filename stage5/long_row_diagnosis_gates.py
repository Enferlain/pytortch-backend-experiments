"""Established 18 gates, with separate evidence and historical-byte restoration."""

import sys
from stage5 import precision_gates as gates


if __name__ == "__main__":
    gates.OUTPUT = gates.ROOT / "artifacts/long-row-diagnosis"
    gates.CACHE = gates.ROOT / "cache/long-row-diagnosis/gates"
    gates.main(
        [
            (
                "precision_matrix",
                [
                    sys.executable,
                    "-m",
                    "stage5.precision_checks",
                    "--output",
                    str(gates.OUTPUT / "precision-regression.json"),
                ],
            )
        ]
        + [
            (name, [sys.executable, "-m", "stage5." + module])
            for name, module in (
                ("precision_harness", "precision_harness_checks"),
                ("frozen_order_qualification", "reduction_order_checks"),
                ("accumulation_checks", "accumulation_checks"),
                ("accumulation_harness", "accumulation_harness_checks"),
                ("long_row_checks", "long_row_checks"),
            )
        ]
    )
