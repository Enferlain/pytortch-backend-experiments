"""One diagnostic cache-policy control using CUDA's existing store intrinsics.

Only the unrelated primer store changes: __stwb versus __stwt. Target kernels,
selector order, allocation, values, byte counts and measurement code are reused.
The device-memory effect of __stwt must be measured; its name is not evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from stage5 import reduction_state_counters as counters


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--store-policy", choices=["wb", "wt"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    original = "q[(unsigned long long)i * stride] = 1u;"
    assert counters.SOURCE.count(original) == 1
    counters.SOURCE = counters.SOURCE.replace(
        original, f"__st{args.store_policy}(p + (unsigned long long)i * stride, 1u);"
    )
    sys.argv = [sys.argv[0], "--output", str(args.output), *remaining]
    try:
        counters.main()
    finally:
        if args.output.is_dir():
            source = Path(__file__).read_bytes()
            (args.output / "store-control.py").write_bytes(source)
            (args.output / "store-control.json").write_text(
                json.dumps(
                    {
                        "policy": args.store_policy,
                        "source_sha256": hashlib.sha256(source).hexdigest(),
                        "argv": sys.argv,
                        "only_source_substitution": [original, f"__st{args.store_policy}(p + (unsigned long long)i * stride, 1u);"],
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
