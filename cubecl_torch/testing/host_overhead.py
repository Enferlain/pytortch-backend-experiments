"""Warm per-launch host overhead: Stage 5 runtime vs this package, same kernels.

Both implementations load the same descriptors and plans (identical PTX) in one
process. Rounds alternate between implementations; each round times a batch of
warm launches (validation, metadata lookup, enqueue; no synchronisation inside
the timed region). Noise is the spread of one implementation's own round
medians. The package passes when its median exceeds Stage 5's by no more than
that noise.

    python -m cubecl_torch.testing.host_overhead --out results/host-overhead.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def kernels():
    """(label, semantic, plan, scalars, tensors) for representative corpus kernels."""
    import torch

    from cubecl_torch.compiler.plan import PhysicalCandidate
    from cubecl_torch.testing.fixtures import captured, descriptor, tensors_for

    def plan(strategy, *args, **fields):
        return PhysicalCandidate(strategy, *args, **fields).to_dict()

    cases = [
        (
            "add-float32 scalar",
            captured("add-float32"),
            plan("scalar_pointwise_baseline"),
            (4099,),
        ),
        (
            "add-float16 vector8",
            captured("add-float16"),
            plan("vector_pointwise", (256, 1, 1), 8),
            (4096,),
        ),
        (
            "sin-float32 scalar",
            captured("sin-float32"),
            plan("scalar_pointwise_baseline"),
            (65539,),
        ),
        (
            "row_sum-float32 tree",
            captured("row_sum-float32"),
            plan("shared_tree_reduction_baseline"),
            (1024, 64),
        ),
        (
            "row_sum-bfloat16 compensated",
            captured("row_sum-bfloat16"),
            plan("shared_tree_compensated"),
            (4099, 16),
        ),
        (
            "column bias",
            descriptor("column"),
            plan("bounded_column_pointwise"),
            (257, 64),
        ),
        (
            "strided pointwise",
            descriptor("pointwise_strided"),
            plan("scalar_pointwise_baseline"),
            (3, 4099),
        ),
    ]
    result = []
    for label, semantic, physical, scalars in cases:
        torch.manual_seed(0)
        result.append(
            (label, semantic, physical, scalars, tensors_for(semantic, scalars))
        )
    return result


def time_launches(call, scalars, tensors, count):
    started = time.perf_counter_ns()
    for _ in range(count):
        call(scalars, *tensors)
    return (time.perf_counter_ns() - started) / count / 1000


def measure(rounds=15, launches=400):
    import torch

    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from cubecl_torch.compiler.nvrtc import STAGE5
    from cubecl_torch.inductor.kernels import CompiledKernel
    from stage5 import artifact

    rows = []
    for label, semantic, plan, scalars, tensors in kernels():
        stage5 = artifact.CompiledKernel(semantic, 0, plan)
        # Byte-identical PTX needs Stage 5's numerics.
        package = CompiledKernel(semantic, 0, plan, numerics=STAGE5)
        assert (
            stage5.directory / "kernel.ptx"
        ).read_bytes() == package.kernel.artifact.ptx
        for call in (stage5, package):
            time_launches(call, scalars, tensors, 50)  # warm layout/metadata caches
        torch.cuda.synchronize()
        samples = {"stage5": [], "package": []}
        for r in range(rounds):
            order = ("stage5", "package") if r % 2 == 0 else ("package", "stage5")
            for name in order:
                call = stage5 if name == "stage5" else package
                samples[name].append(time_launches(call, scalars, tensors, launches))
                torch.cuda.synchronize()
        medians = {k: statistics.median(v) for k, v in samples.items()}
        noise = max(
            statistics.median(abs(v - medians[k]) for v in samples[k]) * 3
            for k in samples
        )
        rows.append(
            {
                "kernel": label,
                "stage5_median_us": medians["stage5"],
                "package_median_us": medians["package"],
                "difference_us": medians["package"] - medians["stage5"],
                "noise_us": noise,
                "within_noise": medians["package"] <= medians["stage5"] + noise,
                "round_medians_us": samples,
            }
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--launches", type=int, default=400)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(REPO))
    rows = measure(args.rounds, args.launches)
    payload = {
        "method": "alternating rounds; per-launch host time of warm launches; noise = 3x the larger median absolute deviation of round medians",
        "rows": rows,
        "all_within_noise": all(r["within_noise"] for r in rows),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    for r in rows:
        print(
            f"{r['kernel']:32s} stage5 {r['stage5_median_us']:7.2f}us  package {r['package_median_us']:7.2f}us"
            f"  diff {r['difference_us']:+6.2f}  noise {r['noise_us']:5.2f}  {'ok' if r['within_noise'] else 'SLOWER'}"
        )
    return 0 if payload["all_within_noise"] else 1


if __name__ == "__main__":
    sys.exit(main())
