"""Bounded physical-order comparison; no accuracy policy is inferred from timing.

Strict mode intentionally fails on the retained eager-reference counterexample.
--qualify-reference is a diagnostic, never a substitute for an acceptance gate.
"""

import json
import argparse
import os
from pathlib import Path
import statistics
import traceback

ROOT = Path(__file__).resolve().parent


def run(args):
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import numpy as np
    import torch
    from stage5.artifact import CompiledKernel, compiler_identity
    from stage5.fixtures import captured
    from stage5.plan import PhysicalCandidate
    from stage5.reduction_order_checks import emulate
    from stage5.accumulation_checks import emulate_compensated
    from stage5.timing import KernelBenchmarker

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = ROOT / "cache/accumulation" / (output.stem + ".progress.json")
    progress.parent.mkdir(parents=True, exist_ok=True)
    plans = [PhysicalCandidate("shared_tree_reduction_baseline")]
    plans += [
        PhysicalCandidate("shared_tree_multi_accumulator", private_accumulators=n)
        for n in (2, 4)
    ]
    plans.append(PhysicalCandidate("shared_tree_compensated"))
    kernels = [
        CompiledKernel(
            captured("row_sum", "float32"), 0, p.to_dict(), compiler_identity(0)
        )
        for p in plans
    ]
    rows = []

    def check(label, x, model=False):
        ref, truth = x.sum(-1), x.double().sum(-1)
        row = {"case": label, "shape": list(x.shape), "plans": []}
        for p, k in zip(plans, kernels):
            y = torch.empty_like(ref)
            k((x.shape[1], x.shape[0]), x, y)
            if model:
                cpu = (
                    emulate_compensated(x.cpu().numpy())
                    if p.strategy == "shared_tree_compensated"
                    else emulate(x.cpu().numpy(), p.private_accumulators)
                )
                assert np.array_equal(cpu, y.cpu().numpy()), p.identity
            row["plans"].append(
                {
                    "private_accumulators": p.private_accumulators,
                    "strategy": p.strategy,
                    "eager_gate_passes": bool(
                        torch.all((y - ref).abs() <= 2e-5 + 2e-5 * ref.abs())
                    ),
                    "max_abs_error_fp64": (y.double() - truth).abs().max().item(),
                    "max_abs_error_eager": (y - ref).abs().max().item(),
                    "values": y.tolist(),
                    "aten": ref.tolist(),
                    "fp64": truth.tolist(),
                    "fp64_gate_passes": bool(
                        torch.all(
                            (y.double() - truth).abs() <= 2e-5 + 2e-5 * truth.abs()
                        )
                    ),
                    "cpu_gpu_bit_exact": model,
                }
            )
        rows.append(row)
        progress.write_text(
            json.dumps({"qualification_complete": False, "rows": rows}, indent=2)
        )

    old = json.loads(
        (ROOT / "artifacts/attribution/cancellation-counterexample.json").read_text()
    )
    torch.manual_seed(old["seed"])
    for shape in old["rng_history"]:
        x = torch.randn(shape, device="cuda")
    check("original_rng_history", x, True)
    for n in (31, 255, 256, 257, 1023, 1024, 1025, 4099, 65539, 131071, 1048576):
        for seed in (9241, 11, 101, 7301):
            torch.manual_seed(seed)
            check(f"random_{seed}", torch.randn((8, n), device="cuda"), n == 65539)
    # Held-out seed sweep, independent of the original failing RNG history.
    for seed in range(10000, 10032):
        torch.manual_seed(seed)
        check(f"heldout_{seed}", torch.randn((8, 65539), device="cuda"))
    for seed in range(30000, 30032):
        torch.manual_seed(seed)
        check(f"validation_{seed}", torch.randn((8, 65539), device="cuda"))
    eager_gate = all(
        v["eager_gate_passes"]
        for r in rows
        for v in r["plans"]
        if v["strategy"] == "shared_tree_compensated"
    )
    if not args.qualify_reference:
        assert eager_gate, (
            "unchanged eager gate fails; use explicit reference qualification to investigate"
        )
    assert all(
        v["fp64_gate_passes"]
        for r in rows
        for v in r["plans"]
        if v["strategy"] == "shared_tree_compensated"
    )
    timings = []
    for shape in ((64, 31), (64, 257), (8, 65539), (64, 65539)):
        x = torch.randn(shape, device="cuda")
        y = torch.empty(shape[0], device="cuda")
        for p, k in zip(plans, kernels):
            with k.kernel.prepared((shape[1], shape[0]), x, y):
                m = KernelBenchmarker().benchmark_kernel_gpu(k.kernel)
            timings.append(
                {
                    "shape": shape,
                    "plan": p.to_dict(),
                    "median_kernel_us": statistics.median(m["kernel_ms"]) * 1000,
                    "attributes": k.kernel.attributes,
                    "resources": k.kernel.resources,
                    "directory": str(k.directory),
                }
            )
    evidence.state("after_accumulation_probe")
    output.write_text(
        json.dumps(
            {
                "rows": rows,
                "timings": timings,
                "independence": evidence.inventory(),
                "qualification_complete": True,
                "selected_candidate_eager_gate_passes": eager_gate,
                "selected_candidate_fp64_gate_passes": True,
                "reference_qualification_only": args.qualify_reference,
            },
            indent=2,
        )
    )
    print("Accumulator comparison captured", output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "cache/accumulation/probe.json"
    )
    parser.add_argument(
        "--qualify-reference",
        action="store_true",
        help="Retain eager-gate failures and qualify them against FP64; diagnostic only",
    )
    args = parser.parse_args()
    progress = ROOT / "cache/accumulation" / (args.output.stem + ".progress.json")
    progress.unlink(missing_ok=True)
    try:
        run(args)
    except Exception as error:
        failed = json.loads(progress.read_text()) if progress.is_file() else {}
        failed.update(
            qualification_complete=False,
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".failed.json").write_text(json.dumps(failed, indent=2))
        raise


if __name__ == "__main__":
    main()
