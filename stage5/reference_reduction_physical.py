"""Diagnostic comparison of existing physical plans on matched reduction inputs.

These direct bridge calls do not change legality or production plan selection.
The existing eager gate remains recorded even when a plan disagrees with it.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import traceback

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combined", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.artifact import CompiledKernel, compiler_identity
    from stage5.fixtures import captured
    from stage5.plan import PhysicalCandidate
    from stage5.timing import KernelBenchmarker
    from stage5.reference_reduction_probe import tensor_sha, sha

    plans = [
        PhysicalCandidate("shared_tree_reduction_baseline"),
        PhysicalCandidate("shared_tree_compensated"),
        PhysicalCandidate("shared_tree_multi_accumulator", private_accumulators=4),
        PhysicalCandidate("plane_shared", plane_width=32),
    ]
    if args.combined:
        plans.append(
            PhysicalCandidate("plane_shared", plane_width=32, private_accumulators=4)
        )
    rows, timings = [], []
    result = {
        "qualification_complete": False,
        "diagnostic_only": True,
        "rows": rows,
        "timings": timings,
    }

    def check(label, x):
        ref, truth = x.sum(-1), x.double().sum(-1)
        row = {
            "case": label,
            "shape": list(x.shape),
            "input_sha256": tensor_sha(x),
            "aten": ref.tolist(),
            "fp64": truth.tolist(),
            "plans": [],
        }
        for plan, kernel in zip(plans, kernels):
            out = torch.empty_like(ref)
            kernel((x.shape[1], x.shape[0]), x, out)
            row["plans"].append(
                {
                    "plan": plan.to_dict(),
                    "values": out.tolist(),
                    "eager_gate_passes": bool(
                        torch.all((out - ref).abs() <= 2e-5 + 2e-5 * ref.abs())
                    ),
                    "max_abs_error_fp64": (out.double() - truth).abs().max().item(),
                }
            )
        rows.append(row)

    try:
        kernels = [
            CompiledKernel(
                captured("row_sum", "float32"), 0, p.to_dict(), compiler_identity(0)
            )
            for p in plans
        ]
        old = json.loads(
            (
                ROOT / "artifacts/attribution/cancellation-counterexample.json"
            ).read_text()
        )
        torch.manual_seed(old["seed"])
        for shape in old["rng_history"]:
            x = torch.randn(shape, device="cuda")
        check("original_rng_history", x)
        for n in (31, 255, 256, 257, 1023, 1024, 1025, 4099, 65539, 131071, 1048576):
            for seed in (9241, 11, 101, 7301):
                torch.manual_seed(seed)
                check(f"random_{seed}_{n}", torch.randn((8, n), device="cuda"))
        for seed in (*range(10000, 10032), *range(30000, 30032)):
            torch.manual_seed(seed)
            check(f"heldout_{seed}", torch.randn((8, 65539), device="cuda"))
        for shape in ((8, 65539), (64, 65539), (8, 131071), (64, 131071), (8, 1048576)):
            torch.manual_seed(7301)
            x = torch.randn(shape, device="cuda")
            y = torch.empty(shape[0], device="cuda")
            # Alternate the original/current plans' positions across shapes;
            # each physical probe has the existing controlled event warmup.
            order = range(len(plans)) if shape[0] == 8 else reversed(range(len(plans)))
            for i in order:
                kernel = kernels[i]
                with kernel.kernel.prepared((shape[1], shape[0]), x, y):
                    measured = KernelBenchmarker().benchmark_kernel_gpu(kernel.kernel)
                timings.append(
                    {
                        "shape": list(shape),
                        "input_sha256": tensor_sha(x),
                        "plan": plans[i].to_dict(),
                        "median_kernel_us": statistics.median(measured["kernel_ms"])
                        * 1000,
                        "samples_kernel_us": [v * 1000 for v in measured["kernel_ms"]],
                        "directory": str(kernel.directory),
                        "attributes": kernel.kernel.attributes,
                        "resources": kernel.kernel.resources,
                        "scope": "prepared native batches; physical component, not end-to-end torch.compile timing",
                    }
                )
                print(
                    shape,
                    plans[i].identity,
                    timings[-1]["median_kernel_us"],
                    flush=True,
                )
        result.update(
            qualification_complete=True,
            harness_sha256=sha(__file__),
            artifacts=[
                {"directory": str(k.directory), "plan": p.to_dict()}
                for p, k in zip(plans, kernels)
            ],
        )
        evidence.state("after_reference_physical")
        result["independence"] = evidence.inventory()
    except Exception as error:
        result.update(failure=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = (
            args.output
            if result["qualification_complete"]
            else args.output.with_suffix(".failed.json")
        )
        output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
