"""Controlled replay of real nine-part descriptors and the one load-batch experiment.

The final packet, allocations, logical split and input bytes stay fixed while
existing partial plans are ablated; --candidate adds only the load-batch plan.
Timings are prepared partial components, never torch.compile graph timings.
"""

import argparse
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/reduction-structure"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ncu-plan", choices=("compensated", "tree", "plane", "four", "batch")
    )
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--rows", type=int, choices=(8, 64), default=8)
    args = parser.parse_args()
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch

    from stage5.artifact import CompiledKernel
    from stage5.plan import PhysicalCandidate
    from stage5.reference_reduction_probe import sha, tensor_sha
    from stage5.timing import KernelBenchmarker

    capture = json.loads((OUTPUT / "cubecl-r0.json").read_text())
    source = next(t for t in capture["timings"] if t["case"] == "float32-8-65539")
    packets = [
        json.loads((Path(k["directory"]) / "descriptor.json").read_text())
        for k in source["selected_kernels"]
    ]
    partial, final = (p["semantic"] for p in packets)
    plans = {
        "compensated": PhysicalCandidate("shared_tree_compensated"),
        "tree": PhysicalCandidate("shared_tree_reduction_baseline"),
        "plane": PhysicalCandidate("plane_shared", plane_width=32),
        "four": PhysicalCandidate(
            "plane_shared", plane_width=32, private_accumulators=4
        ),
    }
    if args.candidate or args.ncu_plan == "batch":
        plans["batch"] = PhysicalCandidate(
            "plane_shared_load_batch", plane_width=32, private_accumulators=4
        )
    if args.ncu_plan:
        plans = {args.ncu_plan: plans[args.ncu_plan]}
    kernels = {
        name: CompiledKernel(partial, 0, plan.to_dict()) for name, plan in plans.items()
    }
    finish = CompiledKernel(final, 0, packets[1]["physical_plan"])
    result = {
        "complete": False,
        "diagnostic_only": True,
        "checks": [],
        "timings": [],
        "partial_source": source["selected_kernels"][0]["directory"],
        "fixed_final": str(finish.directory),
        "kernels": {
            n: {
                "directory": str(k.directory),
                "plan": plans[n].to_dict(),
                "resources": k.kernel.resources,
            }
            for n, k in kernels.items()
        },
    }
    for rows in (args.rows,) if args.ncu_plan else (8, 64):
        torch.manual_seed(7301)
        x = torch.randn((rows, 65539), device="cuda")
        tmp = torch.empty((rows, 9), device="cuda")
        out = torch.empty(rows, device="cuda")
        scalars = (65539, rows)
        final_scalars = (rows,)

        # Runtime layout validation proves the saved symbolic output is B*9.
        def launch(
            k, scalars=scalars, x=x, tmp=tmp, final_scalars=final_scalars, out=out
        ):
            k(scalars, x, tmp)
            finish(final_scalars, tmp, out)

        for name, k in kernels.items():
            for _ in range(20):
                launch(k)
            result["checks"].append(
                {
                    "plan": name,
                    "shape": list(x.shape),
                    "input_sha256": tensor_sha(x),
                    "values": out.tolist(),
                    "aten": x.sum(-1).tolist(),
                    "fp64": x.double().sum(-1).tolist(),
                    "input_pointer": x.data_ptr(),
                    "partial_pointer": tmp.data_ptr(),
                    "output_pointer": out.data_ptr(),
                    "stream": torch.cuda.current_stream().cuda_stream,
                }
            )
        if args.ncu_plan:
            k = kernels[args.ncu_plan]
            torch.cuda._sleep(50_000_000)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            launch(k)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        else:
            names = list(kernels)
            for round_id in range(3):
                for name in names[round_id:] + names[:round_id]:
                    k = kernels[name]
                    with k.kernel.prepared(scalars, x, tmp):
                        measurement = KernelBenchmarker().benchmark_kernel_gpu(k.kernel)
                    result["timings"].append(
                        {
                            "plan": name,
                            "shape": list(x.shape),
                            "round": round_id,
                            "input_sha256": tensor_sha(x),
                            "samples_us": [v * 1000 for v in measurement["kernel_ms"]],
                            "median_us": statistics.median(measurement["kernel_ms"])
                            * 1000,
                            "scope": "prepared partial component; fixed final excluded",
                        }
                    )
                    print(
                        rows,
                        round_id,
                        name,
                        result["timings"][-1]["median_us"],
                        flush=True,
                    )
    evidence.state("after_structure_probe")
    result.update(
        complete=True, independence=evidence.inventory(), harness_sha256=sha(__file__)
    )
    path = OUTPUT / (
        f"component-ncu-{args.ncu_plan}-{args.rows}.json"
        if args.ncu_plan
        else "candidate-components.json"
        if args.candidate
        else "components.json"
    )
    path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
