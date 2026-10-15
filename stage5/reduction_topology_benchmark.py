"""One reference-derived topology, measured against existing diagnostic paths."""

import json
import os
from pathlib import Path

from stage5.reduction_topology_model import OUTPUT, SOURCE
from stage5.reduction_topology_probe import candidate


def main():
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch

    from stage5.artifact import CompiledKernel
    from stage5.attribution import inventory, summary
    from stage5.plan import PhysicalCandidate
    from stage5.reference_reduction_probe import sha, tensor_sha
    from stage5.timing import KernelBenchmarker

    capture = json.loads((SOURCE / "cubecl-r0.json").read_text())
    plans = {
        "baseline": PhysicalCandidate(
            "plane_shared_load_batch", plane_width=32, private_accumulators=4
        ),
        "reference512": candidate(),
        "compensated": PhysicalCandidate("shared_tree_compensated"),
    }
    rows = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        timing = next(
            t
            for t in capture["timings"]
            if t["case"] == f"{str(dtype).removeprefix('torch.')}-8-65539"
        )
        p, f = [
            json.loads((Path(k["directory"]) / "descriptor.json").read_text())
            for k in timing["selected_kernels"]
        ]
        kernels = {
            name: CompiledKernel(p["semantic"], 0, plan.to_dict())
            for name, plan in plans.items()
        }
        final = CompiledKernel(f["semantic"], 0, f["physical_plan"])
        for b, n in ((8, 65539), (64, 65539), (8, 131071), (8, 1048576)):
            torch.manual_seed(7301)
            x = torch.randn((b, n), device="cuda").to(dtype)
            tmp = torch.empty((b, 9), device="cuda")
            out = torch.empty((b,), device="cuda", dtype=dtype)
            case = {
                "dtype": str(dtype),
                "shape": [b, n],
                "input_sha256": tensor_sha(x),
                "logical_split": 9,
                "fixed_final": str(final.directory),
                "plans": {},
                "rounds": [],
            }
            for name, k in kernels.items():
                k((n, b), x, tmp)
                final((b,), tmp, out)
                case["plans"][name] = {
                    "directory": str(k.directory),
                    "plan": plans[name].to_dict(),
                    "resources": k.kernel.resources,
                    "attributes": k.kernel.attributes,
                    "output": out.tolist(),
                }
            for round_id in range(3):
                names = list(kernels)
                for name in names[round_id:] + names[:round_id]:
                    k = kernels[name]
                    with k.kernel.prepared((n, b), x, tmp):
                        measured = KernelBenchmarker().benchmark_kernel_gpu(k.kernel)
                    values = [v * 1000 for v in measured["kernel_ms"]]
                    case["rounds"].append(
                        {
                            "round": round_id,
                            "plan": name,
                            **summary(values),
                            "kernel_count": measured["kernel_count"],
                        }
                    )
            case["comparison"] = {}
            for other in ("baseline", "compensated"):
                round_wins = 0
                round_losses = 0
                samples = {}
                for name in ("reference512", other):
                    samples[name] = [
                        v
                        for r in case["rounds"]
                        if r["plan"] == name
                        for v in r["samples_us"]
                    ]
                for r in range(3):
                    a = next(
                        t
                        for t in case["rounds"]
                        if t["round"] == r and t["plan"] == "reference512"
                    )
                    o = next(
                        t
                        for t in case["rounds"]
                        if t["round"] == r and t["plan"] == other
                    )
                    round_wins += a["median_us"] < o["median_us"]
                    round_losses += a["median_us"] > o["median_us"]
                a, o = summary(samples["reference512"]), summary(samples[other])
                case["comparison"][other] = {
                    "reference512_median_us": a["median_us"],
                    "other_median_us": o["median_us"],
                    "stable_win": round_wins >= 2
                    and a["median_us"] + 3 * a["mad_us"]
                    < o["median_us"] - 3 * o["mad_us"],
                    "stable_loss": round_losses >= 2
                    and o["median_us"] + 3 * o["mad_us"]
                    < a["median_us"] - 3 * a["mad_us"],
                    "round_medians": {
                        name: [
                            t["median_us"] for t in case["rounds"] if t["plan"] == name
                        ]
                        for name in ("reference512", other)
                    },
                }
            rows.append(case)
            print(
                str(dtype),
                b,
                n,
                {
                    k: (
                        v["reference512_median_us"],
                        v["other_median_us"],
                        v["stable_win"],
                        v["stable_loss"],
                    )
                    for k, v in case["comparison"].items()
                },
                flush=True,
            )
            (OUTPUT / "component-timings.json").write_text(
                json.dumps({"complete": False, "cases": rows}, indent=2)
            )
    evidence.state("after_merge_benchmark")
    result = {
        "complete": True,
        "scope": "prepared partial components; all final kernel work excluded",
        "cases": rows,
        "environment": inventory(torch, "cubecl"),
        "harness_sha256": sha(__file__),
    }
    (OUTPUT / "component-timings.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
