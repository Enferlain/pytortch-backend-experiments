"""Trace FP32 reduction rounding against executed, fixed semantic packets.

CPU models are diagnostic arithmetic ablations, not backend candidates. Their
baseline must reproduce both executed partial-buffer and final-output bytes.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts/reduction-structure"
OUTPUT = SOURCE / "rounding"


def butterfly(values, distances):
    import torch

    lanes = torch.arange(values.shape[-1])
    for distance in distances:
        values = values + values[..., lanes ^ distance]
    return values


def finish_model(partials):
    import torch

    values = torch.zeros((*partials.shape[:-1], 256), dtype=torch.float32)
    values[..., : partials.shape[-1]] = partials
    for distance in (128, 64, 32, 16, 8, 4, 2, 1):
        values = values[..., :distance] + values[..., distance : 2 * distance]
    return values[..., 0]


def model(x, *, adjacent=False, descending=False):
    """Exact Stage5 scalar256/four-chain work and neutral-fill semantics."""
    import torch

    assert x.device.type == "cpu" and x.ndim == 2
    b, n = x.shape
    extent = (n + 8) // 9
    padded = torch.zeros((b, extent * 9), dtype=torch.float32)
    padded[:, :n] = x.float()
    data = padded.reshape(b, 9, extent)
    steps = (extent + 1023) // 1024
    tiled = torch.zeros((b, 9, steps * 1024), dtype=torch.float32)
    tiled[..., :extent] = data
    chains = torch.zeros((b, 9, 4, 256), dtype=torch.float32)
    for step in range(steps):
        chains = chains + tiled[..., step * 1024 : (step + 1) * 1024].reshape(
            b, 9, 4, 256
        )
    a0, a1, a2, a3 = chains.unbind(-2)
    private = (a0 + a1) + (a2 + a3) if adjacent else (a0 + a2) + (a1 + a3)
    distances = (16, 8, 4, 2, 1) if descending else (1, 2, 4, 8, 16)
    plane = butterfly(private.reshape(b, 9, 8, 32), distances)[..., 0]
    shared = torch.zeros((b, 9, 32), dtype=torch.float32)
    shared[..., :8] = plane
    partials = butterfly(shared, distances)[..., 0]
    return {
        "chains": chains,
        "private": private,
        "plane": plane,
        "partials": partials,
        "output": finish_model(partials),
        "truth_partials": data.double().sum(-1),
    }


def main():
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch

    from stage5.artifact import CompiledKernel
    from stage5.plan import PhysicalCandidate
    from stage5.reference_reduction_probe import sha, tensor_sha

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    capture = json.loads((SOURCE / "cubecl-r0.json").read_text())
    source = next(t for t in capture["timings"] if t["case"] == "float32-8-65539")
    packets = [
        json.loads((Path(k["directory"]) / "descriptor.json").read_text())
        for k in source["selected_kernels"]
    ]
    plan = PhysicalCandidate(
        "plane_shared_load_batch", plane_width=32, private_accumulators=4
    )
    partial = CompiledKernel(packets[0]["semantic"], 0, plan.to_dict())
    final = CompiledKernel(packets[1]["semantic"], 0, packets[1]["physical_plan"])
    torch.manual_seed(10017)
    x = torch.randn((8, 65539), device="cuda")
    reference = next(
        r
        for r in json.loads((SOURCE / "triton-r0.json").read_text())["rows"]
        if r["case"] == "heldout_10017"
    )
    assert tensor_sha(x) == reference["input_sha256"]
    tmp, out = torch.empty((8, 9), device="cuda"), torch.empty(8, device="cuda")
    partial((65539, 8), x, tmp)
    final((8,), tmp, out)
    models = {
        "baseline": model(x.cpu()),
        "adjacent_private_only": model(x.cpu(), adjacent=True),
        "descending_collective_only": model(x.cpu(), descending=True),
    }
    baseline = models["baseline"]
    assert tensor_sha(baseline["partials"]) == tensor_sha(tmp)
    assert tensor_sha(baseline["output"]) == tensor_sha(out)
    row = 4
    truth = reference["fp64"][row]
    result = {
        "question": "Locate heldout_10017 output4 error without compensation",
        "input_sha256": tensor_sha(x),
        "partial_kernel": str(partial.directory),
        "fixed_final": str(final.directory),
        "partial_sha256": tensor_sha(tmp),
        "output_sha256": tensor_sha(out),
        "baseline_model_partials_and_output_bitwise": True,
        "reference": reference,
        "models": {},
        "harness_sha256": sha(__file__),
    }
    for name, m in models.items():
        p = m["partials"][row]
        tp = m["truth_partials"][row]
        result["models"][name] = {
            "output": m["output"].tolist(),
            "row4_partials": p.tolist(),
            "row4_exact_partials": tp.tolist(),
            "row4_partial_errors": (p.double() - tp).tolist(),
            "row4_partial_errors_sum": (p.double().sum() - tp.sum()).item(),
            "row4_final_rounding": (
                m["output"][row].double() - p.double().sum()
            ).item(),
            "row4_total_error": m["output"][row].item() - truth,
            "row4_planes": m["plane"][row].tolist(),
            "row4_private": m["private"][row].tolist(),
            "row4_cumulative_rounding": {
                "private_chains": m["chains"][row].double().sum().item() - truth,
                "private_merge": m["private"][row].double().sum().item() - truth,
                "intra_plane": m["plane"][row].double().sum().item() - truth,
                "inter_plane": p.double().sum().item() - truth,
                "final": m["output"][row].item() - truth,
            },
        }
    evidence.state("after_rounding_trace")
    result["independence"] = evidence.inventory()
    (OUTPUT / "trace.json").write_text(json.dumps(result, indent=2))
    for name, r in result["models"].items():
        print(
            name,
            {k: v for k, v in r.items() if "row4_" in k and not isinstance(v, list)},
        )


if __name__ == "__main__":
    main()
