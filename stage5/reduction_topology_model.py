"""CPU FP32 model of the retained R2048/16-warp Triton partial.

No backend candidate is registered. Qualification uses the unchanged CubeCL
fixed-nine partial partition and final tree; the reference final is modeled
separately solely to check the reconstruction against executed Triton outputs.
"""

import json
from itertools import chain
from pathlib import Path

from stage5.reduction_rounding_probe import butterfly, finish_model

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts/reduction-structure"
OUTPUT = SOURCE / "rounding/topology"


def model(x):
    import torch

    assert x.device.type == "cpu" and x.ndim == 2
    b, n = x.shape
    extent = (n + 8) // 9
    data = torch.zeros((b, 9 * extent), dtype=torch.float32)
    data[:, :n] = x.float()
    data = data.reshape(b, 9, extent)
    steps = (extent + 2047) // 2048
    tiled = torch.zeros((b, 9, steps * 2048), dtype=torch.float32)
    tiled[..., :extent] = data
    chains = torch.zeros((b, 9, 4, 512), dtype=torch.float32)
    # Thread t, chain j, loop k: r=t+512*j+2048*k. LLVM selects the
    # old chain when r>=extent, but adds a protected zero when part*extent+r>=n.
    coords = torch.arange(2048).reshape(4, 512)
    for k in range(steps):
        pending = tiled[..., k * 2048 : (k + 1) * 2048].reshape(b, 9, 4, 512)
        chains = torch.where(coords + k * 2048 < extent, chains + pending, chains)
    a0, a1, a2, a3 = chains.unbind(-2)
    private = (a0 + a1) + (a2 + a3)
    # Every lane has the warp sum. All 32 lanes store that same value to
    # shared[warp] in the retained code (byte address (tid>>3)&60).
    warps = butterfly(private.reshape(b, 9, 16, 32), (16, 8, 4, 2, 1))
    shared = warps[..., 0]
    # Every warp reloads shared[tid&15]; each half-warp gets all 16 values.
    replicated = shared[..., torch.arange(32) & 15]
    reduced = butterfly(replicated, (8, 4, 2, 1))
    partials = reduced[..., 0]  # Only tid==0 owns the global partial store.
    reference_final = torch.zeros((b, 16), dtype=torch.float32)
    reference_final[:, :9] = partials
    return {
        "chains": chains,
        "private": private,
        "warps": shared,
        "partials": partials,
        "output": finish_model(partials).to(x.dtype),
        "reference_output": butterfly(reference_final, (8, 4, 2, 1))[:, 0].to(x.dtype),
        "truth_partials": data.double().sum(-1),
    }


def main():
    from stage1 import evidence

    evidence.guard()
    import torch

    from stage5.long_row_analyze import reference_for_extent
    from stage5.reduction_merge_qualification import core_inputs, extra_inputs, row
    from stage5.reduction_policy import qualify_case, VERSION
    from stage5.reference_reduction_probe import sha, tensor_sha

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    references = {
        f"triton-{h}": json.loads(
            (ROOT / f"artifacts/long-row/triton-{h}.json").read_text()
        )
        for h in ("short", "mid", "medium", "million")
    }
    references["triton-large"] = json.loads((SOURCE / "triton-r0.json").read_text())
    extra = json.loads((SOURCE / "rounding/new-triton-reference.json").read_text())
    previous = json.loads((SOURCE / "rounding/merge-qualification.json").read_text())
    rows, checks, reconstruction, traces = [], [], [], {}
    for label, x in chain(core_inputs(torch), extra_inputs(torch)):
        reference = next(
            r
            for r in (
                extra
                if label.startswith("new_")
                else references[reference_for_extent(x.shape[1])]
            )["rows"]
            if r["case"] == label
        )
        m = model(x.cpu())
        actual = row(torch, label, x, m["output"])
        outcome = qualify_case(actual, reference)
        prior = next(r for r in previous["rows"] if r["case"] == label)
        assert actual["input_sha256"] == prior["input_sha256"]
        rows.append(actual)
        checks.append(outcome)
        # The retained 65K graph is exactly R2048/16warps/split9. Other
        # corresponding-efficient regimes are qualification references only.
        if x.shape[1] == 65539:
            assert m["reference_output"].tolist() == reference["values"], label
            reconstruction.append(
                {
                    "case": label,
                    "output_bitwise": tensor_sha(m["reference_output"])
                    == tensor_sha(torch.tensor(reference["values"], dtype=x.dtype)),
                }
            )
        if label == "new_existing_finite_adversaries":
            assert m["partials"].tolist() == extra["adversarial_partial"]["values"]
        if not outcome["passes"] or label in (
            "heldout_10017",
            "new_existing_finite_adversaries",
        ):
            truth = torch.tensor(reference["fp64"], dtype=torch.float64)
            traces[label] = {
                "partials": m["partials"].tolist(),
                "truth_partials": m["truth_partials"].tolist(),
                "reference_final_output": m["reference_output"].tolist(),
                "cumulative_error": {
                    key: (
                        m[key].double().reshape(x.shape[0], -1).sum(-1) - truth
                    ).tolist()
                    for key in ("chains", "private", "warps", "partials")
                },
                "fixed_final_rounding": (
                    m["output"].double() - m["partials"].double().sum(-1)
                ).tolist(),
                "reference_final_rounding": (
                    m["reference_output"].double() - m["partials"].double().sum(-1)
                ).tolist(),
            }
    evidence.state("after_cpu_topology_qualification")
    result = {
        "complete": True,
        "base_revision": "b74c15c02aadd8afeb778eb7b6649350c2a73321",
        "policy": VERSION,
        "gpu_candidate_implemented": False,
        "rows": rows,
        "checks": checks,
        "failures": [c for c in checks if not c["passes"]],
        "reference_reconstruction": reconstruction,
        "adversarial_partials_match_executed_reference": True,
        "traces": traces,
        "harness_sha256": sha(__file__),
        "input_sources_sha256": {
            str(p.relative_to(ROOT.parent)): sha(p)
            for p in (
                SOURCE / "triton-r0.json",
                SOURCE / "rounding/new-triton-reference.json",
                SOURCE / "rounding/merge-qualification.json",
                *(
                    ROOT / f"artifacts/long-row/triton-{h}.json"
                    for h in ("short", "mid", "medium", "million")
                ),
            )
        },
        "independence": evidence.inventory(),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "qualification.json").write_text(json.dumps(result, indent=2))
    print(
        len(checks),
        "cases;",
        len(result["failures"]),
        "failed cases;",
        len(reconstruction),
        "reference reconstructions",
        flush=True,
    )
    for c in result["failures"]:
        print(c["case"], [e for e in c["elements"] if not e["passes"]], flush=True)


if __name__ == "__main__":
    main()
