"""Normal-selector admission checks against frozen Stage 5 components.

No eligibility changes or timing: distinguish offline whole-reduction
qualification from the existing selector's per-kernel baseline comparison.
"""

import json
import os
from itertools import chain
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/adoption"


def partial():
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.artifact import CompiledKernel
    from stage5.plan import PhysicalCandidate
    from stage5.reduction_codegen_mechanisms import Component
    from stage5.reduction_merge_qualification import core_inputs, extra_inputs
    from stage5.reference_reduction_probe import sha, tensor_sha
    from stage5.runtime import compare

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "artifacts/reduction-structure"
    qualification = json.loads(
        (source / "rounding/topology/qualification.json").read_text()
    )
    assert qualification["complete"] and not qualification["failures"]
    assert qualification["harness_sha256"] == sha(ROOT / "reduction_topology_model.py")
    rows_by_case = {r["case"]: r for r in qualification["rows"]}
    kernels = {}
    for dtype in ("float32", "float16", "bfloat16"):
        frozen = Component(source / "codegen/uniform" / dtype)
        baseline = CompiledKernel(
            frozen.semantic,
            0,
            PhysicalCandidate("shared_tree_reduction_baseline").to_dict(),
        )
        final = Component(source / "rounding/topology/kernels" / dtype / "fixed_final")
        kernels[dtype] = frozen, baseline, final
    rows = []
    for label, x in chain(core_inputs(torch), extra_inputs(torch)):
        b, n = x.shape
        dtype = str(x.dtype).removeprefix("torch.")
        frozen, baseline, final = kernels[dtype]
        parts = [torch.empty((b, 9), device="cuda") for _ in range(2)]
        frozen((n, b), x, parts[0])
        baseline((n, b), x, parts[1])
        y = torch.empty(b, dtype=x.dtype, device="cuda")
        final((b,), parts[0], y)
        expected = rows_by_case[label]
        assert tensor_sha(x) == expected["input_sha256"], label
        assert y.tolist() == expected["values"], label
        error = None
        try:
            compare(parts[0], parts[1], frozen.semantic, frozen.kernel.plan)
        except AssertionError as caught:
            error = str(caught)
        mismatch = (~torch.isclose(parts[0], parts[1], rtol=2e-5, atol=2e-5)).nonzero()
        rows.append(
            {
                "case": label,
                "dtype": dtype,
                "shape": [b, n],
                "input_sha256": tensor_sha(x),
                "output_sha256": tensor_sha(y),
                "frozen_whole_output_matches_qualified": True,
                "normal_partial_comparison_passes": error is None,
                "error": error,
                "mismatches": [
                    {
                        "coordinate": [int(v) for v in index],
                        "frozen": parts[0][tuple(index)].item(),
                        "normal_baseline": parts[1][tuple(index)].item(),
                    }
                    for index in mismatch.cpu().tolist()
                ],
            }
        )
    evidence.state("after_adoption_partial_controls")
    result = {
        "complete": True,
        "scope": "unchanged normal per-kernel comparison, not the whole-reduction qualification policy",
        "rows": rows,
        "failures": [r["case"] for r in rows if not r["normal_partial_comparison_passes"]],
        "qualification_sha256": sha(source / "rounding/topology/qualification.json"),
        "harness_sha256": sha(__file__),
        "environment": evidence.inventory(),
        "device": str(torch.cuda.get_device_properties(0)),
        "artifacts": {
            d: {
                name: {
                    "directory": str(k.directory),
                    "ptx_sha256": sha(k.directory / "kernel.ptx"),
                }
                for name, k in zip(("frozen", "normal_baseline", "final"), group, strict=True)
            }
            for d, group in kernels.items()
        },
    }
    (OUT / "partial-admission.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(rows)} frozen whole outputs match; {len(result['failures'])} normal partial comparisons fail", flush=True)


if __name__ == "__main__":
    partial()
