"""Investigate validation boundaries using existing frozen reductions only.

No eligibility, algorithm, split, tolerance, runtime or timing changes. Run on
the original RTX3090 so retained CUDA RNG input hashes remain authoritative.
"""

from itertools import chain
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts/reduction-structure"
OUT = ROOT / "artifacts/admission-boundary"


def main():
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.artifact import CompiledKernel
    from stage5.long_row_analyze import reference_for_extent
    from stage5.plan import PhysicalCandidate
    from stage5.reduction_codegen_mechanisms import Component
    from stage5.reduction_merge_qualification import core_inputs, extra_inputs, row
    from stage5.reduction_policy import qualify_case, TOLERANCES, VERSION
    from stage5.reference_reduction_probe import sha, tensor_sha
    from stage5.runtime import compare

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "results.json"
    if destination.exists():
        raise FileExistsError("preserve existing capture: " + str(destination))
    source_paths = [
        SOURCE / "rounding/topology/qualification.json",
        ROOT / "artifacts/adoption/partial-admission.json",
        SOURCE / "rounding/merge-qualification.json",
        SOURCE / "rounding/new-triton-reference.json",
        SOURCE / "triton-r0.json",
        *(
            ROOT / f"artifacts/long-row/triton-{h}.json"
            for h in ("short", "mid", "medium", "million")
        ),
    ]
    topology, admission, merge, extra, large, *other = [
        json.loads(p.read_text()) for p in source_paths
    ]
    assert topology["complete"] and not topology["failures"]
    assert admission["complete"]
    originals = {r["case"]: r for r in topology["rows"]}
    admission_rows = {r["case"]: r for r in admission["rows"]}
    references = {
        f"triton-{h}": data
        for h, data in zip(("short", "mid", "medium", "million"), other, strict=True)
    }
    references["triton-large"] = large
    kernels = {}
    for dtype in ("float32", "float16", "bfloat16"):
        frozen = Component(SOURCE / f"codegen/uniform/{dtype}")
        kernels[dtype] = {
            "reference512": frozen,
            "scalar_baseline": CompiledKernel(
                frozen.semantic,
                0,
                PhysicalCandidate("shared_tree_reduction_baseline").to_dict(),
            ),
            "compensated": CompiledKernel(
                frozen.semantic,
                0,
                PhysicalCandidate("shared_tree_compensated").to_dict(),
            ),
            "final": Component(
                SOURCE / f"rounding/topology/kernels/{dtype}/fixed_final"
            ),
        }

    rows = []
    for label, x in chain(core_inputs(torch), extra_inputs(torch)):
        b, n = x.shape
        dtype = str(x.dtype).removeprefix("torch.")
        group = kernels[dtype]
        reference = next(
            r
            for r in (
                extra
                if label.startswith("new_")
                else references[reference_for_extent(n)]
            )["rows"]
            if r["case"] == label
        )
        expected = originals[label]
        assert tensor_sha(x) == expected["input_sha256"] == reference["input_sha256"], (
            label
        )
        parts, outputs, outcomes, hashes = {}, {}, {}, {}
        for name in ("reference512", "scalar_baseline", "compensated"):
            parts[name] = torch.empty((b, 9), device=x.device)
            outputs[name] = torch.empty(b, dtype=x.dtype, device=x.device)
            group[name]((n, b), x, parts[name])
            group["final"]((b,), parts[name], outputs[name])
            actual = row(torch, label, x, outputs[name])
            outcomes[name] = qualify_case(actual, reference)
            hashes[name] = {
                "partial": tensor_sha(parts[name]),
                "output": tensor_sha(outputs[name]),
            }
        assert outputs["reference512"].tolist() == expected["values"], label
        assert (
            hashes["reference512"]["output"] == admission_rows[label]["output_sha256"]
        ), label
        error = None
        try:
            compare(
                parts["reference512"],
                parts["scalar_baseline"],
                group["reference512"].semantic,
                group["reference512"].kernel.plan,
            )
        except AssertionError as caught:
            error = str(caught)
        assert (error is None) == admission_rows[label][
            "normal_partial_comparison_passes"
        ], label
        final_error = None
        try:
            rtol, atol = TOLERANCES[str(x.dtype)]
            torch.testing.assert_close(
                outputs["reference512"],
                outputs["scalar_baseline"],
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        except AssertionError as caught:
            final_error = str(caught)
        rows.append(
            {
                "case": label,
                "dtype": str(x.dtype),
                "shape": [b, n],
                "input_sha256": tensor_sha(x),
                "hashes": hashes,
                "values": {name: y.tolist() for name, y in outputs.items()},
                "qualification": outcomes,
                "partial_comparison_error": error,
                "whole_old_tree_comparison_error": final_error,
            }
        )
    assert len(rows) == len(originals) == 148
    assert all(r["qualification"]["reference512"]["passes"] for r in rows)
    assert [r["case"] for r in rows if r["partial_comparison_error"]] == admission[
        "failures"
    ]

    # Re-evaluate retained rejected outputs using their original identity fields.
    # These are controls, not additional GPU candidates or changed tolerances.
    merge_rows = {r["case"]: r for r in merge["rows"]}
    negative, corruptions = [], []
    for r in rows:
        label = r["case"]
        old = merge_rows[label]
        reference = next(
            x
            for x in (
                extra
                if label.startswith("new_")
                else references[reference_for_extent(r["shape"][1])]
            )["rows"]
            if x["case"] == label
        )
        negative.append(qualify_case(old, reference))
        corrupt = {**originals[label], "values": list(originals[label]["values"])}
        corrupt["values"][0] += 1_000_000.0
        outcome = qualify_case(corrupt, reference)
        assert not outcome["passes"], label
        corruptions.append({"case": label, "rejected": True})
    retained_failed = [
        r["case"] for r in merge["checks"] if not r["passes"] and r["case"] in originals
    ]
    assert [r["case"] for r in negative if not r["passes"]] == retained_failed
    evidence.state("after_admission_boundary_probe")
    result = {
        "complete": True,
        "policy": VERSION,
        "timing_performed": False,
        "rows": rows,
        "summary": {
            "cases": len(rows),
            "qualification_failures": {
                name: [
                    r["case"] for r in rows if not r["qualification"][name]["passes"]
                ]
                for name in ("reference512", "scalar_baseline", "compensated")
            },
            "partial_old_tree_comparison_failures": [
                r["case"] for r in rows if r["partial_comparison_error"]
            ],
            "whole_old_tree_comparison_failures": [
                r["case"] for r in rows if r["whole_old_tree_comparison_error"]
            ],
            "retained_rejected_merge_failures": retained_failed,
            "corruption_controls_rejected": len(corruptions),
        },
        "negative_merge_checks": negative,
        "corruption_controls": corruptions,
        "artifacts": {
            dtype: {
                name: {
                    "directory": str(k.directory),
                    "ptx_sha256": sha(k.directory / "kernel.ptx"),
                }
                for name, k in group.items()
            }
            for dtype, group in kernels.items()
        },
        "source_sha256": {
            str(p.relative_to(ROOT.parent)): sha(p)
            for p in (
                *source_paths,
                Path(__file__),
                ROOT / "runtime.py",
                ROOT / "reduction_policy.py",
            )
        },
        "device": str(torch.cuda.get_device_properties(0)),
        "environment": evidence.inventory(),
    }
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


def oracle():
    """Measure/check reuse of the existing model; this is not a runtime hook."""
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.reduction_codegen_mechanisms import Component
    from stage5.reduction_merge_qualification import core_inputs, extra_inputs
    from stage5.reduction_topology_model import model
    from stage5.reference_reduction_probe import sha, tensor_sha

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    prior = json.loads((OUT / "results.json").read_text())
    expected = {r["case"]: r for r in prior["rows"]}
    destination = OUT / "oracle.json"
    if destination.exists():
        raise FileExistsError("preserve existing capture: " + str(destination))
    kernels = {
        dtype: Component(SOURCE / f"codegen/uniform/{dtype}")
        for dtype in ("float32", "float16", "bfloat16")
    }
    rows = []
    for label, x in chain(core_inputs(torch), extra_inputs(torch)):
        b, n = x.shape
        assert tensor_sha(x) == expected[label]["input_sha256"], label
        partials = torch.empty((b, 9), device=x.device)
        kernels[str(x.dtype).removeprefix("torch.")]((n, b), x, partials)
        assert (
            tensor_sha(partials) == expected[label]["hashes"]["reference512"]["partial"]
        ), label
        began = time.perf_counter()
        host = x.cpu()
        copied = time.perf_counter()
        m = model(host)
        finished = time.perf_counter()
        assert tensor_sha(m["partials"]) == tensor_sha(partials), label
        assert (
            tensor_sha(m["output"])
            == expected[label]["hashes"]["reference512"]["output"]
        ), label
        corrupted = m["partials"].clone()
        corrupted[0, 0] += 1_000_000.0
        assert tensor_sha(corrupted) != tensor_sha(partials), label
        rows.append(
            {
                "case": label,
                "input_sha256": tensor_sha(x),
                "partial_sha256": tensor_sha(partials),
                "output_sha256": tensor_sha(m["output"]),
                "cpu_model_bitwise": True,
                "corruption_control_rejected": True,
                "host_wall_copy_ms": (copied - began) * 1000,
                "host_wall_model_ms": (finished - copied) * 1000,
            }
        )
    assert len(rows) == 148
    evidence.state("after_admission_oracle_probe")
    destination.write_text(
        json.dumps(
            {
                "complete": True,
                "rows": rows,
                "scope": "existing CPU model reuse under explicit historical CPU FP environment; single host-wall observation per case, not device timing or warm benchmark",
                "production_runtime_hook": False,
                "source_sha256": {
                    str(p.relative_to(ROOT.parent)): sha(p)
                    for p in (
                        Path(__file__),
                        ROOT / "reduction_topology_model.py",
                        ROOT / "reduction_rounding_probe.py",
                        OUT / "results.json",
                    )
                },
                "environment": evidence.inventory(),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "148 original inputs: existing CPU partial/final model bitwise; 148 corruption controls rejected",
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("boundary", "oracle"), default="boundary", nargs="?"
    )
    args = parser.parse_args()
    (main if args.mode == "boundary" else oracle)()
