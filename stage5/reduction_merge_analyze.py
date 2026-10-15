"""Verify the rounding experiment and retain compact, reproducible evidence."""

import difflib
import json
import re
import shutil
from pathlib import Path

from stage5.long_row_analyze import reference_for_extent
from stage5.reduction_policy import TOLERANCES, qualify_case
from stage5.reduction_rounding_probe import OUTPUT, SOURCE
from stage5.reduction_structure_analyze import verify_environment
from stage5.reference_reduction_probe import sha

ROOT = Path(__file__).resolve().parent


def graph():
    capture = json.loads((OUTPUT / "graph-large.json").read_text())
    verify_environment(capture["environment"], True)
    references = {
        f"triton-{h}": json.loads(
            (ROOT / f"artifacts/long-row/triton-{h}.json").read_text()
        )
        for h in ("short", "mid", "large", "medium", "million")
    }
    references["triton-large"] = json.loads((SOURCE / "triton-r0.json").read_text())
    checks = []
    profiles = []
    for row in capture["rows"]:
        reference = next(
            r
            for r in references[reference_for_extent(row["shape"][1])]["rows"]
            if r["case"] == row["case"]
        )
        checks.append(qualify_case(row, reference))
    assert len(checks) == 178 and all(c["passes"] for c in checks)
    for timing in capture["timings"]:
        samples = []
        for sample in timing["profiles"]:
            path = Path(sample["raw_trace"])
            assert sha(path) == sample["sha256"]
            kernels = sorted(
                (
                    e
                    for e in json.loads(path.read_text())["traceEvents"]
                    if e.get("cat") == "kernel"
                ),
                key=lambda e: e["ts"],
            )
            assert [e["name"] for e in kernels] == sample["names"]
            assert len(kernels) == sample["kernels_per_call"]
            assert all(e["name"].startswith("cubecl5_") for e in kernels)
            samples.append(
                {
                    "partial_us": kernels[0]["dur"],
                    "final_us": kernels[1]["dur"] if len(kernels) == 2 else None,
                    "kernel_sum_us": sum(e["dur"] for e in kernels),
                    "kernel_count": len(kernels),
                }
            )
        profiles.append(
            {
                "case": timing["case"],
                "samples": samples,
                "selected_kernels": timing["selected_kernels"],
            }
        )
    return {
        "qualification": checks,
        "profiles": profiles,
        "graphs": capture["graphs"],
        "split_regimes": capture["split_regimes"],
        "note": "Guarded graph capture, not rotated against reference or a fixed-split timing comparison",
    }


def normalization():
    import numpy as np

    capture = json.loads((OUTPUT / "normalization-cubecl.json").read_text())
    reference = json.loads(
        (ROOT / "artifacts/long-row/normalization-triton.json").read_text()
    )
    assert (
        capture["complete"] and capture["cubecl_policy"] == "adjacent_merge_diagnostic"
    )
    verify_environment(capture["environment"], True)
    verify_environment(reference["environment"], False)
    actual_path = ROOT / "cache/reduction-rounding/norm/norm-cubecl.npz"
    reference_path = ROOT / "cache/long-row/norm-triton.npz"
    assert sha(actual_path) == capture["raw_arrays_sha256"]
    assert sha(reference_path) == reference["raw_arrays_sha256"]
    results = []
    with np.load(actual_path) as actual, np.load(reference_path) as expected:
        for c, r in zip(capture["cases"], reference["cases"], strict=True):
            for field in (
                "case",
                "phase",
                "dtype",
                "shape",
                "input_sha256",
                "upstream_sha256",
            ):
                assert c[field] == r[field], field
            assert sha(c["raw_trace"]) == c["trace_sha256"]
            rtol, atol = TOLERANCES[c["dtype"]]
            key = c["case"] + (":out" if c["phase"] == "forward" else ":gradient")
            a, b = actual[key], expected[key]
            assert np.all(np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)), key
            results.append(
                {
                    "case": c["case"],
                    "phase": c["phase"],
                    "passes": True,
                    "max_abs_error": float(np.max(np.abs(a - b))),
                    "rtol": rtol,
                    "atol": atol,
                    "kernel_count": len(c["kernels"]),
                }
            )
    assert len(results) == 18
    return results


def retain_kernels():
    qualification = json.loads((OUTPUT / "merge-qualification.json").read_text())
    retained = {}
    for dtype, kernels in qualification["kernels"].items():
        for name, directory in kernels.items():
            source = Path(directory)
            target = OUTPUT / "kernels" / dtype / name
            target.mkdir(parents=True, exist_ok=True)
            for file in (
                "descriptor.json",
                "semantic.json",
                "physical-plan.json",
                "cubecl.ir",
                "kernel.cu",
                "kernel.ptx",
                "manifest.json",
                "nvrtc.json",
                "resources.json",
                "integrity.json",
            ):
                shutil.copyfile(source / file, target / file)
            retained[f"{dtype}/{name}"] = {
                "source": str(source),
                "retained": str(target),
                "cuda_sha256": sha(target / "kernel.cu"),
                "ptx_sha256": sha(target / "kernel.ptx"),
            }
        a, b = [Path(kernels[n]) / "kernel.cu" for n in ("baseline", "adjacent")]
        normalize = lambda p: re.sub(
            r"cubecl5_[0-9a-f]+", "cubecl5_ENTRY", p.read_text()
        )
        # Same IR operation count/SSA numbering; only the operands of the first
        # two private-merge additions change. Final publication is unchanged.
        before, after = normalize(a), normalize(b)
        diff = list(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="baseline.cu",
                tofile="adjacent.cu",
            )
        )
        (OUTPUT / "kernels" / dtype / "merge.diff").write_text("".join(diff))
        removed = [x for x in diff if x.startswith("-") and not x.startswith("---")]
        added = [x for x in diff if x.startswith("+") and not x.startswith("+++")]
        assert len(removed) == len(added) == 2, (dtype, removed, added)
        assert all(" + " in line for line in removed + added)
        assert re.findall(
            r"add\.f32", (Path(kernels["baseline"]) / "kernel.ptx").read_text()
        ) == re.findall(
            r"add\.f32", (Path(kernels["adjacent"]) / "kernel.ptx").read_text()
        )
        # Rebuilding the fixed final with the changed compiler must not change
        # its generated code; compiler binary identity alone will change caches.
        old = json.loads((SOURCE / "cubecl-r0.json").read_text())
        timed = next(t for t in old["timings"] if t["case"] == f"{dtype}-8-65539")
        historical = Path(timed["selected_kernels"][-1]["directory"])
        for file in ("kernel.cu", "kernel.ptx"):
            assert (historical / file).read_bytes() == (
                Path(kernels["fixed_final"]) / file
            ).read_bytes()
    return retained


def main():
    q = json.loads((OUTPUT / "merge-qualification.json").read_text())
    assert q["complete"] and len(q["checks"]) == 149
    ordinary = [
        c for c in q["checks"] if c["case"] != "new_existing_finite_adversaries"
    ]
    assert len(ordinary) == 148 and all(c["passes"] for c in ordinary)
    assert [c["case"] for c in q["failures"]] == ["new_existing_finite_adversaries"]
    assert q["harness_sha256"] == sha(ROOT / "reduction_merge_qualification.py")
    verify_environment(q["environment"], True)
    new = json.loads((OUTPUT / "new-triton-reference.json").read_text())
    assert new["harness_sha256"] == sha(ROOT / "reduction_merge_qualification.py")
    verify_environment(new["environment"], False)
    assert new["environment"]["torch_commit"] == q["environment"]["torch_commit"]
    assert new["adversarial_partial"]["values"] == q["adversarial_partials"]["baseline"]
    assert [row[0] for row in q["adversarial_partials"]["adjacent"][:3]] == [0, 0, 0]
    assert [row[0] for row in new["adversarial_partial"]["values"][:3]] == [1, -1, 0.25]
    b = json.loads((OUTPUT / "component-timings.json").read_text())
    assert b["complete"] and len(b["cases"]) == 12
    assert b["harness_sha256"] == sha(ROOT / "reduction_merge_benchmark.py")
    verify_environment(b["environment"], True)
    assert all(not c["comparison"]["baseline"]["stable_loss"] for c in b["cases"])
    trace = json.loads((OUTPUT / "trace.json").read_text())
    assert trace["harness_sha256"] == sha(ROOT / "reduction_rounding_probe.py")
    assert trace["baseline_model_partials_and_output_bitwise"]
    result = {
        "complete": True,
        "working_policy": "diagnostic only; adjacent_merge_diagnostic is not a qualified replacement",
        "default_selection": "candidate rejected by existing adversarial qualification; no new default",
        "qualification_failures": q["failures"],
        "trace": {
            name: {k: v for k, v in r.items() if not isinstance(v, list)}
            for name, r in trace["models"].items()
        },
        "component_case_count": 12,
        "direct_qualification_count": len(q["checks"]),
        "graph": graph(),
        "normalization": normalization(),
        "retained_kernels": retain_kernels(),
    }
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2))
    print(
        "148 direct+178 graph pass; added adversarial case fails3 elements;18 RMS phases pass; fixed final/code differences verified"
    )


if __name__ == "__main__":
    main()
