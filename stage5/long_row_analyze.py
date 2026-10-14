"""Per-case qualification and device comparisons with inspectable provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
from stage5.reduction_policy import qualify_case, VERSION
from stage5.reduction_policy import TOLERANCES

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/long-row"
REPORT = ROOT.parent / "STAGE5_LONG_ROW_REPORT.md"
NAMES = (
    "baseline-fixed",
    "compensated-fixed",
    "combined-fixed",
    "combined-guarded-short",
    "combined-guarded-large",
    "production-guarded-short",
    "production-guarded-large",
    "triton-short",
    "triton-large",
    "triton-mid",
    "triton-medium",
    "triton-million",
)


def reference_for_extent(n):
    if n <= 257:
        return "triton-short"
    if n <= 8192:
        return "triton-mid"
    if n <= 73728:
        return "triton-large"
    if n <= 131072:
        return "triton-medium"
    return "triton-million"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timing(row):
    rounds = [
        [p["median_us"] for p in row["profiles"] if p["round"] == r] for r in range(3)
    ]
    assert all(len(r) == 7 for r in rounds), "incomplete timing rounds"
    values = [v for r in rounds for v in r]
    median = statistics.median(values)
    return {
        "median_us": median,
        "mad_us": statistics.median(abs(v - median) for v in values),
        "round_medians_us": list(map(statistics.median, rounds)),
        "samples_us": values,
        "kernels": row["profiles"][0]["kernels_per_call"],
    }


def analyze():
    data = {name: json.loads((OUTPUT / (name + ".json")).read_text()) for name in NAMES}
    truth = data["triton-large"]
    qualified = {}
    for name, capture in data.items():
        assert capture["qualification_complete"] and len(capture["rows"]) == 178
        env = capture["environment"]
        assert env["PYTHON_GIL"] is None and env["gil"] == name.startswith("triton")
        assert (
            env["fp_patch_sha256"]
            == "9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2"
        )
        assert env["torch_commit"] == truth["environment"]["torch_commit"]
        expected_history = {
            "triton-short": "short_first",
            "triton-large": "large_first",
            "triton-mid": "mid_first",
            "triton-medium": "medium_first",
            "triton-million": "million_first",
        }.get(name, "short_first" if name.endswith("-short") else "large_first")
        assert capture["history"] == expected_history, "wrong reference/history role"
        assert capture["implementation"] == (
            "triton" if name.startswith("triton") else "cubecl"
        )
        if not name.startswith("triton"):
            independent = env["independence"]
            assert (
                independent["loaded_modules"] == []
                and independent["native_mappings"] == []
            )
        cases = []
        for index, actual in enumerate(capture["rows"]):
            # Short rows use the nonsplit reference. Long rows use the efficient
            # split reference; the pathological short-first long graph is excluded.
            reference_name = reference_for_extent(actual["shape"][-1])
            reference = data[reference_name]["rows"][index]
            outcome = qualify_case(actual, reference)
            outcome["reference_capture"] = reference_name
            cases.append(outcome)
        qualified[name] = {
            "cases": cases,
            "failing_cases": [c["case"] for c in cases if not c["passes"]],
            "failing_elements": [
                {"case": c["case"], **e}
                for c in cases
                for e in c["elements"]
                if not e["passes"]
            ],
            "eager_diagnostic_failures": [
                r["case"] for r in capture["rows"] if not r["eager_gate_passes"]
            ],
            "all_cases_pass": all(c["passes"] for c in cases),
        }
    perf = {}
    measurements = {
        name: {r["case"]: r for r in d["timings"]} for name, d in data.items()
    }
    for label, baseline in measurements["baseline-fixed"].items():
        reference_name = reference_for_extent(baseline["shape"][-1])
        names = (
            "baseline-fixed",
            "compensated-fixed",
            "combined-fixed",
            "production-guarded-short",
            "production-guarded-large",
            reference_name,
        )
        entries = {}
        for name in names:
            r = measurements[name][label]
            assert r["input_sha256"] == baseline["input_sha256"]
            entries[name] = timing(r)
        candidate, incumbent = entries["combined-fixed"], entries["compensated-fixed"]
        wins = sum(
            a < b
            for a, b in zip(
                candidate["round_medians_us"],
                incumbent["round_medians_us"],
                strict=True,
            )
        )
        stable_win = (
            wins >= 2
            and candidate["median_us"] + 3 * candidate["mad_us"]
            < incumbent["median_us"] - 3 * incumbent["mad_us"]
        )
        stable_slowdown = (
            wins <= 1
            and incumbent["median_us"] + 3 * incumbent["mad_us"]
            < candidate["median_us"] - 3 * candidate["mad_us"]
        )
        perf[label] = {
            "measurements": entries,
            "reference_capture": reference_name,
            "candidate_stable_win": stable_win,
            "candidate_material_slowdown": stable_slowdown,
            "simplification_at_comparable_cost": not stable_win and not stable_slowdown,
        }
    result = {
        "policy": VERSION,
        "qualification": qualified,
        "performance": perf,
        "candidate_default_accepted": False,
        "decision": "Combined candidate is not eligible: per-element qualification fails. Compensation remains experimental continuity; cwk stays open.",
        "source_data_sha256": {name: sha(OUTPUT / (name + ".json")) for name in NAMES},
    }
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2))
    table = [
        "| Input | Baseline fixed µs | Compensated fixed µs | Combined fixed µs | Guarded production short-first µs | Guarded production long-first µs | Efficient Triton µs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, record in perf.items():
        values = record["measurements"]
        table.append(
            "| "
            + label
            + " | "
            + " | ".join(
                f"{values[n]['median_us']:.3f}"
                for n in (
                    "baseline-fixed",
                    "compensated-fixed",
                    "combined-fixed",
                    "production-guarded-short",
                    "production-guarded-large",
                    record["reference_capture"],
                )
            )
            + " |"
        )
    numerical = [
        "| Capture | Per-case failures | Failing output elements | Eager-relative diagnostic failures |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in NAMES:
        q = qualified[name]
        numerical.append(
            f"| {name} | {len(q['failing_cases'])} | {len(q['failing_elements'])} | {len(q['eager_diagnostic_failures'])} |"
        )
    return result, "\n".join(numerical), "\n".join(table), data


def curate(data):
    selected = OUTPUT / "selected"
    artifacts = {}
    for capture in data.values():
        for artifact in capture.get("artifacts", []):
            directory = Path(artifact["directory"])
            artifacts[str(directory)] = artifact
    for directory, artifact in artifacts.items():
        source = Path(directory)
        target = selected / source.name
        target.mkdir(parents=True, exist_ok=True)
        for filename in (
            "descriptor.json",
            "semantic.json",
            "physical-plan.json",
            "manifest.json",
            "kernel.cu",
            "kernel.ptx",
            "cubecl.ir",
            "nvrtc.json",
            "resources.json",
        ):
            shutil.copyfile(source / filename, target / filename)
    # Preserve only the actually dispatched Triton artifacts/wrappers.
    for name in (n for n in data if n.startswith("triton")):
        for case in data[name]["timings"]:
            for kernel in case["selected_kernels"]:
                source = (
                    OUTPUT / "representatives" / name / "triton" / kernel["cache_hash"]
                )
                target = selected / name / kernel["cache_hash"]
                target.mkdir(parents=True, exist_ok=True)
                for p in source.iterdir():
                    if p.suffix in (".ttir", ".ttgir", ".llir", ".ptx", ".json"):
                        shutil.copyfile(p, target / p.name)
                shutil.copyfile(kernel["wrapper"], target / "wrapper.py")
    return {
        str(p.relative_to(OUTPUT)): sha(p) for p in selected.rglob("*") if p.is_file()
    }


def normalization():
    import numpy as np

    captures = {
        n: json.loads((OUTPUT / ("normalization-" + n + ".json")).read_text())
        for n in ("cubecl", "triton")
    }
    arrays = {
        n: np.load(ROOT / "cache/long-row" / ("norm-" + n + ".npz"), allow_pickle=False)
        for n in captures
    }
    rows = []
    for n, capture in captures.items():
        assert capture["complete"] and len(capture["cases"]) == 18
        assert capture["environment"]["gil"] == (n == "triton")
        assert capture["environment"]["PYTHON_GIL"] is None
        assert capture["raw_arrays_sha256"] == sha(
            ROOT / "cache/long-row" / ("norm-" + n + ".npz")
        )
    for actual, reference in zip(
        captures["cubecl"]["cases"], captures["triton"]["cases"], strict=True
    ):
        for field in (
            "case",
            "phase",
            "dtype",
            "shape",
            "input_sha256",
            "upstream_sha256",
        ):
            assert actual[field] == reference[field], field
        suffix = "out" if actual["phase"] == "forward" else "gradient"
        key = actual["case"] + ":" + suffix
        a, t = arrays["cubecl"][key], arrays["triton"][key]
        assert a.shape == t.shape
        rtol, atol = TOLERANCES[actual["dtype"]]
        finite = np.isfinite(a) & np.isfinite(t)
        allowed = atol + rtol * np.abs(t[finite])
        differences = np.abs(a[finite] - t[finite])
        assert np.array_equal(np.isnan(a), np.isnan(t)) and np.array_equal(
            np.isinf(a), np.isinf(t)
        )
        assert np.all(differences <= allowed), key
        rows.append(
            {
                "case": actual["case"],
                "phase": actual["phase"],
                "dtype": actual["dtype"],
                "passes_every_element": True,
                "rtol": rtol,
                "atol": atol,
                "max_abs_difference": float(differences.max()),
                "cubecl_kernels": actual["kernels"],
                "triton_kernels": reference["kernels"],
            }
        )
    for a in arrays.values():
        a.close()
    result = {
        "complete": True,
        "cases": rows,
        "source_sha256": {
            n: sha(OUTPUT / ("normalization-" + n + ".json")) for n in captures
        },
        "meaning": "Configured Triton per-element comparison with arbitrary upstream; no ATen/Triton kernel inside CubeCL profiler regions",
    }
    (OUTPUT / "normalization-comparison.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-report", action="store_true")
    parser.add_argument("--curate", action="store_true")
    args = parser.parse_args()
    result, numerical, performance, data = analyze()
    normalization()
    content = REPORT.read_text()
    for marker, table in (("NUMERICAL", numerical), ("PERFORMANCE", performance)):
        start, end = f"<!-- {marker}_START -->", f"<!-- {marker}_END -->"
        before, remainder = content.split(start, 1)
        old, after = remainder.split(end, 1)
        if args.update_report:
            content = before + start + "\n" + table + "\n" + end + after
        else:
            assert old.strip() == table, "stale report table: " + marker
    if args.update_report:
        REPORT.write_text(content)
    if args.curate:
        (OUTPUT / "file-hashes.json").write_text(
            json.dumps(curate(data), indent=2, sort_keys=True)
        )
    (OUTPUT / "analysis-provenance.json").write_text(
        json.dumps(
            {
                "report_sha256": sha(REPORT),
                "analyzer_sha256": sha(__file__),
                "source_data_sha256": result["source_data_sha256"],
            },
            indent=2,
        )
    )
    print("Long-row analysis PASS", flush=True)


if __name__ == "__main__":
    main()
