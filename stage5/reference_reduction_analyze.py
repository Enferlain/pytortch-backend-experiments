"""Validate matched inputs and curate the bounded reduction comparison."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/reference-reduction"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, atol=2e-5, rtol=2e-5):
    return all(abs(x - y) <= atol + rtol * abs(y) for x, y in zip(a, b, strict=True))


def bind_report(result, physical, update):
    path = ROOT.parent / "STAGE5_REDUCTION_REFERENCE_REPORT.md"
    report = path.read_text()
    tables = []
    labels = [
        ("CubeCL first_use, short first", "cubecl-short-warm"),
        ("CubeCL first_use, long first", "cubecl-large-warm"),
        ("Configured Triton, short first", "triton-short-warm"),
        ("Configured Triton, long first", "triton-large-warm"),
        ("CubeCL explicit baseline, long first", "cubecl-baseline-large"),
    ]
    lines = [
        "| Executed path | Eager-relative failing inputs | FP64 diagnostic failing inputs |",
        "| --- | ---: | ---: |",
    ]
    for label, name in labels:
        n = result["numerical"][name]
        lines.append(
            f"| {label} | {len(n['eager_failures'])} | {len(n['fp64_diagnostic_failures'])} |"
        )
    tables.append("\n".join(lines))
    lines = [
        "| Physical plan | Eager-relative failures | Differences beyond 2e-5 from long-first Triton |",
        "| --- | ---: | ---: |",
    ]
    for label, row in zip(
        (
            "Original shared tree",
            "Compensated shared tree",
            "Four private accumulators + shared tree",
            "Existing scalar plane/shared reduction",
        ),
        result["physical_diagnostics"],
        strict=True,
    ):
        lines.append(
            f"| {label} | {len(row['eager_failures'])} | {len(row['triton_large_differences'])} |"
        )
    tables.append("\n".join(lines))
    lookup = {r["case"]: r for r in result["timing"]}
    lines = [
        "| Input | CubeCL short-first µs | Triton short-first µs | CubeCL long-first µs | Triton long-first µs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, case in (
        ("FP32 8×65539", "float32-8-65539"),
        ("FP32 64×65539", "float32-64-65539"),
        ("FP16 64×65539", "float16-64-65539"),
        ("BF16 64×65539", "bfloat16-64-65539"),
        ("FP32 8×131071", "float32-8-131071"),
        ("FP32 8×1048576", "float32-8-1048576"),
    ):
        values = [
            lookup[case]["measurements"][name]["median_kernel_sum_us"]
            for name in (
                "cubecl-short-warm",
                "triton-short-warm",
                "cubecl-large-warm",
                "triton-large-warm",
            )
        ]
        lines.append(
            "| " + label + " | " + " | ".join(f"{v:.3f}" for v in values) + " |"
        )
    tables.append("\n".join(lines))
    kernels = [k for capture in result["nsight"] for k in capture["kernels"]]
    lines = [
        "| Metric | CubeCL compensated component | Triton partial | Triton final |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, name in (
        ("Grid Size", "Grid Size"),
        ("Block Size", "Block Size"),
        ("Registers Per Thread", "Registers Per Thread"),
        ("Dynamic Shared Memory Per Block (bytes)", "Dynamic Shared Memory Per Block"),
        ("Waves Per SM", "Waves Per SM"),
        ("Duration (ns)", "Duration"),
        ("Compute (SM) Throughput (%)", "Compute (SM) Throughput"),
        ("Memory Throughput (%)", "Memory Throughput"),
    ):
        values = [
            next(m["value"] for m in k["metrics"] if m["name"] == name).replace(",", "")
            for k in kernels
        ]
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    tables.append("\n".join(lines))
    for table in tables:
        header = table.splitlines()[0]
        start = report.index(header)
        end = report.index("\n\n", start)
        if update:
            report = report[:start] + table + report[end:]
        else:
            assert report[start:end] == table, "report table is stale: " + header
    # Bind quoted timing numbers as well as tables; regenerate these small
    # paragraphs so a fresh capture cannot leave an older cost sentence behind.
    m = lookup["float32-8-1048576"]["measurements"]
    small = lookup["float32-8-65539"]["measurements"]
    paragraphs = [
        (
            "The same-topology **explicit CubeCL baseline diagnostic**",
            f"The same-topology **explicit CubeCL baseline diagnostic** performs the long-first\n8×1048576 case in {m['cubecl-baseline-large']['median_kernel_sum_us']:.3f}µs versus {m['cubecl-large-warm']['median_kernel_sum_us']:.3f}µs for first_use and {m['triton-large-warm']['median_kernel_sum_us']:.3f}µs for Triton.\nIts eager/FP64 diagnostic differences remain listed above. At 8×65539, its\n{small['cubecl-baseline-large']['median_kernel_sum_us']:.3f}µs versus Triton's {small['triton-large-warm']['median_kernel_sum_us']:.3f}µs. Its final reduction also differs\nfrom first_use's selection, so this is not an isolated compensation-only ablation.",
        )
    ]
    values = {
        r["plan"]["strategy"]: r["median_kernel_us"]
        for r in physical["timings"]
        if r["shape"] == [8, 65539]
    }
    paragraphs.append(
        (
            "The prepared-native physical component confirms",
            f"The prepared-native physical component confirms the direct arithmetic cost:\n8×65539 is {values['shared_tree_reduction_baseline']:.3f}µs original, {values['shared_tree_compensated']:.3f}µs compensated, {values['shared_tree_multi_accumulator']:.3f}µs with four accumulators,\nand {values['plane_shared']:.3f}µs for the existing scalar plane/shared plan. These component timings\nare kept separate from the compiled graph's multi-kernel sums.",
        )
    )
    for marker, text in paragraphs:
        start = report.index(marker)
        end = report.index("\n\n", start)
        if update:
            report = report[:start] + text + report[end:]
        else:
            assert report[start:end] == text, "report cost paragraph is stale"
    if update:
        path.write_text(report)
    result["report_binding"] = {
        "path": str(path),
        "sha256": sha(path),
        "tables_and_cost_paragraphs_verified": True,
        "numbers_source_sha256": hashlib.sha256(
            json.dumps(
                {
                    "numerical": result["numerical"],
                    "timing": result["timing"],
                    "physical": physical["timings"],
                    "nsight": result["nsight"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }


def analyze(update_report=False):
    names = (
        "cubecl-short-warm",
        "cubecl-large-warm",
        "triton-short-warm",
        "triton-large-warm",
        "cubecl-baseline-large",
    )
    data = {name: json.loads((OUTPUT / (name + ".json")).read_text()) for name in names}
    assert all(d["qualification_complete"] for d in data.values())
    reference = data[names[0]]
    for name, d in data.items():
        assert len(d["graphs"]) == 3, "unexpected Dynamo specialization"
        if name.startswith("cubecl"):
            assert d["cubecl_policy"] == (
                "baseline" if name == "cubecl-baseline-large" else "first_use"
            )
        assert len(d["rows"]) == len(reference["rows"]) == 115
        assert (
            d["environment"]["torch_commit"] == reference["environment"]["torch_commit"]
        )
        assert d["environment"]["PYTHON_GIL"] is None
        assert d["environment"]["gil"] == name.startswith("triton")
        for r, ref in zip(d["rows"], reference["rows"], strict=True):
            assert (
                r["case"],
                r["dtype"],
                r["shape"],
                r["input_sha256"],
                r["aten"],
                r["fp64"],
            ) == (
                ref["case"],
                ref["dtype"],
                ref["shape"],
                ref["input_sha256"],
                ref["aten"],
                ref["fp64"],
            ), (name, r["case"])
        for row, ref in zip(d["timings"], reference["timings"], strict=True):
            assert (row["case"], row["input_sha256"]) == (
                ref["case"],
                ref["input_sha256"],
            )
            assert len(row["profiles"]) == 7
    physical = json.loads((OUTPUT / "physical.json").read_text())
    assert physical["qualification_complete"] and len(physical["rows"]) == 109
    for row, ref in zip(physical["rows"], reference["rows"][:109], strict=True):
        assert (row["case"], row["input_sha256"], row["aten"], row["fp64"]) == (
            ref["case"],
            ref["input_sha256"],
            ref["aten"],
            ref["fp64"],
        )
    numerical = {}
    for name, d in data.items():
        numerical[name] = {
            "core_fp32_cases": 109,
            "eager_failures": [
                r["case"] for r in d["rows"][:109] if not r["eager_gate_passes"]
            ],
            "fp64_diagnostic_failures": [
                r["case"] for r in d["rows"][:109] if not close(r["values"], r["fp64"])
            ],
        }
    direct = []
    for i, plan in enumerate(physical["rows"][0]["plans"]):
        direct.append(
            {
                "plan": plan["plan"],
                "eager_failures": [
                    r["case"]
                    for r in physical["rows"]
                    if not r["plans"][i]["eager_gate_passes"]
                ],
                "triton_large_differences": [
                    r["case"]
                    for r, t in zip(
                        physical["rows"],
                        data["triton-large-warm"]["rows"][:109],
                        strict=True,
                    )
                    if not close(r["plans"][i]["values"], t["values"])
                ],
            }
        )
    timing = []
    for i, base in enumerate(reference["timings"]):
        row = {
            "case": base["case"],
            "shape": base["shape"],
            "dtype": base["dtype"],
            "input_sha256": base["input_sha256"],
            "measurements": {},
        }
        for name, d in data.items():
            r = d["timings"][i]
            values = [p["median_us"] for p in r["profiles"]]
            row["measurements"][name] = {
                "median_kernel_sum_us": statistics.median(values),
                "mad_us": statistics.median(
                    abs(v - statistics.median(values)) for v in values
                ),
                "samples_us": values,
                "kernels_per_call": r["profiles"][0]["kernels_per_call"],
            }
        timing.append(row)
    nsight = []
    for name, expected in (("compensated", 1), ("triton-large-primed", 2)):
        path = ROOT / "cache/reference-reduction" / ("ncu-" + name + ".csv")
        text = path.read_text()
        records = list(csv.DictReader(text[text.index('"ID"') :].splitlines()))
        ids = {r["ID"] for r in records}
        assert len(ids) == expected, "incomplete Nsight capture"
        nsight.append(
            {
                "case": name,
                "raw_csv": str(path),
                "sha256": sha(path),
                "kernels": [
                    {
                        "id": index,
                        "name": next(
                            r["Kernel Name"] for r in records if r["ID"] == index
                        ),
                        "metrics": [
                            {
                                "section": r["Section Name"],
                                "name": r["Metric Name"],
                                "value": r["Metric Value"],
                                "unit": r["Metric Unit"],
                            }
                            for r in records
                            if r["ID"] == index and r["Metric Name"]
                        ],
                    }
                    for index in sorted(ids)
                ],
            }
        )
    # Retain executed kernels, not every compiler cache or profiling binary.
    representatives = OUTPUT / "selected"
    for name, d in data.items():
        if name.startswith("triton"):
            for case in d["timings"]:
                if case["dtype"] != "torch.float32" or case["shape"] != [8, 65539]:
                    continue
                for kernel in case["selected_kernels"]:
                    src = (
                        OUTPUT
                        / "representatives"
                        / name
                        / "triton"
                        / kernel["cache_hash"]
                    )
                    dst = representatives / name / kernel["cache_hash"]
                    dst.mkdir(parents=True, exist_ok=True)
                    for p in src.iterdir():
                        if p.suffix in (".ttir", ".ttgir", ".llir", ".ptx", ".json"):
                            shutil.copyfile(p, dst / p.name)
                    wrapper = Path(kernel["wrapper"])
                    shutil.copyfile(wrapper, dst / "wrapper.py")
        else:
            for item in d["artifacts"]:
                src = Path(item["directory"])
                dst = representatives / name / src.name
                dst.mkdir(parents=True, exist_ok=True)
                for p in src.iterdir():
                    if p.suffix in (".json", ".ir", ".cu", ".ptx", ".txt"):
                        shutil.copyfile(p, dst / p.name)
    for item in physical["artifacts"]:
        src = Path(item["directory"])
        dst = representatives / "physical" / src.name
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if p.suffix in (".json", ".ir", ".cu", ".ptx", ".txt"):
                shutil.copyfile(p, dst / p.name)
    result = {
        "comparison_complete": True,
        "default_policy_accepted": False,
        "inputs_byte_matched": True,
        "numerical": numerical,
        "physical_diagnostics": direct,
        "timing": timing,
        "nsight": nsight,
        "nsight_version": subprocess.check_output(["ncu", "--version"], text=True),
        "sources": {p.name: sha(p) for p in ROOT.glob("reference_reduction_*.py")},
        "capture_hashes": {name: sha(OUTPUT / (name + ".json")) for name in names},
        "interpretation": "diagnostic comparison; existing tolerances and production selection are unchanged",
    }
    bind_report(result, physical, update_report)
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-report", action="store_true")
    analyze(parser.parse_args().update_report)
