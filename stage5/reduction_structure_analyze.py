"""Verify current matched captures and preserve adjacent machine-code evidence."""

import csv
import json
import shutil
from pathlib import Path

from stage5.long_row_analyze import reference_for_extent
from stage5.long_row_diagnose import distribution, instructions
from stage5.reduction_policy import qualify_case
from stage5.reference_reduction_probe import sha

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/reduction-structure"
CACHE = ROOT / "cache/reduction-structure"


def verify_environment(env, cube):
    assert env["gil"] == (not cube) and env["PYTHON_GIL"] is None
    assert (
        env["fp_patch_sha256"]
        == "9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2"
    )
    if cube:
        assert env["independence"]["loaded_modules"] == []
        assert env["independence"]["native_mappings"] == []


def fingerprint(kernel, cube):
    if cube:
        p = Path(kernel["directory"])
        return {
            "descriptor": sha(p / "descriptor.json"),
            "ptx": sha(p / "kernel.ptx"),
            "plan": kernel["plan"],
        }
    return {
        k: kernel[k]
        for k in ("entrypoint", "config", "cache_hash", "registers", "spills", "shared")
    }


def baseline():
    captures = {
        impl: [json.loads((OUTPUT / f"{impl}-r{i}.json").read_text()) for i in range(3)]
        for impl in ("cubecl", "triton")
    }
    source_hashes = {
        f"{impl}-r{i}.json": sha(OUTPUT / f"{impl}-r{i}.json")
        for impl in captures
        for i in range(3)
    }
    for impl, rounds in captures.items():
        for capture in rounds:
            assert capture["qualification_complete"] and len(capture["rows"]) == 178
            verify_environment(capture["environment"], impl == "cubecl")
    result = {
        "scope": "real torch.compile; fixed nine logical partials; fixed final packet per backend/dtype",
        "capture_hashes": source_hashes,
        "cases": [],
        "qualification": [],
    }
    for case in [
        f"{dtype}-{b}-65539"
        for dtype in ("float32", "float16", "bfloat16")
        for b in (8, 64)
    ]:
        row = {"case": case, "implementations": {}}
        input_hash = None
        for impl, rounds in captures.items():
            timed = [next(t for t in c["timings"] if t["case"] == case) for c in rounds]
            ids = [
                [fingerprint(k, impl == "cubecl") for k in t["selected_kernels"]]
                for t in timed
            ]
            assert len(ids[0]) == 2 and all(v == ids[0] for v in ids), (
                "changed partial/final artifact"
            )
            summaries = []
            for timing in timed:
                if input_hash is None:
                    input_hash = timing["input_sha256"]
                assert input_hash == timing["input_sha256"], "unmatched input bytes"
                assert len(timing["profiles"]) == 7
                components, sums = [[], []], []
                for record in timing["profiles"]:
                    p = Path(record["raw_trace"])
                    assert sha(p) == record["sha256"], "modified trace"
                    events = sorted(
                        (
                            e
                            for e in json.loads(p.read_text())["traceEvents"]
                            if e.get("cat") == "kernel"
                        ),
                        key=lambda e: e["ts"],
                    )
                    assert (
                        len(events) == 2
                        and [e["name"] for e in events] == record["names"]
                    )
                    assert all(e["dur"] > 0 for e in events)
                    for j, event in enumerate(events):
                        components[j].append(event["dur"])
                    sums.append(sum(e["dur"] for e in events))
                summaries.append(
                    {
                        "partial": distribution(components[0]),
                        "final": distribution(components[1]),
                        "graph_sum": distribution(sums),
                    }
                )
            row["implementations"][impl] = {"identity": ids[0], "rounds": summaries}
        row["input_sha256"] = input_hash
        result["cases"].append(row)
    # Explicitly verify B changes grid only, not either physical artifact.
    for i in range(0, 6, 2):
        for impl in captures:
            a, b = [
                result["cases"][j]["implementations"][impl]["identity"]
                for j in (i, i + 1)
            ]
            assert a == b, "row count changed partial/final plan"
    references = {
        key: json.loads(path.read_text())
        for key, path in (
            ("triton-" + h, ROOT / f"artifacts/long-row/triton-{h}.json")
            for h in ("short", "mid", "large", "medium", "million")
        )
    }
    for round_id, capture in enumerate(captures["cubecl"]):
        outcomes = []
        for row in capture["rows"]:
            n = row["shape"][-1]
            ref_capture = (
                captures["triton"][round_id]
                if 8192 < n <= 73728
                else references[reference_for_extent(n)]
            )
            reference = next(r for r in ref_capture["rows"] if r["case"] == row["case"])
            outcomes.append(qualify_case(row, reference))
        assert all(o["passes"] for o in outcomes), (
            "post-fix baseline failed per-element qualification"
        )
        result["qualification"].append({"round": round_id, "outcomes": outcomes})
    return result


def profile(name, component=False):
    metrics = CACHE / f"{name}-metrics.csv"
    sass = CACHE / f"{name}-sass.txt"
    rows = list(csv.DictReader(metrics.open()))
    units, rows = rows[0], rows[1:]
    assert len(rows) == 2
    executed = instructions(sass.read_text())
    for row in rows:
        assert sum(executed[row["Kernel Name"]].values()) == float(
            row["smsp__inst_executed.sum"]
        )
    capture_path = OUTPUT / (
        f"component-ncu-{name.split('-')[1]}-8.json"
        if component
        else f"ncu-{name}.json"
    )
    capture = json.loads(capture_path.read_text())
    if not component:
        assert capture["qualification_complete"] and capture["resources_only"]
        verify_environment(capture["environment"], name.startswith("cubecl"))
        selected = capture["selected_kernels"]
        assert [r["Kernel Name"] for r in rows] == [k["entrypoint"] for k in selected]
    assert int(rows[0]["launch__grid_size"].replace(",", "")) == 9 * (
        8 if component else capture["shape"][0]
    )
    shutil.copyfile(sass, OUTPUT / f"{name}-sass.txt")
    return {
        "raw_metrics": str(metrics),
        "raw_sha256": sha(metrics),
        "sass_sha256": sha(sass),
        "capture_sha256": sha(capture_path),
        "units": units,
        "rows": rows,
        "dynamic_warp_instructions": {k: dict(v) for k, v in executed.items()},
    }


def curate():
    paths = set()
    first = json.loads((OUTPUT / "cubecl-r0.json").read_text())
    for timing in first["timings"][:6]:
        paths.update(Path(k["directory"]) for k in timing["selected_kernels"])
    components = json.loads((OUTPUT / "components.json").read_text())
    paths.update(Path(k["directory"]) for k in components["kernels"].values())
    paths.add(Path(components["fixed_final"]))
    names = (
        "descriptor.json",
        "semantic.json",
        "physical-plan.json",
        "cubecl.ir",
        "kernel.cu",
        "kernel.ptx",
        "manifest.json",
        "nvrtc.json",
        "resources.json",
    )
    for directory in paths:
        dst = OUTPUT / "selected" / directory.name
        dst.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copyfile(directory / name, dst / name)
    triton = json.loads((OUTPUT / "triton-r0.json").read_text())
    for timing in triton["timings"][:6]:
        for k in timing["selected_kernels"]:
            root = ROOT / "cache/reference-reduction/triton-r0/triton" / k["cache_hash"]
            dst = OUTPUT / "selected" / k["cache_hash"]
            dst.mkdir(parents=True, exist_ok=True)
            for src in root.iterdir():
                if src.suffix in (".json", ".ptx", ".llir", ".ttir", ".ttgir"):
                    shutil.copyfile(src, dst / src.name)
            shutil.copyfile(k["wrapper"], dst / "wrapper.py")


def candidate():
    capture = json.loads((OUTPUT / "candidate-graph.json").read_text())
    verify_environment(capture["environment"], True)
    assert capture["qualification_complete"] and len(capture["rows"]) == 178
    baseline_capture = json.loads((OUTPUT / "cubecl-r0.json").read_text())
    current_reference = json.loads((OUTPUT / "triton-r0.json").read_text())
    refs = {
        "triton-" + h: json.loads(
            (ROOT / f"artifacts/long-row/triton-{h}.json").read_text()
        )
        for h in ("short", "mid", "large", "medium", "million")
    }
    refs["triton-large"] = current_reference
    outcomes = [
        qualify_case(
            row,
            next(
                r
                for r in refs[reference_for_extent(row["shape"][-1])]["rows"]
                if r["case"] == row["case"]
            ),
        )
        for row in capture["rows"]
    ]
    components = json.loads((OUTPUT / "candidate-components.json").read_text())
    old_components = json.loads((OUTPUT / "components.json").read_text())
    assert components["complete"]
    # Adding the isolated strategy must preserve every pre-existing code packet.
    for name in ("compensated", "tree", "plane", "four"):
        a, b = [
            Path(c["kernels"][name]["directory"]) for c in (old_components, components)
        ]
        for file in ("descriptor.json", "kernel.ptx"):
            assert sha(a / file) == sha(b / file), (name, file)
    comparisons = []
    for rows in (8, 64):
        for base in ("four", "compensated"):
            selected = {
                n: [
                    t
                    for t in components["timings"]
                    if t["shape"][0] == rows and t["plan"] == n
                ]
                for n in ("batch", base)
            }
            assert all(len(v) == 3 for v in selected.values())
            d = {
                n: distribution([v for t in rr for v in t["samples_us"]])
                for n, rr in selected.items()
            }
            wins = sum(
                a["median_us"] < b["median_us"]
                for a, b in zip(selected["batch"], selected[base], strict=True)
            )
            stable = (
                wins >= 2
                and d["batch"]["median_us"] + 3 * d["batch"]["mad_us"]
                < d[base]["median_us"] - 3 * d[base]["mad_us"]
            )
            comparisons.append(
                {
                    "rows": rows,
                    "baseline": base,
                    "rounds_won": wins,
                    "distributions": d,
                    "stable_component_win": stable,
                }
            )
    graph = []
    for t in capture["timings"][:6]:
        original = next(
            r for r in baseline_capture["timings"] if r["case"] == t["case"]
        )
        assert fingerprint(t["selected_kernels"][1], True) == fingerprint(
            original["selected_kernels"][1], True
        )
        assert t["input_sha256"] == original["input_sha256"]
        partial, final = [], []
        for pr in t["profiles"]:
            p = Path(pr["raw_trace"])
            assert sha(p) == pr["sha256"]
            ev = sorted(
                (
                    e
                    for e in json.loads(p.read_text())["traceEvents"]
                    if e.get("cat") == "kernel"
                ),
                key=lambda e: e["ts"],
            )
            assert len(ev) == 2 and [e["name"] for e in ev] == pr["names"]
            partial.append(ev[0]["dur"])
            final.append(ev[1]["dur"])
        graph.append(
            {
                "case": t["case"],
                "partial": distribution(partial),
                "final": distribution(final),
                "note": "candidate graph capture is separate in time from baseline rounds; not an automatic reference-selection comparison",
            }
        )
    return {
        "outcomes": outcomes,
        "failures": [o for o in outcomes if not o["passes"]],
        "component_comparisons": comparisons,
        "graph": graph,
        "baseline_code_packets_unchanged": True,
    }


def normalization():
    import numpy as np

    capture_path = OUTPUT / "normalization-cubecl.json"
    reference_path = ROOT / "artifacts/long-row/normalization-triton.json"
    capture, reference = [
        json.loads(p.read_text()) for p in (capture_path, reference_path)
    ]
    assert capture["complete"] and reference["complete"]
    assert capture["cubecl_policy"] == "load_batch_diagnostic"
    verify_environment(capture["environment"], True)
    verify_environment(reference["environment"], False)
    arrays = CACHE / "norm/norm-cubecl.npz"
    ref_arrays = ROOT / "cache/long-row/norm-triton.npz"
    assert sha(arrays) == capture["raw_arrays_sha256"]
    assert sha(ref_arrays) == reference["raw_arrays_sha256"]
    from stage5.reduction_policy import TOLERANCES

    rows = []
    with np.load(arrays) as actual, np.load(ref_arrays) as expected:
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
            rows.append(
                {
                    "case": c["case"],
                    "phase": c["phase"],
                    "rtol": rtol,
                    "atol": atol,
                    "passes": True,
                    "max_abs_error": float(np.max(np.abs(a - b))),
                    "kernels": c["kernels"],
                }
            )
    assert len(rows) == 18
    result = {
        "complete": True,
        "cubecl_policy": capture["cubecl_policy"],
        "cases": rows,
        "capture_sha256": sha(capture_path),
        "reference_capture_sha256": sha(reference_path),
        "raw_arrays_sha256": sha(arrays),
        "reference_arrays_sha256": sha(ref_arrays),
        "source": "candidate vs retained efficient configured Triton; identical input/upstream bytes",
    }
    (OUTPUT / "normalization-comparison.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    result = {"complete": False, "baseline": baseline(), "profiles": {}}
    for impl in ("cubecl", "triton"):
        for b in (8, 64):
            name = f"{impl}-{b}"
            result["profiles"][name] = profile(name)
    for plan in ("tree", "plane", "four"):
        name = f"component-{plan}-8"
        result["profiles"][name] = profile(name, component=True)
    for b in (8, 64):
        a, t = (result["profiles"][f"{impl}-{b}"] for impl in ("cubecl", "triton"))
        assert a["rows"][0]["launch__grid_size"] == t["rows"][0]["launch__grid_size"]
        ca, ct = (
            json.loads((OUTPUT / f"ncu-{impl}-{b}.json").read_text())
            for impl in ("cubecl", "triton")
        )
        assert ca["input_sha256"] == ct["input_sha256"]
    result["profiles"]["component-batch-8"] = profile(
        "component-batch-8", component=True
    )
    result["candidate"] = candidate()
    result["normalization"] = normalization()
    curate()
    captures = [
        json.loads((OUTPUT / n).read_text())
        for n in ("candidate-components.json", "candidate-graph.json")
    ]
    paths = {Path(k["directory"]) for k in captures[0]["kernels"].values()}
    paths.update(
        Path(k["directory"])
        for t in captures[1]["timings"][:6]
        for k in t["selected_kernels"]
    )
    for directory in paths:
        dst = OUTPUT / "selected" / directory.name
        dst.mkdir(parents=True, exist_ok=True)
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
        ):
            shutil.copyfile(directory / file, dst / file)
    result.update(complete=True, analyzer_sha256=sha(__file__))
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2))
    print(
        "matched baselines, byte identities, fixed packets, qualification and SASS census verified"
    )


if __name__ == "__main__":
    main()
