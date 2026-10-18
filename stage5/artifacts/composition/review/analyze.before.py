"""Audit qualification, original PTX, actual wrappers and all mixed-graph traces."""

import ast
import csv
import gzip
from pathlib import Path
import statistics

from stage5.composition_trial import ROOT, OUT, read, write, sha, initialize, role


def function(path, name):
    return ast.dump(next(n for n in ast.parse(path.read_text()).body
                         if isinstance(n, ast.FunctionDef) and n.name == name))


def main():
    torch = initialize()
    from stage5.composition_run import references, rms_label, column_case
    from stage5.rms_trial import qualify
    from stage5.layout_trial import tensor_sha, check
    from stage5.composition_runtime import gate, FrozenColumn, FrozenPacked
    from stage5.column_layout_runtime import Kernel as OriginalColumn
    from stage5.rms_packed_trial_runtime import Kernel as OriginalPacked
    from stage2.indexing import evaluate, LIMIT

    gate()
    assert FrozenColumn._validate is OriginalColumn._validate
    assert FrozenPacked._validate is OriginalPacked._validate

    plan, gpu, alias, graphs, timing = (read(OUT / f) for f in (
        "plan.json", "gpu.json", "alias-gpu.json", "graphs-gpu.json", "timing.json"))
    assert all(d["complete"] for d in (gpu, alias, graphs, timing))
    assert (len(gpu["rows"]), len(gpu["guards"]), len(gpu["streams"]), len(gpu["mutations"])) == (60, 33, 6, 6)
    assert len(alias["rows"]) == 24 and len(alias["controls"]) == 9
    assert all(r["bitwise"] and r["canaries"] for r in alias["rows"])
    assert all(r["prelaunch"] for r in gpu["guards"])
    assert all(r["passes"] for r in gpu["streams"])
    assert all(r["isolated"] for r in gpu["mutations"])
    assert all(r["rejected"] for r in alias["controls"])
    extent_proof = read(OUT / "extent-alias-cpu.json")
    for row in extent_proof["rows"]:
        n, b = row["scalars"]
        assert 0 < b*n <= LIMIT and row["numel"] == b*n
    assert len(extent_proof["rows"]) == 21
    incoming = read(OUT / "admission-failure.json")["semantic"]
    canonical = next(k["semantic"] for k in read(ROOT / "stage5/artifacts/column-layout/post-review-gpu.json")["kernels"]
                     if k["semantic"]["buffers"][0]["dtype"] == incoming["buffers"][0]["dtype"])
    assert incoming["numel"]["op"] == canonical["numel"]["op"] == "mul"
    assert incoming["numel"]["args"] == list(reversed(canonical["numel"]["args"]))
    for row in extent_proof["rows"]:
        assert evaluate(incoming["numel"], row["scalars"]) == evaluate(canonical["numel"], row["scalars"])
    incoming["numel"] = canonical["numel"]
    assert incoming == canonical
    for name, digest in read(OUT / "sources.json")["files"].items():
        assert sha(ROOT / name) == sha(OUT / "sources" / name) == digest, name
    producer = OUT / "phase-sources/direct-gpu-run.py"
    assert gpu["source_sha256"] == sha(producer)
    assert function(producer, "gpu") == function(ROOT / "stage5/composition_run.py", "gpu")
    assert timing["source_sha256"] == graphs["source_sha256"] == sha(ROOT / "stage5/composition_run.py")
    assert alias["source_sha256"] == sha(ROOT / "stage5/composition_alias_check.py")
    old_rms = read(ROOT / "stage5/artifacts/reuse-first-rms-packed-trial/gpu.json")
    old_col = read(ROOT / "stage5/artifacts/column-layout/post-review-gpu.json")
    for r in gpu["rows"]:
        old = next(q for q in (old_rms if r["kind"] == "rms" else old_col)["rows"]
                   if q.get("case", q.get("label")) == r["label"])
        assert r["input_sha256"] == old["input_sha256"]
        assert r["output_sha256"] == old["output_sha256"]
        assert r["policy"] and r["frozen_bitwise"] and r["canaries"]
    for a in gpu["artifacts"]:
        for name, digest in a["files"].items():
            assert sha(Path(a["directory"]) / name) == sha(Path(a["source"]) / name) == digest
    data = references(torch)
    values = torch.load(OUT / "graph-values.pt", weights_only=True)
    assert graphs["values_sha256"] == sha(OUT / "graph-values.pt")
    assert len(timing["rows"]) == len(plan["cases"]) == len(values) == 9
    table, traces, duration_count, clocks = [], set(), 0, set()
    for row, case in zip(timing["rows"], plan["cases"]):
        assert row["case"] == case
        p = data[rms_label(case)]
        v = values[row["label"]]
        assert qualify(torch, rms_label(case), p["input"], v["ordinary_rms"], p)["passes"]
        assert tensor_sha(v["ordinary_rms"]) == row["output_sha256"]["ordinary"][0]
        assert row["output_sha256"]["column"][0] == row["output_sha256"]["ordinary"][0]
        rms_ref = next(r for r in gpu["rows"] if r["label"] == rms_label(case))
        col_ref = next(r for r in gpu["rows"] if r["label"] == column_case(case)["label"])
        assert row["input_sha256"] == [rms_ref["input_sha256"], *col_ref["input_sha256"]]
        for mode in ("rms", "both"):
            assert row["output_sha256"][mode][0] == rms_ref["output_sha256"]
        for mode in plan["modes"]:
            assert row["output_sha256"][mode][1] == col_ref["output_sha256"]
        check(torch, v["special_column"], v["special_inputs"][0] + v["special_inputs"][1])
        a = row["artifacts"]
        assert a["rms"][1] == a["ordinary"][1] and a["column"][0] == a["ordinary"][0]
        assert a["both"][0] == a["rms"][0] and a["both"][1] == a["column"][1]
        for mode, artifacts in a.items():
            for artifact in artifacts:
                original = Path(artifact["directory"])
                retained = original if original.is_relative_to(OUT) else OUT / "ordinary" / original.name
                assert sha(retained / "kernel.ptx") == artifact["ptx_sha256"]
                assert read(retained / "manifest.json")["entrypoint"] == artifact["entrypoint"]
                assert read(retained / "nvrtc.json")["options"].count("--fmad=false") == 1
            w = row["wrappers"][mode]
            path = OUT / w["path"]
            assert sha(path) == w["sha256"]
            if Path(w["executed_path"]).exists():
                assert path.read_bytes() == Path(w["executed_path"]).read_bytes()
            calls = [n for n in ast.walk(ast.parse(path.read_text()))
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                     and n.func.id == "composition_compile"]
            assert len(calls) == 2
            assert [role(ast.literal_eval(c.args[0])) for c in calls] == ["rms", "column"]
            assert all(ast.literal_eval(c.args[2]) == mode for c in calls)
            for i, (call, artifact) in enumerate(zip(calls, artifacts)):
                original = Path(artifact["directory"])
                retained = original if original.is_relative_to(OUT) else OUT / "ordinary" / original.name
                incoming = ast.literal_eval(call.args[0])
                target = read(retained / "semantic.json")
                if i == 1 and incoming != target:
                    assert incoming["numel"]["args"] == list(reversed(target["numel"]["args"]))
                    incoming["numel"] = target["numel"]
                assert incoming == target
        assert len(row["rounds"]) == 8
        per_kernel = {mode: [[], []] for mode in plan["modes"]}
        for rr, order in zip(row["rounds"], plan["order"]):
            assert rr["order"] == order
            for snapshot in ("activity_before", "activity_after"):
                for fields in list(csv.reader(rr[snapshot]["stdout"].splitlines()))[1:]:
                    if "510ddc27" in ",".join(fields):
                        clocks.add((fields[4].strip(), fields[5].strip()))
            for mode, capture in rr["paths"].items():
                path = OUT / capture["trace"]
                assert str(path) not in traces and sha(path) == capture["sha256"]
                traces.add(str(path))
                import json

                events = [e for e in json.loads(gzip.decompress(path.read_bytes()))["traceEvents"]
                          if e.get("cat") == "kernel"]
                expected = [x["entrypoint"] for x in row["artifacts"][mode]]
                assert len(events) == 32 and [e["name"] for e in events] == expected * 16
                for i, artifact in enumerate(row["artifacts"][mode]):
                    physical = artifact["plan"]
                    count = case["rms_rows"] if i == 0 else case["column_shape"][0] * case["column_shape"][1]
                    span = physical["outputs_per_workgroup"] if i == 0 else (
                        physical["workgroup"][0] * physical["vector_width"] * physical["elements_per_unit"])
                    assert all(e["args"]["grid"] == [(count + span - 1) // span, 1, 1]
                               and e["args"]["block"] == physical["workgroup"] for e in events[i::2])
                sums = [events[2*i]["dur"] + events[2*i+1]["dur"] for i in range(16)]
                assert all(abs(x-y) < 0.004 for x, y in zip(sums, capture["samples_us"]))
                assert statistics.median(capture["samples_us"]) == capture["median_us"]
                for k in (0, 1):
                    per_kernel[mode][k].extend(e["dur"] for e in events[k::2])
                duration_count += len(events)
        med = {mode: statistics.median(x for rr in row["rounds"] for x in rr["paths"][mode]["samples_us"])
               for mode in plan["modes"]}
        table.append(dict(label=row["label"], median_sum_us=med,
                          kernel_median_us={m: [statistics.median(xs) for xs in entries]
                                            for m, entries in per_kernel.items()},
                          both_wins={m: sum(rr["paths"]["both"]["median_us"] < rr["paths"][m]["median_us"]
                                            for rr in row["rounds"]) for m in ("ordinary", "rms", "column")},
                          all_round_interaction_us=[rr["paths"]["both"]["median_us"]
                              - rr["paths"]["rms"]["median_us"] - rr["paths"]["column"]["median_us"]
                              + rr["paths"]["ordinary"]["median_us"] for rr in row["rounds"]]))
    assert len(traces) == 288 and duration_count == 9216
    for name, digest in read(OUT / "preservation.json").items():
        assert sha(ROOT / name) == digest, name
    write("analysis.json", dict(complete=True, rows=table, traces=len(traces),
        kernel_durations=duration_count, clock_snapshots=sorted(clocks),
        both_wins={m: sum(r["both_wins"][m] for r in table) for m in ("ordinary", "rms", "column")},
        preserved_files=len(read(OUT / "preservation.json")), source_sha256=sha(__file__)))
    lines = ["| Case | Ordinary | RMS only | Column only | Both | Both wins vs ordinary |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in table:
        lines.append("| " + r["label"] + " | " + " | ".join(f'{r["median_sum_us"][m]:.3f}' for m in plan["modes"])
                     + f' | {r["both_wins"]["ordinary"]}/8 |')
    (OUT / "timing-table.md").write_text("\n".join(lines) + "\n")
    print("PASS9 mixed graphs/288 traces/9216 durations and all source/PTX/qualification bindings")


if __name__ == "__main__":
    main()
