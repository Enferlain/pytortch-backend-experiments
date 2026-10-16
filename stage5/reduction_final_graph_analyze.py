"""Independent admission of cwk.7 identity, numerical and raw timing evidence."""

import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts/reduction-structure"
OUT = SOURCE / "final/graphs"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    median = statistics.median(values)
    return {
        "count": len(values),
        "median_us": median,
        "mad_us": statistics.median(abs(v - median) for v in values),
        "min_us": min(values),
        "max_us": max(values),
    }


def raw(capture, expected_names):
    path = Path(capture["trace"])
    assert sha(path) == capture["sha256"], path
    events = sorted(
        (
            e
            for e in json.loads(path.read_text())["traceEvents"]
            if e.get("cat") == "kernel"
        ),
        key=lambda e: e["ts"],
    )
    assert [e["name"] for e in events] == expected_names * 16
    partial = [e["dur"] for e in events[::2]] if len(expected_names) == 2 else []
    final = [e["dur"] for e in events[1::2]] if partial else [e["dur"] for e in events]
    assert partial == capture.get("partial_us", []) and final == capture["final_us"]
    return partial, final


def main():
    frozen = json.loads((OUT / "frozen-files.json").read_text())
    assert all(sha(p) == h for p, h in frozen.items())
    cpu = json.loads((OUT / "cpu.json").read_text())
    assert cpu["complete"] and len(cpu["checks"]) == 148
    assert all(c["passes"] for c in cpu["checks"])
    assert len(cpu["neighbors"]) == 1332 and all(c["bitwise"] for c in cpu["neighbors"])
    controls = json.loads((OUT / "controls.json").read_text())
    assert controls["complete"] and len(controls["identities"]) == 9
    assert all(all(c["matches"].values()) for c in controls["identities"])
    assert len(controls["gpu_neighbors"]) == 324
    assert all(c["bitwise"] and c["canaries"] for c in controls["gpu_neighbors"])
    assert len(controls["rejections"]) == 8 and all(
        c["exit_code"] and c["no_target_code"] for c in controls["rejections"]
    )
    qualification = json.loads((OUT / "qualification.json").read_text())
    old = json.loads((SOURCE / "codegen/graph-cubecl.json").read_text())
    assert (
        qualification["qualification_complete"]
        and qualification["frozen_graph_rows_identical"]
    )
    substitutions = [s for s in qualification["substitutions"] if s["substituted"]]
    assert {s["semantic"]["buffers"][-1]["dtype"] for s in substitutions} == {
        "float32",
        "float16",
        "bfloat16",
    }
    assert all(s["plan"]["strategy"] == "independent_planes" for s in substitutions)
    assert len(qualification["rows"]) == 178 and qualification["timings"] == []
    assert qualification["numerical_failures"] == old["numerical_failures"]
    for a, b in zip(qualification["rows"], old["rows"], strict=True):
        for key in ("case", "shape", "dtype", "input_sha256", "values", "aten", "fp64"):
            assert json.dumps(a[key]) == json.dumps(b[key])
    extra = json.loads((OUT / "extra-graphs.json").read_text())
    assert (
        extra["complete"]
        and len(extra["checks"]) == 33
        and extra["offset_current_stream_bitwise"]
    )
    reviewed_extra = json.loads((OUT / "extras-reviewed/extra-graphs.json").read_text())
    assert reviewed_extra["complete"] and reviewed_extra["checks"] == extra["checks"]
    assert reviewed_extra["offset_current_stream_bitwise"]
    assert (
        Path(reviewed_extra["cache_directory"]).resolve()
        == (ROOT / "cache/cwk7-extra/extras-reviewed").resolve()
    )
    for original in extra["artifacts"]:
        repeated = next(
            a
            for a in reviewed_extra["artifacts"]
            if a["semantic"] == original["semantic"] and a["plan"] == original["plan"]
        )
        for name in ("cubecl.ir", "kernel.cu", "kernel.ptx"):
            assert sha(Path(original["directory"]) / name) == sha(
                Path(repeated["directory"]) / name
            )
    inventory = json.loads((SOURCE / "final/inventory.json").read_text())
    assert all(sha(s["path"]) == s["sha256"] for s in inventory["sources"])
    graph = json.loads((OUT / "timing.json").read_text())
    isolated = json.loads((OUT / "isolated.json").read_text())
    assert graph["complete"] and isolated["complete"]
    retained_cube = json.loads(
        (SOURCE / "codegen/fixed-graph-cubecl-r0.json").read_text()
    )
    retained_triton = json.loads(
        (SOURCE / "codegen/fixed-graph-triton-r0.json").read_text()
    )
    result = {
        "complete": False,
        "cpu_cases": 148,
        "cpu_neighbors": 1332,
        "gpu_neighbors": 324,
        "graph_rows": 178,
        "additional_graph_rows": 33,
        "frozen_files": len(frozen),
        "raw_traces": 0,
        "graph_timing": [],
        "isolated_timing": [],
        "retained_files": {},
        "automatic_eligibility": False,
    }
    for case in graph["cases"]:
        b, n = case["shape"]
        key = f"float32-{b}-{n}"
        previous = next(t for t in retained_cube["timings"] if t["case"] == key)
        reference = next(t for t in retained_triton["timings"] if t["case"] == key)
        assert (
            case["input_sha256"]
            == previous["input_sha256"]
            == reference["input_sha256"]
        )
        for tag, artifacts in case["artifacts"].items():
            assert len(artifacts) == 2
            assert artifacts[0]["plan"]["strategy"] == "plane_shared_reference512"
            assert artifacts[1]["plan"]["strategy"] == (
                "independent_planes"
                if tag == "plane"
                else "shared_tree_reduction_baseline"
            )
            for i, artifact in enumerate(artifacts):
                directory = Path(artifact["directory"])
                for path in directory.iterdir():
                    if path.suffix not in (".cu", ".ptx", ".ir", ".json", ".log"):
                        continue
                    dest = OUT / "retained" / key / tag / str(i) / path.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        if path.name == "resources.json":
                            # CompiledKernel refreshes cache-hit/module-load
                            # timing fields whenever another graph loads this
                            # same immutable artifact. Keep the original
                            # snapshot; independently require static facts.
                            before = json.loads(dest.read_text())
                            now = json.loads(path.read_text())
                            for field in (
                                "attributes",
                                "resources",
                                "ptx_sha256",
                                "identity",
                            ):
                                assert before[field] == now[field], (dest, field)
                        else:
                            assert sha(dest) == sha(path), dest
                    else:
                        shutil.copyfile(path, dest)
                    result["retained_files"][str(dest)] = sha(dest)
                if i == 0:
                    for name in ("cubecl.ir", "kernel.cu", "kernel.ptx"):
                        assert sha(directory / name) == sha(
                            Path(previous["selected_kernels"][0]["directory"]) / name
                        )
        groups = []
        for tag in ("frozen", "plane", "triton_device_sequence"):
            captures = [c for c in case["captures"] if c["variant"] == tag]
            assert sorted(c["round"] for c in captures) == list(range(6))
            names = (
                [a["entrypoint"] for a in case["artifacts"][tag]]
                if tag != "triton_device_sequence"
                else [a["entrypoint"] for a in reference["selected_kernels"]]
            )
            partials, finals, sums, gaps, rounds = [], [], [], [], []
            for capture in captures:
                a, z = raw(capture, names)
                pairs = [x + y for x, y in zip(a, z, strict=True)]
                partials.extend(a)
                finals.extend(z)
                sums.extend(pairs)
                gaps.extend(capture["gap_us"])
                rounds.append(
                    {
                        "round": capture["round"],
                        "order": capture["order"],
                        "partial": stats(a),
                        "final": stats(z),
                        "sum": stats(pairs),
                    }
                )
                result["raw_traces"] += 1
            groups.append(
                {
                    "variant": tag,
                    "partial": stats(partials),
                    "final": stats(finals),
                    "sum": stats(sums),
                    "inter_kernel_gap": stats(gaps),
                    "rounds": rounds,
                }
            )
        result["graph_timing"].append({"shape": case["shape"], "groups": groups})
        part = next(c for c in isolated["cases"] if c["shape"] == case["shape"])
        assert (
            part["input_sha256"] == case["input_sha256"]
            and part["output_sha256"] == case["output_sha256"]
        )
        groups = []
        for scope in ("final_only", "same_partial_pair"):
            for tag in ("frozen", "plane", "triton"):
                captures = [
                    c
                    for c in part["captures"]
                    if c["variant"] == tag and c["scope"] == scope
                ]
                assert sorted(c["round"] for c in captures) == list(range(6))
                final_name = (
                    case["artifacts"][tag][1]["entrypoint"]
                    if tag != "triton"
                    else reference["selected_kernels"][1]["entrypoint"]
                )
                names = (
                    [case["artifacts"]["frozen"][0]["entrypoint"]]
                    if scope == "same_partial_pair"
                    else []
                ) + [final_name]
                values = []
                for capture in captures:
                    _, final = raw(capture, names)
                    values.extend(final)
                    result["raw_traces"] += 1
                groups.append({"scope": scope, "variant": tag, "final": stats(values)})
        result["isolated_timing"].append({"shape": case["shape"], "groups": groups})
    assert result["raw_traces"] == 108
    resources = json.loads((OUT / "driver-resources.json").read_text())
    assert resources["complete"] and len(resources["rows"]) == 8
    for resource in resources["rows"]:
        case = next(c for c in graph["cases"] if c["shape"] == resource["shape"])
        artifact = next(
            a
            for a in case["artifacts"][resource["variant"]]
            if a["entrypoint"] == resource["entrypoint"]
        )
        assert sha(Path(artifact["directory"]) / "kernel.ptx") == resource["ptx_sha256"]
        assert resource["attributes"]["local_bytes_per_thread"] == 0
    result["driver_resources"] = resources
    from stage5.reduction_codegen_analyze import actions

    metrics = list(csv.DictReader((OUT / "ncu-metrics.csv").open()))
    executed = actions(OUT / "executed-sass-counts.txt")
    assert len(metrics) == 4 and len(executed) == 3
    result["executed_code"] = []
    for row, code in zip(metrics[1:], executed, strict=True):
        assert row["Kernel Name"] == code["entrypoint"]
        assert sum(code["warp"].values()) == float(
            row["smsp__inst_executed.sum"].replace(",", "")
        )
        result["executed_code"].append(
            {
                **code,
                "metrics": {
                    k: row[k]
                    for k in (
                        "smsp__inst_executed.sum",
                        "launch__registers_per_thread",
                        "launch__registers_per_thread_allocated",
                        "sass__inst_executed_shared_loads",
                        "sass__inst_executed_shared_stores",
                    )
                },
            }
        )
    result["complete"] = True
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2))
    print(
        "cwk.7 evidence validates:",
        result["raw_traces"],
        "raw traces; all frozen partial artifacts match",
    )


if __name__ == "__main__":
    main()
