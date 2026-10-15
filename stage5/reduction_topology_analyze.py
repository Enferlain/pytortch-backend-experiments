"""Check matched inputs, complete graph census, fixed plans and timing scopes."""

import csv
import json
from pathlib import Path

from stage5.long_row_analyze import reference_for_extent
from stage5.long_row_diagnose import distribution, instructions
from stage5.reduction_policy import qualify_case, TOLERANCES
from stage5.reduction_structure_analyze import fingerprint, verify_environment
from stage5.reduction_topology_model import ROOT, SOURCE, OUTPUT
from stage5.reference_reduction_probe import sha


def graphs():
    captures = {
        impl: [
            json.loads((OUTPUT / f"topology-{impl}-r{r}.json").read_text())
            for r in range(3)
        ]
        for impl in ("cubecl", "triton")
    }
    references = {
        f"triton-{h}": json.loads(
            (ROOT / f"artifacts/long-row/triton-{h}.json").read_text()
        )
        for h in ("short", "mid", "medium", "million")
    }
    qualification = []
    for impl, rounds in captures.items():
        for capture in rounds:
            assert capture["qualification_complete"] and len(capture["rows"]) == 178
            verify_environment(capture["environment"], impl == "cubecl")
    for r, capture in enumerate(captures["cubecl"]):
        checks = []
        for row in capture["rows"]:
            ref = (
                captures["triton"][r]
                if 8192 < row["shape"][-1] <= 73728
                else references[reference_for_extent(row["shape"][-1])]
            )
            checks.append(
                qualify_case(
                    row, next(x for x in ref["rows"] if x["case"] == row["case"])
                )
            )
        qualification.append(
            {
                "round": r,
                "checks": checks,
                "failures": [c for c in checks if not c["passes"]],
            }
        )
    cases = []
    for case in [t["case"] for t in captures["cubecl"][0]["timings"]]:
        record = {"case": case, "implementations": {}}
        hashes = set()
        for impl, rounds in captures.items():
            timed = [next(t for t in c["timings"] if t["case"] == case) for c in rounds]
            ids = [
                [fingerprint(k, impl == "cubecl") for k in t["selected_kernels"]]
                for t in timed
            ]
            assert len(ids[0]) == 2 and all(i == ids[0] for i in ids)
            result = []
            for t in timed:
                hashes.add(t["input_sha256"])
                components = [[], []]
                assert len(t["profiles"]) == 7
                for p in t["profiles"]:
                    path = Path(p["raw_trace"])
                    assert sha(path) == p["sha256"]
                    events = sorted(
                        [
                            e
                            for e in json.loads(path.read_text())["traceEvents"]
                            if e.get("cat") == "kernel"
                        ],
                        key=lambda e: e["ts"],
                    )
                    assert (
                        len(events) == 2 and [e["name"] for e in events] == p["names"]
                    )
                    for i, event in enumerate(events):
                        assert event["dur"] > 0
                        components[i].append(event["dur"])
                result.append(
                    {
                        "partial": distribution(components[0]),
                        "final": distribution(components[1]),
                        "graph_sum": distribution(
                            [a + b for a, b in zip(*components, strict=True)]
                        ),
                    }
                )
            record["implementations"][impl] = {"identity": ids[0], "rounds": result}
        assert len(hashes) == 1
        record["input_sha256"] = hashes.pop()
        cases.append(record)
    # For the controlled 65K B=8/64 set the reference must also match the
    # retained efficient config and both CubeCL finals must be byte-identical.
    old = json.loads((SOURCE / "triton-r0.json").read_text())
    for case in cases[:6]:
        old_t = next(t for t in old["timings"] if t["case"] == case["case"])
        assert case["implementations"]["triton"]["identity"] == [
            fingerprint(k, False) for k in old_t["selected_kernels"]
        ]
    for i in (0, 2, 4):
        for impl in captures:
            assert (
                cases[i]["implementations"][impl]["identity"]
                == cases[i + 1]["implementations"][impl]["identity"]
            )
    return {
        "cases": cases,
        "qualification": qualification,
        "scope": "rotated C/T,T/C,C/T generated graph; fixed9 and fixed final per backend; seven independent CUPTI captures per case/round",
        "eager_failures_preserved": {
            impl: [c["numerical_failures"] for c in rounds]
            for impl, rounds in captures.items()
        },
    }


def normalization():
    import numpy as np

    capture = json.loads((OUTPUT / "normalization-cubecl.json").read_text())
    ref = json.loads(
        (ROOT / "artifacts/long-row/normalization-triton.json").read_text()
    )
    assert capture["complete"] and capture["cubecl_policy"] == "reference512_diagnostic"
    verify_environment(capture["environment"], True)
    verify_environment(ref["environment"], False)
    actual_path = ROOT / "cache/reduction-topology/norm/norm-cubecl.npz"
    ref_path = ROOT / "cache/long-row/norm-triton.npz"
    assert sha(actual_path) == capture["raw_arrays_sha256"]
    assert sha(ref_path) == ref["raw_arrays_sha256"]
    checks = []
    with np.load(actual_path) as actual, np.load(ref_path) as expected:
        for c, r in zip(capture["cases"], ref["cases"], strict=True):
            for field in (
                "case",
                "phase",
                "dtype",
                "shape",
                "input_sha256",
                "upstream_sha256",
            ):
                assert c[field] == r[field]
            assert sha(c["raw_trace"]) == c["trace_sha256"]
            key = c["case"] + (":out" if c["phase"] == "forward" else ":gradient")
            a, b = actual[key], expected[key]
            rtol, atol = TOLERANCES[c["dtype"]]
            assert np.all(np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
            assert all(k.startswith("cubecl5_") for k in c["kernels"])
            checks.append(
                {
                    "case": c["case"],
                    "phase": c["phase"],
                    "passes": True,
                    "max_abs_error": float(np.max(np.abs(a - b))),
                    "kernel_count": len(c["kernels"]),
                }
            )
    assert len(checks) == 18
    return checks


def profiles():
    results = {}
    for case in ("float32-8", "float32-64", "float16-8"):
        raw = OUTPUT / f"candidate-{case}-raw.csv"
        rows = list(csv.DictReader(raw.open()))
        executed = instructions((OUTPUT / f"candidate-{case}-sass.txt").read_text())
        # First CSV row holds units; the remaining rows are the partial/final.
        assert len(rows) == 3
        rows = rows[1:]
        assert len(rows) == 2
        capture = json.loads((OUTPUT / f"ncu-{case}.json").read_text())
        verify_environment(capture["environment"], True)
        # The resource-only producer also lists the FP32 warmup graph when
        # collecting FP16. Bind each actual replayed kernel by exact name.
        selected = {k["entrypoint"]: k for k in capture["selected_kernels"]}
        assert len(selected) == len(capture["selected_kernels"])
        assert all(row["Kernel Name"] in selected for row in rows)
        assert (
            selected[rows[0]["Kernel Name"]]["plan"]["strategy"]
            == "plane_shared_reference512"
        )
        assert (
            selected[rows[1]["Kernel Name"]]["plan"]["strategy"]
            == "shared_tree_reduction_baseline"
        )
        assert (
            int(rows[0]["launch__grid_size"].replace(",", ""))
            == int(case.split("-")[1]) * 9
        )
        assert int(rows[0]["launch__block_size"].replace(",", "")) == 512
        for row in rows:
            assert sum(executed[row["Kernel Name"]].values()) == float(
                row["smsp__inst_executed.sum"].replace(",", "")
            )
        results[case] = {
            "raw_path": str(raw),
            "raw_sha256": sha(raw),
            "kernels": [
                {
                    "name": row["Kernel Name"],
                    "instructions": dict(executed[row["Kernel Name"]]),
                    "metrics": {
                        k: v
                        for k, v in row.items()
                        if any(
                            s in k
                            for s in (
                                "register",
                                "local",
                                "spill",
                                "barrier",
                                "warp",
                                "throughput",
                                "time_duration",
                                "shared",
                                "grid_size",
                                "block_size",
                            )
                        )
                    },
                }
                for row in rows
            ],
        }
    return results


def main():
    graph = graphs()
    result = {
        "complete": True,
        "graph": graph,
        "normalization": normalization(),
        "profiles": profiles(),
        "harness_sha256": sha(__file__),
    }
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2))
    print("Graph failures:", [c for r in graph["qualification"] for c in r["failures"]])
    for c in graph["cases"][:6]:
        print(
            c["case"],
            {
                impl: [r["partial"]["median_us"] for r in d["rounds"]]
                for impl, d in c["implementations"].items()
            },
        )


if __name__ == "__main__":
    main()
