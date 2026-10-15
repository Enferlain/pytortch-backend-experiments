"""Verify retained repeatability captures without recollecting GPU work."""

import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/reduction-structure/repeatability"
RUNS = (
    "initial",
    "state-controls-r1",
    "unmonitored",
    "address-controls",
    "module-controls",
    "cadence-controls",
    "gated-controls",
    "headless-3080",
)
INPUT = "199358562e389b69090676348ef457ed0141f285d0609c9dd4ff6bfe3b44e218"
NAMES = {
    "cubecl": ["cubecl5_42026f8357481503", "cubecl5_b1f1603b0e5b0a49"],
    "triton": ["triton_red_fused_sum_0", "triton_per_fused_sum_1"],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values):
    median = statistics.median(values)
    return {
        "count": len(values),
        "median_us": median,
        "mad_us": statistics.median(abs(v - median) for v in values),
        "min_us": min(values),
        "max_us": max(values),
    }


def main():
    result = {"complete": False, "runs": {}, "prequeue_checks": [], "ncu": {}}
    all_artifacts = None
    for name in RUNS:
        path = OUT / name
        data = json.loads((path / "result.json").read_text())
        assert data["complete"] and data["bitwise_partials_and_finals"]
        assert data["shape"] == [64, 65539] and data["split"] == 9
        assert data["input_sha256"] == INPUT
        assert sha(path / "harness.py") == data["harness_sha256"], name
        assert all(sha(p) == h for p, h in data["artifact_sha256"].items())
        if all_artifacts is None:
            all_artifacts = data["artifact_sha256"]
        assert data["artifact_sha256"] == all_artifacts
        groups = {}
        for capture in data["captures"]:
            trace = Path(capture["raw_trace"])
            assert sha(trace) == capture["trace_sha256"]
            events = json.loads(trace.read_text())["traceEvents"]
            kernels = sorted(
                [e for e in events if e.get("cat") == "kernel"], key=lambda e: e["ts"]
            )
            names = NAMES[capture["impl"]]
            measured = [e for e in kernels if e["name"] in names]
            priming = len(capture.get("priming_partial_us", []))
            count = len(capture["partial_us"])
            assert [e["name"] for e in measured] == names * (priming + count)
            raw = [e["dur"] for e in measured[2 * priming :]]
            expected = [
                v
                for pair in zip(capture["partial_us"], capture["final_us"], strict=True)
                for v in pair
            ]
            assert len(raw) == len(expected) and all(
                abs(a - b) < 1e-6 for a, b in zip(raw, expected, strict=True)
            )
            groups.setdefault((capture["mode"], capture["impl"]), []).append(capture)
            if capture["mode"] == "gated_burst":
                spin = next(e for e in kernels if "spin_kernel" in e["name"])
                apis = [
                    e
                    for e in events
                    if "LaunchKernel" in e.get("name", "")
                    and e.get("cat") in ("cuda_driver", "cuda_runtime")
                ]
                assert len(apis) == 2 * count + 1
                lead = spin["ts"] + spin["dur"] - max(e["ts"] + e["dur"] for e in apis)
                assert lead > 0, (name, capture["round"], lead)
                result["prequeue_checks"].append(
                    {
                        "run": name,
                        "impl": capture["impl"],
                        "round": capture["round"],
                        "all_apis_queued_before_spin_end_us": lead,
                        "first_partial_us": capture["partial_us"][0],
                        "later_partials": summarize(capture["partial_us"][2:]),
                    }
                )
        rows = []
        for (mode, impl), captures in groups.items():
            partials = [v for c in captures for v in c["partial_us"]]
            later = [v for c in captures for v in c["partial_us"][2:]]
            rows.append(
                {
                    "mode": mode,
                    "impl": impl,
                    "partials": summarize(partials),
                    "first_partials": summarize([c["partial_us"][0] for c in captures]),
                    "after_first_two": summarize(later) if later else None,
                    "round_partial_medians_us": [
                        statistics.median(c["partial_us"]) for c in captures
                    ],
                    "finals": summarize([v for c in captures for v in c["final_us"]]),
                }
            )
        result["runs"][name] = {
            "device": data["environment"]["device"],
            "capture_count": len(data["captures"]),
            "rows": rows,
        }
    for tag in (
        "ncu-prime",
        "ncu-prime_flush",
        "ncu-app-range2-prime",
        "ncu-app-range2-prime_flush",
    ):
        with (OUT / (tag + "-raw.csv")).open() as stream:
            rows = list(csv.DictReader(stream))
        units, row = rows
        result["ncu"][tag] = {
            "report_sha256": sha(OUT / (tag + ".ncu-repz")),
            "metrics": {
                k: {"value": row[k], "unit": units[k]}
                for k in (
                    "gpu__time_duration.sum",
                    "dram__bytes_read.sum",
                    "lts__t_sector_op_read_hit_rate.pct",
                    "lts__t_sectors_op_read.sum",
                )
            },
            "scope": "Counter/replay control only; not warm CUPTI performance",
        }
    connection = sqlite3.connect(OUT / "nsys-state.sqlite")
    clock_rows = []
    for start, end in connection.execute(
        "select start,end from CUPTI_ACTIVITY_KIND_KERNEL where registersPerThread=40 order by start"
    ):
        values = [
            row[0]
            for row in connection.execute(
                "select value from GPU_METRICS where metricId=0 and timestamp between ? and ?",
                (start, end),
            )
        ]
        clock_rows.append(
            {"partial_us": (end - start) / 1000, "raw_gpc_clock_samples": values}
        )
    assert len(clock_rows) == 5
    result["nsys"] = {
        "report_sha256": sha(OUT / "nsys-state.nsys-rep"),
        "sqlite_sha256": sha(OUT / "nsys-state.sqlite"),
        "partial_clock_samples": clock_rows,
        "unit_note": "Raw exported values; displayed metric name is GPC Clock Frequency [MHz], but raw values are near 1.7e9. No unit conversion assumed.",
        "limit": "Two samples per partial; overlapping raw clock bands do not identify a global clock ramp or a unique internal state.",
    }
    result["fixed_artifact_sha256"] = all_artifacts
    previous = ROOT / "artifacts/reduction-structure/codegen"
    case = next(
        c
        for c in json.loads((previous / "graph-analysis.json").read_text())["cases"]
        if c["case"] == "float32-64-65539"
    )
    capture = next(
        t
        for t in json.loads((previous / "fixed-graph-cubecl-r0.json").read_text())[
            "timings"
        ]
        if t["case"] == "float32-64-65539"
    )
    for selected, identity in zip(
        capture["selected_kernels"],
        case["implementations"]["cubecl"]["identity"],
        strict=True,
    ):
        directory = Path(selected["directory"])
        assert sha(directory / "kernel.ptx") == identity["ptx"]
        assert sha(directory / "descriptor.json") == identity["descriptor"]
    result["cubecl_artifacts_match_prior_graph_fingerprints"] = True
    passes = list(OUT.glob("ncu*-passes/app-*/result.json")) + list(
        OUT.glob("nsys-state-pass/app-*/result.json")
    )
    for path in passes:
        data = json.loads(path.read_text())
        assert data["complete"] and data["bitwise_partials_and_finals"]
        assert (
            data["input_sha256"] == INPUT and data["artifact_sha256"] == all_artifacts
        )
        assert sha(path.parent / "harness.py") == data["harness_sha256"]
    result["verified_counter_application_passes"] = len(passes)
    result["complete"] = True
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2))
    print(
        "Verified",
        len(RUNS),
        "completed capture groups;",
        sum(r["capture_count"] for r in result["runs"].values()),
        "raw traces;",
        len(result["prequeue_checks"]),
        "prequeued controls",
    )


if __name__ == "__main__":
    main()
