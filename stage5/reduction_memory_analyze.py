"""Validate the predeclared A/B/C memory-state interventions without sample trimming."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

from stage5.reduction_state_counter_analyze import (
    sha,
    csv_stream,
    stats,
    verify_primer_source,
    TARGET,
)

OUT = (
    Path(__file__).resolve().parent
    / "artifacts/reduction-structure/state-memory-causal"
)


def capture(directory, result, row):
    trace = Path(row["trace"])
    assert sha(trace) == row["trace_sha256"]
    document = json.loads(trace.read_text())
    assert document["cuda_driver_version"] == 13030
    assert document["cuda_runtime_version"] == 13000
    assert document["cupti_version"] == 130001
    device = next(d for d in document["deviceProperties"] if d["id"] == 0)
    assert device["name"] == result["device_query"]["name"] and device["numSms"] == 82
    events = document["traceEvents"]
    kernels = sorted(
        (e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"]
    )
    assert (
        len(
            {
                (e["args"]["device"], e["args"]["context"], e["args"]["stream"])
                for e in kernels
            }
        )
        == 1
    )
    target = kernels[-1]
    assert target["name"] == TARGET and target["dur"] == row["target_us"]
    for key, value in {
        "grid": [576, 1, 1],
        "block": [512, 1, 1],
        "registers per thread": 40,
        "shared memory": 64,
        "device": 0,
    }.items():
        assert target["args"][key] == value
    assert "spin_kernel" in kernels[0]["name"]
    apis = [
        e
        for e in events
        if "LaunchKernel" in e.get("name", "")
        and e.get("cat") in ("cuda_driver", "cuda_runtime")
    ]
    assert len(apis) == len(kernels)
    lead = kernels[0]["ts"] + kernels[0]["dur"] - max(e["ts"] + e["dur"] for e in apis)
    assert lead == row["prequeue_lead_us"] and lead > 0
    assert result["reset"] == "dense"
    steps = row["condition"]["steps"]
    assert [e["name"] for e in kernels[2:-1]] == ["primer_" + op for op, _, _ in steps]
    assert len(kernels) == 5
    assert all(
        e["args"]["grid"] == [82, 1, 1] and e["args"]["block"] == [256, 1, 1]
        for e in kernels[2:-1]
    )
    gap = target["ts"] - (kernels[-2]["ts"] + kernels[-2]["dur"])
    assert gap == row["target_gap_us"]
    idle_us = row["condition"]["idle_us"]
    if idle_us is not None:
        times = row["idle_callback_monotonic_ns"]
        assert times[1] - times[0] >= idle_us * 1000
        assert gap >= idle_us
    else:
        assert row["idle_callback_monotonic_ns"] is None
    assert row["bitwise_partial_and_output"]
    output = {"target_us": target["dur"], "prequeue_lead_us": lead, "gap_us": gap}
    if "pm_metrics" not in result:
        return output
    prefix = directory / f"r{row['round']}-{row['mode']}"
    decode = json.loads(Path(str(prefix) + ".pm.csv.decode.json").read_text())
    assert (
        decode["result"] == 0 and decode["stop_reason"] == 2 and decode["overflow"] == 0
    )
    with csv_stream(Path(str(prefix) + ".pm-unix.csv")) as stream:
        samples = list(csv.DictReader(stream))
    # The raw PM clock uses PyTorch's registered approximate timestamp callback.
    # Verify the conversion against independent wall-clock brackets in each capture.
    guard = 1500
    if result.get("clock_method") == "c10_deferred":
        # Calibration/quantization residuals widen the window guard; they are
        # not hidden by requiring a spuriously exact zero-residual wall bracket.
        uncertainty = []
        for anchor in row["clock_anchors"]:
            assert (
                anchor["approx_before"]
                <= anchor["cupti_timestamp"]
                <= anchor["approx_after"]
            )
            converted = anchor["converted_cupti_unix_ns"]
            uncertainty.append(
                max(
                    abs(converted - anchor["wall_before_ns"]),
                    abs(converted - anchor["wall_after_ns"]),
                )
            )
        guard += max(uncertainty)
    else:
        for anchor in row.get("clock_anchors", []):
            if "converted_cupti_unix_ns" in anchor:
                assert (
                    anchor["wall_before_ns"]
                    <= anchor["converted_cupti_unix_ns"]
                    <= anchor["wall_after_ns"]
                )
    with csv_stream(Path(str(prefix) + ".pm.csv")) as stream:
        raw_samples = list(csv.DictReader(stream))
    assert len(samples) == len(raw_samples)
    assert all(
        all(converted[k] == raw[k] for k in raw)
        for converted, raw in zip(samples, raw_samples)
    )
    raw_data = Path(str(prefix) + ".pm.csv.counterdata")
    if raw_data.exists():
        assert raw_data.stat().st_size > 0
    else:
        with gzip.open(str(raw_data) + ".gz", "rb") as stream:
            assert stream.read(1)
    start = document["baseTimeNanoseconds"] + round(target["ts"] * 1000)
    end = start + round(target["dur"] * 1000)
    interior = [
        s
        for s in samples
        if int(s["start_unix_ns"]) >= start + guard
        and int(s["end_unix_ns"]) <= end - guard
    ]
    # For additive counters, also retain a conservative upper bound: every
    # interval touching the expanded window. This may contain edge primer work
    # and is never reported as an exact target count. Comparing slow lower to
    # fast upper avoids extrapolating missing boundary intervals.
    overlapping = [
        s
        for s in samples
        if int(s["start_unix_ns"]) < end + guard
        and int(s["end_unix_ns"]) > start - guard
    ]
    assert int(overlapping[0]["start_unix_ns"]) <= start - guard
    assert int(overlapping[-1]["end_unix_ns"]) >= end + guard
    assert all(
        a["end_unix_ns"] == b["start_unix_ns"]
        for a, b in zip(overlapping, overlapping[1:])
    )
    if not interior:
        # Retain the timing and the failure; never shrink an uncertain clock
        # guard to manufacture counter admission. Exclude the whole contrast
        # from coupled causal conclusions, including its favorable prefix.
        output["counter_window_unavailable"] = {
            "reason": "No wholly interior sample under unchanged clock guard",
            "edge_guard_ns": guard,
            "target_start_unix_ns": start,
            "target_end_unix_ns": end,
        }
        return output
    durations = [int(s["end_unix_ns"]) - int(s["start_unix_ns"]) for s in interior]
    assert all(0 < duration < 5000 for duration in durations)
    assert all(
        int(a["end_unix_ns"]) <= int(b["start_unix_ns"])
        for a, b in zip(interior, interior[1:])
    )
    coverage = sum(durations)
    output["counter_window"] = {
        "target_start_unix_ns": start,
        "target_end_unix_ns": end,
        "edge_guard_ns": guard,
        "interior_samples": len(interior),
        "coverage_ns": coverage,
        "metrics": {},
    }
    for metric in result["pm_metrics"]:
        values = [float(s[metric]) for s in interior]
        assert all(math.isfinite(v) for v in values)
        is_sum = metric.endswith(".sum")
        output["counter_window"]["metrics"][metric] = {
            "aggregation": "interior_sum" if is_sum else "time_weighted_interior_mean",
            "value": sum(values)
            if is_sum
            else sum(v * duration for v, duration in zip(values, durations)) / coverage,
            "values": values,
        }
        if is_sum:
            assert all(
                math.isfinite(float(s[metric])) and float(s[metric]) >= 0
                for s in overlapping
            )
            output["counter_window"]["metrics"][metric]["expanded_overlap_upper"] = sum(
                float(s[metric]) for s in overlapping
            )
    return output


def analyze(root=OUT):
    plan_path = root / "predeclared-plan.json"
    plan = json.loads(plan_path.read_text())
    plan_hash = sha(plan_path)
    assert plan_hash == (root / "predeclared-plan.sha256").read_text().strip()
    reference = json.loads(
        (root.parent / "state-counters/baseline-r0/result.json").read_text()
    )
    identity_keys = [
        "input_sha256",
        "artifact_sha256",
        "partial_sha256",
        "output_sha256",
    ]
    sources = json.loads((root / "capture-source-sha256.json").read_text())
    assert all(sha(path) == digest for path, digest in sources.items())
    comparable = None
    rows = []
    pending = []
    for spec in plan["schedule"]:
        directory = root / (spec["id"] + f"-{spec['round']:02d}")
        if not (directory / "result.json").exists():
            pending.append(directory.name)
            continue
        result = json.loads((directory / "result.json").read_text())
        assert not result.get("error"), directory
        if not result["complete"]:
            pending.append(directory.name)
            continue
        assert result["sampler_close_status"] == 0
        assert [result[k] for k in identity_keys] == [
            reference[k] for k in identity_keys
        ]
        assert result["shape"] == [64, 65539] and result["split"] == 9
        assert (
            result["plan_sha256"]
            == sha(directory / "predeclared-plan.json")
            == plan_hash
        )
        assert result["contrast"] == spec["id"]
        assert result["conditions"] == {
            mode: plan["cases"][mode] for mode in spec["modes"]
        }
        assert result["device_query"] == plan["device_query"]
        assert sha(directory / "harness.py") == result["harness_sha256"]
        assert (
            sha(directory / "reduction_memory_idle.cpp") == result["idle_source_sha256"]
        )
        assert sha(directory / "primers/kernel.ptx") == result["primer_ptx_sha256"]
        assert verify_primer_source(directory) == "volatile"
        assert all(
            sha(path) == digest for path, digest in result["artifact_sha256"].items()
        )
        config = {
            k: result[k]
            for k in [
                "pm_metrics",
                "pm_interval_ns",
                "sampler_sha256",
                "clock_method",
                "reset",
                "harness_sha256",
                "primer_iterations",
                "primer_ptx_sha256",
                "idle_library_sha256",
                "idle_source_sha256",
            ]
        }
        assert result["sampler_sha256"] == sha(root / "sampler.so")
        assert result["idle_library_sha256"] == sha(root / "idle.so")
        config["sampler_source_sha256"] = sha(
            directory / "reduction_state_sampling.cpp"
        )
        assert (
            result["pm_metrics"] == plan["pm_metrics"]
            and result["pm_interval_ns"] == plan["pm_interval_ns"]
        )
        assert (
            config["clock_method"] == "c10_deferred"
            and config["primer_iterations"] == 65536
        )
        if comparable is None:
            comparable = config
        assert config == comparable
        spans = result["pointer_spans"]
        assert spans["memory"][1] == spans["B"][1] == plan["allocation_bytes"]
        for i, (ka, (pa, na)) in enumerate(spans.items()):
            for kb, (pb, nb) in list(spans.items())[i + 1 :]:
                assert pa + na <= pb or pb + nb <= pa, (ka, kb)
        prepared = {
            (a["operation"], a["allocation"], a["bytes"]): a
            for a in result["prepared_actions"]
        }
        expected_steps = {
            tuple(s) for mode in spec["modes"] for s in plan["cases"][mode]["steps"]
        }
        assert set(prepared) == expected_steps
        for (op, allocation, size), a in prepared.items():
            assert a["base_ptr"] == spans["memory" if allocation == "A" else "B"][0]
            assert a["sink_ptr"] == spans["sink"][0]
            assert a["count"] * 4 == size and a["stride"] == 1
            assert (
                a["grid"] == plan["primer_grid"] and a["block"] == plan["primer_block"]
            )
        captures = result["captures"]
        assert len(captures) == 2 and set(c["mode"] for c in captures) == set(
            spec["modes"]
        )
        for c in captures:
            assert (
                c["round"] == spec["round"]
                and c["condition"] == plan["cases"][c["mode"]]
            )
            rows.append(
                capture(directory, result, c)
                | {
                    "mode": c["mode"],
                    "contrast": spec["id"],
                    "round": spec["round"],
                    "condition": c["condition"],
                    "directory": directory.name,
                }
            )
    failures = [
        {
            "directory": r["directory"],
            "mode": r["mode"],
            "contrast": r["contrast"],
            **r["counter_window_unavailable"],
        }
        for r in rows
        if "counter_window_unavailable" in r
    ]
    excluded = {r["contrast"] for r in failures}
    admitted = [r for r in rows if r["contrast"] not in excluded]
    raw_unqualified = {}
    for contrast in sorted(excluded):
        raw_unqualified[contrast] = {
            "reason": "At least one counter window fails admission; whole contrast unavailable for causal use",
            "latency_only": {
                mode: stats(
                    [
                        r["target_us"]
                        for r in rows
                        if r["contrast"] == contrast and r["mode"] == mode
                    ]
                )
                for mode in sorted(
                    {r["mode"] for r in rows if r["contrast"] == contrast}
                )
            },
        }
    summary = {}
    for contrast in sorted(set(row["contrast"] for row in admitted)):
        summary[contrast] = {}
        for mode in sorted(
            set(row["mode"] for row in admitted if row["contrast"] == contrast)
        ):
            selected = [
                r for r in admitted if r["contrast"] == contrast and r["mode"] == mode
            ]
            summary[contrast][mode] = {
                "target_us": stats([r["target_us"] for r in selected]),
                "gap_us": stats([r["gap_us"] for r in selected]),
                "write_lower_B": stats(
                    [
                        r["counter_window"]["metrics"]["dram__bytes_write.sum"]["value"]
                        for r in selected
                    ]
                ),
                "write_upper_B": stats(
                    [
                        r["counter_window"]["metrics"]["dram__bytes_write.sum"][
                            "expanded_overlap_upper"
                        ]
                        for r in selected
                    ]
                ),
            }
    paired = {}
    for spec in plan["schedule"]:
        selected = {
            r["mode"]: r
            for r in admitted
            if r["contrast"] == spec["id"] and r["round"] == spec["round"]
        }
        if len(selected) != 2:
            continue
        first, second = [selected[mode] for mode in spec["modes"]]
        metric = "dram__bytes_write.sum"
        a, b = [r["counter_window"]["metrics"][metric] for r in (first, second)]
        paired.setdefault(spec["id"], []).append(
            {
                "round": spec["round"],
                "first": first["mode"],
                "second": second["mode"],
                "latency_first_minus_second_us": first["target_us"]
                - second["target_us"],
                "write_difference_lower_B": a["value"] - b["expanded_overlap_upper"],
                "write_difference_upper_B": a["expanded_overlap_upper"] - b["value"],
            }
        )
    return {
        "complete": not pending and not failures,
        "execution_schedule_complete": not pending,
        "counter_window_failures": failures,
        "unqualified_contrasts": raw_unqualified,
        "qualified_comparison_captures": len(admitted),
        "captures": len(rows),
        "pending": pending,
        "configuration": comparable,
        "summary": summary,
        "rows": rows,
        "paired_first_minus_second": paired,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    result = analyze(args.root)
    (args.root / "analysis.json").write_text(json.dumps(result, indent=2))
    print(
        "Checked activity/identity captures",
        result["captures"],
        "admitted for coupled comparison",
        result["qualified_comparison_captures"],
        "pending pairs",
        len(result["pending"]),
    )
    for c, modes in result["summary"].items():
        print(
            c,
            {
                m: (
                    round(v["target_us"]["median"], 3),
                    round(v["write_lower_B"]["median"]),
                )
                for m, v in modes.items()
            },
        )

    if result["counter_window_failures"]:
        print("COUNTER ADMISSION FAILURE:", result["counter_window_failures"])
        raise SystemExit(2)


if __name__ == "__main__":
    main()
