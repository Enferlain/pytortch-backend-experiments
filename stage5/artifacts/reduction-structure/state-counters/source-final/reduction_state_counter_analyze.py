"""Validate selector-preserving PM captures and keep failed protocols explicit."""

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/reduction-structure/state-counters"
TARGET = "cubecl5_42026f8357481503"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_stream(path):
    return path.open() if path.exists() else gzip.open(str(path) + ".gz", "rt")


def stats(values):
    """Shared distribution helper used by the retained summarize.py driver."""
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def capture(directory, result, row):
    trace = Path(row["trace"])
    assert sha(trace) == row["trace_sha256"]
    document = json.loads(trace.read_text())
    events = document["traceEvents"]
    kernels = sorted(
        (e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"]
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
    expected = {
        "read_write": ["primer_read", "primer_write"],
        "write_read": ["primer_write", "primer_read"],
    }
    if result["reset"] != "dense" or row["mode"] not in expected:
        raise ValueError(
            "Selector proof requires dense reset and read_write/write_read"
        )
    assert [e["name"] for e in kernels[2:-1]] == expected[row["mode"]]
    assert row["bitwise_partial_and_output"]
    output = {"target_us": target["dur"], "prequeue_lead_us": lead}
    if "pm_metrics" not in result:
        return output
    prefix = directory / f"r{row['round']}-{row['mode']}"
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
    assert interior
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


def verify_primer_source(directory):
    tree = ast.parse((directory / "harness.py").read_text())
    source_values = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SOURCE" for t in node.targets)
    ]
    if len(source_values) != 1:
        raise ValueError("Archived harness must contain exactly one literal SOURCE")
    original_source = ast.literal_eval(source_values[0])
    control = directory / "store-control.json"
    expected_source = original_source
    policy = "volatile"
    if control.exists():
        description = json.loads(control.read_text())
        assert sha(directory / "store-control.py") == description["source_sha256"]
        policy = description["policy"]
        assert policy in ("wb", "wt")
        before = "q[(unsigned long long)i * stride] = 1u;"
        after = f"__st{policy}(p + (unsigned long long)i * stride, 1u);"
        assert description["only_source_substitution"] == [before, after]
        assert original_source.count(before) == 1
        expected_source = original_source.replace(before, after)
        ptx = (directory / "primers/kernel.ptx").read_text()
        assert f"st.global.{policy}.u32" in ptx
        other = "wt" if policy == "wb" else "wb"
        assert f"st.global.{other}.u32" not in ptx
    else:
        assert (
            "st.volatile.global.u32" in (directory / "primers/kernel.ptx").read_text()
        )
    assert (directory / "primers/kernel.cu").read_text() == expected_source
    return policy


def analyze(directories):
    reference = json.loads((OUT / "baseline-r0/result.json").read_text())
    identity = [
        reference[k]
        for k in ("input_sha256", "artifact_sha256", "partial_sha256", "output_sha256")
    ]
    result = {"complete": False, "runs": {}}
    comparable = None
    for directory in directories:
        d = json.loads((directory / "result.json").read_text())
        assert d["complete"], directory
        assert [
            d[k]
            for k in (
                "input_sha256",
                "artifact_sha256",
                "partial_sha256",
                "output_sha256",
            )
        ] == identity
        assert d["shape"] == [64, 65539] and d["split"] == 9
        assert d.get("sampler_close_status", 0) == 0, "Sampler cleanup failed"
        assert sha(directory / "harness.py") == d["harness_sha256"]
        assert sha(directory / "primers/kernel.ptx") == d["primer_ptx_sha256"]
        configuration = {
            key: d.get(key)
            for key in (
                "pm_metrics",
                "pm_interval_ns",
                "sampler_sha256",
                "clock_method",
                "reset",
                "harness_sha256",
                "primer_iterations",
                "primer_ptx_sha256",
            )
        }
        sampler_source = directory / "reduction_state_sampling.cpp"
        configuration["sampler_source_sha256"] = (
            sha(sampler_source) if sampler_source.exists() else None
        )
        policy = verify_primer_source(directory)
        configuration["store_policy"] = policy
        if comparable is None:
            comparable = configuration
        assert configuration == comparable, (
            f"Counter configurations differ in {directory.name}",
            configuration,
            comparable,
        )
        for path, digest in d["artifact_sha256"].items():
            assert sha(path) == digest
        rows = [
            capture(directory, d, row)
            | {
                "mode": row["mode"],
                "store_policy": policy,
                "primer_ptx_sha256": d["primer_ptx_sha256"],
            }
            for row in d["captures"]
        ]
        result["runs"][directory.name] = rows
    result["complete"] = True
    result["comparison_configuration"] = comparable
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directories)
    args.output.write_text(json.dumps(result, indent=2))
    print("Validated", len(result["runs"]), "capture groups")


if __name__ == "__main__":
    main()
