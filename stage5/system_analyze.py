"""Nsight stats plus explicit SQL attribution to measured NVTX steps.

API, range and device durations remain distinct. Unlabelled wall time is not
assigned to a causal layer. Profiling overhead is not subtracted from timings.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import subprocess

from stage5.attribution import sha, summary

ROOT = Path(__file__).resolve().parent


def union_ns(intervals):
    total, end = 0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def analyze(path, evidence):
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        strings = dict(connection.execute("SELECT id, value FROM StringIds"))
        ranges = [
            dict(row)
            for row in connection.execute(
                "SELECT start,end,globalTid,text,textId,eventType FROM NVTX_EVENTS WHERE end IS NOT NULL"
            )
        ]
        for r in ranges:
            r["name"] = r["text"] or strings.get(r["textId"], "")
        steps = [r for r in ranges if r["name"].startswith("step:")]
        assert len(steps) == evidence["steps"], "Missing measured NVTX steps"
        apis = [
            dict(row)
            for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        ]
        for a in apis:
            a["name"] = strings[a["nameId"]]
        kernels = [
            dict(row)
            for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL")
        ]
        for k in kernels:
            k["name"] = strings[k["demangledName"]]
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        memcpy = (
            [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY"
                )
            ]
            if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables
            else []
        )
        # Single-process capture; correlationId is unique in this process.
        assert len({a["correlationId"] for a in apis}) == len(apis)
        by_correlation = {a["correlationId"]: a for a in apis}
        records = []
        phases = defaultdict(list)
        exclusive = defaultdict(list)
        library_ranges = defaultdict(list)
        counts = Counter()
        api_by_name = defaultdict(list)
        queue, native_api = [], []
        kernel_names = Counter()
        for step in sorted(steps, key=lambda r: r["start"]):
            begin, end = step["start"], step["end"]
            current_api = [a for a in apis if begin <= a["start"] < end]
            ids = {a["correlationId"] for a in current_api}
            current_kernels = [k for k in kernels if k["correlationId"] in ids]
            generated_correlations = {
                k["correlationId"]
                for k in current_kernels
                if k["name"].startswith("cubecl5_")
            }
            assert current_kernels, "Missing GPU activity for measured step"
            for k in current_kernels:
                a = by_correlation[k["correlationId"]]
                queue.append(max(0, k["start"] - a["end"]) / 1000)
                kernel_names[k["name"]] += 1
            for a in current_api:
                value = (a["end"] - a["start"]) / 1000
                api_by_name[a["name"]].append(value)
                if (
                    a["name"].startswith("cuLaunchKernel")
                    and a["correlationId"] in generated_correlations
                ):
                    native_api.append(value)
            local_ranges = [
                r
                for r in ranges
                if begin <= r["start"] and r["end"] <= end and r is not step
            ]
            for r in local_ranges:
                if r["name"].lower().startswith("cublas"):
                    library_ranges[r["name"]].append((r["end"] - r["start"]) / 1000)
                if r["name"] in ("forward", "backward", "optimizer"):
                    phases[r["name"]].append((r["end"] - r["start"]) / 1000)
                if evidence["deep"] and r["name"] in (
                    "selection_dispatch",
                    "proof_signature",
                    "compiled_dispatch",
                    "kernel_dispatch",
                    "validation",
                    "stream_readiness_lifetime",
                    "record_stream",
                    "current_stream",
                    "capture_query",
                    "event_readiness",
                    "metadata",
                    "argument_binding",
                    "native_enqueue",
                    "lock_wait",
                ):
                    children = [
                        (c["start"], c["end"])
                        for c in local_ranges
                        if c is not r
                        and c["globalTid"] == r["globalTid"]
                        and r["start"] <= c["start"]
                        and c["end"] <= r["end"]
                    ]
                    value = r["end"] - r["start"] - union_ns(children)
                    assert value >= 0
                    exclusive[r["name"]].append(value / 1000)
            generated = sum(k["name"].startswith("cubecl5_") for k in current_kernels)
            dispatch_intervals = [
                (r["start"], r["end"])
                for r in local_ranges
                if r["name"].startswith("generated_dispatch:")
            ]
            within = [
                (max(begin, k["start"]), min(end, k["end"]))
                for k in kernels
                if k["start"] < end and k["end"] > begin
            ]
            records.append(
                {
                    "name": step["name"],
                    "wall_us": (end - begin) / 1000,
                    "generated_dispatch_union_us": union_ns(dispatch_intervals) / 1000,
                    "unlabelled_wall_us": ((end - begin) - union_ns(dispatch_intervals))
                    / 1000,
                    "kernel_count": len(current_kernels),
                    "generated_count": generated,
                    "kernel_duration_sum_us": sum(
                        k["end"] - k["start"] for k in current_kernels
                    )
                    / 1000,
                    "device_busy_overlap_us": union_ns(within) / 1000,
                    "api_duration_sum_us": sum(
                        a["end"] - a["start"] for a in current_api
                    )
                    / 1000,
                    "api_counts": dict(Counter(a["name"] for a in current_api)),
                    "memcpy_count": sum(m["correlationId"] in ids for m in memcpy),
                }
            )
            counts.update(a["name"] for a in current_api)
        generated_total = sum(r["generated_count"] for r in records)
        if evidence["implementation"] in ("cubecl", "reference"):
            assert generated_total == evidence["ordinary_native_launches"]
        result = {
            "sqlite": str(path),
            "sqlite_sha256": sha(path),
            "steps": records,
            "step_wall": summary([r["wall_us"] for r in records]),
            "generated_dispatch_union": summary(
                [r["generated_dispatch_union_us"] for r in records]
            ),
            "kernel_duration_sum": summary(
                [r["kernel_duration_sum_us"] for r in records]
            ),
            "phases": {k: summary(v) for k, v in phases.items()},
            "api": {
                k: {**summary(v), "count": len(v), "sum_us": sum(v)}
                for k, v in api_by_name.items()
            },
            "queue_after_api": summary(queue),
            "native_launch_api": summary(native_api) if native_api else None,
            "exclusive_ranges": {
                k: {
                    **summary(v),
                    "calls": len(v),
                    "sum_us_per_step": sum(v) / len(steps),
                }
                for k, v in exclusive.items()
            },
            "library_ranges": {
                k: {
                    **summary(v),
                    "calls": len(v),
                    "sum_us_per_step": sum(v) / len(steps),
                }
                for k, v in library_ranges.items()
            },
            "kernel_names": dict(kernel_names),
            "tables": tables,
            "qualification": "profiled timestamps; ranges include instrumentation; API durations are not adapter wall time; generated native API population is matched by kernel correlation; kernel_duration_sum includes full correlated kernel durations even across step end, device_busy_overlap clips to step; queue is max(0,kernel start-API end); unlabelled time is unassigned",
        }
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="")
    args = parser.parse_args()
    out = ROOT / "artifacts/attribution"
    result = []
    paths = sorted((ROOT / "cache/attribution").glob("systems-*.nsys-rep"))
    if args.name:
        paths = [p for p in paths if p.stem == args.name]
    assert paths, "No Systems captures found"
    for path in paths:
        command = [
            "nsys",
            "stats",
            "--report",
            "cuda_api_sum,cuda_gpu_kern_sum,cuda_kern_exec_sum,nvtx_pushpop_sum",
            "--format",
            "column",
            str(path),
        ]
        proc = subprocess.run(command, text=True, capture_output=True, check=True)
        (out / (path.stem + "-stats.txt")).write_text(proc.stdout + proc.stderr)
        evidence = json.loads((out / (path.stem + ".json")).read_text())
        data = analyze(path.with_suffix(".sqlite"), evidence)
        data.update(
            name=path.stem, command=command, report_sha256=sha(path), capture=evidence
        )
        result.append(data)
        print(
            path.stem,
            data["step_wall"]["median_us"],
            data["generated_dispatch_union"]["median_us"],
            flush=True,
        )
    target = "systems-analysis" + ("-" + args.name if args.name else "") + ".json"
    (out / target).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
