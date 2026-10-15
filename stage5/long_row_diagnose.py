"""Verify and summarize production partial/final evidence without retiming it."""

import collections
import csv
import io
import json
from pathlib import Path
import re
import shutil
import statistics

from stage5.reference_reduction_probe import sha

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/long-row-diagnosis"
CACHE = ROOT / "cache/long-row-diagnosis"
COUNTER_NAMES = (
    "gpu__time_duration.sum",
    "launch__block_size",
    "launch__grid_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_dynamic",
    "launch__waves_per_multiprocessor",
    "smsp__inst_executed.sum",
    "sass__inst_executed_register_spilling",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
)


def distribution(values):
    assert values and all(v > 0 for v in values), "incomplete/nonpositive samples"
    median = statistics.median(values)
    return {
        "median_us": median,
        "mad_us": statistics.median(abs(v - median) for v in values),
        "samples_us": values,
    }


def instructions(text):
    """NCU source view has dynamic warp and predicated-thread execution counts."""
    totals = collections.defaultdict(collections.Counter)
    current = None
    for line in text.splitlines():
        if line.startswith("Kernel Name"):
            current = line.split()[-1]
        match = re.match(
            r"^0x[0-9a-f]+\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)"
            r".*?\s+(\d+)\s+(\d+)\s*$",
            line,
        )
        if match:
            assert current is not None
            totals[current][match[1].split(".")[0]] += int(match[2])
    assert totals, "missing SASS execution counts"
    return totals


def profile(name):
    raw = CACHE / (name + "-metrics.csv")
    sass = CACHE / (name + "-sass.txt")
    text = raw.read_text()
    rows = list(csv.DictReader(io.StringIO(text)))
    # NCU 2026.3 raw CSV is wide; its first data row contains metric units.
    units, rows = rows[0], rows[1:]
    assert len(rows) == 2 and [r["ID"] for r in rows] == ["0", "1"]
    executed = instructions(sass.read_text())
    capture = json.loads((OUTPUT / (name + ".json")).read_text())
    assert capture["qualification_complete"] and capture["resources_only"]
    assert capture["shape"] == [int(name.split("-")[1]), 65539]
    env = capture["environment"]
    cube = name.startswith("production")
    assert env["gil"] == (not cube) and env["PYTHON_GIL"] is None
    assert (
        env["fp_patch_sha256"]
        == "9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2"
    )
    if cube:
        assert env["independence"]["loaded_modules"] == []
        assert env["independence"]["native_mappings"] == []
    answer = []
    for row in rows:
        kernel = row["Kernel Name"]
        assert kernel.startswith("cubecl5_" if cube else "triton_")
        assert kernel in {k["entrypoint"] for k in capture["selected_kernels"]}
        assert sum(executed[kernel].values()) == float(row["smsp__inst_executed.sum"])
        answer.append(
            {
                "kernel": kernel,
                "metrics": {
                    key: {"value": float(row[key].replace(",", "")), "unit": units[key]}
                    for key in COUNTER_NAMES
                },
                "dynamic_warp_instructions_by_opcode": dict(executed[kernel]),
            }
        )
    # Preserve actual driver-JIT SASS, not a separately assembled approximation.
    shutil.copyfile(sass, OUTPUT / (name + "-sass.txt"))
    return {
        "kernels": answer,
        "input_sha256": capture["input_sha256"],
        "raw_metrics": {"path": str(raw), "sha256": sha(raw), "rows": rows},
        "sass_sha256": sha(sass),
        "units": units,
        "command": json.loads((OUTPUT / (name + "-command.json")).read_text()),
        "environment": env,
    }


def historical_components():
    result = {}
    for name in ("production-guarded-large", "triton-large"):
        path = ROOT / "artifacts/long-row" / (name + ".json")
        capture = json.loads(path.read_text())
        assert capture["qualification_complete"]
        rows = []
        for timing in capture["timings"]:
            if timing["case"] not in ("float32-8-65539", "float32-64-65539"):
                continue
            assert len(timing["profiles"]) == 21
            parts = [[], []]
            samples, traces = [], []
            for record in timing["profiles"]:
                trace = Path(record["raw_trace"])
                assert sha(trace) == record["sha256"], "historical trace changed"
                events = sorted(
                    (
                        e
                        for e in json.loads(trace.read_text())["traceEvents"]
                        if e.get("cat") == "kernel"
                    ),
                    key=lambda e: e["ts"],
                )
                assert len(events) == 2
                assert [e["name"] for e in events] == record["names"]
                assert record["round"] == len(samples) // 7
                for index, event in enumerate(events):
                    parts[index].append(event["dur"])
                samples.append(sum(e["dur"] for e in events))
                traces.append({"path": str(trace), "sha256": sha(trace)})
            # Chrome and Kineto timestamps differ slightly in representation.
            assert abs(statistics.median(samples) - timing["median_us"]) < 0.01
            rows.append(
                {
                    "case": timing["case"],
                    "input_sha256": timing["input_sha256"],
                    "partial": distribution(parts[0]),
                    "final": distribution(parts[1]),
                    "graph_sum": distribution(samples),
                    "traces": traces,
                }
            )
        assert len(rows) == 2
        result[name] = {"capture_sha256": sha(path), "cases": rows}
    for left, right in zip(
        result["production-guarded-large"]["cases"], result["triton-large"]["cases"]
    ):
        assert left["case"] == right["case"]
        assert left["input_sha256"] == right["input_sha256"]
    return result


def main():
    profiles = {
        name: profile(name)
        for name in (
            "production-8",
            "triton-8",
            "production-64",
            "triton-64",
            "production-64-fixed9",
            "production-8-after",
        )
    }
    for count in (8, 64):
        a, b = (profiles[f"{kind}-{count}"] for kind in ("production", "triton"))
        assert a["input_sha256"] == b["input_sha256"]
        assert a["kernels"][0]["metrics"]["launch__grid_size"]["value"] == count * (
            9 if count == 8 else 4
        )
        assert b["kernels"][0]["metrics"]["launch__grid_size"]["value"] == count * 9
    assert (
        profiles["production-64-fixed9"]["input_sha256"]
        == profiles["triton-64"]["input_sha256"]
    )
    assert (
        profiles["production-64-fixed9"]["kernels"][0]["metrics"]["launch__grid_size"][
            "value"
        ]
        == 576
    )
    assert (
        profiles["production-8-after"]["input_sha256"]
        == profiles["production-8"]["input_sha256"]
    )
    result = {
        "complete": True,
        "historical_cupti": historical_components(),
        "nsight": profiles,
        "analyzer_sha256": sha(__file__),
        "scope": "nine-part FP32 comparisons plus actual four-part 64-row CubeCL control; counter durations separate from CUPTI timing",
    }
    assert len({p["environment"]["torch_commit"] for p in profiles.values()}) == 1
    assert len({p["environment"]["device"] for p in profiles.values()}) == 1
    (OUTPUT / "diagnosis.json").write_text(json.dumps(result, indent=2))
    print(
        f"verified historical per-kernel samples and {len(profiles)} two-kernel Nsight captures"
    )


if __name__ == "__main__":
    main()
