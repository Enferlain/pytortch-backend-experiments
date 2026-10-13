"""Summarize qualified timings and selected physical configurations."""

import collections
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
impls = ("cubecl_baseline", "cubecl_stage5", "triton_matched", "triton_default", "aten")
data = {
    i: json.loads((ROOT / "artifacts" / ("baseline-" + i + ".json")).read_text())
    for i in impls
}
by_name = {i: {r["name"]: r for r in d["records"]} for i, d in data.items()}
rows = []
for name, c in by_name["cubecl_stage5"].items():
    row = {k: c[k] for k in ("name", "family", "shape", "dtype", "dynamic")}
    row["implementations"] = {}
    for i in impls:
        r = by_name[i][name]
        row["implementations"][i] = {
            k: r[k]
            for k in (
                "event_interval_median_us",
                "host_enqueue_median_us",
                "cold_compile_first_run_us",
            )
        }
        row["implementations"][i]["profiled_kernel_sum_us"] = r.get("census", {}).get(
            "profiled_kernel_sum_us"
        )
        row["implementations"][i]["kernel_count"] = r.get("census", {}).get(
            "kernel_count"
        )
        row["implementations"][i]["kernel_census"] = r.get("census", {}).get("kernels")
        row["implementations"][i]["host_environment_gil"] = data[i]["environment"][
            "gil_after"
        ]
        for k in (
            "kernel_duration_median_us",
            "kernel_duration_mad_us",
            "kernel_duration_samples_us",
        ):
            row["implementations"][i][k] = r.get(k)
    b = row["implementations"]["cubecl_baseline"]["kernel_duration_median_us"]
    o = row["implementations"]["cubecl_stage5"]["kernel_duration_median_us"]
    row["profiled_baseline_over_optimized"] = b / o if o else None
    rows.append(row)
selections = []
identity = hashlib.sha256(
    b"".join(
        (ROOT / f).read_bytes()
        for f in ("runtime.py", "legality.py", "plan.py", "timing.py")
    )
).hexdigest()
for path in sorted((ROOT / "cache/tuning").glob("*/selection.json")):
    d = json.loads(path.read_text())
    if d["key_data"]["identity"]["tuning_source"] != identity:
        continue
    # Only keep selections whose semantic compiler binary matches the current
    # measured corpus; historical tuning trials remain raw ignored local data.
    current = data["cubecl_stage5"].get("artifacts", [])
    semantics = {json.dumps(a["semantic"], sort_keys=True) for a in current}
    if json.dumps(d["key_data"]["semantic"], sort_keys=True) not in semantics:
        continue
    # Keep finite distributions and hashes, rather than hundreds of raw
    # CUPTI observations for every candidate/round in committed evidence.
    for candidate in d["candidates"]:
        for key in ("round_samples_ms", "enqueue_interval_samples_ms"):
            rounds = candidate.pop(key, [])
            summaries = []
            for values in rounds:
                if values:
                    ordered = sorted(values)
                    median = statistics.median(values)
                    summaries.append(
                        {
                            "count": len(values),
                            "median_ms": median,
                            "mad_ms": statistics.median(
                                abs(v - median) for v in values
                            ),
                            "p10_ms": ordered[len(ordered) // 10],
                            "p90_ms": ordered[9 * len(ordered) // 10],
                            "raw_sha256": hashlib.sha256(
                                json.dumps(values).encode()
                            ).hexdigest(),
                        }
                    )
            candidate[key + "_summary"] = summaries
    selections.append(d)
strategy_counts = collections.Counter(d["winner"]["strategy"] for d in selections)
summary = {
    "records": rows,
    "methodology": {
        "profiled": "Seven actual GPU kernel sums with identical CUDA activity warmup outside each measured region; separate single-call census is structural. No confidence-interval or universal speedup claim. Physical gains use paired rotated candidate trials, not separate-process differences for identical kernels.",
        "events": "Unprofiled whole invocation intervals include Python enqueue gaps. Host Triton comparisons are GIL-qualified.",
        "tuning": "Inductor warmup/event diagnostics plus384 exact named CUPTI kernel durations per round; GPU annotations and driver queue intervals are excluded from selection.",
        "policy": "math_policy.py checks each operation independently. Default Triton is separately labelled, RNG numerical streams not assumed identical.",
    },
    "selections": len(selections),
    "winner_strategies": dict(strategy_counts),
    "environments": {i: d["environment"] for i, d in data.items()},
}
(ROOT / "artifacts/results.json").write_text(json.dumps(summary, indent=2))
(ROOT / "artifacts/tuning.json").write_text(json.dumps(selections, indent=2))
lines = [
    "| Case (FP32) | Baseline GPU kernels µs | Optimized GPU kernels µs | Configured Triton µs | Default Triton µs | ATen µs | Baseline / optimized |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
representatives = (
    "add-float32-1",
    "add-float32-16777219",
    "fused-float32-1048583",
    "bias_row-float32-256x4099",
    "transpose-float32-256x513",
    "row_sum-float32-64x31",
    "row_sum-float32-64x1024",
    "row_sum-float32-8x65539",
    "column_sum-float32-4099x64",
    "rms_inference-float32-64x1024",
    "rms_backward-float32-64x1024",
    "gather-float32-65539",
    "scatter_collision-float32-65539",
    "dropout_backward-float32-65539",
    "training-float32-5x9",
)
for name in representatives:
    r = next(x for x in rows if x["name"] == name)
    values = [r["implementations"][i]["kernel_duration_median_us"] for i in impls]
    lines.append(
        "| "
        + name
        + " | "
        + " | ".join(f"{v:.3f}" for v in values)
        + f" | {r['profiled_baseline_over_optimized']:.2f}× |"
    )
(ROOT / "artifacts/performance-table.md").write_text("\n".join(lines) + "\n")
print("Winner strategies", strategy_counts)
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [next(r for r in rows if r["name"] == n) for n in representatives[:11]]
    fig, ax = plt.subplots(figsize=(13, 5))
    for j, i in enumerate(impls):
        ax.bar(
            [x + j * 0.16 for x in range(len(selected))],
            [r["implementations"][i]["kernel_duration_median_us"] for r in selected],
            width=0.15,
            label=i,
            yerr=[r["implementations"][i]["kernel_duration_mad_us"] for r in selected],
            capsize=2,
        )
    ax.set_yscale("log")
    ax.set_title("FP32: median of 7 warmed profiler captures; error bars show MAD")
    ax.set_ylabel("GPU kernel sum (µs), profiler instrumented")
    ax.set_xticks(
        [x + 0.32 for x in range(len(selected))],
        [r["name"].removesuffix("-float32") for r in selected],
        rotation=45,
        ha="right",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out = ROOT / "artifacts/plots"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "qualified-family-comparison.svg")
    fig.savefig(out / "qualified-family-comparison.png", dpi=150)
except ImportError:
    print(
        "Plot dependencies absent; rerun in a plotting-only environment with matplotlib."
    )
