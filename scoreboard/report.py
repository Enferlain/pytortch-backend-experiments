"""Summarize a scoreboard results directory into coverage and cost tables (Markdown).

Usage: python scoreboard/report.py RESULTS_DIR   (writes RESULTS_DIR/summary.md and prints it)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PATHS = ("eager", "triton", "cubecl")


def load(directory: Path):
    results = {}
    for f in sorted(directory.glob("*.json")):
        r = json.loads(f.read_text())
        results[(r["workload"], r["path"])] = r
    return results


def ms(us):
    return f"{us / 1000:.1f}" if us is not None else "–"


def fmt(x, spec=".3g"):
    return format(x, spec) if isinstance(x, (int, float)) else "–"


def class_us(r, predicate):
    classes = (r.get("gpu_kernels") or {}).get("by_class", {})
    return sum(v["us"] for k, v in classes.items() if predicate(k)) if classes else None


def coverage_table(results):
    lines = [
        "| Workload | Path | Status | GIL at end | Compiled regions | Graph breaks | CubeCL regions | Most frequent rejection / failure |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (w, p), r in sorted(
        results.items(), key=lambda kv: (kv[0][0], PATHS.index(kv[0][1]))
    ):
        cov = r.get("coverage") or {}
        statuses = (
            ", ".join(f"{k} {v}" for k, v in cov.get("by_status", {}).items()) or "–"
        )
        reason = next(iter(cov.get("rejection_reasons", {})), None) or (
            r.get("failure") or {}
        ).get("error", "")
        reason = reason.replace("|", "/").replace("\n", " ")[:160] or "–"
        breaks = sum((r.get("dynamo") or {}).get("graph_breaks", {}).values())
        gil = r["environment"]["gil"].get("end")
        lines.append(
            f"| {w} | {p} | {r['status']} | {'on' if gil else 'off'} | "
            f"{(r.get('dynamo') or {}).get('unique_graphs', '–')} | {breaks} | {statuses} | {reason} |"
        )
    return "\n".join(lines)


def region_reasons(region):
    """``{reason: count}`` for a rejected region; older results kept only the first."""
    if "reasons" in region:
        return region["reasons"]
    return {region.get("reason", "").split("\n")[0][:300]: 1}


def reasons_table(results):
    """Per workload: each distinct rejection reason, the regions that have it and
    how often it was found in them (nodes, for reasons found before fusion).
    Shape symbols are renamed ``sN`` so one reason is one row across regions."""
    lines = [
        "| Workload | Reason | Regions | Found |",
        "|---|---|---|---|",
    ]
    for (w, p), r in sorted(results.items()):
        if p != "cubecl":
            continue
        regions, found = {}, {}
        for region in (r.get("coverage") or {}).get("regions_detail", []):
            if region["status"] != "rejected":
                continue
            for reason, n in region_reasons(region).items():
                # One row per reason, whichever shape symbols (s33, s53, ...) it names.
                reason = re.sub(r"\bs\d+\b", "sN", reason)
                regions[reason] = regions.get(reason, 0) + 1
                found[reason] = found.get(reason, 0) + n
        for reason in sorted(regions, key=lambda k: (-regions[k], k)):
            text = reason.replace("|", "/").replace("\n", " ")[:200]
            lines.append(f"| {w} | {text} | {regions[reason]} | {found[reason]} |")
    return "\n".join(lines)


def cost_table(results):
    lines = [
        (
            "| Workload | Path | Step GPU-elapsed ms (median) | Kernel sum ms | Generated ms | Vendor-library ms "
            "| Attention ms | ATen/other ms | vs eager | vs Triton | First step s | Peak GiB | Output rel err "
            "| Grad-norm rel err (max) |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (w, p), r in sorted(
        results.items(), key=lambda kv: (kv[0][0], PATHS.index(kv[0][1]))
    ):
        step = (r.get("steady_gpu_ms") or {}).get("median")
        ref = {
            q: (results.get((w, q), {}).get("steady_gpu_ms") or {}).get("median")
            for q in ("eager", "triton")
        }
        ratio = {q: f"{step / v:.2f}×" if step and v else "–" for q, v in ref.items()}
        kernels = r.get("gpu_kernels") or {}
        c = r.get("correctness") or {}
        lines.append(
            f"| {w} | {p} | {fmt(step, '.1f')} | {ms(kernels.get('total_us'))} | "
            f"{ms(class_us(r, lambda k: k.startswith(('triton_', 'cubecl'))))} | {ms(class_us(r, lambda k: k == 'vendor_library'))} | "
            f"{ms(class_us(r, lambda k: k == 'attention'))} | {ms(class_us(r, lambda k: k == 'aten_other'))} | {ratio['eager']} | {ratio['triton']} | {fmt(r.get('first_step_s'), '.1f')} | "
            f"{fmt(r.get('peak_memory_gib'), '.1f')} | {fmt(c.get('output_rel_l2_max'))} | {fmt(c.get('grad_norm_rel_max'))} |"
        )
    return "\n".join(lines)


def family_table(results):
    lines = [
        "| Workload | Path | Kernel class | Calls | GPU ms | Share of kernel time |",
        "|---|---|---|---|---|---|",
    ]
    for (w, p), r in sorted(
        results.items(), key=lambda kv: (kv[0][0], PATHS.index(kv[0][1]))
    ):
        kernels = r.get("gpu_kernels") or {}
        total = kernels.get("total_us") or 0
        for k, v in sorted(
            kernels.get("by_class", {}).items(), key=lambda kv: -kv[1]["us"]
        ):
            lines.append(
                f"| {w} | {p} | {k} | {v['calls']} | {ms(v['us'])} | {v['us'] / total:.0%} |"
            )
    return "\n".join(lines)


def notes(results):
    out = []
    for (w, p), r in sorted(results.items()):
        for d in r.get("config_deviations", []):
            out.append(f"- {w}/{p}: {d}")
        settings = r.get("settings") or {}
        if p == "eager" and settings:
            out.append(
                f"- {w} settings: " + ", ".join(f"{k}={v}" for k, v in settings.items())
            )
        kernels = r.get("gpu_kernels") or {}
        if kernels.get("attention_kernels"):
            out.append(
                f"- {w}/{p} attention kernels: "
                + "; ".join(
                    sorted(
                        {
                            k.split("<")[0].split("(")[0].removeprefix("void ")
                            for k in kernels["attention_kernels"]
                        }
                    )
                )
            )
        if kernels.get("cuda_graph_launches"):
            out.append(
                f"- {w}/{p}: {kernels['cuda_graph_launches']} cudaGraphLaunch calls in the profiled step"
            )
    devices = {r["environment"].get("device") for r in results.values()}
    clocks = {
        (w, p): (r.get("clocks_before"), r.get("clocks_after"))
        for (w, p), r in results.items()
    }
    out.append(f"- Devices: {', '.join(sorted(d for d in devices if d))}")
    out.append(
        "- Clocks are not fixed yet (sil.1); before/after nvidia-smi readings are in each JSON. "
        f"Example: {next(iter(clocks.values()), None)}"
    )
    return "\n".join(out)


def render(directory: Path) -> str:
    results = load(directory)
    return (
        "\n\n".join(
            [
                f"# Scoreboard — {directory.name}",
                (
                    "Step time is the CUDA-event interval around one training/optimizer step (includes GPU idle gaps, so host-bound "
                    "steps show up). Kernel columns sum CUPTI kernel durations from one profiled step. *Generated* = Inductor "
                    "kernels (Triton or CubeCL); *ATen/other* mixes eager code outside compiled regions, extern ATen calls and "
                    "fallbacks."
                ),
                "## Coverage",
                coverage_table(results),
                "## Rejection reasons",
                (
                    "Inventory (`--on-reject record`) runs scan each region past its first unsupported "
                    "feature, forward and backward; fail-fast runs and older results list only the first."
                ),
                reasons_table(results),
                "## Cost",
                cost_table(results),
                "## Kernel classes",
                family_table(results),
                "## Notes",
                notes(results),
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    text = render(directory)
    (directory / "summary.md").write_text(text)
    print(text)
