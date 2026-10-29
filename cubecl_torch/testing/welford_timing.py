"""Device time of Welford reductions under CubeCL and Inductor+Triton.

One process measures one backend: the generated kernels' summed device time
per call (PyTorch profiler), the mean over ``CALLS`` back-to-back calls, per
case. The CubeCL process also times every physical plan pinned in turn, so
the cost of the Welford plan rule (welford-reductions 5) is the gap between
the default selection and the fastest plan of any kind. Run the backends
alternately, several rounds, on the non-display GPU, and compare only within
one capture (bd memory measurement-environment)::

    python -m cubecl_torch.testing.welford_timing run --rounds 3 \\
        --out cubecl_torch/testing/results/welford-timing-3080.json

``run`` drives both interpreters; ``measure`` is the per-process worker.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from cubecl_torch.testing.extrema_timing import DEVICES, PYTHON, REPO, clocks, margin

CALLS = 100
CASES = (
    ("var_mean", "float32", (37, 129)),
    ("var_mean", "float32", (300, 4096)),
    ("var_mean", "bfloat16", (300, 4096)),
    # Two sums in both backends (wide-reduction-workgroups 5).
    ("var_mean", "bfloat16", (2048, 8192)),
    ("var_mean", "float32", (8, 300000)),
    ("layer_norm", "bfloat16", (4096, 1280)),
    ("layer_norm", "float32", (4096, 1280)),
    ("layer_norm_mean", "bfloat16", (4096, 1280)),
    ("layer_norm_mean", "float32", (4096, 1280)),
    ("group_norm", "float32", (1, 320, 64, 64)),
    ("group_norm", "bfloat16", (1, 320, 64, 64)),
    # Kernels that read the row again (wide-reduction-workgroups 4.1).
    *(
        (kind, dtype, shape)
        for kind in ("softmax", "sum_consumer", "layer_norm", "layer_norm_affine")
        for dtype in ("float32", "bfloat16")
        for shape in ((300, 4096), (2048, 8192))
    ),
)


def function(kind, x):
    import torch

    if kind == "softmax":
        return lambda x: torch.softmax(x, -1)
    if kind == "sum_consumer":
        return lambda x: x * x.sum(-1, keepdim=True)
    if kind == "layer_norm_affine":
        g = torch.Generator(device=x.device).manual_seed(5)
        weight = torch.randn(x.shape[-1:], generator=g, device=x.device).to(x.dtype)
        bias = torch.randn(x.shape[-1:], generator=g, device=x.device).to(x.dtype)
        return lambda x: torch.ops.aten.native_layer_norm(
            x, x.shape[-1:], weight, bias, 1e-5
        )
    if kind == "var_mean":
        return lambda x: torch.var_mean(x, -1, correction=0)
    if kind == "layer_norm":
        return lambda x: torch.ops.aten.native_layer_norm(
            x, x.shape[-1:], None, None, 1e-5
        )
    if kind == "layer_norm_mean":
        # One statistic: one kernel on both backends, compared kernel to kernel.
        return lambda x: torch.ops.aten.native_layer_norm(
            x, x.shape[-1:], None, None, 1e-5
        )[1]
    n, c = 1, 320
    return lambda x: torch.ops.aten.native_group_norm(
        x, None, None, n, c, 4096, 32, 1e-5
    )


def measure(backend, gpu, only=None):
    if backend == "cubecl":
        from cubecl_torch.runtime import guard

        guard.install()
        import cubecl_torch.inductor as ci

        ci.install()  # before anything imports Inductor (torch.profiler does)
    import torch
    from torch.profiler import ProfilerActivity, profile

    pinned = {}
    if backend == "cubecl":
        from cubecl_torch.inductor import tuning

        compile_backend = ci.compile_backend
        select = tuning.AutotunedKernel.select

        def pinned_select(self, key, key_data, plans, rejected, *args):
            for p in plans:
                if p.identity == pinned.get("plan"):
                    pinned["hit"] = True
                    return p
            # Legal only under other plans: time those kernels as selected.
            return select(self, key, key_data, plans, rejected, *args)

        tuning.AutotunedKernel.select = pinned_select
        # Every reduction plan, the Welford rule's exclusions included.
        from cubecl_torch.compiler.plan import catalog, welford_catalog
        from cubecl_torch.inductor import legality

        rule = legality.welford_plan_reason

        def pinned_rule(*args):
            # The default selection keeps the rule; a pinned plan and the
            # "unrestricted" selection bypass it.
            return (
                None
                if pinned.get("plan") or pinned.get("unrestricted")
                else rule(*args)
            )

        legality.welford_plan_reason = pinned_rule
        plans = [
            p for p in (*catalog(), *welford_catalog()) if "pointwise" not in p.strategy
        ]
    else:
        compile_backend = "inductor"
        plans = []

    device = next(
        i
        for i in range(torch.cuda.device_count())
        if torch.cuda.get_device_name(i) == DEVICES[gpu]
    )
    torch.cuda.set_device(device)

    def spin(seconds):
        import time

        a = torch.randn(2048, 2048, device="cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            a = a @ a
            a = a / a.norm()
        torch.cuda.synchronize()

    def timed(f, x):
        for _ in range(20):  # compile, first-use tuning, warm-up
            f(x)
        spin(0.3)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(CALLS):
                f(x)
            torch.cuda.synchronize()
        kernels = {}
        for e in prof.events():
            if e.device_type.name == "CUDA" and e.device_time > 0:
                kernels[e.name] = kernels.get(e.name, 0.0) + e.device_time / CALLS
        timed.kernels = kernels
        return sum(kernels.values())

    spin(3.0)
    results = {}
    for kind, dtype, shape in CASES:
        name = f"{kind}-{dtype}-" + "x".join(map(str, shape))
        if only and only not in name:
            continue
        x = torch.randn(shape, device="cuda", dtype=getattr(torch, dtype))
        row = {}
        for plan in [None, "unrestricted", *plans]:
            pinned.clear()
            if plan == "unrestricted":
                pinned["unrestricted"] = True
            elif plan is not None:
                pinned["plan"] = plan.identity
            torch._dynamo.reset()
            if backend == "cubecl":
                tuning.CACHE.clear()
            try:
                us = timed(torch.compile(function(kind, x), backend=compile_backend), x)
            except Exception as error:  # noqa: BLE001  a plan refused for this case
                row[plan_name(plan)] = str(error).splitlines()[0][:120]
                continue
            if plan is None or plan == "unrestricted":
                # Per-kernel device time and plans of both selections.
                prefix = "" if plan is None else "unrestricted_"
                row[prefix + "kernels"] = {
                    k: round(v, 3) for k, v in timed.kernels.items()
                }
                if backend == "cubecl":
                    row[prefix + "plans"] = sorted(
                        plan_name(p)
                        for k in tuning.CACHE.values()
                        for p in k.selections.values()
                    )
            if isinstance(plan, str) or plan is None or pinned.get("hit"):
                row[plan_name(plan)] = us
            else:
                # Not legal for any of this case's kernels: nothing was pinned.
                row[plan_name(plan)] = "not legal for this case"
        results[name] = row
    return {"backend": backend, "device": DEVICES[gpu], "us": results}


def plan_name(plan):
    if plan is None:
        return "selected"
    if plan == "unrestricted":
        return "unrestricted"
    return (
        f"{plan.strategy}/{plan.workgroup[0]}x{plan.workgroup[1]}"
        f"a{plan.private_accumulators}v{plan.vector_width}/{plan.private_merge}"
    )


def run(rounds, out, gpu, only=None):
    captures = []
    for r in range(rounds):
        for backend in ("cubecl", "triton") if r % 2 == 0 else ("triton", "cubecl"):
            before = clocks()
            proc = subprocess.run(
                [
                    PYTHON[backend],
                    "-m",
                    "cubecl_torch.testing.welford_timing",
                    "measure",
                    backend,
                    gpu,
                    *(["--only", only] if only else []),
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=True,
            )
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            row.update(round=r, clocks_before=before, clocks_after=clocks())
            captures.append(row)
    summary = summarize(captures)
    Path(out).write_text(
        json.dumps({"captures": captures, "summary": summary}, indent=1) + "\n"
    )
    return summary


def summarize(captures):
    """Medians: Triton, CubeCL's selection, and its fastest pinned plan."""
    summary = {}
    for case in captures[0]["us"]:
        t = [x["us"][case]["selected"] for x in captures if x["backend"] == "triton"]
        cubecl = [x["us"][case] for x in captures if x["backend"] == "cubecl"]
        c = [row["selected"] for row in cubecl]
        plans = {
            name
            for row in cubecl
            for name, v in row.items()
            if name not in ("selected", "unrestricted", "kernels")
            and isinstance(v, float)
        }
        medians = {
            name: statistics.median(
                row[name] for row in cubecl if isinstance(row.get(name), float)
            )
            for name in plans
        }
        fastest = min(medians, key=medians.get) if medians else None
        summary[case] = {
            "triton_median": round(statistics.median(t), 2),
            "cubecl_median": round(statistics.median(c), 2),
            "cubecl_unrestricted_median": round(
                statistics.median(row["unrestricted"] for row in cubecl), 2
            ),
            "cubecl_fastest_plan": fastest,
            "cubecl_fastest_median": round(medians[fastest], 2) if fastest else None,
            "triton_kernels_round0": next(
                x["us"][case]["kernels"] for x in captures if x["backend"] == "triton"
            ),
            "cubecl_kernels_round0": cubecl[0]["kernels"],
            "cubecl_plans_round0": cubecl[0].get("plans"),
            "cubecl_unrestricted_kernels_round0": cubecl[0].get("unrestricted_kernels"),
            "cubecl_unrestricted_plans_round0": cubecl[0].get("unrestricted_plans"),
            "slower_than_triton": statistics.median(c)
            > statistics.median(t) * (1 + margin(c, t)),
            "plans": {
                k: round(v, 2) for k, v in sorted(medians.items(), key=lambda kv: kv[1])
            },
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("backend", choices=sorted(PYTHON))
    m.add_argument("gpu", choices=sorted(DEVICES))
    m.add_argument("--only", help="time only cases whose name contains this")
    r = sub.add_parser("run")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--out", required=True)
    r.add_argument("--gpu", choices=sorted(DEVICES), default="3080")
    r.add_argument("--only", help="time only cases whose name contains this")
    args = parser.parse_args(argv)
    if args.command == "measure":
        print(json.dumps(measure(args.backend, args.gpu, args.only)))
        return 0
    for case, row in run(args.rounds, args.out, args.gpu, args.only).items():
        print(case, {k: v for k, v in row.items() if k != "plans"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
