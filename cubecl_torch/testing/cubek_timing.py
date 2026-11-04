"""Device time of the bridged sums against Triton and our best native plan
(torch-runtime-bridge 4.5).

``ncu`` with its fixed clocks, one warm call of each pinned plan in its own NVTX
range (``--print-nvtx-rename kernel`` names its kernels after the range), the
kernels' durations summed per call. Our best native plan is first-use
selection's with the bridged plans withdrawn, found first without ``ncu``
(selection times with CUPTI, which ``ncu`` excludes). From the repository root::

    stage0/.venv/bin/python -m cubecl_torch.testing.cubek_timing run --gpu 3080 \\
        --rounds 3 --out cubecl_torch/testing/results/cubek-timing-3080.json

``best`` and ``measure`` are the per-process workers.
"""

import argparse
import csv
import io
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from cubecl_torch.testing.extrema_timing import DEVICES, PYTHON, REPO

SHAPES = ((2048, 8192), (2048, 4099), (300, 4096), (7, 2049), (64, 31))
DTYPES = ("float32", "bfloat16")
WARM = 5


def cases():
    return [(dtype, shape) for dtype in DTYPES for shape in SHAPES]


def case_name(dtype, shape):
    return f"{dtype}-{shape[0]}x{shape[1]}"


def admitted(plan, dtype, cols):
    """The bridged plans the band admits for a class (``cubek_numerical_reason``),
    timed whether or not they are held out for speed (``cubek_speed_reason``,
    lifted here so the rule can be re-measured)."""
    if plan.mapping and cols <= 1024:
        return False
    return dtype != "float32" or bool(plan.mapping)


def _device(torch, gpu):
    device = next(
        i
        for i in range(torch.cuda.device_count())
        if torch.cuda.get_device_name(i) == DEVICES[gpu]
    )
    torch.cuda.set_device(device)


def _cubecl():
    from cubecl_torch.runtime import guard

    guard.install()
    import cubecl_torch.inductor as ci

    ci.install()
    return ci


def best(gpu):
    """Our best native plan per case: first-use selection, bridged plans withdrawn."""
    ci = _cubecl()
    import torch

    from cubecl_torch.compiler.plan import CUBEK_REDUCE
    from cubecl_torch.inductor import tuning

    _device(torch, gpu)
    candidates = tuning.candidates

    def native(*args, **kwargs):
        accepted, rejected = candidates(*args, **kwargs)
        return [p for p in accepted if p.strategy != CUBEK_REDUCE], rejected

    tuning.candidates = native
    result = {}
    for dtype, shape in cases():
        torch._dynamo.reset()
        tuning.CACHE.clear()
        x = torch.randn(shape, device="cuda", dtype=getattr(torch, dtype))
        torch.compile(lambda t: t.sum(-1), backend=ci.compile_backend)(x)
        (kernel,) = [k for k in tuning.CACHE.values() if k.semantic["schema"] == 3]
        (plan,) = set(kernel.selections.values())
        result[case_name(dtype, shape)] = plan.to_dict()
    print("BEST", json.dumps(result))


def measure(backend, gpu, best_plans):
    if backend == "cubecl":
        ci = _cubecl()
    import torch

    _device(torch, gpu)
    if backend == "cubecl":
        from cubecl_torch.compiler.plan import PhysicalCandidate, bridge_catalog
        from cubecl_torch.inductor import tuning
        from cubecl_torch.testing.welford_band import plan_label

        pinned = {}
        select = tuning.AutotunedKernel.select

        def pinned_select(self, key, key_data, plans, rejected, *args):
            for p in plans:
                if p == pinned.get("plan"):
                    pinned["hit"] = True
                    return p
            return select(self, key, key_data, plans, rejected, *args)

        tuning.AutotunedKernel.select = pinned_select
        from cubecl_torch.inductor import legality

        legality.cubek_speed_reason = lambda plan: None
        compile_backend = ci.compile_backend
    else:
        compile_backend = "inductor"

    for dtype, shape in cases():
        name = case_name(dtype, shape)
        x = torch.randn(shape, device="cuda", dtype=getattr(torch, dtype))
        runs = [("triton", None)]
        if backend == "cubecl":
            native = PhysicalCandidate.from_dict(best_plans[name])
            runs = [("native " + plan_label(native), native)] + [
                (plan_label(p), p)
                for p in bridge_catalog()
                if admitted(p, dtype, shape[1])
            ]
        for label, plan in runs:
            torch._dynamo.reset()
            if backend == "cubecl":
                tuning.CACHE.clear()
                pinned.clear()
                pinned["plan"] = plan
            f = torch.compile(lambda t: t.sum(-1), backend=compile_backend)
            for _ in range(WARM):
                f(x)
            torch.cuda.synchronize()
            if backend == "cubecl" and not pinned.get("hit"):
                raise RuntimeError(f"{label} was not offered for {name}")
            torch.cuda.nvtx.range_push("target")
            torch.cuda.nvtx.range_push(f"{name}|{label}")
            f(x)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()


def _python(backend):
    return [PYTHON[backend]]


def _env(backend):
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    if backend == "triton":
        env["PYTHON_GIL"] = "0"
    return env


def _ncu(backend, gpu, best_plans):
    """``{case|label: µs}`` from one profiled worker."""
    with tempfile.TemporaryDirectory() as scratch:
        log = Path(scratch) / "ncu.csv"
        plans = Path(scratch) / "best.json"
        plans.write_text(json.dumps(best_plans))
        subprocess.run(
            [
                "ncu",
                "--nvtx",
                "--nvtx-include",
                "target/",
                "--print-nvtx-rename",
                "kernel",
                "--metrics",
                "gpu__time_duration.sum",
                "--csv",
                "--page",
                "raw",
                "--log-file",
                str(log),
                *_python(backend),
                "-m",
                "cubecl_torch.testing.cubek_timing",
                "measure",
                backend,
                gpu,
                str(plans),
            ],
            cwd=REPO,
            env=_env(backend),
            check=True,
            capture_output=True,
        )
        text = log.read_text()
    rows = list(csv.DictReader(io.StringIO(text[text.index('"ID"') :])))
    times = {}
    for row in rows[1:]:  # the second row holds units
        name = row["Kernel Name"]
        duration = float(row["gpu__time_duration.sum"].replace(",", ""))
        times[name] = times.get(name, 0.0) + duration / 1000.0  # ns -> µs
    return times


def run(gpu, rounds, out):
    found = subprocess.run(
        [*_python("cubecl"), "-m", "cubecl_torch.testing.cubek_timing", "best", gpu],
        cwd=REPO,
        env=_env("cubecl"),
        check=True,
        capture_output=True,
        text=True,
    )
    best_plans = json.loads(
        next(x for x in found.stdout.splitlines() if x.startswith("BEST "))[5:]
    )
    samples = {}
    for _ in range(rounds):
        for backend in ("cubecl", "triton"):
            for key, us in _ncu(backend, gpu, best_plans).items():
                samples.setdefault(key, []).append(us)
    result = {
        "device": DEVICES[gpu],
        "best_native": best_plans,
        "us": {
            k: {"runs": v, "median": statistics.median(v)}
            for k, v in sorted(samples.items())
        },
    }
    if out:
        Path(out).write_text(json.dumps(result, indent=1) + "\n")
    for key, value in result["us"].items():
        runs = ", ".join(f"{r:.1f}" for r in value["runs"])
        print(f"{key}: {runs}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gpu", choices=DEVICES, default="3080")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--out")
    b = sub.add_parser("best")
    b.add_argument("gpu", choices=DEVICES)
    m = sub.add_parser("measure")
    m.add_argument("backend", choices=("cubecl", "triton"))
    m.add_argument("gpu", choices=DEVICES)
    m.add_argument("plans")
    options = parser.parse_args(argv)
    if options.command == "run":
        run(options.gpu, options.rounds, options.out)
    elif options.command == "best":
        best(options.gpu)
    else:
        measure(
            options.backend, options.gpu, json.loads(Path(options.plans).read_text())
        )


if __name__ == "__main__":
    sys.exit(main())
