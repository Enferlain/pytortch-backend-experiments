"""Device time of RMSNorm's weight gradient, ``(g * x * rstd[:, None]).sum(0)``,
against Inductor+Triton (fused-reduction-bridge tasks 1.2 and 5.2).

``ncu`` with its fixed clocks. Triton: one warm call per case in its own NVTX
range, its kernels' durations summed. Burn (an external program, ``--burn``):
twenty calls per case; each kernel name's median over the last ten calls,
summed, is one call. From the repository root::

    stage0/.venv/bin/python -m cubecl_torch.testing.fused_reduction_timing run \\
        --gpu 3080 --burn BIN --burn-mode stock --burn-mode PROBE_CREDIT=1 \\
        --out cubecl_torch/testing/results/fused-reduction-timing-3080.json

``measure`` is the per-process Triton worker.
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

SHAPES = ((64, 1024), (8192, 1280))
DTYPES = ("float32", "bfloat16")
WARM = 5


def cases():
    return [(dtype, shape) for dtype in DTYPES for shape in SHAPES]


def case_name(dtype, shape):
    return f"{dtype}-{shape[0]}x{shape[1]}"


def weight_gradient(g, x, rstd):
    return (g.float() * x.float() * rstd.float()[:, None]).sum(0).to(g.dtype)


def _device(torch, gpu):
    index = next(
        i
        for i in range(torch.cuda.device_count())
        if torch.cuda.get_device_name(i) == DEVICES[gpu]
    )
    torch.cuda.set_device(index)


def measure(gpu):
    import torch

    _device(torch, gpu)
    for dtype, shape in cases():
        t = getattr(torch, dtype)
        g = torch.randn(shape, device="cuda", dtype=t)
        x = torch.randn(shape, device="cuda", dtype=t)
        rstd = torch.rand(shape[0], device="cuda", dtype=t) + 0.5
        torch._dynamo.reset()
        f = torch.compile(weight_gradient, dynamic=False)
        for _ in range(WARM):
            f(g, x, rstd)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("target")
        torch.cuda.nvtx.range_push(case_name(dtype, shape))
        f(g, x, rstd)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()


def _rows(text):
    rows = list(csv.DictReader(io.StringIO(text[text.index('"ID"') :])))
    assert rows[0]["gpu__time_duration.sum"] == "ns", rows[0]
    return [
        (r["Kernel Name"], float(r["gpu__time_duration.sum"].replace(",", "")) / 1e3)
        for r in rows[1:]  # the second row holds units
    ]


def triton(gpu):
    """``{case: µs}``."""
    with tempfile.TemporaryDirectory() as scratch:
        log = Path(scratch) / "ncu.csv"
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
                PYTHON["triton"],
                "-m",
                "cubecl_torch.testing.fused_reduction_timing",
                "measure",
                gpu,
            ],
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO), "PYTHON_GIL": "0"},
            check=True,
            capture_output=True,
        )
        times = {}
        for name, us in _rows(log.read_text()):
            times[name] = times.get(name, 0.0) + us
        return times


def _uuid(gpu):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return next(
        line.split(",")[1].strip()
        for line in out.splitlines()
        if line.split(",")[0].strip() == DEVICES[gpu]
    )


def burn(binary, gpu, mode, env_extra):
    """``{case: {"us": µs per call, "kernels": [names]}}`` for one Burn mode."""
    result = {}
    for dtype, shape in cases():
        with tempfile.TemporaryDirectory() as scratch:
            log = Path(scratch) / "ncu.csv"
            subprocess.run(
                [
                    "ncu",
                    "--metrics",
                    "gpu__time_duration.sum",
                    "--csv",
                    "--page",
                    "raw",
                    "--log-file",
                    str(log),
                    binary,
                    str(shape[0]),
                    str(shape[1]),
                    "bf16" if dtype == "bfloat16" else "f32",
                    "time",
                ],
                env={
                    **os.environ,
                    **env_extra,
                    "CUDA_VISIBLE_DEVICES": _uuid(gpu),
                },
                check=True,
                capture_output=True,
            )
            per_name = {}
            for name, us in _rows(log.read_text()):
                per_name.setdefault(name, []).append(us)
        # Each kernel name must launch once per call (setup casts aside): the
        # launch counts are recorded so that can be checked.
        result[case_name(dtype, shape)] = {
            "us": sum(statistics.median(v[-10:]) for v in per_name.values()),
            "kernels": sorted(per_name),
            "launches": {k: len(v) for k, v in sorted(per_name.items())},
        }
    return result


def run(gpu, out, binary=None, modes=()):
    record = {"gpu": DEVICES[gpu], "triton": triton(gpu)}
    for mode in modes:
        extra = dict(m.split("=", 1) for m in mode.split(",") if "=" in m)
        record[f"burn {mode}"] = burn(binary, gpu, mode, extra)
    Path(out).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("gpu", choices=DEVICES)
    r = sub.add_parser("run")
    r.add_argument("--gpu", choices=DEVICES, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--burn")
    r.add_argument("--burn-mode", action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "measure":
        measure(args.gpu)
    else:
        run(args.gpu, args.out, args.burn, args.burn_mode)


if __name__ == "__main__":
    sys.exit(main())
