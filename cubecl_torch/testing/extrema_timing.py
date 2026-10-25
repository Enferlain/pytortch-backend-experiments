"""Device time of max/min reductions under CubeCL and Inductor+Triton.

One process measures one backend: the generated kernels' summed device time
per call (PyTorch profiler), the mean over one session of ``CALLS``
back-to-back calls, per case. Run the
backends alternately, several rounds, on the non-display GPU, and compare only
within one capture (bd memory measurement-environment)::

    python -m cubecl_torch.testing.extrema_timing run --rounds 3 \\
        --out cubecl_torch/testing/results/extrema-timing.json

``run`` drives both interpreters; ``measure`` is the per-process worker.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = {
    "cubecl": str(REPO / "stage0/.venv/bin/python"),
    "triton": "/home/imi/Projects/sd-scripts/.venv-wsl/bin/python",
}
SHAPES = ((37, 129), (300, 4096), (8, 65539), (2, 1048579))
DTYPES = ("float32", "bfloat16")
OPS = ("amax", "amin")
CALLS = 100
DEVICES = {"3080": "NVIDIA GeForce RTX 3080", "3090": "NVIDIA GeForce RTX 3090"}


def measure(backend, gpu):
    if backend == "cubecl":
        import cubecl_torch.inductor as ci

        ci.install()
        compile_backend = ci.compile_backend
    else:
        compile_backend = "inductor"
    import torch
    from torch.profiler import ProfilerActivity, profile

    device = next(
        i
        for i in range(torch.cuda.device_count())
        if torch.cuda.get_device_name(i) == DEVICES[gpu]
    )
    torch.cuda.set_device(device)

    spin = spin_for(torch)
    spin(3.0)
    results = {}
    for op in OPS:
        for dtype in DTYPES:
            for shape in SHAPES:
                torch._dynamo.reset()
                x = torch.randn(shape, device="cuda", dtype=getattr(torch, dtype))
                f = torch.compile(getattr(torch, op), backend=compile_backend)
                for _ in range(20):  # compile, first-use tuning, warm-up
                    f(x, -1)
                spin(0.5)
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(CALLS):
                        f(x, -1)
                    torch.cuda.synchronize()
                total = sum(
                    e.device_time
                    for e in prof.events()
                    if e.device_type.name == "CUDA" and e.device_time > 0
                )
                results[f"{op}-{dtype}-{shape[0]}x{shape[1]}"] = total / CALLS
    return {"backend": backend, "device": DEVICES[gpu], "us": results}


def spin_for(torch):
    """A function holding the GPU clocks up for some seconds: the 3080 idles at
    210 MHz between processes."""

    def spin(seconds):
        a = torch.randn(2048, 2048, device="cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            a = a @ a
            a = a / a.norm()
        torch.cuda.synchronize()

    return spin


def clocks():
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,clocks.sm,clocks.mem,utilization.gpu,display_active",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip().splitlines()


def run(rounds, out, gpu, module="cubecl_torch.testing.extrema_timing"):
    """Alternate the backends over ``rounds`` runs of ``module``'s ``measure``."""
    captures = []
    for r in range(rounds):
        for backend in ("cubecl", "triton") if r % 2 == 0 else ("triton", "cubecl"):
            before = clocks()
            proc = subprocess.run(
                [
                    PYTHON[backend],
                    "-m",
                    module,
                    "measure",
                    backend,
                    gpu,
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
        json.dumps({"captures": captures, "summary": summary}, indent=2)
    )
    return summary


def margin(*samples):
    return max((max(x) - min(x)) / statistics.median(x) for x in samples)


def summarize(captures):
    """Median per backend; slower only beyond both backends' relative spread."""
    summary = {}
    for case in captures[0]["us"]:
        c = [x["us"][case] for x in captures if x["backend"] == "cubecl"]
        t = [x["us"][case] for x in captures if x["backend"] == "triton"]
        summary[case] = {
            "cubecl_us": [round(v, 2) for v in c],
            "triton_us": [round(v, 2) for v in t],
            "cubecl_median": round(statistics.median(c), 2),
            "triton_median": round(statistics.median(t), 2),
            "slower": statistics.median(c) > statistics.median(t) * (1 + margin(c, t)),
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("backend", choices=sorted(PYTHON))
    m.add_argument("gpu", choices=sorted(DEVICES))
    r = sub.add_parser("run")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--out", required=True)
    r.add_argument("--gpu", choices=sorted(DEVICES), default="3080")
    args = parser.parse_args(argv)
    if args.command == "measure":
        print(json.dumps(measure(args.backend, args.gpu)))
        return 0
    for case, row in run(args.rounds, args.out, args.gpu).items():
        print(case, row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
