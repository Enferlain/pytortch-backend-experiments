"""Warm host time of a bridged launch, against Triton's raw launch with the same
arguments, in one process (spec: runtime-bridge, launch host cost).

Run in an environment with Triton, from the repository root:

    PYTHONPATH=. /path/to/python -m cubecl_torch.testing.launch_cost \
        --device cuda:1 --out cubecl_torch/testing/results/launch-cost.json

Both kernels compute ``y = x + 1`` over ``n`` float32 values. Each sample is
the mean host time of ``--launches`` back-to-back launches, with the stream
drained between samples, and the two kernels' samples interleaved; the result
is the median of ``--samples`` samples, with the spread of the middle half as
noise. ``bridge`` launches as PyTorch's ops do, without recording its
tensors on the stream; ``bridge_recorded`` records them (the bridge's opt-in).
``--threads`` adds the same bridged loop run from that many threads at once,
each on its own stream.

``--instructions`` instead counts user-mode instructions per warm launch of
each kernel with ``perf stat``, over the launch loop only (perf's control
file descriptors), one process per count. They vary between processes as time
does (by about 15%), so each kernel is counted in several processes:

    PYTHONPATH=. /path/to/python -m cubecl_torch.testing.launch_cost \
        --instructions bridge bridge_recorded triton torch --device cuda:1
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

from cubecl_torch.runtime import bridge


def samples(launches_by_name, stream, count, launches):
    """Samples of each launch, interleaved so machine noise falls on all alike."""
    out = {name: [] for name in launches_by_name}
    with torch.cuda.stream(stream):
        for launch in launches_by_name.values():
            for _ in range(launches):  # Warm: compiled, cached, clocks up.
                launch()
        stream.synchronize()
        for _ in range(count):
            for name, launch in launches_by_name.items():
                start = time.perf_counter_ns()
                for _ in range(launches):
                    launch()
                out[name].append((time.perf_counter_ns() - start) / launches / 1000)
                stream.synchronize()
    return out


def summary(values):
    q = statistics.quantiles(values, n=4)
    return {"median_us": statistics.median(values), "iqr_us": [q[0], q[2]]}


def bridged(device, n, record_stream=False):
    x = torch.randn(n, device=device)
    y = torch.empty_like(x)
    return lambda: bridge.test_add_one(x, y, record_stream=record_stream)


def triton_add_one(device, n):
    import triton
    import triton.language as tl

    @triton.jit
    def add_one(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        tl.store(y_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + 1, mask=mask)

    x = torch.randn(n, device=device)
    y = torch.empty_like(x)
    grid = (triton.cdiv(n, 256),)
    return lambda: add_one[grid](x, y, n, BLOCK=256)


def torch_add_one(device, n):
    x = torch.randn(n, device=device)
    y = torch.empty_like(x)
    return lambda: torch.add(x, 1, out=y)


KERNELS = {
    "bridge": bridged,
    "bridge_recorded": lambda device, n: bridged(device, n, record_stream=True),
    "triton": triton_add_one,
    "torch": torch_add_one,
}


def count_loop(kernel, device, n, launches):
    """The warm launch loop of ``--count``: perf counts only between the
    enable and disable written to its control descriptor."""
    control, ack = (int(fd) for fd in os.environ["LAUNCH_COST_PERF_FDS"].split(","))

    def perf(command):
        os.write(control, command + b"\n")
        os.read(ack, 5)

    launch = KERNELS[kernel](device, n)
    for _ in range(launches // 10):
        launch()
    torch.cuda.synchronize(device)
    perf(b"enable")
    for i in range(launches):
        launch()
        if i % 2000 == 0:
            torch.cuda.synchronize(device)
    torch.cuda.synchronize(device)
    perf(b"disable")


def instructions(kernel, device="cuda", n=1024, launches=30000, python=None):
    """User-mode instructions per warm launch of ``kernel``, counted by
    ``perf stat`` in a new process over its launch loop only."""
    control_r, control_w = os.pipe()
    ack_r, ack_w = os.pipe()
    fds = (control_r, control_w, ack_r, ack_w)
    try:
        out = subprocess.run(
            [
                "perf",
                "stat",
                "-x,",
                "-e",
                "instructions:u",
                "--delay=-1",
                f"--control=fd:{control_r},{ack_w}",
                python or sys.executable,
                "-m",
                "cubecl_torch.testing.launch_cost",
                "--count",
                kernel,
                "--device",
                str(device),
                "--n",
                str(n),
                "--launches",
                str(launches),
            ],
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "LAUNCH_COST_PERF_FDS": f"{control_w},{ack_r}"},
            pass_fds=fds,
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        for fd in fds:
            os.close(fd)
    line = next(l for l in out.stderr.splitlines() if "instructions" in l)
    return int(line.split(",")[0]) / launches


def threaded(device, n, threads, count, launches):
    results = [None] * threads
    start = threading.Barrier(threads)

    def work(k):
        stream = torch.cuda.Stream(device)
        launch = bridged(device, n)
        start.wait()
        results[k] = samples({"bridge": launch}, stream, count, launches)["bridge"]

    pool = [threading.Thread(target=work, args=(k,)) for k in range(threads)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()
    return [value for result in results for value in result]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--launches", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out")
    parser.add_argument("--instructions", nargs="+", choices=sorted(KERNELS))
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--count", choices=sorted(KERNELS), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    if args.count:
        count_loop(args.count, device, args.n, args.launches)
        return
    result = {
        "device": torch.cuda.get_device_name(device),
        "python": sys.version.split()[0],
        "gil_enabled": sys._is_gil_enabled(),
        "torch": torch.__version__,
        "n": args.n,
    }
    if args.instructions:
        result["launches"] = 30000
        result["instructions_per_launch"] = {
            kernel: [
                round(instructions(kernel, device, args.n))
                for _ in range(args.processes)
            ]
            for kernel in args.instructions
        }
    else:
        stream = torch.cuda.Stream(device)
        names = ("bridge", "bridge_recorded", "triton")
        single = samples(
            {name: KERNELS[name](device, args.n) for name in names},
            stream,
            args.samples,
            args.launches,
        )
        result |= {name: summary(single[name]) for name in names}
        result[f"bridge_{args.threads}_threads"] = summary(
            threaded(device, args.n, args.threads, args.samples, args.launches)
        )
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
