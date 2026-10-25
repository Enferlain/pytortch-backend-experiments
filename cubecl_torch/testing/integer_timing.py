"""Device time of representative integer kernels under CubeCL and Inductor+Triton.

The integer-value-arithmetic timing check: a padding mask (``ops.and_``), an
int64 floor division, a shift/xor mix and an int8 wrap. Same method as
``extrema_timing``: one process measures one backend, the generated kernels'
summed device time per call (PyTorch profiler), the mean over ``CALLS``
back-to-back calls; backends alternate over rounds, and a case is slower only
beyond both backends' relative spread::

    python -m cubecl_torch.testing.integer_timing run --rounds 3 --gpu 3080 \\
        --out cubecl_torch/testing/results/integer-timing-3080.json
"""

import argparse
import sys

from cubecl_torch.testing import extrema_timing
from cubecl_torch.testing.extrema_timing import CALLS, DEVICES, PYTHON

SIZES = {"large": 1 << 22, "small": 1 << 12}


def graphs(torch):
    import torch.nn.functional as F

    def pad(n):
        rows = max(n // 1024, 1)
        return lambda x: F.pad(x, (1, 2, 2, 1), value=0.5), (
            torch.randn(rows, 1024, device="cuda"),
        )

    def floordiv(n):
        a = torch.randint(-(1 << 40), 1 << 40, (n,), device="cuda")
        b = torch.randint(1, 1000, (n,), device="cuda") * (
            torch.randint(0, 2, (n,), device="cuda") * 2 - 1
        )
        return lambda a, b: a // b, (a, b)

    def shifts(n):
        a = torch.randint(-(1 << 30), 1 << 30, (n,), device="cuda")
        b = torch.randint(0, 40, (n,), device="cuda")

        def fn(a, b):
            x, s = a.to(torch.int32), b.to(torch.int32)
            return ((x << s) ^ (x >> 3) ^ ~x).to(torch.int64)

        return fn, (a, b)

    def int8_wrap(n):
        x = torch.randn(n, device="cuda") * 100
        return lambda x: ((x.to(torch.int8) + 100) * 3).to(torch.float32), (x,)

    return {"pad": pad, "floordiv": floordiv, "shifts": shifts, "int8_wrap": int8_wrap}


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
    spin = extrema_timing.spin_for(torch)
    spin(3.0)
    results = {}
    for name, make in graphs(torch).items():
        for size, n in SIZES.items():
            torch._dynamo.reset()
            torch.manual_seed(0)
            fn, args = make(n)
            f = torch.compile(fn, backend=compile_backend)
            for _ in range(20):  # compile, first-use tuning, warm-up
                f(*args)
            spin(0.5)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(CALLS):
                    f(*args)
                torch.cuda.synchronize()
            total = sum(
                e.device_time
                for e in prof.events()
                if e.device_type.name == "CUDA" and e.device_time > 0
            )
            results[f"{name}-{size}"] = total / CALLS
    return {"backend": backend, "device": DEVICES[gpu], "us": results}


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
        import json

        print(json.dumps(measure(args.backend, args.gpu)))
        return 0
    summary = extrema_timing.run(
        args.rounds, args.out, args.gpu, module="cubecl_torch.testing.integer_timing"
    )
    for case, row in summary.items():
        print(case, row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
