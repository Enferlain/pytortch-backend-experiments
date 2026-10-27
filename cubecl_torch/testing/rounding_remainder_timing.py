"""Device time of extrema-family kernels under CubeCL and Inductor+Triton.

The float-extrema-rounding-remainder timing check (x3c):
- fmod and remainder with small (|a/b| up to ~100) and large (up to ~1e10)
  quotients: CubeCL's remainder is exact CUDA fmod, a loop for large quotients;
  Triton's is an inexact frem expansion;
- float floor division at the small scale only (its div_rn + floor does the
  same work at any scale);
- floor and round in float32 and bfloat16;
- the sign and exponent ops copysign, nextafter, ldexp and frexp (frexp's
  exponent widened to int64, int32 not being a storage dtype here).
Same method as ``extrema_timing``: one process measures one backend, the
generated kernels' summed device time per call (PyTorch profiler), the mean over
``CALLS`` back-to-back calls; backends alternate over rounds, and a case is
slower only beyond both backends' relative spread::

    python -m cubecl_torch.testing.rounding_remainder_timing run --rounds 3 --gpu 3080 \\
        --out cubecl_torch/testing/results/rounding-remainder-timing-3080.json
"""

import argparse
import sys

from cubecl_torch.testing import extrema_timing
from cubecl_torch.testing.extrema_timing import CALLS, DEVICES, PYTHON

SIZES = {"large": 1 << 22, "small": 1 << 12}


def graphs(torch):
    def binary(op, scale):
        """``op`` over quotients up to about ``scale``."""

        def make(n):
            a = torch.randn(n, device="cuda") * scale
            b = torch.rand(n, device="cuda") * 3 + 0.1
            return op, (a, b)

        return make

    def unary(op, dtype=torch.float32):
        def make(n):
            x = (torch.randn(n, device="cuda") * 100).to(dtype)
            return op, (x,)

        return make

    def pair(op):
        def make(n):
            return op, (torch.randn(n, device="cuda"), torch.randn(n, device="cuda"))

        return make

    def ldexp(n):
        x = torch.randn(n, device="cuda")
        e = torch.randint(-40, 40, (n,), device="cuda")
        return torch.ldexp, (x, e)

    def frexp(n):
        def fn(x):
            m, e = torch.frexp(x)
            return m, e.to(torch.int64)

        return fn, (torch.randn(n, device="cuda") * 1e3,)

    return {
        "fmod_small_q": binary(torch.fmod, 10.0),
        "fmod_large_q": binary(torch.fmod, 1e9),
        "remainder_small_q": binary(torch.remainder, 10.0),
        "remainder_large_q": binary(torch.remainder, 1e9),
        "floordiv": binary(lambda a, b: a // b, 10.0),
        "floor_f32": unary(torch.floor),
        "round_f32": unary(torch.round),
        "floor_bf16": unary(torch.floor, torch.bfloat16),
        "round_bf16": unary(torch.round, torch.bfloat16),
        "copysign": pair(torch.copysign),
        "nextafter": pair(torch.nextafter),
        "ldexp": ldexp,
        "frexp": frexp,
    }


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
        args.rounds,
        args.out,
        args.gpu,
        module="cubecl_torch.testing.rounding_remainder_timing",
    )
    for case, row in summary.items():
        print(case, row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
