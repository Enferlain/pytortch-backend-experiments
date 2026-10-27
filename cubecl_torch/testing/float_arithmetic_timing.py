"""Device time of representative basic-float kernels under CubeCL and Inductor+Triton.

The float-arithmetic timing check: an Adam-style step (lerp, addcmul, addcdiv;
Triton fuses it into one kernel with three outputs, this backend emits three,
bead 4z8.21) and each of its updates alone, an addcdiv with a value (fma +
div_rn), an addcdiv with value 1 (div_rn here, div.full in Triton: see the
design's Risks) and an abs/relu pointwise kernel.
Same method as ``extrema_timing``: one process measures one backend, the
generated kernels' summed device time per call (PyTorch profiler), the mean over
``CALLS`` back-to-back calls; backends alternate over rounds, and a case is
slower only beyond both backends' relative spread::

    python -m cubecl_torch.testing.float_arithmetic_timing run --rounds 3 --gpu 3080 \\
        --out cubecl_torch/testing/results/float-arithmetic-timing-3080.json
"""

import argparse
import sys

from cubecl_torch.testing import extrema_timing
from cubecl_torch.testing.extrema_timing import CALLS, DEVICES, PYTHON

SIZES = {"large": 1 << 22, "small": 1 << 12}


BETA1, BETA2, STEP_SIZE = 0.9, 0.999, 1e-3


def graphs(torch):
    def adam(n):
        beta1, beta2, step_size = BETA1, BETA2, STEP_SIZE

        def fn(param, grad, exp_avg, exp_avg_sq, denom):
            exp_avg = exp_avg.lerp(grad, 1 - beta1)
            exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(grad, grad, value=1 - beta2)
            param = param.addcdiv(exp_avg, denom, value=-step_size)
            return param, exp_avg, exp_avg_sq

        state = [torch.randn(n, device="cuda") for _ in range(4)]
        return fn, (*state, torch.rand(n, device="cuda") + 1e-3)

    def update(name):
        def make(n):
            a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            denom = torch.rand(n, device="cuda") + 1e-3
            fns = {
                "lerp": (lambda e, g: e.lerp(g, 1 - BETA1), (a, b)),
                "addcmul": (
                    lambda s, g: s.mul(BETA2).addcmul(g, g, value=1 - BETA2),
                    (a, b),
                ),
                "addcdiv": (
                    lambda p, e, d: p.addcdiv(e, d, value=-STEP_SIZE),
                    (a, b, denom),
                ),
            }
            return fns[name]

        return make

    def addcdiv(value):
        def make(n):
            s, a = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            b = torch.rand(n, device="cuda") + 0.5
            return lambda s, a, b: torch.addcdiv(s, a, b, value=value), (s, a, b)

        return make

    def abs_relu(n):
        x = torch.randn(n, device="cuda")
        return lambda x: torch.relu(torch.abs(x) - 0.5), (x,)

    return {
        "adam": adam,
        "adam_lerp": update("lerp"),
        "adam_addcmul": update("addcmul"),
        "adam_addcdiv": update("addcdiv"),
        "addcdiv": addcdiv(0.3),
        "addcdiv_value1": addcdiv(1),
        "abs_relu": abs_relu,
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
        module="cubecl_torch.testing.float_arithmetic_timing",
    )
    for case, row in summary.items():
        print(case, row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
