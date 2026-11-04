"""First-call cost of a sum with the bridged plans offered against today's
(torch-runtime-bridge 4.5): wall time of the first ``torch.compile`` call of
``x.sum(-1)``, compile and first-use tuning included, each in a fresh process
with empty caches. Modes: ``today`` (bridged plans withdrawn), ``bridged``
(all offered), ``mapped`` (only the mapped cube), ``held_out`` (the plans
measured slower than Triton withdrawn: unit everywhere, plane on 2048x4099
and 7x2049) and ``triton`` (Inductor's own backend, in the Triton venv). From
the repository root::

    stage0/.venv/bin/python -m cubecl_torch.testing.cubek_first_call run --gpu 3080 \\
        --rounds 5 --out cubecl_torch/testing/results/cubek-first-call-3080.json

``once`` is the per-process worker.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from cubecl_torch.testing.cubek_timing import (
    PYTHON,
    REPO,
    _device,
    case_name,
    cases,
)

MODES = ("today", "bridged", "mapped", "held_out", "triton")
SLOWER_PLANE = {(2048, 4099), (7, 2049)}


def _offered(mode, plan, shape):
    """Whether ``mode`` offers the bridged ``plan`` for a sum of ``shape``."""
    if mode == "today":
        return False
    if mode == "mapped":
        return bool(plan.mapping)
    if mode == "held_out":
        if plan.routine == "unit":
            return False
        return not (plan.routine == "plane" and shape in SLOWER_PLANE)
    return True


def once(mode, gpu):
    if mode == "triton":
        import torch

        backend = "inductor"
    else:
        from cubecl_torch.runtime import guard

        guard.install()
        import cubecl_torch.inductor as ci

        ci.install()
        import torch

        from cubecl_torch.compiler.plan import CUBEK_REDUCE
        from cubecl_torch.inductor import legality, tuning

        # The modes choose the bridged plans themselves: Burn's grid, held out
        # for speed since this measurement, is offered where a mode offers it.
        legality.cubek_speed_reason = lambda plan: None

        backend = ci.compile_backend
        candidates = tuning.candidates
        current = {}

        def offered(*args, **kwargs):
            accepted, rejected = candidates(*args, **kwargs)
            return [
                p
                for p in accepted
                if p.strategy != CUBEK_REDUCE or _offered(mode, p, current["shape"])
            ], rejected

        tuning.candidates = offered

    _device(torch, gpu)
    # CUDA, the allocator and Dynamo warmed on another graph, so the first call
    # below pays for this sum's compile and tuning only.
    w = torch.ones(8, device="cuda")
    torch.compile(lambda t: t * 2, backend=backend)(w)
    torch.cuda.synchronize()
    result = {}
    for dtype, shape in cases():
        torch._dynamo.reset()
        if mode != "triton":
            current["shape"] = shape
        x = torch.randn(shape, device="cuda", dtype=getattr(torch, dtype))
        f = torch.compile(lambda t: t.sum(-1), backend=backend)
        start = time.perf_counter()
        f(x)
        torch.cuda.synchronize()
        result[case_name(dtype, shape)] = (time.perf_counter() - start) * 1000
    print("FIRST", json.dumps(result))


def run(gpu, rounds, out, modes=MODES):
    samples = {}
    for _ in range(rounds):
        for mode in modes:
            with tempfile.TemporaryDirectory() as cache:
                env = {
                    **os.environ,
                    "PYTHONPATH": str(REPO),
                    "CUBECL_TORCH_CACHE_DIR": cache,
                    "TORCHINDUCTOR_CACHE_DIR": f"{cache}/inductor",
                    "TRITON_CACHE_DIR": f"{cache}/triton",
                }
                if mode == "triton":
                    env["PYTHON_GIL"] = "0"
                found = subprocess.run(
                    [
                        PYTHON["cubecl" if mode != "triton" else "triton"],
                        "-m",
                        "cubecl_torch.testing.cubek_first_call",
                        "once",
                        mode,
                        gpu,
                    ],
                    cwd=REPO,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            line = next(x for x in found.stdout.splitlines() if x.startswith("FIRST "))
            for case, ms in json.loads(line[6:]).items():
                samples.setdefault(case, {}).setdefault(mode, []).append(ms)
    result = {
        case: {
            mode: {"runs": runs, "median": statistics.median(runs)}
            for mode, runs in by_mode.items()
        }
        for case, by_mode in samples.items()
    }
    if out:
        Path(out).write_text(json.dumps(result, indent=1) + "\n")
    for case, by_mode in result.items():
        print(
            case,
            " ".join(f"{m} {v['median']:.0f} ms" for m, v in by_mode.items()),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gpu", choices=("3080", "3090"), default="3080")
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    r.add_argument("--out")
    o = sub.add_parser("once")
    o.add_argument("mode", choices=MODES)
    o.add_argument("gpu", choices=("3080", "3090"))
    options = parser.parse_args(argv)
    if options.command == "run":
        run(options.gpu, options.rounds, options.out, options.modes)
    else:
        once(options.mode, options.gpu)


if __name__ == "__main__":
    sys.exit(main())
