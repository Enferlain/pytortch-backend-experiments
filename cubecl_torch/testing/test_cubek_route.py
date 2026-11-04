"""Bridged sums through the production route (torch-runtime-bridge 4.4).

Each bridged plan is pinned on the classes the band admits it for, at the
shapes of the band, and run through ``torch.compile`` with the runtime guard
installed: every output is within the band's per-row bound of a float64 sum.
The pin also lifts the speed hold-out (task 4.5), so Burn's grid, held out of
selection, is still checked through the route.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

from cubecl_torch.compiler.plan import bridge_catalog
from cubecl_torch.testing.fixtures import REPO

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SHAPES = [(2048, 8192), (2048, 4099), (300, 4096), (7, 2049), (64, 31)]

ROUTE = """
import json
import sys
from cubecl_torch.runtime import guard
guard.install()
import cubecl_torch.inductor as ci
ci.install()
import torch
from cubecl_torch.compiler.plan import PhysicalCandidate
from cubecl_torch.inductor import legality, tuning
from cubecl_torch.testing.cubek_band import pin
from cubecl_torch.testing.welford_band import plan_label

plan = PhysicalCandidate.from_dict(json.loads(sys.argv[1]))
pin(plan.routine, plan.parallel_output, plan.planes, plan.mapping)
results = []
for rows, cols in json.loads(sys.argv[2]):
    for dtype in ("float32", "float16", "bfloat16"):
        # A float16 sum of mean 1000 overflows (to inf, as Triton's): offset data
        # stays finite at mean 4.
        for offset in (0.0, 4.0 if dtype == "float16" else 1000.0):
            torch._dynamo.reset()
            tuning.CACHE.clear()
            x = torch.randn(rows, cols, generator=torch.Generator().manual_seed(rows + cols))
            x = (x + offset).to(getattr(torch, dtype)).cuda()
            f = torch.compile(lambda t: t.sum(-1), backend=ci.compile_backend, dynamic=False)
            y = f(x)
            torch.cuda.synchronize()
            (kernel,) = [k for k in tuning.CACHE.values() if k.semantic["schema"] == 3]
            selected = sorted({plan_label(p) for p in kernel.selections.values()})
            exact = x.double().sum(1)
            bound = 64 * torch.finfo(torch.float32).eps * x.double().abs().sum(1)
            bound += exact.abs() * torch.finfo(y.dtype).eps
            results.append({
                "case": [rows, cols, dtype, offset],
                "selected": selected,
                "admitted": legality.cubek_numerical_reason(
                    plan, kernel.semantic, ()
                ) is None,
                "held_out": legality.cubek_speed_reason(plan) is not None,
                "within": bool(((y.double() - exact).abs() <= bound).all()),
            })
print("RESULTS", json.dumps(results))
"""


def admitted(plan, dtype):
    """The classes the band admitted ``plan`` for (``cubek_numerical_reason``),
    restated so the test checks the rule as well as the route."""
    return dtype != "float32" or bool(plan.mapping)


@pytest.mark.parametrize("plan", bridge_catalog(), ids=lambda p: p.identity[:80])
def test_admitted_classes_through_the_route(tmp_path, plan):
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            ROUTE,
            json.dumps(plan.to_dict()),
            json.dumps(SHAPES),
        ],
        check=False,
        cwd=REPO,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO),
            "CUBECL_TORCH_CACHE_DIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(x for x in run.stdout.splitlines() if x.startswith("RESULTS "))
    from cubecl_torch.testing.welford_band import plan_label

    checked = 0
    for result in json.loads(line.removeprefix("RESULTS ")):
        _, cols, dtype, _ = result["case"]
        if plan.mapping and cols <= 1024:
            continue  # structurally refused: Inductor's persistent rows
        assert result["admitted"] == admitted(plan, dtype), result
        assert result["held_out"] == (not plan.mapping), result
        if not result["admitted"]:
            continue
        assert result["selected"] == [plan_label(plan)], result
        assert result["within"], result
        checked += 1
    assert checked
