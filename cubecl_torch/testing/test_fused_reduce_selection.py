"""The fused ecosystem reduce in first-use selection (fused-reduction-bridge
task 4.1): RMSNorm's weight-gradient kernels are offered CubeK's three fused
routines, the forward and input-gradient row kernels are refused with the
reason, and the compiled gradients match eager float64 (the strict candidate
checks pass on the way). Exactness is task 3.2's test; accuracy against
Triton is the band (group 5)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import json, sys
import cubecl_torch.inductor as ci
ci.install()
import torch
from cubecl_torch.inductor import tuning

seen = []
select = tuning.AutotunedKernel.select
def record(self, key, key_data, plans, rejected, *args):
    winner = select(self, key, key_data, plans, rejected, *args)
    if self.semantic.get("schema") == 3:
        seen.append({
            "outer": self.semantic["outer"].get("value"),
            "extent": self.semantic["reduction_extent"].get("value"),
            "fused": sorted(p.routine for p in plans if p.strategy == "cubek_fused_reduce"),
            "refused": sorted({r["reason"] for r in rejected
                               if r["plan"]["strategy"] == "cubek_fused_reduce"}),
        })
    return winner
tuning.AutotunedKernel.select = record

rows, cols, dtype = json.loads(sys.argv[1])
dtype = getattr(torch, dtype)
def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
g = torch.Generator(device="cuda").manual_seed(0)
x = torch.randn(rows, cols, device="cuda", dtype=dtype, generator=g).requires_grad_()
w = torch.randn(cols, device="cuda", dtype=dtype, generator=g).requires_grad_()
torch.compile(rms, backend=ci.compile_backend, dynamic=False)(x, w).sum().backward()
xr = x.detach().double().requires_grad_()
wr = w.detach().double().requires_grad_()
rms(xr, wr).sum().backward()
torch.cuda.synchronize()
error = {name: ((a.grad.double() - b.grad).abs().max() / b.grad.abs().max()).item()
         for name, a, b in (("x", x, xr), ("w", w, wr))}
print(json.dumps({"kernels": seen, "error": error}))
"""


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("rows, cols", [(64, 1024), (8192, 1280)])
def test_rmsnorm_backward_selection(tmp_path, rows, cols, dtype):
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT, json.dumps([rows, cols, dtype])],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO),
            "CUBECL_TORCH_CACHE_DIR": str(tmp_path / "cache"),
            "CUBECL_TORCH_STRICT_CANDIDATES": "1",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    row = [k for k in result["kernels"] if (k["outer"], k["extent"]) == (rows, cols)]
    column = [k for k in result["kernels"] if k not in row]
    # Forward and input gradient: a per-element pass over the row.
    assert len(row) == 2 and all(
        not k["fused"] and any("reduction-scope store" in r for r in k["refused"])
        for k in row
    ), row
    # The weight gradient (one kernel, or both split stages).
    assert column and all(k["fused"] == ["cube_unit", "plane", "unit"] for k in column)
    limit = {"float32": 1e-5, "bfloat16": 2e-2}[dtype]
    assert max(result["error"].values()) < limit, result["error"]
