"""Isolate long-row accumulation order without changing acceptance tolerances."""

import json
from pathlib import Path
from stage1 import evidence

evidence.guard()
import torch
from stage5.artifact import CompiledKernel, compiler_identity
from stage5.plan import catalog
from stage5.fixtures import captured

ROOT = Path(__file__).resolve().parent
semantic = captured("row_sum", "float32")
identity = compiler_identity(0)
torch.manual_seed(6142)
x = torch.randn(8, 65539, device="cuda")
reference = x.sum(-1)
exact = x.double().sum(-1)
rows = []
for p in catalog():
    if (
        "pointwise" in p.strategy
        or p.vector_width > 1
        or p.strategy == "coalesced_columns"
        or p.strategy == "independent_planes_final9"
    ):
        continue
    k = CompiledKernel(semantic, 0, p.to_dict(), identity)
    y = torch.empty(8, device="cuda")
    k((65539, 8), x, y)
    torch.cuda.synchronize()
    rows.append(
        {
            "plan": p.to_dict(),
            "output": y.tolist(),
            "reference": reference.tolist(),
            "fp64_reference": exact.tolist(),
            "abs_error_fp64": (y.double() - exact).abs().tolist(),
            "ref_abs_error_fp64": (reference.double() - exact).abs().tolist(),
            "private_iterations": (65539 + p.workgroup[0] - 1) // p.workgroup[0],
        }
    )
    print(p.strategy, p.workgroup, (y - reference).abs().max().item(), flush=True)
(ROOT / "artifacts/reduction-accuracy.json").write_text(json.dumps(rows, indent=2))
