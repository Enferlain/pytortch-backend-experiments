"""Every enabled physical family, exact tails, storage and collective contracts."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from stage1 import evidence

evidence.guard()
import torch
from stage5.artifact import CompiledKernel, compiler_identity
from stage5.plan import catalog
from stage5.fixtures import captured
from stage5.legality import reason

torch.manual_seed(6142)
identity = compiler_identity(0)
rows = []
rejections = []
for dtype in ("float32", "float16", "bfloat16"):
    td = getattr(torch, dtype)
    for family in ("add", "row_sum"):
        semantic = captured(family, dtype)
        for plan in catalog():
            if (family == "add") != ("pointwise" in plan.strategy):
                continue
            if dtype == "float32" and plan.vector_width > 4:
                continue
            k = CompiledKernel(semantic, 0, plan.to_dict(), identity)
            if plan.strategy == "independent_planes":
                assert k.kernel.attributes["static_shared_bytes"] == 0
            checks = []
            for n in (0, 1, 31, 32, 33, 255, 256, 257, 4099, 65539):
                outer = (
                    (1 if plan.vector_width > 1 and n % plan.vector_width else 3)
                    if family == "row_sum"
                    else 1
                )
                x = (
                    (
                        torch.arange(outer * n, device="cuda", dtype=torch.int64) % 17
                        - 8
                    ).to(td)
                    / 8
                ).reshape((outer, n) if family == "row_sum" else (n,))
                if family == "row_sum":
                    x = x.as_strided((outer, n), (n, 1))
                y = torch.empty(
                    outer if family == "row_sum" else n, device="cuda", dtype=td
                )
                if family == "row_sum":
                    # Live row fixture slots are N then B; verify expressions.
                    scalars = (n, outer)
                    expected = x.view(outer, n).sum(-1)
                else:
                    scalars = (n,)
                    expected = x + 1
                why = reason(plan, semantic, scalars, (x, y), k.kernel.resources)
                if plan.vector_width > 1 and why:
                    try:
                        k(scalars, x, y)
                    except ValueError as error:
                        assert why in str(error)
                    else:
                        raise AssertionError("unsupported vector plan accepted")
                    rejections.append(
                        {
                            "dtype": dtype,
                            "family": family,
                            "plan": plan.to_dict(),
                            "n": n,
                            "reason": why,
                        }
                    )
                    continue
                k(scalars, x, y)
                torch.testing.assert_close(
                    y,
                    expected,
                    rtol=0.016
                    if td == torch.bfloat16
                    else 0.002
                    if td == torch.float16
                    else 2e-5,
                    atol=0.008
                    if td == torch.bfloat16
                    else 0.001
                    if td == torch.float16
                    else 2e-5,
                    equal_nan=True,
                )
                checks.append(n)
            rows.append(
                {
                    "family": family,
                    "dtype": dtype,
                    "plan": plan.to_dict(),
                    "tails": checks,
                    "directory": str(k.directory),
                    "attributes": k.kernel.attributes,
                    "resources": k.kernel.resources,
                }
            )
            print(f"{family} {dtype} {plan.identity} PASS", flush=True)
evidence.state("after_stage5_components")
(ROOT / "artifacts/components.json").write_text(
    json.dumps(
        {
            "checks": rows,
            "rejections": rejections,
            "independence": evidence.inventory(),
        },
        indent=2,
    )
)
