"""Load layouts for the fused ecosystem reduce (fused-reduction-bridge task 2.1,
design 5): Inductor's block analysis per load, and the layout the full-size
loads of the reduction operand agree on."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cubecl_torch.inductor.operand_layout import normalized, operand_layout

REPO = Path(__file__).resolve().parents[2]


def layout(outer, outer_strides, reduction, reduction_strides, offset=0):
    return {
        "outer": {"dims": outer, "strides": outer_strides},
        "reduction": {"dims": reduction, "strides": reduction_strides},
        "offset": offset,
    }


def test_normalized_drops_unit_dims_and_merges_contiguous_ones():
    assert normalized([256, 1, 2], [256, 0, 128]) == ([512], [128])
    assert normalized([32, 1280], [327680, 1]) == ([32, 1280], [327680, 1])
    assert normalized([4, 8], [8, 1]) == ([32], [1])


def test_broadcast_loads_do_not_compete():
    g = layout([32, 1280], [327680, 1], [256], [1280])
    rstd = layout([32, 1280], [256, 0], [256], [1])
    agreed, reason = operand_layout([g, g, rstd])
    assert reason is None
    assert agreed == {
        "outer": {"dims": [32, 1280], "strides": [327680, 1]},
        "reduction": {"dims": [256], "strides": [1280]},
    }


def test_offsets_are_not_layout():
    a = layout([64], [1024], [1024], [1], offset=0)
    b = layout([64], [1024], [1024], [1], offset=7)
    assert operand_layout([a, b])[1] is None


def test_disagreeing_full_size_loads_are_refused_with_both_layouts():
    a = layout([256], [256], [256], [1])
    b = layout([256], [1], [256], [256])
    agreed, reason = operand_layout([a, b])
    assert agreed is None
    assert reason.startswith("loads disagree on layout")
    assert "[256]" in reason and "[1]" in reason


def test_a_load_without_layout_refuses():
    assert operand_layout([layout([64], [1], [8], [64]), None]) == (
        None,
        "a load of the reduction operand has no block layout",
    )


def test_a_reduction_over_several_memory_dims_refuses():
    a = layout([8], [1], [4, 16], [1000, 8])
    assert operand_layout([a]) == (
        None,
        "the reduction spans several memory dimensions",
    )


# Compiles in its own process: the Inductor overlay must be installed before
# anything imports Inductor, which other tests in a worker already did.
RECORD = """
import json, sys
import cubecl_torch.inductor as ci
ci.install()
import torch
from cubecl_torch.inductor import tuning

seen = []
init = tuning.AutotunedKernel.__init__
def record(self, semantic, *args, **kwargs):
    init(self, semantic, *args, **kwargs)
    if semantic.get("schema") == 3:
        loads = [op for op in semantic["ops"] if op["op"] == "load"]
        seen.append([semantic["outer"].get("value"),
                     semantic["reduction_extent"].get("value"),
                     [self.layouts.get(op["id"]) for op in loads],
                     semantic, self.layouts])
tuning.AutotunedKernel.__init__ = record

def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

case, rows, cols, dtype = json.loads(sys.argv[1])
dtype = getattr(torch, dtype)
compiled = lambda fn: torch.compile(fn, backend=ci.compile_backend, dynamic=False)
if case == "rms_backward":
    x = torch.randn(rows, cols, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(cols, device="cuda", dtype=dtype, requires_grad=True)
    compiled(rms)(x, w).sum().backward()
elif case == "transposed":
    a = torch.randn(rows, cols, device="cuda", dtype=dtype)
    b = torch.randn(cols, rows, device="cuda", dtype=dtype)
    compiled(lambda a, b: (a * b.t()).sum(-1))(a, b)
elif case == "rolled":
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    compiled(lambda x: x.roll(1, -1).mul(x).sum(-1))(x)
torch.cuda.synchronize()
print(json.dumps(seen))
"""


@pytest.fixture
def recorded(tmp_path):
    """Compile a named case and return, per reduction kernel keyed by (outer,
    extent), the layouts of its loads in op order."""

    def run(case, rows, cols, dtype=torch.float32, full=False):
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                RECORD,
                json.dumps([case, rows, cols, str(dtype).removeprefix("torch.")]),
            ],
            cwd=REPO,
            env={
                **os.environ,
                "PYTHONPATH": str(REPO),
                "CUBECL_TORCH_CACHE_DIR": str(tmp_path / "cache"),
            },
            capture_output=True,
            text=True,
            check=True,
        )
        seen = json.loads(out.stdout.strip().splitlines()[-1])
        if full:
            return [(o, e, semantic, layouts) for o, e, _, semantic, layouts in seen]
        return {(o, e): loads for o, e, loads, _, _ in seen}

    return run


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_split_weight_gradient(recorded, dtype):
    rows, cols = 8192, 1280
    kernels = recorded("rms_backward", rows, cols, dtype)
    # Inductor picks the split count from the GPU (32 on the RTX 3080).
    (splits,) = [
        o // cols
        for o, e in kernels
        if o > cols and o % cols == 0 and o // cols * e == rows
    ]
    first = kernels[(splits * cols, rows // splits)]
    agreed, reason = operand_layout(first)
    assert reason is None
    assert agreed == {
        "outer": {"dims": [splits, cols], "strides": [rows // splits * cols, 1]},
        "reduction": {"dims": [rows // splits], "strides": [cols]},
    }
    # rstd[r] is read as a broadcast over the columns.
    assert first[2]["outer"]["strides"] == [rows // splits, 0]
    # The second stage sums each column's partials.
    assert operand_layout(kernels[(cols, splits)])[0] == {
        "outer": {"dims": [cols], "strides": [1]},
        "reduction": {"dims": [splits], "strides": [cols]},
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unsplit_weight_gradient(recorded, dtype):
    kernels = recorded("rms_backward", 64, 1024, dtype)
    assert operand_layout(kernels[(1024, 64)]) == (
        {
            "outer": {"dims": [1024], "strides": [1]},
            "reduction": {"dims": [64], "strides": [1024]},
        },
        None,
    )


def test_transposed_product_disagrees(recorded):
    kernels = recorded("transposed", 256, 256)
    reasons = [operand_layout(loads)[1] or "" for loads in kernels.values()]
    assert any(r.startswith("loads disagree on layout") for r in reasons)


def test_rolled_index_has_no_layout(recorded):
    # roll reads ModularIndexing(r0 + 1, 1, cols): no block pattern matches it.
    kernels = recorded("rolled", 64, 512)
    assert kernels
    assert all(
        operand_layout(loads)[1]
        == "a load of the reduction operand has no block layout"
        for loads in kernels.values()
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows, cols", [(64, 1024), (8192, 1280)])
def test_rmsnorm_coverage(recorded, rows, cols, dtype):
    """The weight gradient's kernels are covered; the forward and input-gradient
    kernels (a per-element pass over the row) are refused with that reason."""
    from cubecl_torch.inductor.legality import fused_reduce_reason

    covered, refused = [], []
    for outer, extent, semantic, layouts in recorded(
        "rms_backward", rows, cols, dtype, full=True
    ):
        reason = fused_reduce_reason(semantic, layouts)
        (covered if reason is None else refused).append((outer, extent, reason))
    row_kernels = [k for k in refused if (k[0], k[1]) == (rows, cols)]
    assert len(row_kernels) == 2, refused  # forward and input gradient
    assert all("reduction-scope store" in r for _, _, r in row_kernels)
    # Every column kernel of the weight gradient (one, or both split stages).
    assert covered and all(o % cols == 0 and o != rows for o, _, _ in covered), covered
    assert refused == row_kernels, refused


def test_disagreeing_and_unmatched_layouts_refuse(recorded):
    from cubecl_torch.inductor.legality import fused_reduce_reason

    reasons = [
        fused_reduce_reason(s, layouts)
        for _, _, s, layouts in recorded("transposed", 256, 256, full=True)
    ]
    assert any("loads disagree on layout" in (r or "") for r in reasons), reasons
    reasons = [
        fused_reduce_reason(s, layouts)
        for _, _, s, layouts in recorded("rolled", 64, 512, full=True)
    ]
    assert reasons and all("no block layout" in (r or "") for r in reasons), reasons
