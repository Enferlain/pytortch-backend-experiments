"""Kernels with several buffers and outputs (OpenSpec multi-output-kernels)."""

import copy

import pytest
import torch

from cubecl_torch.testing.fixtures import captured

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def chained_sum(inputs):
    """A pointwise descriptor loading ``inputs`` float32 buffers and storing
    their left-to-right sum, built from the captured ``add-float32``."""
    base = captured("add-float32")
    read, write = base["buffers"]
    load = next(op for op in base["ops"] if op["op"] == "load")
    store = next(op for op in base["ops"] if op["op"] == "store")
    ops = [{**copy.deepcopy(load), "id": f"x{i}", "buffer": i} for i in range(inputs)]
    total = "x0"
    for i in range(1, inputs):
        ops.append(
            {
                "args": [total, f"x{i}"],
                "dtype": "float32",
                "id": f"s{i}",
                "op": "add",
                "semantic_dtype": "float32",
            }
        )
        total = f"s{i}"
    ops.append({**copy.deepcopy(store), "args": [total], "buffer": inputs})
    return {
        **base,
        "buffers": [copy.deepcopy(read) for _ in range(inputs)] + [write],
        "ops": ops,
    }


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("multi-output-cache")


def test_norm_training_kernels_fit_the_buffer_limit(cache_dir):
    """LayerNorm and GroupNorm (affine, then SiLU) backward kernels read 6 to 9
    buffers (Triton's, measured 2026-10-27); none is rejected for its count."""
    from cubecl_torch.testing.fixtures import run_python

    result = run_python(
        """
import torch.nn.functional as F
import torch._functorch.config
torch._functorch.config.force_non_lazy_backward_lowering = True
reasons = []
def backend(gm, inputs):
    try:
        return ci.compile_backend(gm, inputs, rejections="all")
    except Exception as e:
        reasons.extend(getattr(e, "reasons", None) or [str(e)])
        return gm.forward
def layer_norm(x, w, b):
    return F.layer_norm(x, x.shape[-1:], w, b, 1e-5).square().sum()
def group_norm(x, w, b):
    return F.silu(F.group_norm(x, 8, w, b, 1e-5)).square().sum()
for fn, shape, channels in ((layer_norm, (64, 96), 96), (group_norm, (2, 32, 8, 8), 32)):
    torch._dynamo.reset()
    args = [torch.randn(shape, device="cuda", requires_grad=True)]
    args += [torch.randn(channels, device="cuda", requires_grad=True) for _ in range(2)]
    torch.compile(fn, backend=backend)(*args).backward()
print(json.dumps(sorted(set(reasons))))
""",
        cache_dir,
    )
    assert not [r for r in result if "inputs" in r or "buffers per kernel" in r], result


FUSION_LIMIT = """
from cubecl_torch.runtime import trace
def fn(*xs):
    a, b = xs[:{a}], xs[{a}:]
    row = sum(a[1:], a[0]).sum(-1, keepdim=True)
    return row + sum(b[1:], b[0])
xs = [torch.randn(33, 70, device="cuda") for _ in range({a} + {b})]
torch._dynamo.reset()
with trace.scoped():
    out = torch.compile(fn, backend=ci.compile_backend)(*xs)
    reasons = [e["reason"] for e in trace.EVENTS
               if e.get("boundary") == "fusion" and e.get("reason")]
    kernels = sum(e.get("event") == "kernel" for e in trace.EVENTS)
print(json.dumps({{"close": torch.allclose(out, fn(*xs), rtol=1e-5, atol=1e-4),
                  "reasons": sorted(set(reasons)), "kernels": kernels}}))
"""


def test_fusion_over_the_buffer_limit_is_rejected_and_still_compiles(cache_dir):
    """A row reduction over ``a`` inputs fused with a pointwise epilogue over
    ``b`` more: 9 + 7 inputs and one output need 17 buffers; 8 + 7 fit in 16,
    a reduction kernel with 15 inputs checked against eager."""
    from cubecl_torch.testing.fixtures import run_python

    over = run_python(FUSION_LIMIT.format(a=9, b=7), cache_dir)
    assert over["close"] and over["kernels"] == 2, over
    assert "more than 16 buffers per kernel unsupported" in over["reasons"], over
    fits = run_python(FUSION_LIMIT.format(a=8, b=7), cache_dir)
    assert fits["close"] and fits["kernels"] == 1, fits
    assert not [r for r in fits["reasons"] if "buffers" in r], fits


def test_sixteen_buffer_pointwise_kernel_equals_eager():
    from cubecl_torch.inductor import tuning

    kernel = tuning.compile_kernel(chained_sum(15), 0)
    n = 4099
    xs = [torch.randn(n, device="cuda") for _ in range(15)]
    out = torch.empty_like(xs[0])
    kernel((n,), *xs, out)
    expected = xs[0]
    for x in xs[1:]:
        expected = expected + x
    assert torch.equal(out, expected)
