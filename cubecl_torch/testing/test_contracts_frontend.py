"""Ported live front-end contracts: rejection, IEEE specials, reduction reuse.

Sources: stage2/verify.py (negatives, special values, comparisons, signed zero,
mixed promotion, boolean storage) and stage3/verify.py (row-sum dynamic reuse,
specials and cancellation, accumulation and empty mean). The Stage 5 corpus
compares the package with Stage 5 on random inputs only; these compare with
eager on the special values and rejections the corpus does not exercise.
"""

import pytest
import torch

from cubecl_torch.testing.fixtures import run_python

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HELPERS = """
import functools
from cubecl_torch.inductor import kernels
torch._dynamo.config.recompile_limit = 128
torch.manual_seed(23)
TOL = {torch.float32: (2e-6, 2e-6), torch.float16: (0.002, 0.001), torch.bfloat16: (0.016, 0.008)}

def compiled(fn, dynamic=True):
    return torch.compile(fn, backend=functools.partial(ci.compile_backend, tuning="baseline"),
                         fullgraph=True, dynamic=dynamic)

def check(a, b):
    rtol, atol = TOL.get(a.dtype, (0, 0))
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, equal_nan=True)

def close(a, b):
    check(a, b)
    assert a.dtype == b.dtype
    assert torch.equal(torch.isnan(a), torch.isnan(b))
    assert torch.equal(torch.isposinf(a), torch.isposinf(b))
    assert torch.equal(torch.isneginf(a), torch.isneginf(b))

SPECIALS = [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, -1, 1]
"""


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("frontend-cache")


def run(code, cache_dir):
    return run_python(HELPERS + code, cache_dir)


def test_unsupported_inputs_rejected_without_compiling(cache_dir):
    code = """
x = lambda *shape, dtype=torch.float32: torch.randn(*shape, device="cuda", dtype=dtype)
cases = [
    ("float64", lambda x: x + 1, [x(257, dtype=torch.float64)]),
    ("integer_tensor", lambda x: x + 1, [torch.ones(257, device="cuda", dtype=torch.int64)]),
    ("rank5", lambda x: torch.sin(x) + 1, [x(2, 2, 2, 2, 2)]),
    ("unknown_tan", lambda x: torch.tan(x), [x(257)]),
    ("masked_unknown_tan", lambda x: torch.cat([torch.tan(x), x[:1]]) + 1, [x(257)]),
    ("bool_arithmetic", lambda x: x + x, [torch.ones(257, device="cuda", dtype=torch.bool)]),
    ("float64_cast", lambda x: x.to(torch.float64), [x(257)]),
    ("minimum", lambda x: torch.minimum(x, x), [x(257)]),
]
result = {}
for name, fn, args in cases:
    before = kernels.COMPILER_INVOCATIONS
    try:
        compiled(fn)(*args)
        result[name] = "accepted"
    except Exception as error:
        assert "unsupported" in str(error) or "required" in str(error), (name, str(error))
        assert kernels.COMPILER_INVOCATIONS == before, name
        result[name] = "rejected"
# Stage 2 also rejected these; later stages support them (full reductions,
# five inputs, specialised runtime scalars). What survives is a correct result.
a = x(257)
check(compiled(lambda x: x.sum())(a), a.sum())
args = [x(257) for _ in range(5)]
check(compiled(lambda a, b, c, d, e: a + b + c + d + e)(*args), sum(args))
check(compiled(lambda x, scale: x * scale)(a, 3), a * 3)
print(json.dumps(result))
"""
    result = run(code, cache_dir)
    assert set(result.values()) == {"rejected"}, result


def test_pointwise_special_values_signed_zero_and_storage(cache_dir):
    code = """
binary = lambda x, y: (x - y) / (y + 2)
unary = lambda x: torch.where(x <= 0, -x, torch.sqrt(torch.exp(x) - 1))
for f in (binary, unary):
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        specials = torch.tensor(SPECIALS + [1e-30, -1e-30], device="cuda", dtype=dtype)
        args = [specials, specials] if f is binary else [specials]
        check(compiled(f)(*args), f(*args))
for cmp in ("eq", "ne", "lt", "le", "gt", "ge"):
    f = lambda x, cmp=cmp: getattr(torch, cmp)(x, 0)
    s = torch.tensor(SPECIALS, device="cuda")
    actual = compiled(f)(s)
    assert actual.dtype == torch.bool
    torch.testing.assert_close(actual, f(s), rtol=0, atol=0)
for name, f in [
    ("sub_self", lambda x: x - x),
    ("add_zero", lambda x: x + 0.0),
    ("mul_zero", lambda x: x * 0.0),
    ("zero_div", lambda x: 0.0 / x),
    ("neg", lambda x: -x),
    ("nonfinite_constant", lambda x: x + float("inf")),
]:
    s = torch.tensor(SPECIALS, device="cuda")
    a, b = compiled(f)(s), f(s)
    check(a, b)
    finite = torch.isfinite(b)
    assert torch.equal(torch.signbit(a[finite]), torch.signbit(b[finite])), (name, a, b)
for left, right in ((torch.float16, torch.float32), (torch.float16, torch.bfloat16)):
    a = torch.randn(257, device="cuda", dtype=left)
    b = torch.randn(257, device="cuda", dtype=right)
    y = compiled(lambda a, b: a + b)(a, b)
    check(y, a + b)
    assert y.dtype == torch.float32
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    mask = torch.arange(257, device="cuda") % 2 == 0
    v = torch.randn(257, device="cuda", dtype=dtype)
    f = lambda mask, x: torch.where(mask, x, -x)
    check(compiled(f)(mask, v), f(mask, v))
s = torch.tensor(SPECIALS, device="cuda")
torch.testing.assert_close(compiled(lambda x: x.to(torch.bool))(s), s.to(torch.bool), rtol=0, atol=0)
print(json.dumps({"ok": True}))
"""
    assert run(code, cache_dir) == {"ok": True}


def test_row_reduction_reuse_specials_and_accumulation(cache_dir):
    code = """
TOL[torch.float32] = (2e-5, 2e-5)  # Stage 3's float32 reduction tolerance
row = lambda x: x.sum(-1)
mean = lambda x: x.mean(-1) + 1
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    f = compiled(row)
    shapes = [(3, 31), (4, 32), (2, 33), (5, 255), (3, 256), (4, 257), (2, 4099),
              (3, 65539), (3, 1), (3, 0), (0, 257), (3, 257)]
    compiles = []
    for b, n in shapes:
        x = torch.randn(b, n, device="cuda", dtype=dtype, requires_grad=True)
        before = kernels.COMPILER_INVOCATIONS
        y = f(x)
        close(y, row(x))
        upstream = torch.randn_like(y)
        ref = x.detach().requires_grad_()
        close(torch.autograd.grad(y, x, upstream)[0], torch.autograd.grad(row(ref), ref, upstream)[0])
        compiles.append(kernels.COMPILER_INVOCATIONS - before)
    # One dynamic reduction kernel serves every ordinary row shape after the
    # first. (3, 65539) is past Stage 5's split threshold and compiles the split
    # reduction, which Stage 3 did not have.
    assert all(c == 0 for c in compiles[1:7]), (str(dtype), compiles)
    for n in (2, 31, 32, 33, 255, 256, 257, 4099):
        x = torch.zeros(8, n, device="cuda", dtype=dtype)
        x[1].fill_(-0.0)
        x[2, 0] = float("nan")
        x[3, 0] = float("inf")
        x[4, 0] = -float("inf")
        x[5, 0] = float("inf")
        x[5, 1] = -float("inf")
        x[6, ::2] = 100
        x[6, 1::2] = -100
        x[7, ::4] = 10000
        x[7, 1::4] = 1
        x[7, 2::4] = -10000
        y = f(x)
        close(y, row(x))
        assert not torch.signbit(y[:2]).any(), (str(dtype), n, y[:2])
    g = compiled(mean)
    for n in (0, 1, 257):
        x = torch.full((3, n), 1000, device="cuda", dtype=dtype)
        close(g(x), mean(x))
    if dtype == torch.float16:
        x = torch.full((3, 257), 1000, device="cuda", dtype=dtype)
        summed = f(x)
        close(summed, row(x))
        assert torch.isposinf(summed).all()
        assert torch.isfinite(g(x)).all()
print(json.dumps({"ok": True}))
"""
    assert run(code, cache_dir) == {"ok": True}
