"""Integer and bool value arithmetic (OpenSpec integer-value-arithmetic, bead 4x9.1)."""

import re
from pathlib import Path

import pytest
import torch

from cubecl_torch.compiler import descriptor
from cubecl_torch.testing.fixtures import run_python

NATIVE = Path(__file__).resolve().parents[1] / "native" / "src"


def test_integer_dtype_table_matches_the_compiler():
    source = (NATIVE / "ir.rs").read_text()
    table = source[source.index("pub(crate) const INTEGERS") :]
    table = table[: table.index("];")]
    rust = {
        name: (signed == "true", int(width))
        for name, signed, width in re.findall(
            r'\("(\w+)", (true|false), (\d+)\)', table
        )
    }
    assert rust == descriptor.INTEGERS


# A compiled function's warm launches and its agreement with eager, per case.
LAUNCHES = """
from cubecl_torch.testing.fixtures import native_launches
def launches_and_result(fn, *args):
    torch._dynamo.reset()
    f = torch.compile(fn, backend=ci.compile_backend)
    f(*args)
    torch.cuda.synchronize()
    before = native_launches()
    y = f(*args)
    torch.cuda.synchronize()
    return native_launches() - before, y
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bool_predicates_fuse_into_pointwise_and_reduction_kernels(tmp_path):
    code = (
        LAUNCHES
        + """
a = torch.randn(64, 1000, device="cuda")
b = torch.randn(64, 1000, device="cuda")
a[0, :7] = float("nan"); b[1, :5] = float("nan")
out = {}
for name, fn in {
    "fmax_sum": lambda a, b: torch.fmax(a, b).sum(-1),
    "fmax_scaled": lambda a, b: torch.fmax(a, b) * 2,
}.items():
    n, y = launches_and_result(fn, a, b)
    expected = fn(a, b)
    out[name] = [n, bool(torch.allclose(y, expected, rtol=1e-5, atol=1e-4, equal_nan=True))]
print(json.dumps(out))
"""
    )
    result = run_python(code, tmp_path)
    assert result["fmax_sum"] == [1, True]
    assert result["fmax_scaled"] == [1, True]


# Compiled vs eager, bit for bit, per named case; prints the mismatching names.
EAGER = """
def mismatches(cases):
    bad = {}
    for name, (fn, args) in cases.items():
        torch._dynamo.reset()
        try:
            y = torch.compile(fn, backend=ci.compile_backend)(*args)
        except Exception as e:
            bad[name] = type(e).__name__ + ": " + str(e).splitlines()[0][:200]
            continue
        want = fn(*args)
        ys, wants = (y, want) if isinstance(want, tuple) else ((y,), (want,))
        for i, (y, want) in enumerate(zip(ys, wants, strict=True)):
            if y.dtype != want.dtype or not torch.equal(y, want):
                where = (y != want).nonzero()[:4].flatten().tolist() if y.shape == want.shape else "shape"
                bad[f"{name}[{i}]"] = f"{y.dtype}/{want.dtype} at {where}"
    return bad
def pairs(values):
    a = torch.tensor([x for x in values for _ in values], device="cuda")
    b = torch.tensor([y for _ in values for y in values], device="cuda")
    return a, b
INTS = {"int8": torch.int8, "int16": torch.int16, "int32": torch.int32,
        "int64": torch.int64, "uint8": torch.uint8}
FLOATS = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
def ints(dtype):
    info = torch.iinfo(dtype)
    v = [0, 1, -1, 2, -2, 7, 100, -100, info.min, info.max, info.min + 1, info.max - 1]
    return [x for x in v if info.min <= x <= info.max]
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_casts_between_bool_integer_and_float(tmp_path):
    code = (
        EAGER
        + """
cases = {}
wide = torch.tensor(ints(torch.int64) + [2**40 + 5, -(2**40) - 5, 2**24 + 1, 300, -300, 70000], device="cuda")
for name, t in INTS.items():
    # Narrowing wraps; widening back to int64 observes it.
    cases[f"int64->{name}->int64"] = (lambda x, t=t: x.to(t).to(torch.int64), (wide,))
    # Integer to float rounds to nearest.
    for fname, f in FLOATS.items():
        cases[f"int64->{name}->{fname}"] = (lambda x, t=t, f=f: x.to(t).to(f), (wide,))
    cases[f"int64->{name}->bool"] = (lambda x, t=t: x.to(t).to(torch.bool), (wide,))
    cases[f"bool->{name}->int64"] = (
        lambda b, t=t: b.to(t).to(torch.int64),
        (torch.tensor([True, False, True], device="cuda"),),
    )
for fname, f in FLOATS.items():
    values = torch.tensor([0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.7, -2.7, 100.9, -100.9, 126.0, -127.5],
                          device="cuda").to(f)
    for name, t in INTS.items():
        # Float to integer truncates toward zero; to uint8 wraps through int16, as eager.
        cases[f"{fname}->{name}->int64"] = (lambda x, t=t: x.to(t).to(torch.int64), (values,))
    specials = torch.tensor([0.0, -0.0, 0.1, float("nan"), float("inf"), -float("inf")], device="cuda").to(f)
    cases[f"{fname}->bool"] = (lambda x: x.to(torch.bool), (specials,))
    cases[f"bool->{fname}"] = (lambda x, f=f: (x > 0).to(f), (values,))
print(json.dumps(mismatches(cases)))
"""
    )
    assert run_python(code, tmp_path) == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_integer_arithmetic_comparisons_and_selection_every_width(tmp_path):
    code = (
        EAGER
        + """
cases = {}
for name, t in INTS.items():
    a, b = pairs(ints(t))
    def arithmetic(a, b, t=t):
        a, b = a.to(t), b.to(t)
        wide = lambda v: v.to(torch.int64)
        return (wide(a + b), wide(a - b), wide(a * b), wide(-a), wide(a + 3), wide(a * -1),
                wide(torch.maximum(a, b)), wide(torch.minimum(a, b)),
                wide(torch.where(a > b, a, b)), wide(torch.clamp(a, -2, 5)))
    def comparisons(a, b, t=t):
        a, b = a.to(t), b.to(t)
        return (a == b, a != b, a < b, a <= b, a > b, a >= b, a > 1)
    cases[f"{name} arithmetic"] = (arithmetic, (a, b))
    cases[f"{name} comparisons"] = (comparisons, (a, b))
a, b = pairs([-128, -1, 0, 1, 127])
cases["int8 with uint8 promotes to int16"] = (
    lambda a, b: (a.to(torch.int8) + b.to(torch.uint8)).to(torch.int64), (a, b))
x = torch.tensor([0.0, 27.0, 28.0, 100.0, -100.0, -128.0, 127.0], device="cuda")
cases["int8 wraps before the comparison"] = (lambda x: (x.to(torch.int8) + 100) > 0, (x,))
p, q = pairs([-1.0, 0.0, 2.0])
cases["bool comparisons and extrema"] = (
    lambda p, q: ((p > 0) == (q > 0), (p > 0) != (q > 0), (p > 0) < (q > 0), (p > 0) >= (q > 0),
                  torch.maximum(p > 0, q > 0), torch.minimum(p > 0, q > 0)), (p, q))
i = torch.arange(-5, 6, device="cuda")
cases["masked int64 fill"] = (lambda i: torch.nn.functional.pad(i, (2, 3), value=7), (i,))
cases["masked bool fill"] = (lambda i: torch.nn.functional.pad(i > 0, (1, 2), value=True), (i,))
cases["where over int64"] = (lambda i: torch.where(i > 0, i * 2, -i), (i,))
print(json.dumps(mismatches(cases)))
"""
    )
    assert run_python(code, tmp_path) == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_logical_and_bitwise_operations(tmp_path):
    code = (
        EAGER
        + """
cases = {}
for name, t in INTS.items():
    a, b = pairs(ints(t))
    def bitwise(a, b, t=t):
        a, b = a.to(t), b.to(t)
        wide = lambda v: v.to(torch.int64)
        return wide(a & b), wide(a | b), wide(a ^ b), wide(~a), wide(a & 6)
    cases[f"{name} bitwise"] = (bitwise, (a, b))
p, q = pairs([-1.0, 0.0, 2.0, float("nan")])
cases["bool bitwise"] = (
    lambda p, q: ((p > 0) & (q > 0), (p > 0) | (q > 0), (p > 0) ^ (q > 0), ~(p > 0)), (p, q))
cases["logical"] = (
    lambda p, q: (torch.logical_and(p > 0, q < 1) | torch.logical_not(q > 0),
                  torch.logical_xor(p, q), torch.logical_not(p), torch.logical_or(p, q)), (p, q))
x = torch.randn(4, 9, device="cuda")
cases["constant pad mask"] = (lambda x: torch.nn.functional.pad(x, (1, 2), mode="constant", value=0.5), (x,))
img = torch.randn(2, 3, 9, 9, device="cuda")
# Two bounds per dimension: Inductor joins them with ops.and_. (avg_pool2d's
# padded window also does, but stops at Min() in its index, bead 4z8.18.)
cases["2-D constant pad mask"] = (
    lambda img: torch.nn.functional.pad(img, (1, 2, 2, 1), value=-1.0), (img,))
print(json.dumps(mismatches(cases)))
"""
    )
    assert run_python(code, tmp_path) == {}
