"""Basic float arithmetic (OpenSpec basic-float-arithmetic, bead 4x9.2).

Results against the Inductor+Triton and eager outputs recorded by
float_arithmetic_reference.py, compared bit for bit.
"""

import pytest
import torch

from cubecl_torch.testing.fixtures import run_python

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Per recorded case, the indices where CubeCL differs from Triton and from
# eager: bits differ, except where both are NaN (counted apart, "nan_bits"),
# and the NaN results with the sign bit set. Integer tensors other than int64
# are not storage dtypes here, so those cases read and write int64 and convert
# inside the kernel, which leaves every value unchanged.
REFERENCE = """
from cubecl_torch.testing import float_arithmetic_reference as ref
data = torch.load(ref.DATA)["cases"]
NARROW = (torch.int8, torch.int16, torch.int32, torch.uint8)
def widened(fn, args):
    dtypes = [a.dtype for a in args]
    if not any(d in NARROW for d in dtypes):
        return fn, args
    def wide(*xs):
        y = fn(*(x.to(d) for x, d in zip(xs, dtypes)))
        return y.to(torch.int64) if y.dtype in NARROW else y
    return wide, [a.to(torch.int64) if a.dtype in NARROW else a for a in args]
def bits(t):
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])
def differences(y, want):
    nan = y.isnan() & want.isnan() if y.is_floating_point() else torch.zeros_like(y, dtype=torch.bool)
    differ = bits(y) != bits(want)
    return {"values": (differ & ~nan).nonzero().flatten().tolist(),
            "nan_bits": (differ & nan).nonzero().flatten().tolist()}
def against_reference(select):
    rows = {}
    for case in ref.cases():
        if not select(case):
            continue
        torch._dynamo.reset()
        args = ref.case_inputs(case)
        expected = data[case["name"]]
        if [ref.digest(a) for a in args] != expected["input_sha256"]:
            rows[case["name"]] = "input digest"
            continue
        fn, inputs = widened(ref.function(case), args)
        y = torch.compile(fn, backend=ci.compile_backend)(*(a.cuda() for a in inputs)).cpu()
        if y.dtype != expected["output"].dtype:
            expected = {k: v.to(y.dtype) for k, v in expected.items() if k != "input_sha256"}
        negative_nan = y.isnan() & y.signbit() if y.is_floating_point() else y != y
        rows[case["name"]] = {
            "triton": differences(y, expected["output"]),
            "eager": differences(y, expected["eager"]),
            "negative_nan": negative_nan.nonzero().flatten().tolist(),
        }
    return rows
"""


def test_abs_square_relu_against_triton_and_eager(tmp_path):
    code = (
        REFERENCE
        + 'print(json.dumps(against_reference(lambda c: c["kind"] == "unary")))'
    )
    result = run_python(code, tmp_path)
    assert all(row != "input digest" for row in result.values()), result
    clean = {"values": [], "nan_bits": []}
    for name, row in result.items():
        op, dtype = name.split("-")
        floating = dtype.startswith(("float", "bfloat"))
        # Every non-NaN result is Triton's, bit for bit.
        assert row["triton"]["values"] == [], (name, row)
        if op == "abs":
            # NaN payload open (abs.f32 canonicalizes, as eager; Triton's mask
            # keeps it), sign cleared. Integer abs wraps at the minimum, as eager
            # (integer-arithmetic).
            assert row["negative_nan"] == [], (name, row)
            assert row["eager"]["values"] == [], (name, row)
        elif op == "square":
            assert row["triton"] == clean and row["eager"] == clean, (name, row)
        else:
            # relu: NaN stays NaN, payload open (float-extrema); -0.0 (index 2
            # of the float specials) gives +0.0 as Triton, where eager gives -0.0.
            assert row["eager"]["values"] == ([2] if floating else []), (name, row)


def test_integer_abs_emits_no_cpp_abs(tmp_path):
    """C++ abs on int and long long is undefined at the minimum; integer abs is
    a compare and select over a wrapping negation."""
    code = """
x = torch.tensor([-2**63, 2**63 - 1, -5, 0, 7], device="cuda")
for t in (torch.int8, torch.int16, torch.int32, torch.int64):
    torch._dynamo.reset()
    fn = lambda x, t=t: torch.abs(x.to(t)).to(torch.int64)
    assert torch.equal(torch.compile(fn, backend=ci.compile_backend)(x), fn(x)), t
import re
sources = list(Path(os.environ["CUBECL_TORCH_CACHE_DIR"]).rglob("kernel.cu"))
calls = [line.strip() for source in sources for line in source.read_text().splitlines()
         if re.search(r"\\babs\\(", line)]
print(json.dumps({"kernels": len(sources), "calls": calls}))
"""
    result = run_python("from pathlib import Path\n" + code, tmp_path)
    assert result["kernels"] >= 4, result
    assert result["calls"] == [], result


def test_basic_ops_on_every_kernel_path(tmp_path):
    """abs, square and relu fused into pointwise, reduction and indirect
    kernels, float and integer: each compiles through CubeCL and equals eager
    (bit for bit, or within float32 tolerance where a reduction reorders or
    NVRTC contracts the square with the following add, as Triton does)."""
    code = """
g = torch.Generator(device="cuda").manual_seed(0)
x = torch.randn(64, 300, device="cuda", generator=g)
w = torch.randn(300, device="cuda", generator=g)
i = torch.randint(0, 64, (128,), device="cuda", generator=g)
n = torch.tensor([-2**63, 2**63 - 1, -5, 0, 7, -2**31, 2**31 - 1], device="cuda")
cases = {
    "abs pointwise": (lambda x: torch.abs(x) * 2, (x,), True),
    "square pointwise": (lambda x: torch.square(x) + 1, (x,), False),
    "relu pointwise": (lambda x: torch.relu(x - 0.5), (x,), True),
    "abs in sum": (lambda x, w: (torch.abs(x) * w).sum(-1), (x, w), False),
    "square in sum": (lambda x: torch.square(x).sum(-1), (x,), False),
    "relu in amax": (lambda x: torch.relu(x).amax(-1), (x,), True),
    "abs of gather": (lambda x, i: torch.abs(x[i]), (x, i), True),
    "square of gather": (lambda x, i: torch.square(x[i]), (x, i), True),
    "relu of gather": (lambda x, i: torch.relu(x[i] * 3), (x, i), True),
    "int64 square": (torch.square, (n,), True),
    "int32 relu": (lambda n: torch.relu(n.to(torch.int32)).to(torch.int64), (n,), True),
    "addcmul of gather": (
        lambda x, i: torch.addcmul(x[i], x[i], x[i] * 2, value=0.3), (x, i), True),
    "addcdiv of gather": (
        lambda x, i: torch.addcdiv(x[i], x[i], x[i] + 3, value=0.3), (x, i), True),
    "add alpha with int64": (lambda x, i: torch.add(x[0, :128], i, alpha=0.3), (x, i), True),
    # Indirect kernels write floating storage only.
    "int32 abs of gather": (
        lambda x, i: torch.abs((x[i] * 1e6).to(torch.int32)).to(torch.float32), (x, i), True),
}
torch.manual_seed(0)
rng = lambda x: torch.addcdiv(x, torch.rand_like(x), x.abs() + 1, value=0.3)
torch._dynamo.reset()
y = torch.compile(rng, backend=ci.compile_backend)(x)
bad = {} if bool(torch.isfinite(y).all()) and y.shape == x.shape else {"rng addcdiv": "bad"}
for name, (fn, args, exact) in cases.items():
    torch._dynamo.reset()
    y, want = torch.compile(fn, backend=ci.compile_backend)(*args), fn(*args)
    ok = torch.equal(y, want) if exact else torch.allclose(y, want, rtol=1e-5, atol=1e-5)
    if not ok:
        bad[name] = float((y.double() - want.double()).abs().max())
print(json.dumps(bad))
"""
    assert run_python(code, tmp_path) == {}


def test_short_row_variance_within_tritons_band(tmp_path):
    """var_mean over rows of at most 8 elements (Inductor's two-step sum with
    square): per-row error against fp64 no worse than Triton's, by
    welford_band's sign and paired mean-difference tests."""
    code = (
        REFERENCE
        + """
from cubecl_torch.testing import welford_band as band
out = {}
for case in ref.cases():
    if case["kind"] != "short_rows":
        continue
    torch._dynamo.reset()
    (x,) = ref.case_inputs(case)
    if ref.digest(x) != data[case["name"]]["input_sha256"][0]:
        out[case["name"]] = "input digest"
        continue
    y = torch.compile(ref.function(case), backend=ci.compile_backend)(x.cuda()).cpu()
    exact = torch.stack(torch.var_mean(x.double(), -1, correction=0))
    triton = data[case["name"]]["output"]
    verdict = {}
    for k, name in enumerate(("var", "mean")):
        c = (y[k].double() - exact[k]).abs().tolist()
        t = (triton[k].double() - exact[k]).abs().tolist()
        test = band.sign_test(c, t)
        test["p_mean"] = band.mean_difference(c, t)
        verdict[name] = band.outside(test)
    out[case["name"]] = verdict
print(json.dumps(out))
"""
    )
    result = run_python(code, tmp_path)
    assert len(result) == 3, result
    assert all(row != "input digest" for row in result.values()), result
    assert all(not v for row in result.values() for v in row.values()), result


FUSED = ("alpha", "addcmul", "addcdiv", "division_mix", "lerp")


def test_fused_forms_against_triton_and_eager(tmp_path):
    """add/sub with alpha, addcmul and addcdiv (value 1 and 0.3, divisor
    specials), addcdiv next to a plain division, and lerp with scalar and
    tensor weights: every non-NaN result equals eager and Triton bit for bit,
    with two exceptions. The plain division follows the division policy, as
    Triton's (not eager). addcdiv(value=1) divides with div_rn, so it equals
    eager where Triton's div.full may round differently (bead 4jf)."""
    code = REFERENCE + (
        f"print(json.dumps(against_reference(lambda c: c['kind'] in {FUSED!r})))"
    )
    result = run_python(code, tmp_path)
    assert len(result) == 26, sorted(result)
    for name, row in result.items():
        assert row != "input digest", name
        if name.startswith("addcdiv_and_division"):
            assert row["triton"]["values"] == [], (name, row)
            assert row["eager"]["values"], (name, "plain division matched eager")
        elif name.startswith("addcdiv-value1"):
            assert row["eager"]["values"] == [], (name, row)
        else:
            assert row["triton"]["values"] == [], (name, row)
            assert row["eager"]["values"] == [], (name, row)


def test_single_rounding_with_precision_emulation(tmp_path):
    """add with alpha (an fma) rounds once under emulate_precision_casts too,
    where float16/bfloat16 values are rounded between ops and contraction is
    off. sub with alpha is a multiply and a subtract, not an fma, so it rounds
    twice there, as Triton's does under the same setting."""
    code = (
        REFERENCE
        + """
import torch._inductor.config as inductor_config
inductor_config.emulate_precision_casts = True
print(json.dumps(against_reference(lambda c: c["kind"] == "alpha" and c["op"] == "add")))
"""
    )
    result = run_python(code, tmp_path)
    assert len(result) == 3, sorted(result)
    for name, row in result.items():
        assert row["eager"]["values"] == [] and row["triton"]["values"] == [], (
            name,
            row,
        )


def test_explicit_rounding_cuda_calls(tmp_path):
    """The kernels call fma, __fmul_rn and __fdiv_rn, which --fmad and
    --prec-div leave alone; a constant fma folds to the singly rounded value."""
    code = """
import re
x, y, z = (torch.randn(4096, device="cuda") + 2 for _ in range(3))
def sources():
    root = Path(os.environ["CUBECL_TORCH_CACHE_DIR"])
    return {p: p.read_text() for p in root.rglob("kernel.cu")}
calls = {}
for name, fn in {
    "add_alpha": lambda x, y, z: torch.add(x, y, alpha=0.3),
    "addcmul": lambda x, y, z: torch.addcmul(x, y, z, value=0.3),
    "addcdiv": lambda x, y, z: torch.addcdiv(x, y, z, value=0.3),
    "addcdiv_value1": lambda x, y, z: torch.addcdiv(x, y, z, value=1),
}.items():
    torch._dynamo.reset()
    before = sources()
    torch.compile(fn, backend=ci.compile_backend)(x, y, z)
    new = "".join(t for p, t in sources().items() if p not in before)
    calls[name] = sorted({c for c in ("fma(", "__fmul_rn(", "__fdiv_rn(") if c in new})
# (1 + 2**-12)**2 - (1 + 2**-11) is 2**-24 rounded once and 0 rounded twice.
b, c = 1 + 2**-12, -(1 + 2**-11)
fold = lambda x: torch.add(torch.full_like(x, c), torch.full_like(x, b), alpha=b)
torch._dynamo.reset()
before = sources()
folded = torch.compile(fold, backend=ci.compile_backend)(x)
fold_source = "".join(t for p, t in sources().items() if p not in before)
print(json.dumps({"calls": calls, "folded": sorted(set(folded.tolist())),
                  "eager": sorted(set(fold(x).tolist())),
                  "fold_kernel": bool(fold_source), "fold_has_fma": "fma(" in fold_source}))
"""
    result = run_python("from pathlib import Path\n" + code, tmp_path)
    assert result["calls"] == {
        "add_alpha": ["fma("],
        "addcmul": ["__fmul_rn(", "fma("],
        "addcdiv": ["__fdiv_rn(", "fma("],
        # The `* 1` survives (joint-graph constant folding is off), so Inductor
        # rebuilds addcdiv: div_rn and no fma.
        "addcdiv_value1": ["__fdiv_rn("],
    }, result
    # Three constant fma operands fold in CubeCL, once rounded.
    assert result["folded"] == result["eager"] == [2.0**-24], result
    assert result["fold_kernel"] and not result["fold_has_fma"], result


def test_adam_step_and_fused_reductions(tmp_path):
    """An Adam-style step compiles through CubeCL and equals eager bit for bit;
    abs and addcmul fused into a row sum run as one kernel each."""
    code = """
from cubecl_torch.runtime import trace
def kernels(fn, *args):
    torch._dynamo.reset()
    start = len(trace.EVENTS)
    with trace.scoped():
        out = torch.compile(fn, backend=ci.compile_backend, dynamic=False)(*args)
    return out, sum(e["event"] == "kernel" for e in list(trace.EVENTS)[start:])
beta1, beta2, step_size = 0.9, 0.999, 1e-3
def adam(param, grad, exp_avg, exp_avg_sq, denom):
    exp_avg = exp_avg.lerp(grad, 1 - beta1)
    exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(grad, grad, value=1 - beta2)
    param = param.addcdiv(exp_avg, denom, value=-step_size)
    return param, exp_avg, exp_avg_sq
g = torch.Generator(device="cuda").manual_seed(0)
state = [torch.randn(1 << 16, device="cuda", generator=g) for _ in range(4)]
state.append(torch.rand(1 << 16, device="cuda", generator=g) + 1e-3)
out, launched = kernels(adam, *state)
adam_exact = [torch.equal(a, b) for a, b in zip(out, adam(*state))]
x = torch.randn(64, 300, device="cuda", generator=g)
w = torch.randn(300, device="cuda", generator=g)
reductions = {}
for name, fn in {
    "abs": lambda x, w: (torch.abs(x) * w).sum(-1),
    "addcmul": lambda x, w: torch.addcmul(x, x, w, value=0.5).sum(-1),
}.items():
    y, count = kernels(fn, x, w)
    reductions[name] = [count, bool(torch.allclose(y, fn(x, w), rtol=1e-5, atol=1e-4))]
print(json.dumps({"adam": adam_exact, "adam_kernels": launched, "reductions": reductions}))
"""
    result = run_python(code, tmp_path)
    assert result["adam"] == [True, True, True], result
    # One kernel per update until multi-output fusion (bead 4z8.21); Triton
    # fuses the three into one.
    assert result["adam_kernels"] == 3, result
    assert result["reductions"] == {"abs": [1, True], "addcmul": [1, True]}, result
