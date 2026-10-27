"""Float extrema contract: openspec float-extrema (generic-reduction-combiners).

NaN propagation, ordered signed zero (-0 < +0), NaN-ignoring fmax/fmin through
Inductor's decomposition, isnan, and bitwise agreement with Inductor+Triton
(cubecl_torch/testing/data/extrema_triton.pt, see extrema_reference.py).
"""

import pytest
import torch

from cubecl_torch.testing.fixtures import run_python

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

COMPARE = """
from cubecl_torch.testing import extrema_reference as ref
BITS = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16}
def mismatch(y, t):
    if y.dtype != t.dtype or y.shape != t.shape:
        return "dtype/shape"
    if not y.is_floating_point():
        return None if torch.equal(y, t) else "values"
    nan = torch.isnan(y)
    if not torch.equal(nan, torch.isnan(t)):
        return "NaN positions"
    bits = BITS[y.dtype]
    if not torch.equal(y[~nan].view(bits), t[~nan].view(bits)):
        return "non-NaN bits"
    return None
def against_triton(kind):
    data = torch.load(ref.DATA)["cases"]
    rows = {}
    for case in ref.cases():
        if case["kind"] != kind:
            continue
        torch._dynamo.reset()
        args = ref.case_inputs(case)
        expected = data[case["name"]]
        if [ref.digest(a) for a in args] != expected["input_sha256"]:
            rows[case["name"]] = "input digest"
            continue
        f = torch.compile(ref.function(case), backend=ci.compile_backend, dynamic=case["dynamic"])
        y = ref.run(f, case, args)
        if (why := mismatch(y, expected["output"])) is not None:
            rows[case["name"]] = why
    return rows
"""


def test_pointwise_extrema_match_triton_bitwise(tmp_path):
    code = COMPARE + 'print(json.dumps(against_triton("pointwise")))'
    assert run_python(code, tmp_path) == {}


def test_pointwise_extrema_scenarios(tmp_path):
    code = """
import math
nan = float("nan")
def signs(t):
    return ["nan" if math.isnan(v) else ("-" if math.copysign(1, v) < 0 else "+") + str(abs(v))
            for v in t.float().tolist()]
a = torch.tensor([-0.0, 0.0, nan, 1.0, nan, 2.0], device="cuda")
b = torch.tensor([0.0, -0.0, 1.0, nan, nan, -3.0], device="cuda")
x = torch.tensor([nan, float("inf"), -float("inf"), 0.0, -0.0, 1.5], device="cuda")
out = {}
for dt in (torch.float32, torch.float16, torch.bfloat16):
    a_, b_, x_ = a.to(dt), b.to(dt), x.to(dt)
    for name, f in (("maximum", torch.maximum), ("minimum", torch.minimum),
                    ("fmax", torch.fmax), ("fmin", torch.fmin)):
        torch._dynamo.reset()
        out[f"{name}-{dt}"] = signs(torch.compile(f, backend=ci.compile_backend)(a_, b_))
    torch._dynamo.reset()
    out[f"isnan-{dt}"] = torch.equal(torch.compile(torch.isnan, backend=ci.compile_backend)(x_), torch.isnan(x_))
    # Bool and/or on the general pointwise path.
    pick = lambda a, b: torch.where((a > 0) & (b < 1) | torch.isnan(a), b, a)
    torch._dynamo.reset()
    out[f"predicates-{dt}"] = signs(torch.compile(pick, backend=ci.compile_backend)(a_, b_)) == signs(pick(a_, b_))
# fmax feeding a reduction: the predicate stays out of the reduction kernel.
torch._dynamo.reset()
fused = lambda a, b: torch.fmax(a, b).sum(-1)
got = torch.compile(fused, backend=ci.compile_backend)(a.view(2, 3), b.view(2, 3))
want = fused(a.view(2, 3), b.view(2, 3))
out["fmax_then_sum"] = signs(got) == signs(want)
print(json.dumps(out))
"""
    result = run_python(code, tmp_path)
    for dt in ("torch.float32", "torch.float16", "torch.bfloat16"):
        # (-0,+0) and (+0,-0): ordered; one NaN or two: NaN; finite: as usual.
        assert result[f"maximum-{dt}"] == ["+0.0", "+0.0", "nan", "nan", "nan", "+2.0"]
        assert result[f"minimum-{dt}"] == ["-0.0", "-0.0", "nan", "nan", "nan", "-3.0"]
        # NaN-ignoring: one NaN yields the other operand, two NaNs yield NaN.
        assert result[f"fmax-{dt}"][2:] == ["+1.0", "+1.0", "nan", "+2.0"]
        assert result[f"fmin-{dt}"][2:] == ["+1.0", "+1.0", "nan", "-3.0"]
        assert result[f"isnan-{dt}"] is True
        assert result[f"predicates-{dt}"] is True
    assert result["fmax_then_sum"] is True


def test_extrema_lower_to_nan_propagating_ptx(tmp_path):
    code = """
x = torch.randn(1000, device="cuda")
torch.compile(lambda a, b: torch.maximum(a, b) * torch.minimum(a, b), backend=ci.compile_backend)(x, x.flip(0))
ptx = [p.read_text() for p in (Path(os.environ["CUBECL_TORCH_CACHE_DIR"]) / "artifacts").glob("*/kernel.ptx")]
print(json.dumps({"max": sum("max.NaN.f32" in p for p in ptx), "min": sum("min.NaN.f32" in p for p in ptx)}))
"""
    result = run_python("from pathlib import Path\n" + code, tmp_path)
    assert result["max"] >= 1 and result["min"] >= 1


def test_isinf(tmp_path):
    """isinf (float-extrema delta of x3c) against eager and the Triton outputs
    recorded by rounding_remainder_reference.py, bit for bit; integer operands
    (Inductor's lowering gives false) and a finite mask fused into a sum."""
    from cubecl_torch.testing import rounding_remainder_reference as ref

    names = [f"isinf-{ref.name(d)}" for d in ref.FLOATS]
    out = tmp_path / "ours.pt"
    run_python(ref.compile_code(names, out), tmp_path)
    ours = torch.load(out)
    data = torch.load(ref.DATA)["cases"]
    for name in names:
        assert ours[name] is not None, f"{name}: input digest changed"
        for wanted in (data[name]["output"][0], data[name]["eager"][0]):
            assert torch.equal(ours[name][0], wanted), name
    code = """
from cubecl_torch.testing.fixtures import native_launches
i = torch.tensor([-5, 0, 7, -2**63, 2**63 - 1], device="cuda")
x = torch.randn(64, 300, device="cuda")
x[3, 7], x[9, 0], x[20, 299] = float("inf"), -float("inf"), float("nan")
masked = lambda x: torch.where(torch.isinf(x), 0.0, x).sum(-1)
compiled = torch.compile(masked, backend=ci.compile_backend)
compiled(x)
before = native_launches()
y = compiled(x)
launches = native_launches() - before
torch._dynamo.reset()
integer = torch.compile(torch.isinf, backend=ci.compile_backend)(i)
print(json.dumps({
    "integer": torch.equal(integer, torch.isinf(i)),
    "masked": torch.allclose(y, masked(x), rtol=1e-5, atol=1e-4, equal_nan=True),
    "launches": launches,
}))
"""
    result = run_python(code, tmp_path)
    assert result == {"integer": True, "masked": True, "launches": 1}, result
