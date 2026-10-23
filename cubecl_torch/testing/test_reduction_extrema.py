"""Max/min reductions: openspec reduction-kernels and float-extrema.

Bitwise agreement with Inductor+Triton (extrema_reference.py) over direct,
fused and split plans, static and dynamic; the row patterns' contract checked
on CubeCL's own output; strategy eligibility under first-use tuning; and named
rejections for reduction types and dtypes not admitted.
"""

import pytest
import torch

from cubecl_torch.testing.fixtures import run_python
from cubecl_torch.testing.test_float_extrema import COMPARE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

PATTERNS = """
import math
def pattern_violations(case, out):
    # The contract per row pattern, on CubeCL's own (input, {amax, amin}, row) output.
    bad = []
    for i in range(out.shape[0]):
        for r in range(out.shape[2]):
            pattern = ref.PATTERNS[(r + i) % len(ref.PATTERNS)]
            hi, lo = (float(v) for v in out[i, :, r])
            if pattern.startswith("nan"):
                ok = math.isnan(hi) and math.isnan(lo)
            elif pattern in ("zeros_mixed", "negative_zeros_one_positive"):
                ok = hi == 0 and lo == 0 and math.copysign(1, hi) > 0 and math.copysign(1, lo) < 0
            elif pattern == "negative_infinity":
                ok = hi == lo == -math.inf
            elif pattern == "infinities":
                ok = hi == math.inf and lo == -math.inf
            else:
                ok = math.isfinite(hi) and math.isfinite(lo) and lo <= hi
            if not ok:
                bad.append([case["name"], i, r, pattern, hi, lo])
    return bad
def reductions(dtype):
    data = torch.load(ref.DATA)["cases"]
    rows = {}
    for case in ref.cases():
        if case["kind"] != "reduction" or case["dtype"] != dtype:
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
        elif bad := pattern_violations(case, y):
            rows[case["name"]] = bad[:5]
    return rows
"""


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_reductions_match_triton_and_the_extrema_contract(tmp_path, dtype):
    code = COMPARE + PATTERNS + f"print(json.dumps(reductions(torch.{dtype})))"
    assert run_python(code, tmp_path) == {}


def test_strategy_eligibility_and_plans(tmp_path):
    code = """
from functools import partial
from cubecl_torch.inductor import legality, tuning
def kernels():
    return [k for k in tuning.CACHE.values() if k.semantic["schema"] == 3]
out = {}
x = torch.randn(300, 4096, device="cuda")
torch.compile(lambda x: x.amax(-1), backend=ci.compile_backend)(x)
accepted, rejected = set(), {}
for k in kernels():
    for plans, refused in k._legality.values():
        accepted |= {p.strategy for p in plans}
        rejected.update({r["plan"]["strategy"]: r["reason"] for r in refused})
out["accepted"] = sorted(accepted)
out["compensated"] = rejected.get("shared_tree_compensated")
out["selected"] = sorted({p.strategy for k in kernels() for p in k.selections.values()})
tuning.CACHE.clear(); torch._dynamo.reset()
y = torch.compile(lambda x: x.amin(-1), backend=partial(ci.compile_backend, tuning="baseline"))(x)
out["baseline_equal"] = torch.equal(y, x.amin(-1))
out["baseline_selected"] = [p.strategy for k in kernels() for p in k.selections.values()]
for dynamic in (False, True):
    tuning.CACHE.clear(); torch._dynamo.reset()
    z = torch.randn(8, 65539, device="cuda")
    m = torch.compile(lambda z: z.amax(-1), backend=ci.compile_backend, dynamic=dynamic)(z)
    out[f"split_equal_{dynamic}"] = torch.equal(m, z.amax(-1))
    out[f"split_stages_{dynamic}"] = len(kernels())
torch._dynamo.reset()
fused = lambda x: (x * 2).amax(-1) + 1
out["fused_equal"] = torch.equal(torch.compile(fused, backend=ci.compile_backend)(x), fused(x))
print(json.dumps(out))
"""
    result = run_python(code, tmp_path)
    sum_only = {
        "shared_tree_compensated",
        "plane_shared_reference512",
        "independent_planes_final9",
    }
    assert result["accepted"] and not set(result["accepted"]) & sum_only
    assert result["compensated"] == (
        "strategy shared_tree_compensated does not support reduction type max"
    )
    assert set(result["selected"]) <= set(result["accepted"])
    assert result["baseline_equal"] and result["baseline_selected"] == []
    for dynamic in ("False", "True"):
        assert result[f"split_equal_{dynamic}"]
        assert result[f"split_stages_{dynamic}"] == 2
    assert result["fused_equal"]


def test_reductions_not_admitted_name_their_type(tmp_path):
    code = """
from cubecl_torch.inductor import UnsupportedCubeCL
x = torch.randn(8, 129, device="cuda")
cases = {
    "prod": (lambda x: x.prod(-1), x),
    "argmax": (lambda x: x.argmax(-1), x),
    "welford_reduce": (lambda x: x.var(-1), x),
    "any": (lambda x: (x > 0).any(-1), x),
    "int64": (lambda x: x.amax(-1), torch.arange(8 * 129, device="cuda").view(8, 129)),
    "int32": (lambda x: x.amax(-1), torch.arange(8 * 129, device="cuda", dtype=torch.int32).view(8, 129)),
}
out = {}
for name, (f, arg) in cases.items():
    torch._dynamo.reset()
    try:
        torch.compile(f, backend=ci.compile_backend)(arg)
        out[name] = "accepted"
    except Exception as error:
        found = error
        while found is not None and not isinstance(found, UnsupportedCubeCL):
            found = getattr(found, "inner_exception", None) or found.__cause__
        out[name] = str(found)
print(json.dumps(out))
"""
    result = run_python(code, tmp_path)
    for name, message in result.items():
        assert name in message, (name, message)


def test_softmax_compiles_with_dependent_max_and_sum(tmp_path):
    """softmax's max feeds its sum: split, short and long unsplit rows (no compensation)."""
    code = """
from cubecl_torch.inductor import tuning
out = {}
for shape in ((64, 1024), (8, 65539), (2048, 8192)):
    for dt in (torch.float32, torch.bfloat16):
        torch._dynamo.reset(); tuning.CACHE.clear()
        x = torch.randn(shape, device="cuda", dtype=dt)
        y = torch.compile(lambda x: torch.softmax(x, -1), backend=ci.compile_backend)(x)
        try:
            torch.testing.assert_close(y, torch.softmax(x, -1))
            out[f"{shape}-{dt}"] = len(tuning.CACHE)
        except AssertionError as error:
            out[f"{shape}-{dt}"] = str(error)[:200]
print(json.dumps(out))
"""
    result = run_python(code, tmp_path)
    assert all(isinstance(v, int) and v >= 1 for v in result.values()), result
