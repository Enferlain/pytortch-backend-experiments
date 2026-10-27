"""Sign and exponent ops (OpenSpec float-extrema-rounding-remainder, spec
float-sign-exponent, bead x3c): sign, signbit, copysign, nextafter, ldexp and
frexp.

Results against the Inductor+Triton and eager outputs recorded by
rounding_remainder_reference.py, compared bit for bit. Where Triton is wrong
(nextafter in float32 rather than the operands' format, flushed subnormals in
ldexp and frexp, frexp(-0.0)) the reference is eager.
"""

import pytest
import torch

from cubecl_torch.testing import rounding_remainder_reference as ref
from cubecl_torch.testing.fixtures import run_python

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

KINDS = ("copysign", "nextafter", "ldexp", "frexp")
CASES = [
    c
    for c in ref.cases()
    if c["kind"] in KINDS or c["name"].split("-")[0] in ("sign", "signbit")
]


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    """Every case compiled through CubeCL: name -> (inputs, ours, record)."""
    tmp = tmp_path_factory.mktemp("sign_exponent")
    out = tmp / "ours.pt"
    run_python(ref.compile_code([c["name"] for c in CASES], out), tmp)
    ours = torch.load(out)
    data = torch.load(ref.DATA)["cases"]
    rows = {}
    for case in CASES:
        assert ours[case["name"]] is not None, f"{case['name']}: input digest changed"
        rows[case["name"]] = (
            ref.case_inputs(case),
            ours[case["name"]],
            data[case["name"]],
        )
    return rows


def against(ours, record, key):
    """Per output, the indices where ours differs from the recorded ``key``."""
    return [
        set(ref.differences(o, r)["values"])
        for o, r in zip(ours, record[key], strict=True)
    ]


@pytest.mark.parametrize("name", [c["name"] for c in CASES if c["kind"] != "copysign"])
def test_equal_to_eager_and_to_triton_where_triton_is_right(compiled, name):
    """Every output equals eager bit for bit, and differs from Triton exactly
    where Triton differs from eager."""
    _, ours, record = compiled[name]
    assert against(ours, record, "eager") == [set()] * len(ours)
    triton_wrong = [
        set(ref.differences(t, e)["values"])
        for t, e in zip(record["output"], record["eager"], strict=True)
    ]
    assert against(ours, record, "output") == triton_wrong
    if name.startswith(("nextafter", "frexp")) or name in (
        "ldexp-float32-int32",
        "ldexp-bfloat16-int64",
    ):
        assert any(triton_wrong), name  # recorded: Triton is wrong here


@pytest.mark.parametrize("dtype", [*ref.FLOATS, torch.int32], ids=ref.name)
def test_copysign(compiled, dtype):
    """Every non-NaN result is eager's and Triton's bit for bit, except for a
    float16 negative-NaN sign source: CUDA's half-to-float load conversion
    makes every float16 NaN the positive canonical one, so that sign is gone
    before the kernel reads it (eager and Triton read the 16 bits). NaN results
    are NaN; their sign survives the store in float32 only."""
    (x, y), ours, record = compiled[f"copysign-{ref.name(dtype)}"]
    (diff,) = against(ours, record, "eager")
    assert against(ours, record, "output") == [diff]
    if dtype == torch.float16:
        negative_nan = y.isnan() & y.signbit()
        assert diff == ref.indices(negative_nan & ~x.isnan()) and diff
    else:
        assert diff == set()
    (out,) = ours
    if dtype == torch.float32:
        assert torch.equal(out.isnan() & out.signbit(), x.isnan() & y.signbit())
    elif dtype != torch.int32:
        assert torch.equal(out.isnan(), x.isnan())


def test_named_values(tmp_path):
    """The spec's explicit cases: nextafter where Triton is wrong, frexp with
    one output used or the exponent widened, nextafter of a value computed in
    the kernel, and ldexp with a float exponent (Inductor's fallback needs
    pow, the transcendental family)."""
    code = """
def compile(fn, *args):
    torch._dynamo.reset()
    return torch.compile(fn, backend=ci.compile_backend)(*args)
out = {}
f32 = torch.tensor([0.0], device="cuda")
out["nextafter float32 (0, 1)"] = compile(torch.nextafter, f32, f32 + 1).item()
for dt in ("float16", "bfloat16"):
    one = torch.ones(1, device="cuda", dtype=getattr(torch, dt))
    out[f"nextafter {dt} (1, 2)"] = compile(torch.nextafter, one, one * 2).item()
x = torch.cat([torch.randn(200, device="cuda"), torch.tensor([0.0, -0.0, 1e-40, float("inf")], device="cuda")])
both = lambda x: (torch.frexp(x)[0], torch.frexp(x)[1].to(torch.int64))
m, e = compile(both, x)
want_m, want_e = torch.frexp(x)
out["frexp both"] = torch.equal(m, want_m) and torch.equal(e, want_e.to(torch.int64))
out["frexp mantissa"] = torch.equal(compile(lambda x: torch.frexp(x)[0] * 2, x), want_m * 2)
out["frexp exponent"] = torch.equal(
    compile(lambda x: torch.frexp(x)[1].to(torch.float32), x), want_e.float())
h = torch.randn(300, device="cuda", dtype=torch.float16)
computed = lambda h: torch.nextafter(h * 3, torch.ones_like(h))
out["nextafter computed"] = torch.equal(compile(computed, h), computed(h))
# A float16 product rounded to its format before frexp, as eager stores it.
for dt in (torch.float16, torch.bfloat16):
    v = torch.linspace(0.3, 0.34, 300, device="cuda", dtype=dt)
    product = lambda v: tuple(torch.frexp(v * 3))[0]
    exponent = lambda v: torch.frexp(v * 3)[1].to(torch.int64)
    out[f"frexp computed {dt}"] = torch.equal(compile(product, v), product(v)) and torch.equal(
        compile(exponent, v), exponent(v))
try:
    n = torch.randint(-5, 5, (8,), device="cuda").float()
    compile(torch.ldexp, x[:8], n)
    out["ldexp float exponent"] = "accepted"
except Exception as error:
    out["ldexp float exponent"] = [l for l in str(error).splitlines() if "CubeCL" in l][0]
print(json.dumps(out))
"""
    result = run_python(code, tmp_path)
    assert result["nextafter float32 (0, 1)"] == 2.0**-149
    assert result["nextafter float16 (1, 2)"] == 1 + 2.0**-10
    assert result["nextafter bfloat16 (1, 2)"] == 1 + 2.0**-7
    for name in (
        "frexp both",
        "frexp mantissa",
        "frexp exponent",
        "nextafter computed",
    ):
        assert result[name] is True, (name, result)
    assert "operation pow unsupported" in result["ldexp float exponent"], result


def test_sign_and_exponent_ops_on_every_kernel_path(tmp_path):
    """Fused into a reduction (one generated kernel each) and applied to a
    gather: compiles through CubeCL and equals eager."""
    code = """
from cubecl_torch.testing.fixtures import native_launches
g = torch.Generator(device="cuda").manual_seed(0)
x = torch.randn(64, 300, device="cuda", generator=g)
w = torch.randn(300, device="cuda", generator=g)
i = torch.randint(0, 64, (128,), device="cuda", generator=g)
cases = {
    "copysign in sum": (lambda x, w: torch.copysign(x, w).sum(-1), (x, w), False),
    # The exponent computed in the kernel: reductions read float storage only.
    "ldexp in amax": (lambda x: torch.ldexp(x, (x * 8).to(torch.int32)).amax(-1), (x,), True),
    "sign of gather": (lambda x, i: torch.sign(x[i]), (x, i), True),
    "nextafter of gather": (lambda x, i: torch.nextafter(x[i], x[i] * 2), (x, i), True),
}
out = {}
for name, (fn, args, exact) in cases.items():
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend=ci.compile_backend)
    compiled(*args)
    before = native_launches()
    y = compiled(*args)
    launches = native_launches() - before
    e = fn(*args)
    same = torch.equal(y, e) if exact else torch.allclose(y, e, rtol=1e-5, atol=1e-4)
    out[name] = [bool(same), launches]
print(json.dumps(out))
"""
    result = run_python(code, tmp_path)
    for name, (same, launches) in result.items():
        assert same, (name, result)
        if name.endswith(("sum", "amax")):
            assert launches == 1, (name, result)
