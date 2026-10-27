"""Float remainder and floor division (OpenSpec float-extrema-rounding-remainder,
spec float-remainder, bead x3c).

Results against the Inductor+Triton and eager outputs recorded by
rounding_remainder_reference.py, compared bit for bit. Where Triton is wrong
(its frem-based remainder; float32/bfloat16 subnormal operands, which its
libdevice flushes) the reference is eager.
"""

import pytest
import torch

from cubecl_torch.testing import rounding_remainder_reference as ref
from cubecl_torch.testing.fixtures import run_python

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Float floor division needs floor: its cases join in group 4.
BINARY = [c for c in ref.cases() if c["kind"] == "binary" and c["op"] != "floordiv"]


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    """Every binary case compiled through CubeCL: name -> (inputs, ours, record)."""
    tmp = tmp_path_factory.mktemp("remainder")
    out = tmp / "ours.pt"
    run_python(ref.compile_code([c["name"] for c in BINARY], out), tmp)
    ours = torch.load(out)
    data = torch.load(ref.DATA)["cases"]
    rows = {}
    for case in BINARY:
        assert ours[case["name"]] is not None, f"{case['name']}: input digest changed"
        rows[case["name"]] = (
            ref.case_inputs(case),
            ours[case["name"]][0],
            data[case["name"]],
        )
    return rows


def flushed(dtype, *values):
    """Elements where Triton's libdevice sees a flushed operand: a float32 or
    bfloat16 subnormal (float16 subnormals are normal in float32 compute)."""
    if dtype == torch.float16:
        return torch.zeros(values[0].shape, dtype=torch.bool)
    tiny = torch.finfo(torch.float32).smallest_normal
    mask = torch.zeros(values[0].shape, dtype=torch.bool)
    for v in values:
        v = v.float()
        mask |= (v != 0) & (v.abs() < tiny)
    return mask


def indices(mask):
    return set(mask.nonzero().flatten().tolist())


@pytest.mark.parametrize("dtype", ref.FLOATS, ids=ref.name)
def test_fmod_is_eager_and_triton_apart_from_flushed_operands(compiled, dtype):
    (a, b), ours, record = compiled[f"fmod-{ref.name(dtype)}"]
    assert ref.differences(ours, record["eager"][0])["values"] == []
    against_triton = set(ref.differences(ours, record["output"][0])["values"])
    assert against_triton <= indices(flushed(dtype, a, b))
    if dtype != torch.float16:
        # fmod(0, 2^-149) is NaN in Triton, 0 here and in eager.
        assert against_triton


@pytest.mark.parametrize("dtype", ref.FLOATS, ids=ref.name)
def test_remainder_follows_eager(compiled, dtype):
    _, ours, record = compiled[f"remainder-{ref.name(dtype)}"]
    assert ref.differences(ours, record["eager"][0])["values"] == []
    # This differs from Triton exactly where Triton differs from eager (its
    # frem remainder): elsewhere the results are Triton's bit for bit.
    against_triton = set(ref.differences(ours, record["output"][0])["values"])
    triton_wrong = set(
        ref.differences(record["output"][0], record["eager"][0])["values"]
    )
    assert against_triton == triton_wrong
    assert against_triton


def test_named_values(tmp_path):
    """The spec's explicit pairs: literal values, and eager's where named."""
    code = """
inf = float("inf")
def run(fn, pairs, dtype=torch.float32):
    a = torch.tensor([p[0] for p in pairs], device="cuda", dtype=dtype)
    b = torch.tensor([p[1] for p in pairs], device="cuda", dtype=dtype)
    torch._dynamo.reset()
    ours = torch.compile(fn, backend=ci.compile_backend)(a, b)
    return [repr(v) for v in ours.tolist()], [repr(v) for v in fn(a, b).tolist()]
out = {
    "fmod large": run(torch.fmod, [(1e10, 3), (-1e10, 3), (3e38, 1.1), (3e38, 3), (1e10, 0.1)]),
    "fmod subnormal": run(torch.fmod, [(0.0, 2**-149), (2**-149, inf)]),
    "remainder triton wrong": run(torch.remainder, [(1e10, 3), (-1e10, 3), (3e38, 1.1), (1e7, 0.1), (-0.0, 2), (-4.0, 2)]),
    "remainder infinite divisor": run(torch.remainder, [(1, inf), (-1, inf), (0, -inf), (3, -inf)]),
    "operator": run(lambda a, b: a % b, [(-5.5, 2), (5.5, -2)]),
}
print(json.dumps(out))
"""
    result = run_python(code, tmp_path)
    for name in (
        "fmod large",
        "fmod subnormal",
        "remainder triton wrong",
        "remainder infinite divisor",
        "operator",
    ):
        ours, eager = result[name]
        assert ours == eager, name
    assert result["fmod large"][0][:2] == ["1.0", "-1.0"]
    assert result["fmod subnormal"][0] == ["0.0", repr(2.0**-149)]
    ours = result["remainder triton wrong"][0]
    # Triton: 256, -253, 9.7e30, -0.049, 0.0 and 0.0.
    assert ours == [
        "1.0",
        "2.0",
        "0.4410529136657715",
        "0.050988391041755676",
        "-0.0",
        "-0.0",
    ]
    assert result["remainder infinite divisor"][0] == ["1.0", "inf", "0.0", "-inf"]


def test_remainders_on_every_kernel_path(tmp_path):
    """Fused into a reduction (one generated kernel each) and applied to a
    gather: compiles through CubeCL and equals eager."""
    code = """
from cubecl_torch.testing.fixtures import native_launches
g = torch.Generator(device="cuda").manual_seed(0)
x = torch.randn(64, 300, device="cuda", generator=g) * 100
w = torch.rand(300, device="cuda", generator=g) * 3 + 0.1
i = torch.randint(0, 64, (128,), device="cuda", generator=g)
cases = {
    "remainder in sum": (lambda x: torch.remainder(x, 1.0).sum(-1), (x,), False),
    "fmod in sum": (lambda x, w: torch.fmod(x, w).sum(-1), (x, w), False),
    "remainder of gather": (lambda x, i: torch.remainder(x[i], 3.0), (x, i), True),
    "bfloat16 remainder": (lambda x, w: torch.remainder(x.bfloat16(), w.bfloat16()), (x, w), True),
    "float16 fmod": (lambda x, w: torch.fmod(x.half(), w.half()), (x, w), True),
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
        if "sum" in name:
            assert launches == 1, (name, result)


def test_fmod_is_cuda_fmod_and_bool_is_refused(tmp_path):
    """Remainders print CUDA fmod, never the inexact floor-modulo formula
    a - floor(a/b)*b; bool operands are refused by the op table, which names
    the op and bool."""
    from cubecl_torch.inductor.admission import OP_RULES

    for op in ("mod", "remainder", "fmod"):
        allowed, _, reason = OP_RULES[op]
        assert torch.bool not in allowed and op in reason and "bool" in reason, op
    code = r"""
import re
from pathlib import Path
x = torch.randn(1000, device="cuda") * 1e6
for fn in (lambda x: torch.fmod(x, 3.0), lambda x: torch.remainder(x, 0.1)):
    torch._dynamo.reset()
    torch.compile(fn, backend=ci.compile_backend)(x)
sources = [s.read_text() for s in Path(os.environ["CUBECL_TORCH_CACHE_DIR"]).rglob("kernel.cu")]
print(json.dumps({
    "kernels": len(sources),
    "fmod": sum(bool(re.search(r"\bfmod\(", s)) for s in sources),
    "floor": sum(bool(re.search(r"\bfloor\(", s)) for s in sources),
}))
"""
    result = run_python(code, tmp_path)
    # Every tuning candidate of both regions calls fmod; none floors.
    assert result["kernels"] >= 2 and result["fmod"] == result["kernels"], result
    assert result["floor"] == 0, result
