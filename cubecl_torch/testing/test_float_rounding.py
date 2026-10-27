"""Float rounding to integral values (OpenSpec float-extrema-rounding-remainder,
spec float-rounding, bead x3c).

Results against the Inductor+Triton and eager outputs recorded by
rounding_remainder_reference.py, compared bit for bit. Triton's libdevice
flushes float32/bfloat16 subnormal operands; there the reference is eager.
"""

import pytest
import torch

from cubecl_torch.testing import rounding_remainder_reference as ref
from cubecl_torch.testing.fixtures import run_python

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

ROUNDING = [c for c in ref.cases() if c["kind"] in ("rounding", "round_decimals")]


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    """Every rounding case compiled through CubeCL: name -> (inputs, ours, record)."""
    tmp = tmp_path_factory.mktemp("rounding")
    out = tmp / "ours.pt"
    run_python(ref.compile_code([c["name"] for c in ROUNDING], out), tmp)
    ours = torch.load(out)
    data = torch.load(ref.DATA)["cases"]
    rows = {}
    for case in ROUNDING:
        assert ours[case["name"]] is not None, f"{case['name']}: input digest changed"
        rows[case["name"]] = (
            ref.case_inputs(case),
            ours[case["name"]][0],
            data[case["name"]],
        )
    return rows


@pytest.mark.parametrize("op", ref.TO_INT)
@pytest.mark.parametrize("dtype", ref.FLOATS, ids=ref.name)
def test_rounding_is_eager_and_triton_apart_from_flushed_operands(compiled, op, dtype):
    (x,), ours, record = compiled[f"{op}-{ref.name(dtype)}"]
    assert ref.differences(ours, record["eager"][0])["values"] == []
    against_triton = set(ref.differences(ours, record["output"][0])["values"])
    assert against_triton <= ref.indices(ref.flushed(dtype, x))
    if op in ("floor", "ceil") and dtype != torch.float16:
        # floor(-2^-149) is -0.0 in Triton, -1 here and in eager.
        assert against_triton


def test_round_with_decimals_matches_triton(compiled):
    """Inductor's decomposition round(x * 100) * 0.01, as Triton runs it
    (eager rounds differently)."""
    _, ours, record = compiled["round_decimals-float32"]
    assert ref.differences(ours, record["output"][0]) == {"values": [], "nan_bits": []}


def test_constant_ties_fold_to_even(tmp_path):
    """round of a constant tie folds in CubeCL to the executed value: no rint
    call remains in the generated CUDA."""
    code = r"""
import re
from pathlib import Path
x = torch.zeros(1000, device="cuda")
out = {}
for c in (0.5, 1.5, 2.5, -2.5):
    torch._dynamo.reset()
    fn = lambda x, c=c: x + torch.round(torch.full_like(x, c))
    out[repr(c)] = torch.compile(fn, backend=ci.compile_backend)(x)[0].item()
cache = Path(os.environ["CUBECL_TORCH_CACHE_DIR"])
rint = lambda: sum(bool(re.search(r"\brint\(", s.read_text())) for s in cache.rglob("kernel.cu"))
out["kernels"] = len(list(cache.rglob("kernel.cu")))
out["rint"] = rint()
# Control: an unfolded round does print rint.
torch._dynamo.reset()
torch.compile(torch.round, backend=ci.compile_backend)(x + 0.5)
out["rint after control"] = rint()
print(json.dumps(out))
"""
    result = run_python(code, tmp_path)
    assert result["kernels"] >= 4 and result["rint"] == 0, result
    assert result["rint after control"] >= 1, result
    assert [result[k] for k in ("0.5", "1.5", "2.5", "-2.5")] == [0.0, 2.0, 2.0, -2.0]


def test_rounding_to_integer_dtypes(tmp_path):
    """Ops-level *_to_int (no aten lowering emits them) through test ops: the
    rounding op, then the cast, as eager rounds and casts."""
    code = """
from cubecl_torch.testing import rounding_remainder_reference as ref
ops = ref.register_to_int()
x = torch.tensor([0.5, 1.5, 2.5, -0.5, -2.5, -0.0, 3.7, -3.7, 1e9, -1e9, 7.0], device="cuda")
# An integer operand: the rounding op returns it unchanged (Inductor's aten
# rounding lowerings clone integers, so only *_to_int reaches that branch).
i = torch.tensor([-2**40, -7, 0, 7, 2**40], device="cuda")
out = {}
for op in ref.TO_INT:
    for bits in (64, 32):
        fn = getattr(ops, f"{op}_to_int{bits}")
        for kind, v in (("float", x), ("int", i)):
            torch._dynamo.reset()
            y = torch.compile(lambda v: fn(v), backend=ci.compile_backend)(v)
            out[f"{op}{bits} {kind}"] = [y.tolist(), fn(v).tolist()]
print(json.dumps(out))
"""
    result = run_python(code, tmp_path)
    assert len(result) == 16
    for name, (ours, eager) in result.items():
        assert ours == eager, (name, ours, eager)


def test_rounding_on_every_kernel_path(tmp_path):
    """Fused into a reduction (one generated kernel each) and applied to a
    gather: compiles through CubeCL and equals eager."""
    code = """
from cubecl_torch.testing.fixtures import native_launches
g = torch.Generator(device="cuda").manual_seed(0)
x = torch.randn(64, 300, device="cuda", generator=g) * 10
i = torch.randint(0, 64, (128,), device="cuda", generator=g)
cases = {
    "floor in sum": (lambda x: torch.floor(x * 4).sum(-1), (x,), False),
    "round in amax": (lambda x: torch.round(x).amax(-1), (x,), True),
    "ceil of gather": (lambda x, i: torch.ceil(x[i]), (x, i), True),
    "bfloat16 trunc": (lambda x: torch.trunc(x.bfloat16()), (x,), True),
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
