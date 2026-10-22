"""Float64 runtime scalar values: Dynamo's lifted Python floats (bead 4z8.6).

Under ``dynamic=True`` Dynamo passes a Python float as a 0-d CPU float64 tensor;
Inductor reads it in GPU kernels as a by-value scalar ("unspec arg"). Here it
is a ``float64`` scalar declaration whose bits ride in an i64 metadata slot and
a ``scalar_value`` op that may only feed a cast. Values are compared bitwise
with PyTorch's own float64 -> compute-dtype conversion.
"""

import copy

import pytest
import torch

from cubecl_torch.compiler import descriptor as descriptors
from cubecl_torch.compiler.errors import MalformedDescriptor
from cubecl_torch.runtime.artifact import ScalarParam
from cubecl_torch.runtime.launch import LaunchContractError
from cubecl_torch.testing.fixtures import native_launches, physical, run_python
from cubecl_torch.testing.test_contracts_descriptors import (
    LINEAR,
    op,
    pointwise,
    reduction,
    rust,
    scalar,
    store,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VALUES = [1e-5, 0.1, -0.0, 0.0, -3.5, float("nan"), float("inf"), -float("inf"),
          1e300, 5e-324, 2.0**-149]  # fmt: skip


def with_float(semantic, *ops_before_store):
    """Append a float64 scalar slot; ``ops_before_store`` go before the store."""
    semantic = copy.deepcopy(semantic)
    slot = len(semantic["scalars"])
    semantic["scalars"].append({"id": slot, "dtype": "float64", "positive": False})
    semantic["ops"][-1:-1] = [dict(o) for o in ops_before_store]
    return semantic, slot


def add_scalar():
    semantic, slot = with_float(
        pointwise([op("x", "load", buffer=0, index=LINEAR), store("y")]),
        op("s", "scalar_value", dtype="float64", slot=1),
        op("c", "cast", ["s"]),
        op("y", "add", ["x", "c"]),
    )
    assert slot == 1
    return semantic


def expected(x, value):
    return x + torch.tensor(value, dtype=torch.float64).to(torch.float32).to(x.device)


def same(actual, wanted):
    assert torch.equal(torch.isnan(actual), torch.isnan(wanted))
    keep = ~torch.isnan(wanted)
    assert torch.equal(actual[keep].view(torch.int32), wanted[keep].view(torch.int32))


@pytest.mark.parametrize(
    ("strategy", "args"),
    [
        ("scalar_pointwise_baseline", ()),
        ("scalar_pointwise", ((64, 1, 1),)),
        ("vector_pointwise", ((256, 1, 1), 4)),
    ],
)
def test_float_scalar_value_pointwise(strategy, args):
    kernel = physical(add_scalar(), strategy, *args)
    x = torch.randn(1031, device="cuda")
    x[:4] = torch.tensor([0.0, -0.0, 1e-30, -1e30])
    for value in VALUES:
        y = torch.empty_like(x)
        kernel((x.numel(), value), x, y)
        same(y, expected(x, value))


def test_float_scalar_value_after_reduction():
    semantic = reduction("float32")
    semantic["scalars"].append({"id": 2, "dtype": "float64", "positive": False})
    *body, cast_out, out = semantic["ops"]
    semantic["ops"] = [
        *body,
        op("s", "scalar_value", dtype="float64", slot=2),
        op("c", "cast", ["s"]),
        op("scaled", "mul", ["sum", "c"]),
        {**cast_out, "args": ["scaled"]},
        out,
    ]
    kernel = physical(semantic, "shared_tree_reduction_baseline")
    x = torch.randn(37, 129, device="cuda")
    for value in (0.5, -3.25, 1e-5):
        y = torch.empty(37, device="cuda")
        kernel((37, 129, value), x.flatten(), y)
        torch.testing.assert_close(
            y, x.sum(-1) * torch.tensor(value).float().item(), rtol=2e-5, atol=2e-5
        )


def test_float_bits_are_passed_not_truncated():
    kernel = physical(add_scalar(), "scalar_pointwise_baseline")
    assert kernel.kernel.artifact.scalars == (
        ScalarParam(),
        ScalarParam(dtype="float64"),
    )
    # The int64 slot keeps int() conversion; the float slot carries IEEE bits.
    assert kernel.kernel.slots((5, 1.0)) == (5, 0x3FF0000000000000)
    assert kernel.kernel.slots((5, -0.0)) == (5, -(1 << 63))
    x = torch.zeros(5, device="cuda")
    y = torch.empty_like(x)
    before = native_launches()
    for bad in ((5, "not a number"), (5, 10**400), (float("inf"), 1.0)):
        with pytest.raises(LaunchContractError, match="scalar mismatch"):
            kernel(bad, x, y)
    assert native_launches() == before


def bad_descriptors():
    index_float = add_scalar()
    index_float["numel"] = scalar(1)
    int_slot = add_scalar()
    int_slot["ops"][1]["slot"] = 0
    direct_math = add_scalar()
    direct_math["ops"][3]["args"] = ["x", "s"]
    positive = add_scalar()
    positive["scalars"][1]["positive"] = True
    int_cast = add_scalar()
    int_cast["ops"][2]["dtype"] = int_cast["ops"][2]["semantic_dtype"] = "int64"
    # A masked region must not carry the float64 value past its cast.
    masked = add_scalar()
    masked["ops"][1:2] = [
        op("i", "index", dtype="int64", index=LINEAR),
        op("five", "constant", dtype="int64", value=5),
        op("m", "lt", ["i", "five"], dtype="bool"),
        op("s", "masked", ["m"], dtype="float64", result="v", other=0.0,
           body=[op("v", "scalar_value", dtype="float64", slot=1)]),
    ]  # fmt: skip
    return {
        "index_float": (index_float, "float scalar"),
        "int_slot": (int_slot, "float64 scalar slot"),
        "direct_math": (direct_math, "explicit float32"),
        "positive": (positive, "scalar schema"),
        "int_cast": (int_cast, "floating compute"),
        "masked": (masked, "mask result dtype float64"),
    }


@pytest.mark.parametrize("name", sorted(bad_descriptors()))
def test_rust_rejects_float_scalar_misuse(tmp_path, name):
    semantic, message = bad_descriptors()[name]
    assert message in rust(tmp_path, semantic)


@pytest.mark.parametrize("name", ["index_float", "int_slot", "positive"])
def test_python_validation_rejects_float_scalar_misuse(name):
    semantic, _ = bad_descriptors()[name]
    with pytest.raises(MalformedDescriptor):
        descriptors.validate(semantic)


def test_float_scalar_parameter_is_never_a_positive_domain():
    with pytest.raises(ValueError):
        ScalarParam(positive=True, dtype="float64")
    with pytest.raises(ValueError):
        ScalarParam(dtype="float32")


FRONTEND = """
import functools
from torch._dynamo.utils import counters
from cubecl_torch.inductor import kernels
torch._dynamo.config.recompile_limit = 64

def compiled(fn, tuning="baseline"):
    return torch.compile(fn, backend=functools.partial(ci.compile_backend, tuning=tuning),
                         dynamic=True)

def same(a, b):
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)

def fused(a, b, c):
    # One FP32 rounding of a*b+c (the FMA the kernel contracts to): FP32 inputs
    # multiply exactly in FP64, and FP64 (53 bits) then FP32 rounding of the
    # sum is the single FP32 rounding for these seeded inputs.
    f32 = lambda v: torch.as_tensor(v, dtype=torch.float32, device="cuda").double()
    return (f32(a) * f32(b) + f32(c)).float()

# Dynamo lifts finite floats; it specializes NaN and inf (a recompile on any backend).
values = [1e-5, 0.1, -0.0, 3.0, 1e300, 5e-324]
result = {}

# Pointwise: one compile serves every value; results equal the singly rounded
# a*b+c bit for bit (the value is rounded to FP32 compute first).
torch.manual_seed(29)
f = lambda x, eps: x * x + eps
x = torch.randn(1031, device="cuda")
for tuning in ("baseline", "first_use"):
    g = compiled(f, tuning)
    before = kernels.COMPILER_INVOCATIONS
    graphs = counters["stats"]["unique_graphs"]
    same(g(x, values[0]), fused(x, x, values[0]))
    first = kernels.COMPILER_INVOCATIONS
    for value in values[1:]:
        same(g(x, value), fused(x, x, value))
    result[tuning + "_graphs"] = counters["stats"]["unique_graphs"] - graphs
    result[tuning + "_first_value_compiled"] = first > before
    # Every later value reuses the first value's kernels.
    result[tuning + "_later_compiles"] = kernels.COMPILER_INVOCATIONS - first
    for value in (float("nan"), float("inf"), -float("inf")):
        same(g(x, value), f(x, value))

# Low-precision storage: the float64 value meets bf16 data in float32 compute.
# Inductor's IR for a tensorified scalar has no bf16 barrier after the multiply
# (even with emulate_precision_casts), so it rounds once, not twice like eager;
# Inductor+Triton gives the same bits (checked 2026-10-20).
h = lambda x, scale, shift: (x * scale + shift).to(torch.bfloat16)
xb = torch.randn(4, 257, device="cuda", dtype=torch.bfloat16)
for scale, shift in ((0.5, 1e-3), (-3.0, 0.1)):
    same(compiled(h)(xb, scale, shift), fused(xb.float(), scale, shift).to(torch.bfloat16))

# The value feeds a reduction kernel's epilogue.
r = lambda x, scale: x.sum(-1) * scale + 1.0
xr = torch.randn(9, 513, device="cuda")
for scale in (0.25, 1e-5):
    torch.testing.assert_close(compiled(r)(xr, scale), r(xr, scale), rtol=2e-5, atol=2e-5)

# Explicit 0-d CPU tensors that are not lifted floats stay rejected.
for name, t in (("int64", torch.tensor(3)), ("float32", torch.tensor(0.5))):
    before = kernels.COMPILER_INVOCATIONS
    try:
        compiled(lambda x, t: x + t)(x, t)
        result[name] = "accepted"
    except Exception as error:
        assert "unsupported" in str(error) or "required" in str(error), str(error)
        assert kernels.COMPILER_INVOCATIONS == before
        result[name] = "rejected"
print(json.dumps(result))
"""


def test_lifted_python_floats_compile_through_cubecl(tmp_path):
    result = run_python(FRONTEND, tmp_path)
    assert result == {
        "baseline_graphs": 1,
        "baseline_first_value_compiled": True,
        "baseline_later_compiles": 0,
        "first_use_graphs": 1,
        "first_use_first_value_compiled": True,
        "first_use_later_compiles": 0,
        "int64": "rejected",
        "float32": "rejected",
    }
