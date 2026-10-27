"""Inductor+Triton reference outputs for the extrema family (float-remainder,
float-rounding, float-sign-exponent and the float-extrema isinf delta).

Covers fmod, remainder and float floor division over every sign combination,
zero, infinite and NaN operands, subnormals and large quotients; floor, ceil,
trunc and round (also with decimals) over ties and special values; isinf, sign,
signbit, copysign, nextafter, ldexp and frexp. Inputs are rebuilt from seeds
with the CPU generator and from fixed special values, so only their SHA-256 and
the outputs (Triton's and eager's, each a list: frexp has two) are stored.

Where the specs say Triton is wrong (float remainder, nextafter, subnormal
ldexp and frexp), eager is the reference; the generator checks eager in float64
where an exact answer is cheap to state:
- fmod of float32/16/bf16 operands is exact in float64 and representable in the
  operand dtype, so eager's fmod must equal the float64 fmod;
- ldexp is ``x * 2**n`` exactly in float64 (n wrapped to int32 as eager does),
  rounded once to the dtype;
- frexp's mantissa and exponent recombine to ``x`` in float64;
- a nextafter step from a nonzero finite ``x`` moves its bits by one.

Regenerate the data file with the Triton environment from the repository root::

    /home/imi/Projects/sd-scripts/.venv-wsl/bin/python -m cubecl_torch.testing.rounding_remainder_reference

Imports only torch: the CubeCL backend never loads here.
"""

import hashlib
import io
import itertools
import sys
from pathlib import Path

import torch

DATA = Path(__file__).resolve().parent / "data" / "rounding_remainder_triton.pt"
FLOATS = (torch.float32, torch.float16, torch.bfloat16)
N = 1024
INF, NAN = float("inf"), float("nan")
BITS = {
    torch.float32: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
}


def name(dtype):
    return str(dtype).removeprefix("torch.")


def generator(seed):
    return torch.Generator().manual_seed(seed)


def bits(t):
    return t.view(BITS[t.dtype])


def signed_nans(dtype):
    """Quiet NaN of both signs as this dtype's bits: converting a float32 NaN
    to bfloat16 on the CPU does not keep its sign."""
    raw = {
        torch.float32: [0x7FC00000, -0x400000],
        torch.float16: [0x7E00, -0x200],
        torch.bfloat16: [0x7FC0, -0x40],
    }[dtype]
    return torch.tensor(raw, dtype=BITS[dtype]).view(dtype)


def subnormal(dtype):
    info = torch.finfo(dtype)
    return info.smallest_normal * info.eps


def large(dtype):
    """A dividend far beyond the dtype's integer range, and the largest scale."""
    top = torch.finfo(dtype).max
    return min(1e10, top * 0.5), top * 0.9


def values(dtype, xs):
    return torch.tensor(xs, dtype=torch.float32).to(dtype)


def pad(head, tail):
    return torch.cat([head, tail[: N - len(head)]])


def scaled_randn(count, seed, low, high):
    """randn times 10**u, u uniform in [low, high): magnitudes over many decades."""
    g = generator(seed)
    exponent = torch.rand(count, generator=g) * (high - low) + low
    return torch.randn(count, generator=g) * 10**exponent


def float_specials(dtype):
    info = torch.finfo(dtype)
    sub = subnormal(dtype)
    ordinary = [1.0, -1.0, 0.5, -2.5, 3.0, 1e-3, -7.25]
    extremes = [info.smallest_normal, info.max, -info.max]
    head = values(dtype, [-0.0, 0.0, INF, -INF, sub, -sub, *extremes, *ordinary])
    return torch.cat([signed_nans(dtype), head])


# Remainder and floor division


def binary_inputs(dtype, seed):
    """Every (dividend, divisor) pair of two special lists, then random pairs
    whose quotients span from below 1 to beyond the dtype's integer range."""
    sub = subnormal(dtype)
    big, huge = large(dtype)
    dividends = [-0.0, 0.0, INF, -INF, NAN, sub, -sub, 1.0, -1.0, 5.5, -5.5]
    dividends += [-4.0, 3.0, 7.0, big, -big, huge, 3e38, 1e7, -1e7, 1e-30, -1e-30, 1e-3]
    divisors = [0.0, -0.0, INF, -INF, NAN, sub, -sub, 3.0, -3.0, 0.1, -0.1]
    divisors += [1.1, 2.0, -0.7, 1e-3]
    a, b = zip(*itertools.product(dividends, divisors), strict=True)
    head_a, head_b = values(dtype, list(a)), values(dtype, list(b))
    top = 10 if dtype != torch.float16 else 4
    tail_a = scaled_randn(N, seed, -2, top).to(dtype)
    tail_b = scaled_randn(N, seed + 1, -2, 2).to(dtype)
    return [pad(head_a, tail_a), pad(head_b, tail_b)]


def check_fmod(a, b, eager):
    exact = torch.fmod(a.double(), b.double()).to(a.dtype)
    nan = exact.isnan()
    assert torch.equal(bits(eager[~nan]), bits(exact[~nan])), "eager fmod is not exact"
    assert eager[nan].isnan().all()


# Rounding


def rounding_inputs(dtype, seed):
    """Ties, values next to integers, and values at and past the dtype's last
    fractional bit (2**23 for float32, 2**10 for float16, 2**7 for bfloat16)."""
    mantissa = {torch.float32: 23, torch.float16: 10, torch.bfloat16: 7}[dtype]
    edge = 2.0**mantissa
    ties = [0.4, 0.5, 1.5, 2.5, 3.5, 0.49999997, 1.0 - 2**-mantissa, 1.0 + 2**-mantissa]
    ties += [edge - 0.5, edge, edge + 1, edge * 4 + 4]
    head = torch.cat([float_specials(dtype), values(dtype, ties + [-t for t in ties])])
    g = generator(seed)
    halves = torch.randint(-4000, 4000, (N // 2,), generator=g) / 2
    tail = torch.cat([halves, scaled_randn(N // 2, seed + 1, -2, 4)]).to(dtype)
    return [pad(head, tail)]


# Sign and exponent


def unary_specials_inputs(dtype, seed):
    return [pad(float_specials(dtype), scaled_randn(N, seed, -3, 3).to(dtype))]


def pair_inputs(dtype, seed):
    """Every pair of the specials list, then random pairs."""
    specials = float_specials(dtype)
    a, b = zip(*itertools.product(specials, specials), strict=True)
    head_a, head_b = torch.stack(a), torch.stack(b)
    tail_a = scaled_randn(N, seed, -3, 3).to(dtype)
    tail_b = scaled_randn(N, seed + 1, -3, 3).to(dtype)
    return [pad(head_a, tail_a), pad(head_b, tail_b)]


def copysign_int_inputs(seed):
    g = generator(seed)
    head = torch.tensor(
        [[-5, 1], [0, -1], [7, -1], [0, 0], [-(2**31), 1], [2**31 - 1, -3]]
    )
    a = torch.randint(-1000, 1000, (N,), generator=g)
    b = torch.randint(-3, 3, (N,), generator=g)
    return [
        pad(head[:, 0], a).to(torch.int32),
        pad(head[:, 1], b).to(torch.int32),
    ]


def nextafter_inputs(dtype, seed):
    """Every pair of the step specials, both directions, equal operands
    included; then random ``x`` with ``y`` above, below or equal."""
    info = torch.finfo(dtype)
    sub = subnormal(dtype)
    steps = [0.0, -0.0, sub, -sub, info.smallest_normal, 1.0, -1.0, 2.0]
    steps += [info.max, -info.max, INF, -INF, NAN]
    a, b = zip(*itertools.product(steps, steps), strict=True)
    g = generator(seed)
    x = scaled_randn(N, seed, -3, 3)
    direction = torch.randint(-1, 2, (N,), generator=g).float()
    y = x + direction * x.abs().clamp(min=1)
    return [
        pad(values(dtype, list(a)), x.to(dtype)),
        pad(values(dtype, list(b)), y.to(dtype)),
    ]


def check_nextafter(x, y, eager):
    """A step from a nonzero finite x toward a different y moves the bits by one."""
    moved = (x != 0) & x.isfinite() & (x != y) & ~y.isnan()
    step = (bits(eager).long() - bits(x).long()).abs()
    assert (step[moved] == 1).all(), "eager nextafter is not one step"


def ldexp_inputs(dtype, exponent_dtype, seed):
    """Special and random mantissas with exponents across the dtype's range,
    beyond it, and beyond int32 (wrapped, as eager); plus small mantissas
    scaled into the subnormal range."""
    xs = float_specials(dtype)
    ns = [0, 1, -1, 16, 128, 200, -149, -160, 300, -300]
    if exponent_dtype == torch.int64:
        ns += [2**32 + 3, -(2**40), 2**31, -(2**31) - 5]
    grid_x, grid_n = zip(*itertools.product(xs, ns), strict=True)
    # From just above the smallest normal down past the smallest subnormal.
    info = torch.finfo(dtype)
    lowest = round(torch.tensor(info.smallest_normal).log2().item())
    mantissa = round(-torch.tensor(info.eps).log2().item())
    sub_x, sub_n = zip(
        *itertools.product(
            [1.0, 1.5, 1.75, 3.0, 1.0078125],
            range(lowest + 6, lowest - mantissa - 3, -1),
        ),
        strict=True,
    )
    head_x = torch.cat([torch.stack(grid_x), values(dtype, list(sub_x))])
    head_n = torch.tensor(list(grid_n) + list(sub_n), dtype=torch.int64)
    g = generator(seed)
    tail_x = scaled_randn(N, seed, -3, 3).to(dtype)
    tail_n = torch.randint(-300, 301, (N,), generator=g)
    return [pad(head_x, tail_x), pad(head_n, tail_n).to(exponent_dtype)]


def check_ldexp(x, n, eager):
    wrapped = n.to(torch.int64).to(torch.int32).double()
    exact = (x.double() * torch.pow(2.0, wrapped)).to(x.dtype)
    nan = exact.isnan()
    assert torch.equal(bits(eager[~nan]), bits(exact[~nan])), "eager ldexp is not exact"
    # 0 * 2**n and inf * 2**-n are NaN in float64 for wrapped n near 2**31.
    kept = (x == 0) | x.isinf()
    assert torch.equal(bits(eager[kept]), bits(x[kept])), "ldexp changed 0 or inf"


def frexp_inputs(dtype, seed):
    info = torch.finfo(dtype)
    g = generator(seed)
    subnormals = (torch.rand(64, generator=g) * info.smallest_normal).to(dtype)
    powers = values(dtype, [2.0**k for k in (-10, -1, 0, 1, 7, 15)])
    head = torch.cat([float_specials(dtype), subnormals, -subnormals, powers])
    return [pad(head, scaled_randn(N, seed + 1, -30, 30).to(dtype))]


def check_frexp(x, eager):
    mantissa, exponent = eager
    finite = x.isfinite() & (x != 0)
    recombined = mantissa.double() * torch.pow(2.0, exponent.double())
    assert torch.equal(recombined[finite], x.double()[finite]), "eager frexp loses x"
    magnitude = mantissa[finite].double().abs()
    assert ((magnitude >= 0.5) & (magnitude < 1)).all()


UNARY = {
    "floor": torch.floor,
    "ceil": torch.ceil,
    "trunc": torch.trunc,
    "round": torch.round,
    "isinf": torch.isinf,
    "sign": torch.sign,
    "signbit": torch.signbit,
}
BINARY = {
    "fmod": torch.fmod,
    "remainder": torch.remainder,
    "floordiv": lambda a, b: a // b,
}


def cases():
    seed = 0

    def case(case_name, kind, **fields):
        nonlocal seed
        seed += 10
        return {"name": case_name, "kind": kind, "seed": seed, **fields}

    for op in BINARY:
        for dtype in FLOATS:
            yield case(f"{op}-{name(dtype)}", "binary", op=op, dtype=dtype)
    for op in ("floor", "ceil", "trunc", "round"):
        for dtype in FLOATS:
            yield case(f"{op}-{name(dtype)}", "rounding", op=op, dtype=dtype)
    yield case("round_decimals-float32", "round_decimals", dtype=torch.float32)
    for op in ("isinf", "sign", "signbit"):
        for dtype in FLOATS:
            yield case(f"{op}-{name(dtype)}", "unary", op=op, dtype=dtype)
    for dtype in FLOATS:
        yield case(f"copysign-{name(dtype)}", "copysign", dtype=dtype)
    yield case("copysign-int32", "copysign", dtype=torch.int32)
    for dtype in FLOATS:
        yield case(f"nextafter-{name(dtype)}", "nextafter", dtype=dtype)
    for dtype in FLOATS:
        for exponent in (torch.int32, torch.int64):
            yield case(
                f"ldexp-{name(dtype)}-{name(exponent)}",
                "ldexp",
                dtype=dtype,
                exponent=exponent,
            )
    for dtype in FLOATS:
        yield case(f"frexp-{name(dtype)}", "frexp", dtype=dtype)


def function(case):
    kind = case["kind"]
    if kind == "binary":
        return BINARY[case["op"]]
    if kind in ("rounding", "unary"):
        return UNARY[case["op"]]
    if kind == "round_decimals":
        return lambda x: torch.round(x, decimals=2)
    if kind == "copysign":
        return torch.copysign
    if kind == "nextafter":
        return torch.nextafter
    if kind == "ldexp":
        return torch.ldexp
    return torch.frexp


def case_inputs(case):
    kind, seed, dtype = case["kind"], case["seed"], case["dtype"]
    if kind == "binary":
        return binary_inputs(dtype, seed)
    if kind == "rounding":
        return rounding_inputs(dtype, seed)
    if kind == "round_decimals":
        return [scaled_randn(N, seed, -2, 3)]
    if kind == "unary":
        return unary_specials_inputs(dtype, seed)
    if kind == "copysign":
        if dtype == torch.int32:
            return copysign_int_inputs(seed)
        return pair_inputs(dtype, seed)
    if kind == "nextafter":
        return nextafter_inputs(dtype, seed)
    if kind == "ldexp":
        return ldexp_inputs(dtype, case["exponent"], seed)
    return frexp_inputs(dtype, seed)


def outputs(result):
    """Every output as a list of CPU tensors: frexp returns two."""
    if isinstance(result, torch.Tensor):
        return [result.cpu()]
    return [r.cpu() for r in result]


def check_eager(case, args, eager):
    """The float64 checks of the module docstring, on eager's outputs."""
    kind = case["kind"]
    if kind == "binary" and case["op"] == "fmod":
        check_fmod(*args, eager[0])
    elif kind == "nextafter":
        check_nextafter(*args, eager[0])
    elif kind == "ldexp":
        check_ldexp(*args, eager[0])
    elif kind == "frexp":
        check_frexp(args[0], eager)


def digest(t):
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


TO_INT = ("floor", "ceil", "trunc", "round")


def register_to_int():
    """Test ops ``cubecl_round_test.<op>_to_int<bits>`` lowered to Inductor's
    ``ops.<op>_to_int``, which no aten lowering of the pinned Inductor emits
    (FloorToInt and friends in index expressions are printed). An int32 target
    is widened to int64 storage; int32 is not a storage dtype here."""
    if hasattr(torch.ops, "cubecl_round_test") and hasattr(
        torch.ops.cubecl_round_test, "floor_to_int64"
    ):
        return torch.ops.cubecl_round_test
    from torch._inductor import lowering
    from torch._inductor.virtualized import ops

    lib = torch.library.Library("cubecl_round_test", "DEF")
    for op in TO_INT:
        for bits, target in ((64, torch.int64), (32, torch.int32)):
            name = f"{op}_to_int{bits}"
            lib.define(f"{name}(Tensor a) -> Tensor")
            lib.impl(
                name,
                lambda a, op=op, target=target: (
                    getattr(torch, op)(a).to(target).to(torch.int64)
                ),
                "CUDA",
            )
            torch.library.register_fake(f"cubecl_round_test::{name}")(
                lambda a: torch.empty_like(a, dtype=torch.int64)
            )

            def lower(x, op=op, target=target):
                value = getattr(ops, f"{op}_to_int")(x, target)
                return ops.to_dtype(value, torch.int64)

            lowering.register_lowering(getattr(torch.ops.cubecl_round_test, name))(
                lowering.make_pointwise(lower, override_return_dtype=torch.int64)
            )
    register_to_int.lib = lib  # keep the library alive
    return torch.ops.cubecl_round_test


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


def compile_code(names, out):
    """Code for a backend subprocess (fixtures.run_python): compile the named
    cases with CubeCL on the recorded inputs and save every output list, by
    name, to ``out``; a case whose inputs no longer match the recorded digest
    is saved as None."""
    return f"""
from cubecl_torch.testing import rounding_remainder_reference as ref
data = torch.load(ref.DATA)["cases"]
ours = {{}}
for case in ref.cases():
    if case["name"] not in {sorted(names)!r}:
        continue
    torch._dynamo.reset()
    args = ref.case_inputs(case)
    if [ref.digest(a) for a in args] != data[case["name"]]["input_sha256"]:
        ours[case["name"]] = None
        continue
    fn = torch.compile(ref.function(case), backend=ci.compile_backend)
    ours[case["name"]] = ref.outputs(fn(*(a.cuda() for a in args)))
torch.save(ours, {str(out)!r})
print(json.dumps(sorted(ours)))
"""


def differences(actual, wanted):
    """Indices where ``actual`` and ``wanted`` differ bit for bit, apart from
    elements NaN in both (whose differing bits are listed as ``nan_bits``)."""
    width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    a = actual.view(width[actual.element_size()])
    w = wanted.to(actual.dtype).view(width[actual.element_size()])
    nan = (
        actual.isnan() & wanted.isnan()
        if actual.is_floating_point()
        else torch.zeros_like(a, dtype=torch.bool)
    )
    return {
        "values": ((a != w) & ~nan).nonzero().flatten().tolist(),
        "nan_bits": ((a != w) & nan).nonzero().flatten().tolist(),
    }


def generate(path=DATA):
    import triton  # noqa: F401  the reference must come from Inductor+Triton

    # Inductor+Triton faulted on cuda:1 tensors without the current device set.
    torch.cuda.set_device(0)
    records = {}
    for case in cases():
        torch._dynamo.reset()
        args = case_inputs(case)
        assert all(len(a) == N for a in args), case["name"]
        compiled = torch.compile(function(case))
        eager = outputs(function(case)(*(a.cuda() for a in args)))
        check_eager(case, args, eager)
        records[case["name"]] = {
            "input_sha256": [digest(a) for a in args],
            "output": outputs(compiled(*(a.cuda() for a in args))),
            "eager": eager,
        }
    payload = {"torch": str(torch.__version__), "cases": records}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path.write_bytes(buffer.getvalue())
    return path


if __name__ == "__main__":
    sys.stdout.write(f"wrote {generate()}\n")
