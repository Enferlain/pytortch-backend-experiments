"""Inductor+Triton reference outputs for basic float arithmetic (float-arithmetic).

Covers abs, square and relu over special values, and the fused forms Inductor
lowers to fma, mul_rn and div_rn on CUDA: add/sub with alpha, addcmul, addcdiv
(also next to a plain division) and both lerp forms. Inputs are rebuilt from
seeds with the CPU generator and from fixed special values, so only their
SHA-256 and the outputs (Triton's and eager's) are stored.

For float32, rows that expose rounding are constructed by cancellation and
checked in float64 when built (exact there: the products have at most 48 bits):
- single rounding: the addend cancels the rounded product, so a singly rounded
  fma returns the product's rounding error where two roundings return 0;
- rounded product: the addcmul products are inexact in float32, so leaving the
  ``mul_rn`` product unrounded changes the result.
float16 and bfloat16 cases use the same construction but promise no such rows:
their products are exact in float32, and rounding the result to 16 bits mostly
hides one float32 rounding more or less.

Regenerate the data file with the Triton environment from the repository root::

    /home/imi/Projects/sd-scripts/.venv-wsl/bin/python -m cubecl_torch.testing.float_arithmetic_reference

Imports only torch: the CubeCL backend never loads here.
"""

import hashlib
import io
import sys
from pathlib import Path

import torch

DATA = Path(__file__).resolve().parent / "data" / "float_arithmetic_triton.pt"
FLOATS = (torch.float32, torch.float16, torch.bfloat16)
INTS = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
N = 1024
ALPHA = 0.3
LERP_WEIGHTS = (0.3, 0.7, -0.6)


def name(dtype):
    return str(dtype).removeprefix("torch.")


def generator(seed):
    return torch.Generator().manual_seed(seed)


def float_specials(dtype):
    """NaN of both signs, signed zeros, infinities, the smallest subnormal and
    the largest finite value, then ordinary values."""
    info = torch.finfo(dtype)
    subnormal = info.smallest_normal * info.eps
    # Quiet NaN of both signs as this dtype's bits: converting a float32 NaN to
    # bfloat16 on the CPU does not keep its sign.
    nan_bits = {
        torch.float32: ([0x7FC00000, -0x400000], torch.int32),
        torch.float16: ([0x7E00, -0x200], torch.int16),
        torch.bfloat16: ([0x7FC0, -0x40], torch.int16),
    }
    bits, carrier = nan_bits[dtype]
    nan = torch.tensor(bits, dtype=carrier).view(dtype)
    root = info.max**0.5
    values = torch.tensor(
        [
            *(-0.0, 0.0, float("inf"), -float("inf"), subnormal, -subnormal),
            *(info.max, -info.max, 1.0, -1.0, 0.5, -2.5, 3.0, 1e-3, -7.25),
            # Squares that overflow to inf in this dtype.
            *(root * 2, -root * 3),
        ],
        dtype=torch.float32,
    )
    return torch.cat([nan, values.to(dtype)])


def unary_float_inputs(dtype, seed):
    head = float_specials(dtype)
    tail = (torch.randn(N - len(head), generator=generator(seed)) * 4).to(dtype)
    return [torch.cat([head, tail])]


def unary_int_inputs(dtype, seed):
    info = torch.iinfo(dtype)
    head = [info.min, info.max, 0, 1, -1, info.min + 1, 5, -5]
    # Squares that wrap in int32 and int64.
    head += [50000, -46341, 2**33 + 7, -(2**32) - 1]
    head = [v for v in head if info.min <= v <= info.max]
    low, high = max(info.min, -1000), min(info.max, 1000)
    tail = torch.randint(low, high + 1, (N - len(head),), generator=generator(seed))
    return [torch.cat([torch.tensor(head, dtype=torch.int64), tail]).to(dtype)]


def f32(x):
    return x.to(torch.float32)


def cancelling(a, b, value):
    """Addends ``s`` for which ``fma(value, a*b, s)`` (product rounded) is the
    rounding error of ``value * fl(a*b)``: zero when rounded twice."""
    product = f32(a) * f32(b)
    if value == 1:
        return -product
    return -(product * torch.tensor(value, dtype=torch.float32))


def sensitive(a, b, s, value):
    """Rows that distinguish one rounding from two, with a product inexact in
    float32. Exact in float64: the products have at most 48 bits and ``s``
    cancels their leading bits."""
    exact = a.double() * b.double()
    rounded = (f32(a) * f32(b)).double()
    if value == 1:
        once = exact + s.double()
    else:
        once = torch.tensor(value, dtype=torch.float32).double() * rounded + s.double()
    return (once != 0) & (exact != rounded)


def sensitive_rows(count, seed, value, factor=None):
    """float32 ``(a, b, s)`` rows, all ``sensitive``; ``b`` is ``factor`` when given."""
    g = generator(seed)
    parts = []
    while sum(len(p[0]) for p in parts) < count:
        a = torch.randn(count, generator=g)
        b = (
            torch.randn(count, generator=g)
            if factor is None
            else torch.full((count,), factor)
        )
        s = cancelling(a, b, value)
        keep = sensitive(a, b, s, value)
        parts.append((a[keep], b[keep], s[keep]))
    a, b, s = (torch.cat(column)[:count] for column in zip(*parts, strict=True))
    assert sensitive(a, b, s, value).all()
    return a, b, s


def alpha_inputs(dtype, seed):
    """``add(c, a, alpha)`` is ``fma(a, alpha, c)``: half the rows cancel the
    rounded ``a * alpha`` against ``c``. ``sub`` negates ``alpha``, so there
    those rows add instead of cancelling; one rounding still differs from two."""
    half = N // 2
    a, _, c = sensitive_rows(half, seed, 1, factor=ALPHA)
    g = generator(seed + 1)
    tail_c, tail_a = (
        torch.randn(N - half, generator=g),
        torch.randn(N - half, generator=g),
    )
    return [torch.cat([c, tail_c]).to(dtype), torch.cat([a, tail_a]).to(dtype)]


def addcmul_inputs(dtype, value, seed):
    half = N // 2
    a, b, s = sensitive_rows(half, seed, value)
    g = generator(seed + 1)
    tail = [torch.randn(N - half, generator=g) for _ in range(3)]
    return [
        torch.cat([s, tail[0]]).to(dtype),
        torch.cat([a, tail[1]]).to(dtype),
        torch.cat([b, tail[2]]).to(dtype),
    ]


def divisor_specials(dtype):
    """(dividend, divisor) pairs: zero, infinite and NaN divisors, a NaN
    dividend, and quotients that are subnormal in ``dtype``."""
    inf, nan = float("inf"), float("nan")
    dividends = [1.0, -1.0, 0.0, -0.0, 1.0, -1.0, inf, 1.0, 0.0, nan]
    divisors = [0.0, 0.0, 0.0, 0.0, inf, -inf, inf, nan, nan, 2.0]
    if dtype == torch.float16:
        dividends += [1e-4, -3e-3, 6e-5]
        divisors += [7.0, 500.0, 3.0]
    else:
        dividends += [3e-30, -1e-35, 1e-38]
        divisors += [1e10, 7e3, 3.0]
    num, den = torch.tensor(dividends).to(dtype), torch.tensor(divisors).to(dtype)
    quotient = (f32(num[-3:]) / f32(den[-3:])).to(dtype)
    assert (quotient.abs() < torch.finfo(dtype).smallest_normal).all()
    assert (quotient != 0).all()
    return num, den


def addcdiv_inputs(dtype, value, seed):
    """The special rows add to -0.0, so their quotient is the result. For
    ``value`` other than 1, half the ordinary rows cancel the rounded
    ``value * (a / b)`` against ``s`` (checked in float64, as ``sensitive``)."""
    num, den = divisor_specials(dtype)
    g = generator(seed)
    count = N - len(num)
    a = torch.randn(count, generator=g)
    b = (
        torch.randn(count, generator=g)
        + torch.sign(torch.randn(count, generator=g)) * 0.25
    )
    s = torch.randn(count, generator=g)
    if value != 1:
        v = torch.tensor(value, dtype=torch.float32)
        q = a / b
        cancel = -(q * v)
        once = v.double() * q.double() + cancel.double()
        rows = (once != 0).nonzero().flatten()[: count // 2]
        s[rows] = cancel[rows]
        assert len(rows) == count // 2
    s_head = torch.full((len(num),), -0.0, dtype=dtype)
    return [
        torch.cat([s_head, s.to(dtype)]),
        torch.cat([num, a.to(dtype)]),
        torch.cat([den, b.to(dtype)]),
    ]


def division_mix_inputs(seed):
    s, a, b = addcdiv_inputs(torch.float32, ALPHA, seed)
    g = generator(seed + 1)
    c = torch.randn(N, generator=g)
    d = torch.randn(N, generator=g) + 0.25
    return [s, a, b, c, d]


def lerp_inputs(dtype, seed, tensor_weight):
    g = generator(seed)
    start = torch.randn(N, generator=g) * 3
    end = torch.randn(N, generator=g) * 3
    inputs = [start.to(dtype), end.to(dtype)]
    if tensor_weight:
        # Both sides of +-0.5, with the boundary values themselves.
        w = torch.rand(N, generator=g) * 3 - 1.5
        w[:6] = torch.tensor([0.5, -0.5, 0.49999997, -0.49999997, 0.0, 1.0])
        inputs.append(w.to(dtype))
    return inputs


UNARY = {
    "abs": torch.abs,
    "square": torch.square,
    "relu": torch.relu,
}


def cases():
    seed = 0
    for op in UNARY:
        for dtype in FLOATS:
            seed += 1
            yield {
                "name": f"{op}-{name(dtype)}",
                "kind": "unary",
                "op": op,
                "dtype": dtype,
                "seed": seed,
            }
    for op, dtypes in (
        ("abs", INTS),
        ("square", (torch.int32, torch.int64)),
        ("relu", (torch.int32,)),
    ):
        for dtype in dtypes:
            seed += 1
            yield {
                "name": f"{op}-{name(dtype)}",
                "kind": "unary",
                "op": op,
                "dtype": dtype,
                "seed": seed,
            }
    for op in ("add", "sub"):
        for dtype in FLOATS:
            seed += 1
            yield {
                "name": f"{op}_alpha-{name(dtype)}",
                "kind": "alpha",
                "op": op,
                "dtype": dtype,
                "seed": seed,
            }
    for op in ("addcmul", "addcdiv"):
        for value in (1, ALPHA):
            for dtype in FLOATS:
                seed += 1
                yield {
                    "name": f"{op}-value{value}-{name(dtype)}",
                    "kind": op,
                    "value": value,
                    "dtype": dtype,
                    "seed": seed,
                }
    seed += 1
    yield {"name": "addcdiv_and_division-float32", "kind": "division_mix", "seed": seed}
    for weight in LERP_WEIGHTS:
        for dtype in (torch.float32, torch.bfloat16):
            seed += 1
            yield {
                "name": f"lerp-scalar{weight}-{name(dtype)}",
                "kind": "lerp",
                "weight": weight,
                "dtype": dtype,
                "seed": seed,
            }
    seed += 1
    yield {
        "name": "lerp-tensor-float32",
        "kind": "lerp",
        "weight": None,
        "dtype": torch.float32,
        "seed": seed,
    }


def function(case):
    kind = case["kind"]
    if kind == "unary":
        return UNARY[case["op"]]
    if kind == "alpha":
        op = torch.add if case["op"] == "add" else torch.sub
        return lambda c, a: op(c, a, alpha=ALPHA)
    if kind == "addcmul":
        return lambda s, a, b: torch.addcmul(s, a, b, value=case["value"])
    if kind == "addcdiv":
        return lambda s, a, b: torch.addcdiv(s, a, b, value=case["value"])
    if kind == "division_mix":
        return lambda s, a, b, c, d: torch.addcdiv(s, a, b, value=ALPHA) - c / d
    if case["weight"] is None:
        return torch.lerp
    return lambda start, end: torch.lerp(start, end, case["weight"])


def case_inputs(case):
    kind, seed = case["kind"], case["seed"]
    if kind == "unary":
        dtype = case["dtype"]
        if dtype.is_floating_point:
            return unary_float_inputs(dtype, seed)
        return unary_int_inputs(dtype, seed)
    if kind == "alpha":
        return alpha_inputs(case["dtype"], seed)
    if kind == "addcmul":
        return addcmul_inputs(case["dtype"], case["value"], seed)
    if kind == "addcdiv":
        return addcdiv_inputs(case["dtype"], case["value"], seed)
    if kind == "division_mix":
        return division_mix_inputs(seed)
    return lerp_inputs(case["dtype"], seed, case["weight"] is None)


def digest(t):
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def generate(path=DATA):
    import triton  # noqa: F401  the reference must come from Inductor+Triton

    records = {}
    for case in cases():
        torch._dynamo.reset()
        args = case_inputs(case)
        compiled = torch.compile(function(case))
        records[case["name"]] = {
            "input_sha256": [digest(a) for a in args],
            "output": compiled(*(a.cuda() for a in args)).cpu(),
            "eager": function(case)(*(a.cuda() for a in args)).cpu(),
        }
    payload = {"torch": str(torch.__version__), "cases": records}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path.write_bytes(buffer.getvalue())
    return path


if __name__ == "__main__":
    sys.stdout.write(f"wrote {generate()}\n")
