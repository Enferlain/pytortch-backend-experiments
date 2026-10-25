"""Inductor+Triton reference outputs for integer edge cases (integer-value-arithmetic).

The cases where eager and Triton may differ, or where the contract defers to
Triton: shifts by every amount (eager defines out-of-range amounts, Triton's
plain shift does not), and float to integer conversions of NaN, infinities and
out-of-range values (hardware conversion in both). Inputs are built from fixed
lists, so only their SHA-256 and the outputs are stored. Regenerate the data
file with the Triton environment from the repository root::

    /home/imi/Projects/sd-scripts/.venv-wsl/bin/python -m cubecl_torch.testing.integer_reference

Imports only torch: the CubeCL backend never loads here.
"""

import hashlib
import io
import sys
from pathlib import Path

import torch

DATA = Path(__file__).resolve().parent / "data" / "integer_triton.pt"
INTS = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
FLOATS = (torch.float32, torch.float16, torch.bfloat16)


def name(dtype):
    return str(dtype).removeprefix("torch.")


def width(dtype):
    return torch.iinfo(dtype).bits


def shift_inputs(dtype):
    """Every (value, amount) pair; amounts span negative, in-range and too wide."""
    info = torch.iinfo(dtype)
    values = [0, 1, -1, 5, -5, 3, info.min, info.max, info.min + 1, info.max - 1]
    values = [v for v in values if info.min <= v <= info.max]
    w = width(dtype)
    amounts = [-1, 0, 1, 3, w - 1, w, w + 1, 100]
    a = torch.tensor([v for v in values for _ in amounts], dtype=torch.int64)
    b = torch.tensor([s for _ in values for s in amounts], dtype=torch.int64)
    return [a, b]


def cast_inputs(dtype):
    big = [1e10, -1e10, 3e9, -3e9, 70000.0, -70000.0, 300.0, -300.0, 255.5, -1.5]
    specials = [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 2.5]
    return [torch.tensor(big + specials, dtype=torch.float32).to(dtype)]


def shift(op, dtype):
    def fn(a, b):
        a, b = a.to(dtype), b.to(dtype)
        return (a << b if op == "lshift" else a >> b).to(torch.int64)

    return fn


def cast(dtype):
    return lambda x: x.to(dtype).to(torch.int64)


TRUEDIV_INPUTS = (
    [7, -7, 1, 0, 2**40 + 3, -(2**31), 123456789, 1000001, -16777217],
    [2, 2, 3, 5, 7, 3, -1000, 3, 7],
)


def register_int_truediv():
    """A test op lowered to ops.int_truediv, which no aten lowering of the
    pinned Inductor calls (IntTrueDiv in value expressions is printed)."""
    if hasattr(torch.ops, "cubecl_int_test") and hasattr(
        torch.ops.cubecl_int_test, "int_truediv"
    ):
        return torch.ops.cubecl_int_test.int_truediv
    from torch._inductor import lowering
    from torch._inductor.virtualized import ops

    lib = torch.library.Library("cubecl_int_test", "DEF")
    lib.define("int_truediv(Tensor a, Tensor b) -> Tensor")
    lib.impl(
        "int_truediv",
        lambda a, b: a.to(torch.float32) / b.to(torch.float32),
        "CUDA",
    )
    torch.library.register_fake("cubecl_int_test::int_truediv")(
        lambda a, b: torch.empty_like(a, dtype=torch.float32)
    )
    lowering.register_lowering(torch.ops.cubecl_int_test.int_truediv)(
        lowering.make_pointwise(ops.int_truediv, override_return_dtype=torch.float32)
    )
    register_int_truediv.lib = lib  # keep the library alive
    return torch.ops.cubecl_int_test.int_truediv


def cases():
    yield {"name": "int_truediv-int64", "kind": "int_truediv"}
    for dtype in INTS:
        for op in ("lshift", "rshift"):
            yield {
                "name": f"{op}-{name(dtype)}",
                "kind": "shift",
                "op": op,
                "dtype": dtype,
            }
    for source in FLOATS:
        for dtype in INTS:
            yield {
                "name": f"cast-{name(source)}-{name(dtype)}",
                "kind": "cast",
                "source": source,
                "dtype": dtype,
            }


def function(case):
    if case["kind"] == "int_truediv":
        return register_int_truediv()
    if case["kind"] == "shift":
        return shift(case["op"], case["dtype"])
    return cast(case["dtype"])


def case_inputs(case):
    if case["kind"] == "int_truediv":
        return [torch.tensor(v, dtype=torch.int64) for v in TRUEDIV_INPUTS]
    if case["kind"] == "shift":
        return shift_inputs(case["dtype"])
    return cast_inputs(case["source"])


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
