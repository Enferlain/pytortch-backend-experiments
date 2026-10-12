"""Narrow descriptor/IR execution tests for widths, nested masks and masked stores."""

import ctypes as C
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage1 import evidence

evidence.guard()
import torch
from stage1.plan import PhysicalKernelPlan
from stage2.runtime import compile_kernel, EVENTS
from stage2.indexing import evaluate

ROOT = Path(__file__).resolve().parent
const = lambda v: {"op": "constant", "value": v}
scalar = lambda s: {"op": "scalar", "slot": s}
linear = {"op": "linear"}


def op(id, name, args=(), dtype="float32", **extra):
    return {
        "id": id,
        "op": name,
        "args": list(args),
        "dtype": dtype,
        "semantic_dtype": dtype,
        **extra,
    }


def store(value, index=linear):
    return {
        "op": "store",
        "args": [value],
        "dtype": "float32",
        "semantic_dtype": "float32",
        "buffer": 1,
        "index": index,
    }


def descriptor(ops, scalars=1):
    return {
        "schema": 2,
        "index_dtype": "int64",
        "numel": scalar(0),
        "scalars": [{"id": i, "dtype": "int64"} for i in range(scalars)],
        "buffers": [
            {
                "access": a,
                "dtype": "float32",
                "size": [scalar(0)],
                "stride": [const(1)],
                "offset": const(0),
            }
            for a in ("read", "write")
        ],
        "ops": ops,
    }


plan = PhysicalKernelPlan().to_dict()
result = {}
# Signed floor/mod expression with negative operands and symbols above 32-bit.
base = {"op": "add", "args": [scalar(1), linear, const(-9)]}
floor = {"op": "floor_div", "args": [base, const(7)]}
residual = {"op": "add", "args": [base, {"op": "mul", "args": [floor, const(-7)]}]}
d = descriptor(
    [
        op("a", "index", dtype="int64", index=residual),
        op("b", "cast", ["a"]),
        store("b"),
    ],
    2,
)
k = compile_kernel(d, 0, plan)
records = []
for v in (-(2**32) - 257, -9, 2**31 + 1, 2**32 + 257):
    x = torch.zeros(257, device="cuda")
    y = torch.empty_like(x)
    k((257, v), x, y)
    expected = torch.tensor(
        [evaluate(residual, [257, v], i) for i in range(257)],
        dtype=torch.float32,
        device="cuda",
    )
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    records.append(v)
result["signed_i64_floor_mod_gpu"] = {
    "runtime_values": records,
    "directory": str(k.directory),
}
# Floor quotients near the signed-i64 lower boundary without allocating huge storage.
wide = {
    "op": "mod",
    "args": [{"op": "floor_div", "args": [base, const(13)]}, const(257)],
}
wide_kernel = compile_kernel(
    descriptor(
        [
            op("a", "index", dtype="int64", index=wide),
            op("b", "cast", ["a"]),
            store("b"),
        ],
        2,
    ),
    0,
    plan,
)
for v in (-(2**63) + 10, 2**63 - 512):
    x = torch.zeros(257, device="cuda")
    y = torch.empty_like(x)
    wide_kernel((257, v), x, y)
    expected = torch.tensor(
        [evaluate(wide, [257, v], i) for i in range(257)],
        device="cuda",
        dtype=torch.float32,
    )
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
result["near_i64_boundary_floor_mod_gpu"] = {
    "directory": str(wide_kernel.directory),
    "values": [-(2**63) + 10, 2**63 - 512],
}

# Empty runtime extents skip native kernel enqueue even when null data pointers agree.
x = torch.empty(0, device="cuda")
y = torch.empty_like(x)
lib = C.CDLL(str(ROOT.parent / "stage0/libstage0_launcher.so"))
lib.stage0_kernel_launches.restype = C.c_uint64
before = lib.stage0_kernel_launches()
k((0, 2**32 + 257), x, y)
assert lib.stage0_kernel_launches() == before and y.numel() == 0
result["empty_runtime_descriptor"] = (
    "validated empty buffers; zero native kernel launches"
)
# Nested branches protect potentially invalid memory. Initial unconditional store
# is overwritten only for predicate=true; no final unconditional masked-result store.
ops = [
    op("i", "index", dtype="int64", index=linear),
    op("limit", "constant", dtype="int64", value=128),
    op("limit2", "constant", dtype="int64", value=64),
    op("mask", "lt", ["i", "limit"], dtype="bool"),
    op("mask2", "lt", ["i", "limit2"], dtype="bool"),
    op("fill", "constant", value=-7.0),
    store("fill"),
    op(
        "outer",
        "masked",
        ["mask"],
        body=[
            op(
                "inner",
                "masked",
                ["mask2"],
                body=[
                    op(
                        "load",
                        "load",
                        buffer=0,
                        index={"op": "mul", "args": [linear, const(2)]},
                    )
                ],
                result="load",
                other=9.0,
            ),
            store("inner"),
        ],
        result="inner",
        other=-11.0,
    ),
]
k = compile_kernel(descriptor(ops), 0, plan)
x = torch.arange(257, dtype=torch.float32, device="cuda")
y = torch.empty_like(x)
k((257,), x, y)
expected = torch.full_like(y, -7)
expected[:64] = x[:128:2]
expected[64:128] = 9
torch.testing.assert_close(y, expected, rtol=0, atol=0)
result["nested_lazy_load_and_predicated_store_gpu"] = {
    "directory": str(k.directory),
    "predicate_ranges": [64, 128],
    "physical_tail": 257,
}
# Compiler rejects invalid incoming type/SSA descriptions before writing IR.
import subprocess
from stage0.bridge import native_path

bad = descriptor([op("a", "add", ["not_defined", "not_defined"]), store("a")])
from stage2.runtime import bridge_description

out = ROOT / "artifacts/invalid-descriptor"
out.mkdir(exist_ok=True)
(out / "descriptor.json").write_text(json.dumps(bridge_description(bad, plan)))
p = subprocess.run(
    [str(native_path("cubecl-stage0")), str(out), "cpp", str(out / "descriptor.json")],
    capture_output=True,
    text=True,
)
assert (
    p.returncode
    and "unknown SSA operand" in p.stderr
    and not (out / "cubecl.ir").exists()
), p.stderr
result["rust_validation_before_ir"] = p.stderr.strip()
# Malformed divisor descriptions reject before IR; symbolic positivity is also
# enforced at launch, rather than trusting a descriptor declaration alone.
rejections = []
for divisor in (const(0), const(-7), scalar(0)):
    bad = descriptor(
        [
            op(
                "a",
                "index",
                dtype="int64",
                index={"op": "floor_div", "args": [linear, divisor]},
            ),
            op("b", "cast", ["a"]),
            store("b"),
        ]
    )
    (out / "descriptor.json").write_text(json.dumps(bridge_description(bad, plan)))
    p = subprocess.run(
        [
            str(native_path("cubecl-stage0")),
            str(out),
            "cpp",
            str(out / "descriptor.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert (
        p.returncode and "positive" in p.stderr and not (out / "cubecl.ir").exists()
    ), p.stderr
    rejections.append(divisor)
positive = descriptor(
    [
        op(
            "a",
            "index",
            dtype="int64",
            index={"op": "floor_div", "args": [linear, scalar(1)]},
        ),
        op("b", "cast", ["a"]),
        store("b"),
    ],
    2,
)
positive["scalars"][1]["positive"] = True
positive_kernel = compile_kernel(positive, 0, plan)
x = torch.zeros(257, device="cuda")
y = torch.empty_like(x)
positive_kernel((257, 7), x, y)
torch.testing.assert_close(
    y, torch.arange(257, device="cuda").div(7, rounding_mode="floor").float()
)
for invalid in (0, -7):
    before = lib.stage0_kernel_launches()
    try:
        positive_kernel((257, invalid), x, y)
    except ValueError as error:
        assert "positive index domain" in str(error)
    else:
        raise AssertionError("nonpositive runtime divisor accepted")
    assert lib.stage0_kernel_launches() == before
for method in (positive_kernel.kernel.prepare, positive_kernel.kernel.prepare_capture):
    try:
        method(257)
    except NotImplementedError as error:
        assert "Stage 2" in str(error)
    else:
        raise AssertionError("unsupported preparation API accepted")
result["positive_divisor_contract"] = {
    "compiler_rejections": rejections,
    "runtime_rejections": [0, -7],
    "valid_divisor": 7,
}
result["unsupported_capture_api"] = "explicit errors before metadata preparation"
evidence.state("after_components")
result["independence"] = evidence.inventory()
result["events"] = EVENTS
(ROOT / "artifacts/components.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
