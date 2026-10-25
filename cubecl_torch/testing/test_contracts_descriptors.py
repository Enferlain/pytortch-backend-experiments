"""Ported descriptor-level device contracts (see contract_inventory.md).

Sources: stage2/components.py (Stage 2 index arithmetic, nested lazy regions,
Rust validation, positive divisors) and stage3/components.py (reduction tails,
extent guard, uniform masked reductions), compiled as schema-5 packets.
"""

import copy
import json
import math
import subprocess

import pytest
import torch

from cubecl_torch.compiler.build import compiler_path
from cubecl_torch.compiler.indexing import evaluate
from cubecl_torch.runtime.launch import LaunchContractError
from cubecl_torch.testing.fixtures import native_launches, physical

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def const(v):
    return {"op": "constant", "value": v}


def scalar(s):
    return {"op": "scalar", "slot": s}


LINEAR = {"op": "linear"}


def op(id, name, args=(), dtype="float32", **extra):
    return {
        "id": id,
        "op": name,
        "args": list(args),
        "dtype": dtype,
        "semantic_dtype": dtype,
        **extra,
    }


def store(value, index=LINEAR):
    return {
        "op": "store",
        "args": [value],
        "dtype": "float32",
        "semantic_dtype": "float32",
        "buffer": 1,
        "index": index,
    }


def pointwise(ops, scalars=1):
    """stage2/components.py descriptor() as schema 4 (schema 2 plus effects)."""
    return {
        "schema": 4,
        "effects": {"rng": None, "mutation": False},
        "index_dtype": "int64",
        "numel": scalar(0),
        "scalars": [
            {"id": i, "dtype": "int64", "positive": False} for i in range(scalars)
        ],
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


BASE = {"op": "add", "args": [scalar(1), LINEAR, const(-9)]}


def index_kernel(index):
    return physical(
        pointwise(
            [
                op("a", "index", dtype="int64", index=index),
                op("b", "cast", ["a"]),
                store("b"),
            ],
            2,
        ),
        "scalar_pointwise_baseline",
    )


def expected_index(index, scalars):
    return torch.tensor(
        [evaluate(index, list(scalars), i) for i in range(257)],
        dtype=torch.float32,
        device="cuda",
    )


def test_signed_i64_floor_mod_on_device():
    floor = {"op": "floor_div", "args": [BASE, const(7)]}
    residual = {"op": "add", "args": [BASE, {"op": "mul", "args": [floor, const(-7)]}]}
    kernel = index_kernel(residual)
    for v in (-(2**32) - 257, -9, 2**31 + 1, 2**32 + 257):
        x = torch.zeros(257, device="cuda")
        y = torch.empty_like(x)
        kernel((257, v), x, y)
        assert torch.equal(y, expected_index(residual, (257, v)))
    # Empty extents skip the native enqueue even with large scalars.
    before = native_launches()
    x = torch.empty(0, device="cuda")
    kernel((0, 2**32 + 257), x, torch.empty_like(x))
    assert native_launches() == before


def test_floor_mod_near_i64_boundary():
    wide = {
        "op": "mod",
        "args": [{"op": "floor_div", "args": [BASE, const(13)]}, const(257)],
    }
    kernel = index_kernel(wide)
    for v in (-(2**63) + 10, 2**63 - 512):
        x = torch.zeros(257, device="cuda")
        y = torch.empty_like(x)
        kernel((257, v), x, y)
        assert torch.equal(y, expected_index(wide, (257, v)))


def test_nested_lazy_load_and_predicated_store():
    ops = [
        op("i", "index", dtype="int64", index=LINEAR),
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
                            index={"op": "mul", "args": [LINEAR, const(2)]},
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
    kernel = physical(pointwise(ops), "scalar_pointwise_baseline")
    x = torch.arange(257, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    kernel((257,), x, y)
    expected = torch.full_like(y, -7)
    expected[:64] = x[:128:2]
    expected[64:128] = 9
    assert torch.equal(y, expected)


def rust(tmp_path, semantic):
    from cubecl_torch.compiler.plan import PhysicalCandidate

    packet = {
        "schema": 5,
        "semantic": semantic,
        "physical_plan": PhysicalCandidate("scalar_pointwise_baseline").to_dict(),
        "entrypoint": "cubecl5_validation",
    }
    (tmp_path / "descriptor.json").write_text(json.dumps(packet))
    proc = subprocess.run(
        [str(compiler_path()), str(tmp_path), "cpp", str(tmp_path / "descriptor.json")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode and not (tmp_path / "cubecl.ir").exists(), proc.stderr
    return proc.stderr


def test_rust_rejects_unknown_ssa_operand(tmp_path):
    bad = pointwise([op("a", "add", ["not_defined", "not_defined"]), store("a")])
    assert "unknown SSA operand" in rust(tmp_path, bad)


@pytest.mark.parametrize("divisor", [const(0), const(-7), scalar(0)])
def test_rust_rejects_unproven_positive_divisor(tmp_path, divisor):
    index = {"op": "floor_div", "args": [LINEAR, divisor]}
    bad = pointwise(
        [
            op("a", "index", dtype="int64", index=index),
            op("b", "cast", ["a"]),
            store("b"),
        ]
    )
    assert "positive" in rust(tmp_path, bad)


def test_positive_runtime_divisor_enforced_at_launch():
    index = {"op": "floor_div", "args": [LINEAR, scalar(1)]}
    semantic = pointwise(
        [
            op("a", "index", dtype="int64", index=index),
            op("b", "cast", ["a"]),
            store("b"),
        ],
        2,
    )
    semantic["scalars"][1]["positive"] = True
    kernel = physical(semantic, "scalar_pointwise_baseline")
    x = torch.zeros(257, device="cuda")
    y = torch.empty_like(x)
    kernel((257, 7), x, y)
    assert torch.equal(
        y, torch.arange(257, device="cuda").div(7, rounding_mode="floor").float()
    )
    for invalid in (0, -7):
        before = native_launches()
        with pytest.raises(ValueError, match="positive index domain"):
            kernel((257, invalid), x, y)
        assert native_launches() == before


def reduction(dtype):
    """stage3/components.py descriptor(): flat row sum, outer scalar 0, extent scalar 1."""
    return {
        "schema": 3,
        "index_dtype": "int64",
        "numel": scalar(0),
        "outer": scalar(0),
        "reduction_extent": scalar(1),
        "scalars": [{"dtype": "int64", "id": 0, "positive": True}, {"dtype": "int64", "id": 1, "positive": False}],
        "buffers": [
            {"access": "read", "dtype": dtype, "size": [{"op": "mul", "args": [scalar(0), scalar(1)]}], "stride": [const(1)], "offset": const(0)},
            {"access": "write", "dtype": dtype, "size": [scalar(0)], "stride": [const(1)], "offset": const(0)},
        ],
        "ops": [
            {"id": "x", "op": "load", "args": [], "dtype": dtype, "semantic_dtype": dtype, "buffer": 0,
             "index": {"op": "add", "args": [{"op": "mul", "args": [LINEAR, scalar(1)]}, {"op": "reduction"}]}},
            {"id": "wide", "op": "cast", "args": ["x"], "dtype": "float32", "semantic_dtype": dtype},
            {"id": "sum", "op": "reduction", "args": ["wide"], "dtype": "float32", "semantic_dtype": dtype,
             "source_dtype": dtype, "reduction_type": "sum", "accumulator_dtype": "float32"},
            {"id": "out", "op": "cast", "args": ["sum"], "dtype": dtype, "semantic_dtype": dtype},
            {"op": "store", "args": ["out"], "dtype": dtype, "semantic_dtype": dtype, "buffer": 1,
             "index": {"op": "linear"}, "iteration_scope": "outer"},
        ],
    }  # fmt: skip


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reduction_tails_canaries_and_extent_guard(dtype):
    from cubecl_torch.inductor import kernels

    name = str(dtype).removeprefix("torch.")
    kernel = physical(reduction(name), "shared_tree_reduction_baseline")
    rtol, atol = {torch.float32: (2e-5, 2e-5), torch.float16: (0.002, 0.001)}.get(
        dtype, (0.016, 0.008)
    )
    before = kernels.COMPILER_INVOCATIONS
    for n in (0, 1, 31, 32, 33, 255, 256, 257, 4099, 65539):
        x = torch.randn(3 * n, device="cuda", dtype=dtype)
        backing = torch.full((5,), 123, device="cuda", dtype=dtype)
        y = backing[1:4]
        kernel((3, n), x, y)
        # Rounding error of an fp32 sum grows like sqrt(n): scale atol past 256.
        expected = x.view(3, n).double().sum(-1).to(dtype)
        scale = math.sqrt(max(n, 256) / 256)
        torch.testing.assert_close(y, expected, rtol=rtol, atol=atol * scale)
        assert torch.equal(
            backing[[0, 4]], torch.full((2,), 123, device="cuda", dtype=dtype)
        )
    assert before == kernels.COMPILER_INVOCATIONS
    with pytest.raises(
        LaunchContractError, match="unsupported physical reduction extent"
    ):
        kernel.kernel._layout((3, 1048577))


@pytest.mark.parametrize(
    ("nested", "outer_true", "inner_true"),
    [
        (False, True, True),
        (False, False, True),
        (True, True, True),
        (True, True, False),
        (True, False, True),
    ],
)
def test_uniform_masked_values_feed_reduction(nested, outer_true, inner_true):
    semantic = reduction("float32")
    prefix = [
        {
            "id": n,
            "op": "constant",
            "args": [],
            "dtype": d,
            "semantic_dtype": d,
            "value": v,
        }
        for n, d, v in (
            ("uniform", "float32", 7.0),
            ("outer_predicate", "bool", outer_true),
            ("inner_predicate", "bool", inner_true),
        )
    ]
    inner = {"id": "inner", "op": "masked", "args": ["inner_predicate"], "dtype": "float32",
             "semantic_dtype": "float32", "body": [], "result": "uniform", "other": 0.0}  # fmt: skip
    mask = {"id": "masked", "op": "masked", "args": ["outer_predicate"], "dtype": "float32",
            "semantic_dtype": "float32", "body": [inner] if nested else [],
            "result": "inner" if nested else "uniform", "other": 0.0}  # fmt: skip
    suffix = copy.deepcopy(semantic["ops"][2:])
    suffix[0]["args"] = ["masked"]
    semantic["ops"] = [*prefix, mask, *suffix]
    kernel = physical(semantic, "shared_tree_reduction_baseline")
    x = torch.empty(3 * 257, device="cuda")
    backing = torch.full((5,), 123.0, device="cuda")
    y = backing[1:4]
    kernel((3, 257), x, y)
    expected = 7 * 257 if outer_true and (inner_true or not nested) else 0
    assert torch.equal(y, torch.full_like(y, expected))
    assert torch.equal(backing[[0, 4]], torch.full((2,), 123.0, device="cuda"))
