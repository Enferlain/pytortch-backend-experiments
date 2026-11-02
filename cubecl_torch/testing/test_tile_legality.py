"""Which kernels are offered tile plans, in both legality layers
(cubecl-tile-stage-1 tasks 6.1 and 6.2). Python's side runs without a GPU."""

import copy

import pytest
import torch

from cubecl_torch.inductor import legality

TARGET = {"device_max_threads": 1024, "plane_width": 32}


def row_sum(
    dtype="float32", kind="sum", reductions=1, offset=0, out=None, square=False
):
    """``x.sum(-1)`` over ``[outer, extent]``: scalar 0 is outer, 1 the extent;
    ``square`` sums ``x * x`` in float32, ``out`` stores another dtype."""
    index = {
        "op": "add",
        "args": [
            {"op": "reduction"},
            {"op": "mul", "args": [{"op": "linear"}, {"op": "scalar", "slot": 1}]},
            {"op": "constant", "value": offset},
        ],
    }
    ops = [
        {
            "op": "load",
            "id": "x",
            "args": [],
            "buffer": 0,
            "dtype": dtype,
            "semantic_dtype": "float32",
            "index": index,
        }
    ]
    if square:
        ops.append({"op": "cast", "id": "f", "args": ["x"], "dtype": "float32"})
        ops.append({"op": "mul", "id": "q", "args": ["f", "f"], "dtype": "float32"})
    for i in range(reductions):
        ops.append(
            {
                "op": "reduction",
                "id": f"s{i}",
                "args": ["q" if square else "x"],
                "dtype": "float32",
                "accumulator_dtype": "float32",
                "reduction_type": kind,
            }
        )
    ops.append(
        {
            "op": "store",
            "args": ["s0"],
            "buffer": 1,
            "dtype": out or dtype,
            "index": {"op": "linear"},
            "iteration_scope": "outer",
        }
    )
    return {
        "schema": 3,
        "ops": ops,
        "outer": {"op": "scalar", "slot": 0},
        "reduction_extent": {"op": "scalar", "slot": 1},
        "scalars": [
            {"dtype": "int64", "id": 0, "positive": True},
            {"dtype": "int64", "id": 1, "positive": True},
        ],
        "buffers": [],
    }


def tensors(semantic, rows, extent, *, misaligned=False):
    dtype = getattr(
        torch, next(op["dtype"] for op in semantic["ops"] if op["op"] == "load")
    )
    x = torch.empty(rows * extent + 1, dtype=dtype)
    x = x[1:] if misaligned else x[: rows * extent]
    return [x.view(rows, extent), torch.empty(rows, dtype=dtype)]


def tile_offers(semantic, rows, extent, **options):
    accepted, rejected = legality.candidates(
        semantic, (rows, extent), tensors(semantic, rows, extent, **options), TARGET
    )
    offered = sorted(p.vector_width for p in accepted if p.strategy == legality.TILE)
    reasons = {
        r["plan"]["vector_width"]: r["reason"]
        for r in rejected
        if r["plan"]["strategy"] == legality.TILE
    }
    return offered, reasons


@pytest.mark.parametrize(
    ("dtype", "extent", "offered"),
    [
        ("float32", 1024, []),
        ("float32", 2048, [4]),
        ("float32", 4096, [4]),
        ("float32", 8192, [4]),
        ("float32", 4099, []),
        ("float32", 16384, []),
        ("bfloat16", 2048, [8]),
        ("bfloat16", 8192, [8]),
        ("float16", 4099, []),
        ("float16", 16384, []),
    ],
)
def test_tile_plans_are_offered_on_measured_classes(dtype, extent, offered):
    found, reasons = tile_offers(row_sum(dtype), 64, extent)
    assert found == offered
    # Every tile plan not offered is recorded with its reason.
    assert sorted(found + list(reasons)) == [1, 4, 8]


def test_reasons_name_the_class():
    _, reasons = tile_offers(row_sum(), 64, 8192)
    assert (
        reasons[1] == "tile sums of float32 data are in Triton's band only at width 4"
    )
    _, reasons = tile_offers(row_sum("bfloat16"), 64, 8192)
    assert reasons[1] == "tile sums of 16-bit values are offered at width 8 only"
    _, reasons = tile_offers(row_sum(), 64, 16384)
    assert reasons[4] == "tile rows past 8192 are outside Triton's band or unmeasured"
    _, reasons = tile_offers(row_sum(), 64, 4099)
    assert reasons[4] == "tile vector plan requires a row of whole steps"
    _, reasons = tile_offers(row_sum(), 64, 1024)
    assert reasons[4].startswith("512-unit plans only for rows of at least 2048")


@pytest.mark.parametrize(
    ("semantic", "reason"),
    [
        (
            row_sum(kind="max"),
            "strategy tile_row_sum does not support reduction type max",
        ),
        (row_sum(reductions=2), "tile plans reduce exactly one sum"),
    ],
)
def test_other_kernels_are_refused(semantic, reason):
    found, reasons = tile_offers(semantic, 64, 8192)
    assert found == []
    assert reasons[4] == reason


def test_mask_region_is_refused():
    semantic = row_sum()
    load = semantic["ops"][0]
    semantic["ops"][0] = {"op": "masked", "id": "m", "args": [], "body": [load]}
    plan = next(p for p in legality.catalog() if p.strategy == legality.TILE)
    assert (
        legality.tile_reason(plan, semantic, (64, 8192))
        == "tile plans take no mask region"
    )


def test_reference512_split_producers_are_refused():
    """A reference512 final keeps its frozen arithmetic by the existing rule."""
    semantic = row_sum()
    accepted, rejected = legality.candidates(
        semantic,
        (64, 8192),
        tensors(semantic, 64, 8192),
        TARGET,
        reduction_context={"role": "producer"},
    )
    assert not [p for p in accepted if p.strategy == legality.TILE]
    assert {
        r["reason"] for r in rejected if r["plan"]["strategy"] == legality.TILE
    } == {"reference512 split pairs are not tile kernels"}


@pytest.mark.parametrize(
    ("semantic", "offered"),
    [
        # 16-bit values summed, a float32 sum stored: the 16-bit class.
        (row_sum("bfloat16", out="float32"), [8]),
        # 16-bit loads summed as float32 data (x * x): the float32 class.
        (row_sum("bfloat16", square=True), [4]),
        (row_sum("float16", square=True, out="float32"), [4]),
        # Float32 values summed, a 16-bit result stored: the float32 class.
        (row_sum("float32", out="bfloat16"), [4]),
    ],
)
def test_the_summed_values_choose_the_class(semantic, offered):
    found, _ = tile_offers(semantic, 64, 8192)
    assert found == offered


def test_misaligned_vector_rows_are_refused():
    found, reasons = tile_offers(row_sum(), 64, 8192, misaligned=True)
    assert found == []
    assert reasons[4] == "runtime pointer is misaligned"
    found, reasons = tile_offers(row_sum(offset=1), 64, 8192)
    assert found == []
    assert reasons[4] == "reduction vector row base/stride not aligned"


def test_signature_separates_tile_classes():
    def facts(extent):
        semantic = row_sum()
        return legality.signature(semantic, (64, extent), tensors(semantic, 64, extent))

    assert facts(4096) != facts(4099)
    assert facts(8192) != facts(8320)
    assert facts(2048) != facts(2176)  # whole steps at width 8, not only 4
    assert copy.deepcopy(facts(4096)) == facts(4096)
