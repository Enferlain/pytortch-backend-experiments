"""Refusals of the fused ecosystem reduce that compiles don't easily produce
(fused-reduction-bridge task 2.2). Runs without a GPU."""

import copy

from cubecl_torch.inductor.legality import fused_reduce_reason
from cubecl_torch.testing.test_cubek_legality import EXTENT, ROWS
from cubecl_torch.testing.test_cubek_legality import row_sum as dynamic_row_sum


def static(v, sizes):
    """``v`` with each scalar slot replaced by its constant (Inductor's
    static sizes)."""
    if isinstance(v, dict):
        if v.get("op") == "scalar":
            return {"op": "constant", "value": sizes[v["slot"]]}
        return {k: static(x, sizes) for k, x in v.items()}
    if isinstance(v, list):
        return [static(x, sizes) for x in v]
    return v


def row_sum(**kwargs):
    """``test_cubek_legality.row_sum`` with static sizes: the fused reduce
    takes no runtime scalars yet."""
    s = dynamic_row_sum(**kwargs)
    s = static({k: v for k, v in s.items() if k != "scalars"}, (ROWS, EXTENT))
    return {**s, "scalars": []}


LAYOUT = {
    "outer": {"dims": [64], "strides": [1024]},
    "reduction": {"dims": [1024], "strides": [1]},
    "offset": 0,
}


def layouts(semantic):
    return {op["id"]: LAYOUT for op in semantic["ops"] if op["op"] == "load"}


def test_plain_sum_is_covered():
    s = row_sum()
    assert fused_reduce_reason(s, layouts(s)) is None


def test_several_reductions_refuse():
    s = row_sum(reductions=2)
    assert "exactly one reduction" in fused_reduce_reason(s, layouts(s))


def test_non_sum_refuses():
    s = row_sum(kind="max")
    assert "a max reduction is not covered" in fused_reduce_reason(s, layouts(s))


def test_welford_component_refuses():
    s = row_sum()
    reduction = next(op for op in s["ops"] if op["op"] == "reduction")
    reduction["op"] = "reduction_component"
    assert "exactly one reduction" in fused_reduce_reason(s, layouts(s))


def test_mask_region_refuses():
    s = row_sum()
    s["ops"].insert(
        0, {"id": "m", "op": "masked", "args": [], "dtype": "float32", "body": []}
    )
    assert "mask region" in fused_reduce_reason(s, layouts(s))


def test_reduction_scope_store_refuses():
    s = row_sum()
    store = copy.deepcopy(next(op for op in s["ops"] if op["op"] == "store"))
    store["iteration_scope"] = "reduction"
    s["ops"].append(store)
    assert "reduction-scope store" in fused_reduce_reason(s, layouts(s))


def test_missing_layout_refuses():
    s = row_sum()
    assert "no block layout" in fused_reduce_reason(s, {})


def test_random_numbers_refuse():
    s = row_sum()
    s["ops"].insert(0, {"id": "seed", "op": "load_seed", "args": [], "dtype": "int64"})
    assert "random numbers" in fused_reduce_reason(s, layouts(s))


def test_indirect_index_refuses():
    s = row_sum()
    s["ops"].insert(0, {"id": "i", "op": "indirect", "args": [], "dtype": "int64"})
    assert "indirect index" in fused_reduce_reason(s, layouts(s))


def test_float64_accumulator_refuses():
    s = row_sum()
    reduction = next(op for op in s["ops"] if op["op"] == "reduction")
    reduction["accumulator_dtype"] = "float64"
    assert "accumulated in float32" in fused_reduce_reason(s, layouts(s))


def test_epilogue_reading_a_row_varying_value_refuses():
    s = row_sum()
    load = next(op for op in s["ops"] if op["op"] == "load")
    store = next(op for op in s["ops"] if op["op"] == "store")
    s["ops"].insert(
        s["ops"].index(store),
        {
            "id": "late",
            "op": "add",
            "args": [store["args"][0], load["id"]],
            "dtype": "float32",
        },
    )
    store["args"] = ["late"]
    assert "varies along it" in fused_reduce_reason(s, layouts(s))


def test_runtime_scalars_refuse():
    s = dynamic_row_sum()
    assert "runtime scalars" in fused_reduce_reason(s, layouts(s))
