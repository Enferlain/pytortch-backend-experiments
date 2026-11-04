"""Which kernels the bridged ``cubek_reduce`` plans may run (torch-runtime-bridge
task 4.1, design decision 8). Runs without a GPU."""

import copy

import pytest
import torch

from cubecl_torch.compiler.plan import CUBEK_REDUCE, PhysicalCandidate, bridge_catalog
from cubecl_torch.inductor import legality

TARGET = {"device_max_threads": 1024, "plane_width": 32}
ROWS, EXTENT = 64, 1024


def row_sum(dtype="float32", out=None, *, kind="sum", reductions=1):
    """``x.sum(-1)`` over ``[outer, extent]`` as Inductor describes it: the
    load, a cast to float32 for 16-bit storage, the sum, a cast to the stored
    dtype when it differs. Scalar 0 is outer, 1 the extent."""
    out = out or dtype
    index = {
        "op": "add",
        "args": [
            {"op": "reduction"},
            {"op": "mul", "args": [{"op": "scalar", "slot": 1}, {"op": "linear"}]},
        ],
    }
    ops = [{"id": "x", "op": "load", "args": [], "dtype": dtype, "index": index}]
    value = "x"
    if dtype != "float32":
        ops.append({"id": "f", "op": "cast", "args": ["x"], "dtype": "float32"})
        value = "f"
    for i in range(reductions):
        ops.append(
            {
                "id": f"s{i}",
                "op": "reduction",
                "args": [value],
                "dtype": "float32",
                "reduction_type": kind,
                "accumulator_dtype": "float32",
            }
        )
    result = "s0"
    if out != "float32":
        ops.append({"id": "o", "op": "cast", "args": ["s0"], "dtype": out})
        result = "o"
    ops.append(
        {
            "op": "store",
            "args": [result],
            "dtype": out,
            "index": {"op": "linear"},
            "iteration_scope": "outer",
        }
    )
    for i, op in enumerate(o for o in ops if o["op"] in ("load", "store")):
        op["buffer"] = i
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


def tensors(semantic, rows=ROWS, extent=EXTENT, *, misaligned=False):
    load = next(legality.memory_ops(semantic["ops"], "load"))["dtype"]
    store = next(legality.memory_ops(semantic["ops"], "store"))["dtype"]
    x = torch.empty(rows * extent + 1, dtype=getattr(torch, load))
    x = x[1:] if misaligned else x[: rows * extent]
    return [x.view(rows, extent), torch.empty(rows, dtype=getattr(torch, store))]


def structural(semantic, **options):
    """The structural reason of every plan of Burn's grid (one reason for
    all); the Inductor mapping adds its own rule, tested apart."""
    args = tensors(semantic, **options)
    # A descriptor with a third buffer (a second output) gets its tensor.
    if any(op.get("buffer") == 2 for op in semantic["ops"]):
        args.append(torch.empty(ROWS))
    reasons = {
        legality.structural_reason(plan, semantic, (ROWS, EXTENT), args, TARGET)
        for plan in bridge_catalog()
        if not plan.mapping
    }
    assert len(reasons) == 1
    return reasons.pop()


def test_bridge_catalog_is_burns_grid_and_inductors_mapping():
    """Unit, plane (independent accumulators) and cube (with planes), each
    with and without parallel output vectorization, as Burn tunes a reduction
    along contiguous rows, then the cube mapped as Inductor's config maps a
    row; none in the native catalog."""
    plans = bridge_catalog()
    assert {(p.routine, p.parallel_output) for p in plans if not p.mapping} == {
        (r, v) for r in ("unit", "plane", "cube") for v in (True, False)
    }
    assert [p for p in plans if p.mapping] == [
        PhysicalCandidate(CUBEK_REDUCE, routine="cube", mapping="inductor")
    ]
    assert all(p.strategy == CUBEK_REDUCE for p in plans)
    assert not set(plans) & set(legality.catalog())
    for plan in plans:
        assert PhysicalCandidate.from_dict(plan.to_dict()) == plan


def test_forced_planes_are_part_of_the_identity():
    """A diagnostic plan forcing CubeK's plane count (``cubek_band --planes``)
    is its own plan, never read back from a record; the catalog's leave the
    field out."""
    forced = PhysicalCandidate(CUBEK_REDUCE, routine="cube", planes=16)
    assert forced.to_dict()["planes"] == 16
    assert forced != PhysicalCandidate(CUBEK_REDUCE, routine="cube")
    with pytest.raises(ValueError, match="unsupported physical plan"):
        PhysicalCandidate.from_dict(forced.to_dict())
    assert all("planes" not in p.to_dict() for p in bridge_catalog())


def test_inductor_mapping_takes_looped_rows():
    """Inductor loops over rows longer than 1024 elements with one fixed
    config, which the mapping follows; shorter rows are refused with the
    reason, the other bridged plans still admitted structurally."""
    mapped = PhysicalCandidate(CUBEK_REDUCE, routine="cube", mapping="inductor")
    burn = PhysicalCandidate(CUBEK_REDUCE, routine="cube")
    for cols, admitted in ((1024, False), (1025, True), (4099, True)):
        semantic = row_sum()
        args = tensors(semantic, ROWS, cols)
        reason = legality.structural_reason(
            mapped, semantic, (ROWS, cols), args, TARGET
        )
        assert (reason is None) == admitted, reason
        if not admitted:
            assert "over 1024 elements" in reason
        assert (
            legality.structural_reason(burn, semantic, (ROWS, cols), args, TARGET)
            is None
        )


def test_native_plan_identities_are_unchanged():
    """The bridged fields are left out of native plans' dictionaries."""
    for plan in legality.catalog():
        assert "routine" not in plan.to_dict()
        assert "parallel_output" not in plan.to_dict()


@pytest.mark.parametrize(
    ("dtype", "out"),
    [
        ("float32", None),
        ("float16", None),
        ("bfloat16", None),
        ("bfloat16", "float32"),
        ("float32", "bfloat16"),
    ],
)
def test_plain_row_sums_are_structurally_legal(dtype, out):
    assert structural(row_sum(dtype, out)) is None


def test_load_without_cast_is_legal():
    """A 16-bit load summed directly (the load widens it) is the same kernel."""
    semantic = row_sum("bfloat16")
    semantic["ops"] = [op for op in semantic["ops"] if op.get("id") != "f"]
    semantic["ops"][1]["args"] = ["x"]
    assert structural(semantic) is None


def fused_body():
    """``(x * x).sum(-1)``: another operation in the body."""
    semantic = row_sum()
    semantic["ops"].insert(
        1, {"id": "q", "op": "mul", "args": ["x", "x"], "dtype": "float32"}
    )
    semantic["ops"][2]["args"] = ["q"]
    return semantic


def narrowed_before_sum():
    """``x.bfloat16().float().sum(-1)``: a rounding cast before the sum."""
    semantic = row_sum("float32", "float32")
    semantic["ops"][1:1] = [
        {"id": "b", "op": "cast", "args": ["x"], "dtype": "bfloat16"},
        {"id": "f", "op": "cast", "args": ["b"], "dtype": "float32"},
    ]
    semantic["ops"][3]["args"] = ["f"]
    return semantic


def masked():
    semantic = row_sum()
    load = semantic["ops"][0]
    semantic["ops"][0] = {"op": "masked", "id": "m", "args": [], "body": [load]}
    return semantic


def offset_rows():
    semantic = row_sum()
    semantic["ops"][0]["index"]["args"].append({"op": "constant", "value": 1})
    return semantic


def strided_rows():
    semantic = row_sum()
    semantic["ops"][0]["index"]["args"][1]["args"][0] = {
        "op": "constant",
        "value": 2048,
    }
    return semantic


def two_outputs():
    semantic = row_sum()
    store = copy.deepcopy(semantic["ops"][-1])
    store["buffer"] = 2
    semantic["ops"].append(store)
    return semantic


@pytest.mark.parametrize(
    ("semantic", "reason"),
    [
        (
            fused_body(),
            "cubek_reduce sums a loaded input with nothing else in the body",
        ),
        (
            narrowed_before_sum(),
            "cubek_reduce sums a loaded input with nothing else in the body",
        ),
        (masked(), "cubek_reduce takes no mask region"),
        (row_sum(reductions=2), "cubek_reduce reduces exactly one sum"),
        (offset_rows(), "cubek_reduce requires contiguous non-empty rows"),
        (strided_rows(), "cubek_reduce requires contiguous non-empty rows"),
        (two_outputs(), "cubek_reduce sums one loaded input into one output"),
        (
            row_sum(kind="max"),
            "strategy cubek_reduce does not support reduction type max",
        ),
        (
            row_sum(kind="welford_reduce"),
            "strategy cubek_reduce does not support reduction type welford_reduce",
        ),
    ],
    ids=[
        "fused_body",
        "narrowing_cast",
        "mask_region",
        "several_reductions",
        "row_offset",
        "row_stride",
        "two_outputs",
        "max",
        "welford",
    ],
)
def test_other_kernels_are_refused(semantic, reason):
    assert structural(semantic) == reason


def test_effectful_and_in_place_kernels_are_refused():
    """An effectful kernel keeps the frozen-tuning reason; an output must be
    a buffer apart from its input."""
    semantic = row_sum()
    semantic["effects"] = {"mutation": True}
    assert structural(semantic) == (
        "effectful/checked-indirect automatic tuning stays frozen"
    )
    semantic = row_sum()
    semantic["ops"][-1]["buffer"] = 0
    reasons = {
        legality.structural_reason(
            plan, semantic, (ROWS, EXTENT), tensors(semantic)[:1], TARGET
        )
        for plan in bridge_catalog()
    }
    assert reasons == {"cubek_reduce writes an output apart from its input"}


def test_misaligned_rows_are_refused():
    assert (
        structural(row_sum(), misaligned=True)
        == "cubek_reduce requires 16-byte aligned rows and output"
    )


def test_empty_rows_are_refused():
    semantic = row_sum()
    reasons = {
        legality.structural_reason(
            plan, semantic, (ROWS, 0), tensors(semantic, extent=0), TARGET
        )
        for plan in bridge_catalog()
    }
    assert reasons == {"cubek_reduce requires contiguous non-empty rows"}


@pytest.mark.parametrize(
    ("dtype", "out", "extent", "admitted"),
    [
        # 16-bit sums stored as loaded: the mapping, on rows Inductor loops
        # over up to 8192; Burn's grid is held out for speed (task 4.5).
        ("bfloat16", None, 2048, "mapped"),
        ("float16", None, 8192, "mapped"),
        ("bfloat16", None, 1024, "none"),
        ("bfloat16", None, 16384, "none"),
        ("bfloat16", "float32", 2048, "none"),
        # Float32: Inductor's mapping only, rows up to 32768.
        ("float32", None, 2048, "mapped"),
        ("float32", None, 32768, "mapped"),
        ("float32", None, 65536, "none"),
        ("float32", None, 1024, "none"),
    ],
)
def test_admission_follows_the_band(dtype, out, extent, admitted):
    """Bridged plans are admitted on the classes measured in Triton's band
    (task 4.2) where they were not held out for speed (task 4.5), each other
    kernel recording why not; the existing plans are unchanged."""
    semantic = row_sum(dtype, out)
    accepted, rejected = legality.candidates(
        semantic, (ROWS, extent), tensors(semantic, ROWS, extent), TARGET
    )
    bridged = [p for p in accepted if p.strategy == CUBEK_REDUCE]
    expected = {
        "mapped": [p for p in bridge_catalog() if p.mapping],
        "none": [],
    }[admitted]
    assert sorted(bridged, key=str) == sorted(expected, key=str)
    reasons = [r for r in rejected if r["plan"]["strategy"] == CUBEK_REDUCE]
    assert len(reasons) + len(bridged) == len(bridge_catalog())
    assert all(r["reason"].startswith("cubek_reduce") for r in reasons)


def test_burns_grid_is_held_out_for_speed():
    """Where the band admits Burn's grid (16-bit, stored as loaded), it is held
    out with the speed reason (task 4.5); elsewhere the band's reason stands."""
    for dtype, held in (("bfloat16", True), ("float32", False)):
        semantic = row_sum(dtype, None)
        _, rejected = legality.candidates(
            semantic, (ROWS, 2048), tensors(semantic, ROWS, 2048), TARGET
        )
        reasons = {
            r["reason"]
            for r in rejected
            if r["plan"]["strategy"] == CUBEK_REDUCE and not r["plan"].get("mapping")
        }
        assert len(reasons) == 1, reasons
        assert ("held out" in reasons.pop()) == held


def test_reference512_split_pairs_are_refused():
    semantic = row_sum()
    _, rejected = legality.candidates(
        semantic,
        (ROWS, EXTENT),
        tensors(semantic),
        TARGET,
        reduction_context={"role": "producer"},
    )
    assert {r["reason"] for r in rejected if r["plan"]["strategy"] == CUBEK_REDUCE} == {
        "reference512 split pairs are not bridged kernels"
    }


def test_signature_separates_bridged_rows():
    """The runtime parts of the rule are legality-cache facts: the same
    descriptor with rows that stop being contiguous, or that Inductor loops
    over, gets its own class."""
    semantic = row_sum()
    index = semantic["ops"][0]["index"]
    index["args"][1]["args"][0] = {"op": "scalar", "slot": 2}
    semantic["scalars"].append({"dtype": "int64", "id": 2, "positive": True})
    t = tensors(semantic)
    contiguous = legality.signature(semantic, (ROWS, EXTENT, EXTENT), t)
    strided = legality.signature(semantic, (ROWS, EXTENT, 2 * EXTENT), t)
    assert contiguous[-3] is True and strided[-3] is False
    # Rows Inductor loops over (the Inductor mapping) are a fact of their own.
    long = legality.signature(
        semantic, (ROWS, 2048, 2048), tensors(semantic, ROWS, 2048)
    )
    assert contiguous[-2] is False and long[-2] is True
