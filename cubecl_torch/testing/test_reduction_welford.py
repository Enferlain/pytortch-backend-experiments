"""Welford reductions from hand-authored descriptors (welford-reductions 3).

welford_reduce lifts each element to (x, 0, 1); welford_combine combines
(mean, m2, weight) partials. Both run under every eligible physical plan and
are compared with fp64 statistics.
"""

import pytest
import torch

from cubecl_torch.compiler.build import compile_descriptor
from cubecl_torch.compiler.plan import catalog
from cubecl_torch.inductor import legality
from cubecl_torch.runtime.cache import ArtifactCache
from cubecl_torch.testing.fixtures import physical

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

OUTER = {"op": "scalar", "slot": 0}
EXTENT = {"op": "scalar", "slot": 1}
ROW = {
    "op": "add",
    "args": [{"op": "reduction"}, {"op": "mul", "args": [{"op": "linear"}, EXTENT]}],
}
# Ragged and short extents, and aligned ones for the vectorized plans.
SHAPES = (1, 5, 31, 33, 256, 257, 4096, 4099)


def buffer(access, rows=True):
    size, stride = (
        ([OUTER, EXTENT], [EXTENT, {"op": "constant", "value": 1}])
        if rows
        else ([OUTER], [{"op": "constant", "value": 1}])
    )
    return {
        "access": access,
        "dtype": "float32",
        "offset": {"op": "constant", "value": 0},
        "size": size,
        "stride": stride,
    }


def value(op, id, args, **fields):
    return {
        "op": op,
        "id": id,
        "args": args,
        "dtype": "float32",
        "semantic_dtype": "float32",
        **fields,
    }


def reduction(kind, args):
    return value(
        "reduction",
        "r",
        args,
        reduction_type=kind,
        source_dtype="float32",
        accumulator_dtype="float32",
    )


def components():
    return [value("reduction_component", f"c{k}", ["r"], index=k) for k in range(3)]


def store(arg, buffer_index, scope="outer"):
    return {
        "op": "store",
        "args": [arg],
        "buffer": buffer_index,
        "dtype": "float32",
        "semantic_dtype": "float32",
        "index": {"op": "linear"} if scope == "outer" else ROW,
        "iteration_scope": scope,
    }


def descriptor(ops, buffers):
    return {
        "schema": 3,
        "index_dtype": "int64",
        "buffers": buffers,
        "numel": OUTER,
        "outer": OUTER,
        "reduction_extent": EXTENT,
        "scalars": [
            {"dtype": "int64", "id": 0, "positive": True},
            {"dtype": "int64", "id": 1, "positive": True},
        ],
        "ops": ops,
    }


def reduce_descriptor(component):
    """welford_reduce over x[outer, n], storing one component per row."""
    return descriptor(
        [
            value("load", "x", [], buffer=0, index=ROW),
            reduction("welford_reduce", ["x"]),
            *components(),
            store(f"c{component}", 1),
        ],
        [buffer("read"), buffer("write", rows=False)],
    )


def combine_descriptor(component):
    """welford_combine over (mean, m2, weight) partials, one component per row."""
    return descriptor(
        [
            value("load", "m", [], buffer=0, index=ROW),
            value("load", "q", [], buffer=1, index=ROW),
            value("load", "w", [], buffer=2, index=ROW),
            reduction("welford_combine", ["m", "q", "w"]),
            *components(),
            store(f"c{component}", 3),
        ],
        [buffer("read"), buffer("read"), buffer("read"), buffer("write", rows=False)],
    )


def row_mean_descriptor():
    """mean + (x - x) stored for every element: exactly the mean each unit read."""
    return descriptor(
        [
            value("load", "x", [], buffer=0, index=ROW),
            reduction("welford_reduce", ["x"]),
            *components(),
            value("sub", "z", ["x", "x"]),
            value("add", "e", ["c0", "z"]),
            store("e", 1, scope="reduction"),
        ],
        [buffer("read"), buffer("write")],
    )


def reduction_plans():
    return [
        p
        for p in catalog()
        if "pointwise" not in p.strategy and p.strategy != legality.FINAL9
    ]


def kernel(semantic, plan):
    return physical(
        semantic,
        plan.strategy,
        plan.workgroup,
        plan.vector_width,
        plan.elements_per_unit,
        plan.outputs_per_workgroup,
        plan.plane_width,
        private_accumulators=plan.private_accumulators,
        private_merge=plan.private_merge,
    )


def eligible(plan):
    return plan.strategy not in legality.SUM_ONLY_STRATEGIES


def run(k, plan, semantic, scalars, *tensors):
    """Launch, or return the runtime proof that excludes this plan here."""
    why = legality.reason(plan, semantic, scalars, tensors, k.kernel.resources)
    try:
        k(scalars, *tensors)
    except ValueError as error:
        assert why and why in str(error), (plan.strategy, why, error)
        return why
    return None


def statistics(mean, m2, weight):
    """fp64 Welford combine of partials along the last axis."""
    mean, m2, weight = (t.double() for t in (mean, m2, weight))
    total = weight.sum(-1)
    safe = torch.where(total == 0, 1, total)
    combined = (weight * mean).sum(-1) / safe
    spread = m2.sum(-1) + (weight * (mean - combined[:, None]) ** 2).sum(-1)
    return (
        torch.where(total == 0, 0, combined),
        torch.where(total == 0, 0, spread),
        total,
    )


def close(actual, expected, atol):
    torch.testing.assert_close(actual.double().cpu(), expected, rtol=1e-5, atol=atol)


def check(y, expected, component, n, plan):
    """Weight exact; mean to float32 rounding; m2 relative to its size."""
    y = y.cpu().double()
    if component == 2:
        assert torch.equal(y, expected[2]), (plan.strategy, n)
    elif component == 0:
        close(y, expected[0], 1e-6)
    else:
        close(y, expected[1], 1e-5 * max(1.0, float(expected[1].abs().max())))


@pytest.mark.parametrize("component", [0, 1, 2])
def test_welford_reduce_under_every_eligible_plan(component):
    semantic = reduce_descriptor(component)
    generator = torch.Generator().manual_seed(1009 + component)
    launched = set()
    for plan in filter(eligible, reduction_plans()):
        k = kernel(semantic, plan)
        for n in SHAPES:
            x = torch.randn(3, n, generator=generator) * 2 + 3
            expected = statistics(x, torch.zeros_like(x), torch.ones_like(x))
            y = torch.full((3,), float("nan"), device="cuda")
            if run(k, plan, semantic, (3, n), x.cuda(), y):
                continue
            launched.add(plan.identity)
            assert not y.isnan().any(), (plan.strategy, n)
            check(y, expected, component, n, plan)
    assert launched == {p.identity for p in filter(eligible, reduction_plans())}


@pytest.mark.parametrize("component", [0, 1, 2])
def test_welford_combine_with_empty_partials(component):
    """Some partials have weight 0, as masked split tails do; none give NaN."""
    semantic = combine_descriptor(component)
    generator = torch.Generator().manual_seed(2003 + component)
    launched = set()
    for plan in filter(eligible, reduction_plans()):
        k = kernel(semantic, plan)
        for n in SHAPES:
            weight = torch.randint(0, 4, (3, n), generator=generator).float()
            mean = torch.where(weight > 0, torch.randn(3, n, generator=generator), 0)
            m2 = torch.where(weight > 1, torch.rand(3, n, generator=generator), 0)
            expected = statistics(mean, m2, weight)
            y = torch.full((3,), float("nan"), device="cuda")
            tensors = (mean.cuda(), m2.cuda(), weight.cuda(), y)
            if run(k, plan, semantic, (3, n), *tensors):
                continue
            launched.add(plan.identity)
            assert not y.isnan().any(), (plan.strategy, n)
            check(y, expected, component, n, plan)
    assert launched == {p.identity for p in filter(eligible, reduction_plans())}


def test_every_unit_reads_one_mean_per_row():
    """A reduction-scope epilogue sees one mean bit pattern per row."""
    semantic = row_mean_descriptor()
    generator = torch.Generator().manual_seed(3001)
    launched = set()
    for plan in filter(eligible, reduction_plans()):
        k = kernel(semantic, plan)
        for n in (1, 33, 256, 257, 4099):
            x = torch.randn(3, n, generator=generator) * 2 + 3
            y = torch.empty(3, n, device="cuda")
            if run(k, plan, semantic, (3, n), x.cuda(), y):
                continue
            launched.add(plan.identity)
            y = y.cpu()
            assert torch.equal(y, y[:, :1].expand(3, n)), (plan.strategy, n)
            close(y[:, 0], x.double().mean(-1), 1e-5)
    assert launched == {p.identity for p in filter(eligible, reduction_plans())}


@pytest.mark.parametrize("kind", ["welford_reduce", "welford_combine"])
def test_sum_only_plans_reject_welford(tmp_path, kind):
    semantic = (
        reduce_descriptor(0) if kind == "welford_reduce" else combine_descriptor(0)
    )
    for i, plan in enumerate(reduction_plans()):
        if eligible(plan):
            continue
        cache = ArtifactCache(tmp_path / str(i))
        with pytest.raises((ValueError, RuntimeError)) as error:
            compile_descriptor(semantic, plan.to_dict(), 0, cache=cache)
        assert (
            f"strategy {plan.strategy} does not support reduction type {kind}"
            in str(error.value)
        )
        assert not list((tmp_path / str(i)).rglob("kernel.ptx"))


@pytest.mark.parametrize(
    "edit, message",
    [
        (
            lambda d: d["ops"][3].update(args=["m"]),
            "reduction type welford_combine takes 3 operand(s), got 1",
        ),
        (
            lambda d: d["ops"][4].update(index=3),
            "reduction type welford_combine has no component 3",
        ),
    ],
)
def test_malformed_welford_descriptors_rejected(tmp_path, edit, message):
    semantic = combine_descriptor(0)
    edit(semantic)
    plan = next(
        p for p in reduction_plans() if p.strategy == "shared_tree_reduction_baseline"
    )
    with pytest.raises((ValueError, RuntimeError)) as error:
        compile_descriptor(semantic, plan.to_dict(), 0, cache=ArtifactCache(tmp_path))
    assert message in str(error.value)
    assert not list(tmp_path.rglob("kernel.ptx"))


def test_welford_reduce_takes_one_operand(tmp_path):
    semantic = reduce_descriptor(0)
    semantic["ops"][1]["args"] = ["x", "x", "x"]
    plan = next(
        p for p in reduction_plans() if p.strategy == "shared_tree_reduction_baseline"
    )
    with pytest.raises((ValueError, RuntimeError)) as error:
        compile_descriptor(semantic, plan.to_dict(), 0, cache=ArtifactCache(tmp_path))
    assert "reduction type welford_reduce takes 1 operand(s), got 3" in str(error.value)
