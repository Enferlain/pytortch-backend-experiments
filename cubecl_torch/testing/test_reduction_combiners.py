"""Reduction combiners: strategy eligibility per reduction type.

OpenSpec change generic-reduction-combiners; the table mirrors Combiner in
native/src/descriptor3.rs.
"""

import pytest
import torch

from cubecl_torch.compiler.build import compile_descriptor
from cubecl_torch.compiler.plan import catalog
from cubecl_torch.inductor import legality
from cubecl_torch.runtime.cache import ArtifactCache
from cubecl_torch.testing.fixtures import descriptor


def semantic(kind):
    return {"schema": 3, "ops": [{"op": "reduction", "reduction_type": kind}]}


def reduction_plans():
    return [p for p in catalog() if "pointwise" not in p.strategy]


def test_sum_is_eligible_for_every_reduction_strategy():
    plans = reduction_plans()
    assert plans
    for plan in plans:
        assert legality.combiner_reason(plan, semantic("sum")) is None, plan.strategy


def test_unknown_reduction_type_is_named():
    for plan in reduction_plans():
        assert (
            legality.combiner_reason(plan, semantic("prod"))
            == "reduction type prod unsupported"
        )


def test_pointwise_descriptors_are_not_checked():
    plan = next(p for p in catalog() if "pointwise" in p.strategy)
    assert legality.combiner_reason(plan, {"schema": 2, "ops": []}) is None


def test_sum_only_strategies_exist_in_the_catalog():
    strategies = {p.strategy for p in catalog()}
    assert legality.SUM_ONLY_STRATEGIES <= strategies


def test_sum_only_strategies_reject_other_types(monkeypatch):
    monkeypatch.setattr(legality, "REDUCTION_TYPES", frozenset({"sum", "probe"}))
    for plan in reduction_plans():
        reason = legality.combiner_reason(plan, semantic("probe"))
        if plan.strategy in legality.SUM_ONLY_STRATEGIES:
            assert reason == (
                f"strategy {plan.strategy} does not support reduction type probe"
            )
        else:
            assert reason is None, plan.strategy


def as_type(semantic, kind):
    for op in semantic["ops"]:
        if op["op"] == "reduction":
            op["reduction_type"] = kind
    return semantic


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("kind", ["max", "min"])
def test_hand_authored_descriptor_follows_strategy_eligibility(tmp_path, kind):
    """A max/min descriptor compiles wherever sum does, except sum-only plans."""
    outcomes = {}
    for i, plan in enumerate(reduction_plans()):
        results = []
        for reduction in ("sum", kind):
            cache = ArtifactCache(tmp_path / f"{i}-{reduction}")
            semantic = as_type(descriptor("reduction"), reduction)
            try:
                compile_descriptor(semantic, plan.to_dict(), 0, cache=cache)
                results.append("compiled")
            except (ValueError, RuntimeError) as error:
                written = list((tmp_path / f"{i}-{reduction}").rglob("kernel.ptx"))
                results.append(("rejected", str(error), bool(written)))
        outcomes[plan.identity] = (plan.strategy, *results)
    for strategy, summed, other in outcomes.values():
        if summed != "compiled":
            continue
        if strategy in legality.SUM_ONLY_STRATEGIES - {legality.FINAL9}:
            assert other[0] == "rejected" and not other[2], strategy
            assert f"does not support reduction type {kind}" in other[1], other[1]
        elif strategy != legality.FINAL9:
            assert other == "compiled", (strategy, other)
    compiled = {s for s, summed, other in outcomes.values() if other == "compiled"}
    assert "shared_tree_compensated" in {
        s for s, summed, other in outcomes.values() if summed == "compiled"
    }
    assert len(compiled) >= 4, compiled
