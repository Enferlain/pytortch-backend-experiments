"""Reduction combiners: strategy eligibility per reduction type.

OpenSpec change generic-reduction-combiners; the table mirrors Combiner in
native/src/descriptor3.rs.
"""

from cubecl_torch.compiler.plan import catalog
from cubecl_torch.inductor import legality


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
