"""The coverage checklist classifies exactly the pinned Inductor's items, and its
support claims match the operations the backend's capability checks accept."""

import ast

from cubecl_torch.inductor.semantic_checks import OPS
from cubecl_torch.testing import coverage_inventory as inventory

STATUSES = set(inventory.STATUS_MEANING)
# Kernel hooks the capability checks see as subblocks/indexing, not call_method ops.
HOOKS = {"indirect_indexing", "check_bounds"}


def accepted_ops():
    """Op names the capability checks accept: OPS plus every extra_ops tuple."""
    package = inventory.HERE.parent / "inductor"
    names = set(OPS)
    for module in ("effects.py", "reduction_checks.py"):
        for node in ast.walk(ast.parse((package / module).read_text())):
            if isinstance(node, ast.keyword) and node.arg == "extra_ops":
                names |= {e.value for e in node.value.elts}
    return names


def test_checklist_matches_pinned_inductor():
    ops, reductions, features = inventory.pinned_items()
    assert set(inventory.OPS) == set(ops)
    assert set(inventory.REDUCTIONS) == set(reductions)
    assert set(inventory.FEATURES) == set(features)


def test_families_and_statuses_are_declared():
    families = inventory.FAMILIES
    for table in (
        inventory.OPS,
        inventory.REDUCTIONS,
        inventory.FEATURES,
        inventory.SURFACES,
    ):
        for name, (family, status, _) in table.items():
            assert family in families, name
            assert status in STATUSES, name
    for family, (_, prerequisites, _, _) in families.items():
        assert set(prerequisites) <= set(families), family
        assert family not in prerequisites


def test_prerequisites_are_acyclic():
    order, pending = [], dict(inventory.FAMILIES)
    while pending:
        ready = [f for f, (_, pre, _, _) in pending.items() if set(pre) <= set(order)]
        assert ready, f"prerequisite cycle among {sorted(pending)}"
        order += ready
        for f in ready:
            del pending[f]


def test_support_claims_match_the_capability_checks():
    accepted = accepted_ops()
    for name, (_, status, _) in inventory.OPS.items():
        if name in HOOKS:
            continue
        if status in ("implemented", "complete"):
            assert name in accepted, f"{name} marked {status} but not accepted"
        elif status == "missing":
            assert name not in accepted, f"{name} accepted but marked missing"


def test_implemented_items_declare_their_domain():
    """'implemented' means a declared domain, never the whole contract."""
    for table in (
        inventory.OPS,
        inventory.REDUCTIONS,
        inventory.FEATURES,
        inventory.SURFACES,
    ):
        for name, (_, status, note) in table.items():
            if status == "implemented":
                assert note.strip(), f"{name}: implemented without a declared domain"
            # 'complete' needs per-item tests of the whole contract and the
            # numerical policy (4z8.17); claiming it requires extending this test.
            assert status != "complete", f"{name}: complete is not backed by tests yet"


def test_markdown_is_current():
    assert inventory.MARKDOWN.read_text() == inventory.render()
