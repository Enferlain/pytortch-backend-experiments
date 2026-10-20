"""Every frozen gate assertion is classified, and live ones link to real tests (6.1/6.2)."""

import ast

from cubecl_torch.testing import contract_inventory
from cubecl_torch.testing.gate_assertions import SCRIPTS, sites

HERE = contract_inventory.HERE


def test_every_site_classified():
    missing = [
        f"{script}:{s['line']}"
        for script, rows in sites().items()
        for s in rows
        if contract_inventory.classify(script, s["line"]) is None
    ]
    assert not missing
    assert set(contract_inventory.RULES) == set(SCRIPTS)


def test_live_entries_link_to_existing_tests():
    defined = {}
    for path in HERE.glob("test_*.py"):
        tree = ast.parse(path.read_text())
        defined[path.name] = {
            n.name for n in tree.body if isinstance(n, ast.FunctionDef)
        }
    for script, rules in contract_inventory.RULES.items():
        for _, _, target, *_ in rules:
            if target in contract_inventory.REASONS:
                continue
            file, _, name = target.partition("::")
            assert name in defined.get(file, ()), f"{script}: {target} not found"


def test_markdown_is_current():
    assert contract_inventory.MARKDOWN.read_text() == contract_inventory.render()
