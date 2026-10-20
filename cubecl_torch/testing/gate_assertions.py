"""Enumerate assertion sites in the frozen Stage 0-5 gate scripts.

Feeds ``contract_inventory.md`` and its completeness test. A site is an
``assert`` statement, a ``raise`` of AssertionError/SystemExit, a call to a
checking helper (``assert_close``, ``assert_equal``, ``expect``, ``rejects``,
``check``, ``require``) or, in Rust, an ``assert!``-family macro.
"""

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = (
    # Stage 5 final gates.
    "stage5/components.py",
    "stage5/controls.py",
    "stage5/proof_controls.py",
    "stage5/descriptor_controls.py",
    "stage5/launch_controls.py",
    "stage5/cache_probe.py",
    "stage5/compiler_compare.py",
    # Stage 4 gates.
    "stage4/verify.py",
    "stage4/runtime_checks.py",
    "stage4/verify_bounds.py",
    "stage4/counter_guard_probe.py",
    "stage4/components.py",
    # stage4/regressions.py -> stage3/regressions.py chain (stages 0-3).
    "stage0/verify.py",
    "stage1/verify.py",
    "stage1/verify_boundaries.py",
    "stage2/verify.py",
    "stage2/components.py",
    "stage0/tests/fp_rewrites.rs",
    "fp_hardening/verify.py",
    "fp_hardening/verify_inductor.py",
    "stage3/verify.py",
    "stage3/components.py",
)
HELPERS = {"assert_close", "assert_equal", "expect", "rejects", "check", "require"}
RUST = re.compile(r"\b(assert|assert_eq|assert_ne|debug_assert)!\s*\(")


def _function_name(stack):
    return next((n.name for n in reversed(stack) if hasattr(n, "name")), "<module>")


def python_sites(path):
    source = path.read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    sites = []

    def visit(node, stack):
        kind = None
        if isinstance(node, ast.Assert):
            kind = "assert"
        elif isinstance(node, ast.Raise) and node.exc is not None:
            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if getattr(target, "id", getattr(target, "attr", None)) in (
                "AssertionError",
                "SystemExit",
            ):
                kind = "raise"
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name in HELPERS:
                kind = "call:" + name
        if kind:
            sites.append(
                {
                    "line": node.lineno,
                    "kind": kind,
                    "function": _function_name(stack),
                    "text": lines[node.lineno - 1].strip()[:160],
                }
            )
        for child in ast.iter_child_nodes(node):
            visit(child, [*stack, node] if hasattr(node, "name") else stack)

    visit(tree, [])
    return sorted(sites, key=lambda s: s["line"])


def rust_sites(path):
    sites, function = [], "<module>"
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if m := re.match(r"\s*(?:pub\s+)?fn\s+(\w+)", line):
            function = m[1]
        if RUST.search(line):
            sites.append(
                {
                    "line": number,
                    "kind": "assert!",
                    "function": function,
                    "text": line.strip()[:160],
                }
            )
    return sites


def sites():
    """``{script: [site, ...]}`` for every gate script."""
    result = {}
    for script in SCRIPTS:
        path = REPO / script
        result[script] = (
            rust_sites(path) if path.suffix == ".rs" else python_sites(path)
        )
    return result


if __name__ == "__main__":
    total = 0
    for script, found in sites().items():
        total += len(found)
        print(f"## {script} ({len(found)})")
        for s in found:
            print(f"{s['line']:5d} {s['kind']:18s} {s['function']:28s} {s['text']}")
    print("total", total)
