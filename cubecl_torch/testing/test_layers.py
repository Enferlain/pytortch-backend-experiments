"""Layer dependency rule: runtime <- compiler <- inductor/ops.

Checked statically (every import statement in a layer) and dynamically (a fresh
process importing every module of a layer loads nothing above it).
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent
FORBIDDEN_TORCH = ("torch._inductor", "torch._dynamo")
# Modules a layer must never import, beyond Inductor and Dynamo.
FORBIDDEN = {
    "runtime": ("cubecl_torch.compiler", "cubecl_torch.inductor", "cubecl_torch.ops"),
    "compiler": ("cubecl_torch.inductor", "cubecl_torch.ops"),
}


def layer_modules(layer):
    for path in sorted((PACKAGE / layer).rglob("*.py")):
        parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
        yield path, ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imported_names(path, module):
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - node.level + 1]
                base = ".".join([*anchor, base] if base else anchor)
            yield base
            yield from (f"{base}.{alias.name}" for alias in node.names)


def matches(name, prefixes):
    return any(name == p or name.startswith(p + ".") for p in prefixes)


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_static_imports(layer):
    forbidden = (*FORBIDDEN[layer], *FORBIDDEN_TORCH)
    found = [
        f"{module}: {name}"
        for path, module in layer_modules(layer)
        for name in imported_names(path, module)
        if matches(name, forbidden)
    ]
    assert not found, f"{layer}/ imports above its layer: {found}"


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_fresh_process_imports(layer):
    modules = [module for _, module in layer_modules(layer)]
    script = (
        "import importlib, json, sys\n"
        f"for m in {modules!r}: importlib.import_module(m)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    loaded = json.loads(out.splitlines()[-1])
    forbidden = (*FORBIDDEN[layer], *FORBIDDEN_TORCH, "triton")
    assert not [m for m in loaded if matches(m, forbidden)]
