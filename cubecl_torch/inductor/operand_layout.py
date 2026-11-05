"""Memory layouts of a reduction kernel's loads, for the fused ecosystem reduce
(openspec fused-reduction-bridge, design 5).

``load_layout`` runs Inductor's own block-pointer analysis, the steps of
``TritonKernel.codegen_indexing``'s ``match_block_expr``: the index is split
into the terms of each range tree, mixed terms are rejected, and each tree's
terms are matched as affine (``s * xindex``) or as FloorDiv/ModularIndexing
blocks, giving that tree's dimensions and strides. ``operand_layout`` then
takes the layout every full-size load of the reduction operand agrees on. It
never chooses between loads that disagree: that is physical policy, left to
Burn's reference selection or a CubeK abstraction when a case needs it.

Layouts are plain JSON: ``{"outer": {"dims": [...], "strides": [...]},
"reduction": {...}, "offset": int}``, dimensions outermost first. Only static
sizes are described; a symbolic size gives no layout.
"""

# Inductor is imported inside the functions that use it: this module must be
# importable before the Inductor overlay is installed.


def _static(values):
    try:
        return [int(v) for v in values]
    except TypeError:
        return None


def _tree_layout(expr, tree):
    """Dimensions and strides of one range tree's terms, or ``None``."""
    import sympy
    from torch._inductor.codegen.block_analysis import BlockPatternMatcher
    from torch._inductor.virtualized import V
    from torch.utils._sympy.functions import FloorDiv, ModularIndexing

    sizevars = V.graph.sizevars
    symbol = tree.symbol()
    numel = sizevars.remove_precomputed_replacements(tree.numel)
    if expr == 0:
        # The load does not move along this tree: a broadcast.
        dims = _static([numel])
        return dims and {"dims": dims, "strides": [0]}
    factored = BlockPatternMatcher.factor_index_expr(expr, symbol)
    stride = BlockPatternMatcher.match_affine_block_expr(factored, symbol)
    if stride is not None:
        dims, strides = _static([numel]), _static([stride])
    else:
        denom, modulo = sympy.symbols(
            "denom modulo", cls=lambda n: sympy.Wild(n, exclude=[symbol])
        )
        num_dims = max(
            2,
            len(tree.nodes),
            factored.count(FloorDiv(symbol, denom))
            + factored.count(ModularIndexing(symbol, denom, modulo)),
        )
        match = BlockPatternMatcher.match_mod_div_block_expr(
            factored, symbol, numel, num_dims
        )
        if match is None:
            return None
        dims, strides = _static(match[0]), _static(match[1])
    if dims is None or strides is None:
        return None
    return {"dims": dims, "strides": strides}


def load_layout(kernel, index):
    """The layout of a load at ``index`` in reduction ``kernel``, or ``None``."""
    from torch._inductor.codegen.block_analysis import BlockPatternMatcher
    from torch._inductor.utils import sympy_subs
    from torch._inductor.virtualized import V

    sizevars = V.graph.sizevars
    relative = sympy_subs(
        index, {v: t.expr for v, t in kernel.range_tree_nodes.items()}
    )
    relative = sizevars.remove_precomputed_replacements(relative)
    trees = kernel.range_trees
    symbols = {t.symbol() for t in trees}
    parts = [
        BlockPatternMatcher.get_subexpr_involving_symbol(relative, t.symbol())
        for t in trees
    ]
    layouts = []
    for tree, part in zip(trees, parts, strict=True):
        if len(symbols & part.free_symbols) > 1:
            return None
        layout = _tree_layout(part, tree)
        if layout is None:
            return None
        layouts.append(layout)
    offset = _static([relative - sum(parts)])
    if offset is None:
        return None
    outer, reduction = layouts
    return {"outer": outer, "reduction": reduction, "offset": offset[0]}


def normalized(dims, strides):
    """``dims``/``strides`` without size-one dimensions and with adjacent
    dimensions merged where the outer one steps exactly over the inner one."""
    kept = [(d, s) for d, s in zip(dims, strides, strict=True) if d != 1]
    merged = []
    for d, s in kept:
        if merged and merged[-1][1] == s * d:
            merged[-1] = (merged[-1][0] * d, s)
        else:
            merged.append((d, s))
    return [d for d, _ in merged], [s for _, s in merged]


def _full_size(layout):
    return all(
        s != 0
        for part in ("outer", "reduction")
        for d, s in zip(layout[part]["dims"], layout[part]["strides"], strict=True)
        if d != 1
    )


def operand_layout(loads):
    """``(layout, None)`` agreed by the full-size ``loads`` (their layouts, or
    ``None`` for a load without one), or ``(None, reason)``."""
    if any(layout is None for layout in loads):
        return None, "a load of the reduction operand has no block layout"
    full = []
    for layout in loads:
        if not _full_size(layout):
            continue  # a broadcast load does not compete
        outer = normalized(layout["outer"]["dims"], layout["outer"]["strides"])
        reduction = normalized(
            layout["reduction"]["dims"], layout["reduction"]["strides"]
        )
        # Offsets are each buffer's own; the replay keeps the recorded indices.
        full.append(
            {
                "outer": {"dims": outer[0], "strides": outer[1]},
                "reduction": {"dims": reduction[0], "strides": reduction[1]},
            }
        )
    if not full:
        return None, "no full-size load of the reduction operand"
    for other in full[1:]:
        if other != full[0]:
            return None, f"loads disagree on layout: {full[0]} and {other}"
    agreed = full[0]
    if len(agreed["reduction"]["dims"]) != 1:
        return None, "the reduction spans several memory dimensions"
    return agreed, None
