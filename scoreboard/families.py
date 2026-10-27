"""Map a CubeCL rejection reason to its coverage-inventory family and tracking bead.

Operation and reduction reasons are looked up in ``coverage_inventory`` (the
checklist derived from the pinned Inductor); structural reasons (indexing,
dtypes, kernel shape, effects) match the patterns below. A reason nothing
matches maps to ``unmapped``: either a new gap to classify, or a pattern to add.
"""

from __future__ import annotations

import re

from cubecl_torch.testing.coverage_inventory import FAMILIES, OPS, REDUCTIONS

# (pattern, family, bead or ""); first match wins, so specific patterns come first.
PATTERNS = [
    (r"index (divisor|modulus) must be provably positive", "index", "4z8.22"),
    (r"\bMax\(|\bMin\(", "index", "4z8.18"),
    (r"\bAbs\(", "index", "4z8.26"),
    # Before 2026-10-26 every direct check_bounds was refused.
    (r"semantic operation check_bounds unsupported", "core", "4z8.27"),
    (r"check_bounds with requested bounds", "core", "4z8.30"),
    (r"one logical reduction dimension required", "index", "4z8.7"),
    # Before 2026-10-26 the indirect path refused floating index values.
    (r"non-int64 symbolic value expression", "index", "4z8.8"),
    (r"unsupported (direct )?index expression|layout/index expression", "index", ""),
    (
        r"one disjoint (external )?output required|one output per node",
        "reduce",
        "4z8.21",
    ),
    # Before 2026-10-25 one message covered both arity limits.
    (r"one to four direct inputs and one output", "core", "4z8.21, 4z8.23"),
    (
        r"more than \d+ direct inputs|one to (four|five) (direct|external) inputs",
        "core",
        "4z8.23",
    ),
    # Before 2026-10-27 (float remainders admitted by x3c).
    (r"float mod/fmod unsupported", "extrema", "x3c"),
    (r"one to four reductions per cooperative schedule", "reduce", ""),
    (r"different reduction domains|reduction fusion iteration domain", "reduce", ""),
    (r"physical reduction extent", "reduce", ""),
    # Before 2026-10-26 (stage-era path option).
    (r"^(?:.*: )?reductions unsupported", "reduce", ""),
    (r"storage dtype torch\.complex", "core", "4z8.24"),
    (r"masked reduction/effect", "core", ""),
    (r"user-visible mutation", "effects", "4z8.9"),
    (r"atomic", "effects", ""),
    (r"\bRNG\b|random", "rng", ""),
    (r"scan, sort, bucketize", "scan", "4x9.13"),
    (r"indirect reduction", "core", "4z8.25"),
    (r"indirect", "core", ""),
    (
        (
            r"integer/bool arithmetic|bitwise|signed-i64|signed index values"
            r"|bool add/sub/mul/neg|bool abs|bool square/relu|comparison operands|logical operands"
            r"|integer division operands|shift operands|int_truediv operands|to_dtype_bitcast"
        ),
        "int",
        "",
    ),
    (r"float64", "dtypes", ""),
    # int64 storage is supported, except in reduction kernels (index buffers there).
    (r"tensor storage dtype torch\.int64", "core", "4z8.25"),
    (r"dtype for \w+ unsupported|storage dtype|tensor storage dtype", "dtypes", ""),
    (r"CUDA tensor inputs required", "runtime", "4z8.3"),
    (r"templates", "templates", ""),
]


def family(reason: str) -> tuple[str, str]:
    """``(family, beads)``: a reason-specific bead, else the family's beads (may be empty)."""
    name, bead = _match(reason)
    # core and index collect unrelated surfaces: their bead lists say nothing
    # about one reason.
    if not bead and name in FAMILIES and name not in ("core", "index"):
        bead = ", ".join(FAMILIES[name][2])
    return name, bead


def _match(reason):
    op = None
    if m := re.search(r"semantic operation (\w+) unsupported", reason):
        op = OPS.get(m.group(1))
        # An implemented op rejected here hit a context limit: a pattern may name it.
        if not op or op[1] != "implemented":
            return (op[0], "") if op else ("unmapped", "")
    if m := re.search(r"unsupported reduction (\w+) over (\S+?):", reason):
        kind, dtype = m.groups()
        entry = REDUCTIONS.get(kind)
        if entry and entry[1] == "implemented":
            # An implemented reduction over an unsupported dtype: integer combiners
            # (4x9.18) or float64.
            if "int" in dtype or "bool" in dtype:
                return ("reduce", "4x9.18")
            return ("dtypes", "")
        return (entry[0], "") if entry else ("unmapped", "")
    for pattern, name, bead in PATTERNS:
        if re.search(pattern, reason):
            return name, bead
    return (op[0], "") if op else ("unmapped", "")
