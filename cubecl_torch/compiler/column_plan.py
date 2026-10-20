"""Initial normal admission of the qualified scalar column layout.

These complete templates and finite extents are a qualification boundary.
No general layout recognition or symbolic bound inference is claimed.
"""

import json
from functools import cache
from pathlib import Path

STRATEGY = "bounded_column_pointwise"
SHAPES = frozenset(((3, 255), (64, 256), (64, 257)))


@cache
def _templates():
    return frozenset(
        json.dumps(t, sort_keys=True)
        for t in json.loads(
            Path(__file__).with_name("column_templates.json").read_text()
        )
    )


def semantic_reason(semantic):
    try:
        matches = (
            isinstance(semantic, dict)
            and json.dumps(semantic, sort_keys=True) in _templates()
        )
    except (TypeError, ValueError):
        matches = False
    if not matches:
        return "column requires an exact qualified FP16/BF16 template"
    return None


def runtime_reason(semantic, scalars):
    why = semantic_reason(semantic)
    if why:
        return why
    return extent_reason(scalars)


def extent_reason(scalars):
    if len(scalars) != 2 or (int(scalars[1]), int(scalars[0])) not in SHAPES:
        return "column extents outside initial qualified shapes"
    return None


def metadata(scalars):
    # Literal translation of the already-qualified U64 FastDivmod parameters.
    # Domain is independently enforced by admission and live Kernel validation.
    n, b = map(int, scalars)
    if (b, n) not in SHAPES:
        raise ValueError("column extents outside initial qualified shapes")
    values = []
    for divisor in (b, n):
        shift = (divisor - 1).bit_length()
        multiplier = ((1 << 64) * ((1 << shift) - divisor)) // divisor + 1
        values.extend(
            (multiplier if multiplier < 1 << 63 else multiplier - (1 << 64), shift)
        )
    return tuple(values)
