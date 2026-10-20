"""Integer rules over launch scalars: the runtime's only expression language.

A rule is a JSON tree evaluated with exact signed-i64 arithmetic:

- ``{"op": "constant", "value": int}`` and ``{"op": "scalar", "slot": int}``;
- ``add``, ``mul``, ``floor_div`` and ``mod`` over ``"args"``;
- ``{"op": "table", "args": [...], "entries": [[[key...], value], ...]}``, a
  finite lookup that fails for keys outside the table.

Producers translate their own expressions into rules; the runtime never sees
descriptors.
"""

import math

LIMIT = (1 << 63) - 1


class RuleDomainError(ValueError):
    """A rule was evaluated outside its domain: table miss, zero divisor, overflow."""


def constant(value):
    return {"op": "constant", "value": value}


def scalar(slot):
    return {"op": "scalar", "slot": slot}


def ceil_div(rule, divisor):
    return {
        "op": "floor_div",
        "args": [
            {"op": "add", "args": [rule, constant(divisor - 1)]},
            constant(divisor),
        ],
    }


def evaluate(tree, scalars):
    op = tree["op"]
    if op == "constant":
        result = tree["value"]
    elif op == "scalar":
        result = scalars[tree["slot"]]
    elif op == "table":
        key = [evaluate(a, scalars) for a in tree["args"]]
        for entry, value in tree["entries"]:
            if entry == key:
                result = value
                break
        else:
            raise RuleDomainError(f"table rule has no entry for {key}")
    else:
        args = [evaluate(a, scalars) for a in tree["args"]]
        if op == "add":
            result = sum(args)
        elif op == "mul":
            result = math.prod(args)
        elif op in ("floor_div", "mod"):
            if args[1] == 0:
                raise RuleDomainError(f"{op} rule divides by zero")
            result = args[0] // args[1] if op == "floor_div" else args[0] % args[1]
        else:
            raise ValueError(f"unknown rule op {op!r}")
    if not -LIMIT <= result <= LIMIT:
        raise RuleDomainError("signed index arithmetic overflow")
    return result
