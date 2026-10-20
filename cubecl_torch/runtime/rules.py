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
    """A table rule was evaluated outside its finite domain."""


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
        elif op == "floor_div":
            result = args[0] // args[1]
        elif op == "mod":
            result = args[0] % args[1]
        else:
            raise ValueError(f"unknown rule op {op!r}")
    if not -LIMIT <= result <= LIMIT:
        raise ValueError("signed index arithmetic overflow")
    return result
