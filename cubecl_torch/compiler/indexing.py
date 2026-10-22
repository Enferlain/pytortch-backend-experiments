"""Target-neutral direct signed-i64 index trees; no tensor-derived indices."""

import sympy
from torch.utils._sympy.functions import FloorDiv, Identity, ModularIndexing

from cubecl_torch.compiler.descriptor import MAX_RANK
from cubecl_torch.compiler.errors import UnsupportedCubeCL

LIMIT = (1 << 63) - 1


def encode(expr, symbols, linear=None):
    expr = sympy.sympify(expr)
    if expr == linear:
        return {"op": "linear"}
    if isinstance(expr, sympy.Integer):
        value = int(expr)
        if not -LIMIT <= value <= LIMIT:
            raise UnsupportedCubeCL("index constant outside bounded signed i64")
        return {"op": "constant", "value": value}
    if isinstance(expr, sympy.Symbol):
        return {"op": "scalar", "slot": symbols(expr)}
    if isinstance(expr, sympy.Pow):
        base, exponent = expr.args
        # Repeated dimensions are canonicalized by SymPy (e.g. B*B -> B**2), so
        # the exponent is bounded by the rank.
        if not isinstance(exponent, sympy.Integer) or not 0 <= exponent <= MAX_RANK:
            raise UnsupportedCubeCL(f"unsupported index power: {expr}")
        if exponent == 0:
            return {"op": "constant", "value": 1}
        if exponent == 1:
            return encode(base, symbols, linear)
        return {
            "op": "mul",
            "args": [encode(base, symbols, linear) for _ in range(int(exponent))],
        }
    if isinstance(expr, Identity):
        return encode(expr.args[0], symbols, linear)
    if isinstance(expr, ModularIndexing):
        a, b, c = expr.args
        if c.is_positive is not True:
            raise UnsupportedCubeCL(f"index modulus must be provably positive: {expr}")
        return {
            "op": "mod",
            "args": [
                encode(FloorDiv(a, b), symbols, linear),
                encode(c, symbols, linear),
            ],
        }
    kinds = {
        sympy.Add: "add",
        sympy.Mul: "mul",
        FloorDiv: "floor_div",
        sympy.Mod: "mod",
    }
    kind = next((v for t, v in kinds.items() if isinstance(expr, t)), None)
    if kind:
        args = list(expr.args)
        if kind in ("floor_div", "mod") and args[1].is_positive is not True:
            raise UnsupportedCubeCL(f"index divisor must be provably positive: {expr}")
        return {"op": kind, "args": [encode(a, symbols, linear) for a in args]}
    raise UnsupportedCubeCL(
        f"unsupported direct index expression: {expr} ({type(expr).__name__})"
    )


def evaluate(tree, scalars, linear=0):
    op = tree["op"]
    if op == "constant":
        result = tree["value"]
    elif op == "linear":
        result = linear
    elif op == "scalar":
        result = scalars[tree["slot"]]
    else:
        args = [evaluate(a, scalars, linear) for a in tree["args"]]
        if op == "add":
            result = sum(args)
        elif op == "mul":
            result = __import__("math").prod(args)
        elif op == "floor_div":
            result = args[0] // args[1]
        elif op == "mod":
            result = args[0] % args[1]
        else:
            raise ValueError(op)
    if not -LIMIT <= result <= LIMIT:
        raise ValueError("signed index arithmetic overflow")
    return result
