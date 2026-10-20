"""Semantic descriptor validation and the schema-5 packet sent to the compiler.

A semantic descriptor (schema 3 or 4) states what a kernel computes: buffers with
symbolic layouts, i64 runtime scalars, SSA ops and extents. The packet adds the
physical plan and a content-derived entry point.
"""

import hashlib
import json

from cubecl_torch.compiler.errors import MalformedDescriptor

# Semantic schemas the schema-5 packet replays (pointwise 4, reduction 3).
SCHEMAS = (3, 4)
DTYPES = ("float32", "float16", "bfloat16", "bool", "int64")
RULE_OPS = ("constant", "scalar", "add", "mul", "floor_div", "mod")
MAX_BUFFERS = 6


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _rule(tree, scalars, where):
    if not isinstance(tree, dict) or tree.get("op") not in RULE_OPS:
        raise MalformedDescriptor(f"{where}: not a layout expression: {tree!r}")
    op = tree["op"]
    if op == "constant":
        if type(tree.get("value")) is not int:
            raise MalformedDescriptor(f"{where}: constant without an integer value")
    elif op == "scalar":
        if type(tree.get("slot")) is not int or not 0 <= tree["slot"] < scalars:
            raise MalformedDescriptor(
                f"{where}: scalar slot {tree.get('slot')!r} outside the {scalars} declared scalars"
            )
    else:
        args = tree.get("args")
        if not isinstance(args, list) or not args:
            raise MalformedDescriptor(f"{where}: {op} without arguments")
        for i, arg in enumerate(args):
            _rule(arg, scalars, f"{where}.args[{i}]")


def validate(semantic):
    """Raise MalformedDescriptor naming the first schema violation."""
    if not isinstance(semantic, dict):
        raise MalformedDescriptor("descriptor is not an object")
    for name in ("schema", "buffers", "scalars", "ops", "numel", "index_dtype"):
        if name not in semantic:
            raise MalformedDescriptor(f"descriptor lacks required field {name!r}")
    if semantic["schema"] not in SCHEMAS:
        raise MalformedDescriptor(f"unsupported semantic schema {semantic['schema']!r}")
    if semantic["index_dtype"] != "int64":
        raise MalformedDescriptor("index_dtype must be int64")
    scalars = semantic["scalars"]
    if not isinstance(scalars, list) or any(
        not isinstance(s, dict)
        or s.get("dtype") != "int64"
        or s.get("id") != i
        or type(s.get("positive")) is not bool
        for i, s in enumerate(scalars)
    ):
        raise MalformedDescriptor("scalars must be int64 declarations with ids 0..n-1")
    buffers = semantic["buffers"]
    if not isinstance(buffers, list) or not 1 <= len(buffers) <= MAX_BUFFERS:
        raise MalformedDescriptor(f"descriptor must declare 1..{MAX_BUFFERS} buffers")
    for i, b in enumerate(buffers):
        where = f"buffers[{i}]"
        if not isinstance(b, dict):
            raise MalformedDescriptor(f"{where} is not an object")
        for name in ("access", "dtype", "size", "stride", "offset"):
            if name not in b:
                raise MalformedDescriptor(f"{where} lacks required field {name!r}")
        if b["dtype"] not in DTYPES:
            raise MalformedDescriptor(f"{where}: unsupported dtype {b['dtype']!r}")
        if not isinstance(b["size"], list) or not isinstance(b["stride"], list):
            raise MalformedDescriptor(f"{where}: size and stride must be lists")
        if len(b["size"]) != len(b["stride"]):
            raise MalformedDescriptor(
                f"{where}: size rank {len(b['size'])} != stride rank {len(b['stride'])}"
            )
        for name in ("size", "stride"):
            for j, rule in enumerate(b[name]):
                _rule(rule, len(scalars), f"{where}.{name}[{j}]")
        _rule(b["offset"], len(scalars), f"{where}.offset")
    if any(
        b["access"] != ("write" if i == len(buffers) - 1 else "read")
        for i, b in enumerate(buffers)
    ):
        raise MalformedDescriptor(
            "unsupported input contract: exactly one final output buffer"
        )
    _rule(semantic["numel"], len(scalars), "numel")
    if semantic["schema"] == 3:
        for name in ("outer", "reduction_extent"):
            if name not in semantic:
                raise MalformedDescriptor(
                    f"reduction descriptor lacks required field {name!r}"
                )
            _rule(semantic[name], len(scalars), name)
    ops = semantic["ops"]
    if (
        not isinstance(ops, list)
        or not ops
        or any(not isinstance(o, dict) or "op" not in o for o in ops)
    ):
        raise MalformedDescriptor("ops must be a non-empty list of operations")
    for i, op in enumerate(ops):
        if op["op"] in ("load", "store", "atomic_add") and not (
            type(op.get("buffer")) is int and 0 <= op["buffer"] < len(buffers)
        ):
            raise MalformedDescriptor(
                f"ops[{i}]: {op['op']} references buffer {op.get('buffer')!r} outside the {len(buffers)} declared"
            )
    if not any(op["op"] in ("store", "atomic_add") for op in ops):
        raise MalformedDescriptor(
            "descriptor writes no buffer (no store or atomic_add)"
        )
    if semantic["schema"] == 3 and {
        op.get("iteration_scope") for op in ops if op["op"] == "store"
    } not in ({"outer"}, {"reduction"}):
        raise MalformedDescriptor("reduction output logical domain mismatch")
    effects = semantic.get("effects", {})
    if not isinstance(effects, dict):
        raise MalformedDescriptor("effects must be an object")


def packet(semantic, plan):
    """Schema-5 compiler packet and its content-derived entry point."""
    value = {"schema": 5, "semantic": semantic, "physical_plan": plan}
    entry = "cubecl5_" + digest(canonical(value).encode())[:16]
    value["entrypoint"] = entry
    return value, entry
