"""Bind a descriptor and physical plan into the runtime launch contract.

This is the descriptor evaluation the Stage 2-5 runtimes did on every launch,
expressed once as runtime rules over the launch scalars. The resulting guards
and messages match ``stage5/artifact.py``.
"""

from cubecl_torch.compiler import column_plan
from cubecl_torch.compiler.errors import MalformedDescriptor
from cubecl_torch.runtime import rules
from cubecl_torch.runtime.artifact import Guard, LaunchShape, ScalarParam, TensorParam

REDUCTION_EXTENT = (
    "unsupported physical reduction extent (outer grid or reduction > 1048576)"
)
REDUCTION_DOMAIN = "reduction output logical domain mismatch"
POINTWISE_EXTENT = "pointwise output extent mismatch"
RNG_EXTENT = "unsupported RNG physical extent: signed-i32 logical counter domain"
COLUMN_EXTENT = "unsupported column physical contract: column extents outside initial qualified shapes"


def _mul(*args):
    return {"op": "mul", "args": list(args)}


def _column_rules():
    """FastDivmod metadata for the finite qualified column shapes, as tables."""
    keys = sorted(column_plan.SHAPES)  # (b, n) == (scalar 1, scalar 0)
    args = [rules.scalar(1), rules.scalar(0)]
    words = {key: column_plan.metadata((key[1], key[0])) for key in keys}
    return [
        {"op": "table", "args": args, "entries": [[list(k), words[k][i]] for k in keys]}
        for i in range(4)
    ]


def bind(semantic, plan, manifest):
    """Contract fields for ``KernelArtifact`` (everything but PTX, entry, provenance)."""
    effects = semantic.get("effects", {})
    reduction = semantic["schema"] == 3
    guards = []
    if reduction:
        outer = semantic["outer"]
        extent = semantic["reduction_extent"]
        guards += [
            Guard("range", (outer,), REDUCTION_EXTENT, 0, 2147483647),
            Guard("range", (extent,), REDUCTION_EXTENT, 0, 1048576),
        ]
        scopes = {
            op["iteration_scope"] for op in semantic["ops"] if op["op"] == "store"
        }
        if scopes not in ({"outer"}, {"reduction"}):
            raise MalformedDescriptor(REDUCTION_DOMAIN)
        expected = _mul(outer, extent) if scopes == {"reduction"} else outer
        guards.append(Guard("output_numel", (expected,), REDUCTION_DOMAIN))
        n = _mul(outer, rules.constant(256))
        grid = rules.ceil_div(outer, plan["outputs_per_workgroup"])
    else:
        n = semantic["numel"]
        if not effects.get("mutation"):
            guards.append(Guard("output_numel", (n,), POINTWISE_EXTENT))
        x, y, z = plan["workgroup"]
        span = x * y * z * plan["vector_width"] * plan["elements_per_unit"]
        grid = rules.ceil_div(n, span)
    if effects.get("rng") and not effects.get("mutation"):
        guards.append(Guard("range", (n,), RNG_EXTENT, maximum=(1 << 31) - 1))
    count = len(semantic["scalars"])
    info = [n, *(rules.scalar(i) for i in range(count))]
    if plan["strategy"] == column_plan.STRATEGY:
        if count != 2:
            raise MalformedDescriptor("column plan requires exactly two scalars")
        guards.append(
            Guard(
                "member",
                (rules.scalar(1), rules.scalar(0)),
                COLUMN_EXTENT,
                allowed=tuple(sorted(column_plan.SHAPES)),
            )
        )
        info += _column_rules()
    metadata = manifest["metadata"]
    if metadata["bytes"] != 8 * (len(semantic["buffers"]) + len(info)):
        raise MalformedDescriptor("InfoBuilder metadata ABI mismatch")
    x, y, z = plan["workgroup"]
    return {
        "tensors": tuple(
            TensorParam(
                b["dtype"],
                tuple(b["size"]),
                tuple(b["stride"]),
                b["offset"],
                b["access"],
            )
            for b in semantic["buffers"]
        ),
        "scalars": tuple(ScalarParam(s["positive"]) for s in semantic["scalars"]),
        "extent": n,
        "info_scalars": tuple(info),
        "metadata_words": metadata["bytes"] // 8,
        "launch": LaunchShape(grid, (x, y, z), manifest["dynamic_shared_bytes"]),
        "guards": tuple(guards),
        "threads": x * y * z,
        "plane_width": plan["plane_width"],
        "repeatable": not (effects.get("rng") or effects.get("mutation")),
    }
