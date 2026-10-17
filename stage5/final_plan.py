"""Admission contract for the qualified descending-plane nine-input final.

All new facts are descriptor-static; live storage/shape/stream validation still
belongs to Kernel. Nearby extents remain diagnostic, not automatic candidates.
The native descriptor adapter independently enforces this same contract.
"""

STRATEGY = "independent_planes_final9"


def semantic_reason(d):
    if not isinstance(d, dict):
        return "qualified final requires a semantic object"
    c = lambda n: {"op": "constant", "value": n}
    outer = {"op": "scalar", "slot": 0}
    effects = d.get("effects", {})
    if (
        d.get("schema") != 3
        or d.get("index_dtype") != "int64"
        or d.get("scalars") != [{"dtype": "int64", "id": 0, "positive": True}]
        or d.get("outer") != outer
        or d.get("numel") != outer
        or d.get("reduction_extent") != c(9)
        or not isinstance(effects, dict)
        or any(v is not False for v in effects.values())
    ):
        return "qualified final requires pure constant9 FP32 sum with scalar outer"
    buffers, ops = d.get("buffers", []), d.get("ops", [])
    if (
        not isinstance(buffers, list)
        or not isinstance(ops, list)
        or len(buffers) != 2
        or any(not isinstance(b, dict) for b in buffers)
        or any(not isinstance(op, dict) for op in ops)
    ):
        return "qualified final requires exactly two buffers"
    ty = buffers[1].get("dtype")
    if ty not in ("float32", "float16", "bfloat16") or buffers != [
        {
            "access": "read",
            "dtype": "float32",
            "offset": c(0),
            "size": [outer, c(9)],
            "stride": [c(9), c(1)],
        },
        {
            "access": "write",
            "dtype": ty,
            "offset": c(0),
            "size": [outer],
            "stride": [c(1)],
        },
    ]:
        return "qualified final requires contiguous FP32 partials and supported output storage"
    if len(ops) != (3 if ty == "float32" else 7):
        return (
            "qualified final requires the qualified load, sum, cast and owner sequence"
        )
    ids = [op.get("id") for op in ops[:-1]]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        return "qualified final requires unique SSA values"
    load, total = ids[0], ids[1 if ty == "float32" else 2]
    expected = [
        {
            "op": "load",
            "id": load,
            "args": [],
            "buffer": 0,
            "dtype": "float32",
            "semantic_dtype": "float32",
            "index": {
                "op": "add",
                "args": [
                    {"op": "reduction"},
                    {"op": "mul", "args": [c(9), {"op": "linear"}]},
                ],
            },
        },
    ]
    if ty != "float32":
        expected.append(
            {
                "op": "cast",
                "id": ids[1],
                "args": [load],
                "dtype": "float32",
                "semantic_dtype": ty,
            }
        )
    expected.append(
        {
            "op": "reduction",
            "id": total,
            "args": [load if ty == "float32" else ids[1]],
            "dtype": "float32",
            "semantic_dtype": "float32",
            "accumulator_dtype": "float32",
            "source_dtype": "float32",
            "reduction_type": "sum",
        }
    )
    if ty != "float32":
        expected[-1].update(semantic_dtype=ty, source_dtype=ty)
        for i, cast_type in ((3, ty), (4, "float32"), (5, ty)):
            expected.append(
                {
                    "op": "cast",
                    "id": ids[i],
                    "args": [ids[i - 1]],
                    "dtype": cast_type,
                    "semantic_dtype": ty,
                }
            )
    expected.append(
        {
            "op": "store",
            "args": [ids[-1]],
            "buffer": 1,
            "dtype": ty,
            "semantic_dtype": ty,
            "index": {"op": "linear"},
            "iteration_scope": "outer",
        }
    )
    if ops != expected:
        return (
            "qualified final requires direct FP32 sum and unmodified outer owner store"
        )
    return None
