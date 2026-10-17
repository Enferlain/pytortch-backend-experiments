"""Admission contract for the qualified descending-plane nine-input final.

All new facts are descriptor-static; live storage/shape/stream validation still
belongs to Kernel. Nearby extents remain diagnostic, not automatic candidates.
The native descriptor adapter independently enforces this same contract.
"""

STRATEGY = "independent_planes_final9"


def semantic_reason(d):
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
    if len(buffers) != 2 or len(ops) != 3:
        return "qualified final requires exactly one load, sum and owner store"
    ty = buffers[1].get("dtype")
    if ty not in ("float32", "float16", "bfloat16") or buffers != [
        {"access": "read", "dtype": "float32", "offset": c(0),
         "size": [outer, c(9)], "stride": [c(9), c(1)]},
        {"access": "write", "dtype": ty, "offset": c(0),
         "size": [outer], "stride": [c(1)]},
    ]:
        return "qualified final requires contiguous FP32 partials and supported output storage"
    load, total = ops[0].get("id"), ops[1].get("id")
    if not isinstance(load, str) or not isinstance(total, str) or load == total or ops != [
        {"op": "load", "id": load, "args": [], "buffer": 0,
         "dtype": "float32", "semantic_dtype": "float32",
         "index": {"op": "add", "args": [{"op": "reduction"},
                   {"op": "mul", "args": [c(9), {"op": "linear"}]}]}},
        {"op": "reduction", "id": total, "args": [load],
         "dtype": "float32", "semantic_dtype": "float32",
         "accumulator_dtype": "float32", "source_dtype": "float32",
         "reduction_type": "sum"},
        {"op": "store", "args": [total], "buffer": 1,
         "dtype": ty, "semantic_dtype": ty, "index": {"op": "linear"},
         "iteration_scope": "outer"},
    ]:
        return "qualified final requires direct FP32 sum and unmodified outer owner store"
    return None
