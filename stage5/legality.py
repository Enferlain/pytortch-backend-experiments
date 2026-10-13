"""Bounded runtime proofs. Failure excludes a physical candidate, not semantics."""

from stage2.indexing import evaluate
from stage5.plan import catalog


def affine(tree, scalars):
    """Exact affine coefficients (constant, outer, reduction); no sampled proof."""
    op = tree["op"]
    if op in ("constant", "scalar"):
        return (evaluate(tree, scalars), 0, 0)
    if op == "linear":
        return (0, 1, 0)
    if op == "reduction":
        return (0, 0, 1)
    args = [affine(x, scalars) for x in tree.get("args", ())]
    if op == "add":
        return tuple(sum(x[i] for x in args) for i in range(3))
    if op == "mul":
        result = (1, 0, 0)
        for a in args:
            if a[1:] != (0, 0) and result[1:] != (0, 0):
                raise ValueError("non-affine product")
            if a[1:] == (0, 0):
                result = tuple(x * a[0] for x in result)
            else:
                result = tuple(x * result[0] for x in a)
        return result
    raise ValueError("not a direct affine physical access")


def reason(plan, semantic, scalars, tensors, target):
    reduction = semantic["schema"] == 3
    if reduction == ("pointwise" in plan.strategy):
        return "wrong semantic family"
    if plan.units > target["device_max_threads"] or (
        plan.plane_width and plan.plane_width != target["plane_width"]
    ):
        return "unsupported queried target topology"
    effects = semantic.get("effects", {})
    if (
        effects.get("rng")
        or effects.get("mutation")
        or any(op["op"] == "indirect" for op in semantic["ops"])
    ):
        if plan.strategy != "scalar_pointwise_baseline":
            return "effectful/checked-indirect automatic tuning stays frozen"
    if reduction and plan.strategy == "coalesced_columns":
        try:
            loads = [
                affine(op["index"], scalars)
                for op in semantic["ops"]
                if op["op"] == "load"
            ]
            if not loads or any(x[1] not in (0, 1) for x in loads):
                return "column candidate requires neighboring outer addresses"
        except ValueError:
            return "column access not proven affine"
    if reduction:
        cooperation = (
            plan.workgroup[1]
            if plan.strategy == "coalesced_columns"
            else plan.workgroup[0]
        )
        extent = evaluate(semantic["reduction_extent"], scalars)
        if cooperation < 256 and (extent + cooperation - 1) // cooperation > 256:
            return "bounded candidate: more than 256 private FP32 additions with reduced cooperation; retain frozen/256-unit assignment"
    if plan.vector_width == 1:
        return None
    width = plan.vector_width
    if any(
        op["op"] in ("masked", "indirect", "atomic_add", "rand", "load_seed")
        for op in semantic["ops"]
    ):
        return "vector lazy/effect region unsupported"
    if not reduction and sum(op["op"] == "store" for op in semantic["ops"]) != 1:
        return "vector adapter requires one store"
    for op in semantic["ops"]:
        if op["op"] not in ("load", "store") or (reduction and op["op"] == "store"):
            continue
        tensor = tensors[op["buffer"]]
        if (
            op["dtype"] not in ("float32", "float16", "bfloat16")
            or tensor.element_size() * width > 16
        ):
            return "unsupported storage transaction width"
        try:
            if reduction:
                if reduction_coefficient(op["index"]) not in (0, 1):
                    return (
                        "reduction vector requires a literal zero/one axis coefficient"
                    )
                offset, outer_stride, r_stride = affine(op["index"], scalars)
                if r_stride == 0:  # uniform scalar input, no vector memory transaction
                    continue
                if (
                    r_stride != 1
                    or offset % width
                    or (
                        evaluate(semantic["outer"], scalars) > 1
                        and outer_stride % width
                    )
                ):
                    return "reduction vector row base/stride not aligned"
            elif op["op"] == "load" and op["index"]["op"] == "constant":
                continue
            elif op["index"] != {"op": "linear"}:
                return "pointwise vector direct-linear address not proven"
            if tensor.data_ptr() % (width * tensor.element_size()):
                return "runtime pointer is misaligned"
        except ValueError:
            return "vector access not proven affine"
    return None


def candidates(semantic, scalars, tensors, target):
    accepted, rejected = [], []
    for p in catalog():
        why = reason(p, semantic, scalars, tensors, target)
        if why:
            rejected.append({"plan": p.to_dict(), "reason": why})
        else:
            accepted.append(p)
    if not accepted:
        raise ValueError("unsupported physical schedule: no legal frozen baseline")
    return accepted, rejected


def signature(semantic, scalars, tensors):
    """Finite classes for every runtime fact used by reason(), never raw sizes.

    Affine coefficients are evaluated once, then classified by the actual
    vector/coalescing predicates. This caches proofs without specializing kernels
    to every shape. Non-affine descriptors have a fixed rejection class.
    """
    alignment = tuple(
        max(a for a in (16, 8, 4, 2, 1) if t.data_ptr() % a == 0) for t in tensors
    )
    if semantic["schema"] != 3:
        return (alignment,)
    access = []
    for op in semantic["ops"]:
        if op["op"] != "load":
            continue
        try:
            offset, outer, r = affine(op["index"], scalars)
            access.append(
                (
                    offset % 8,
                    outer % 8,
                    outer in (0, 1),
                    0 if r == 0 else 1 if r == 1 else 2,
                )
            )
        except ValueError:
            access.append(None)
    extent = evaluate(semantic["reduction_extent"], scalars)
    bounded = tuple(
        (extent + width - 1) // width <= 256 for width in (4, 8, 32, 64, 128)
    )
    outer = evaluate(semantic["outer"], scalars)
    return (alignment, tuple(access), bounded, outer <= 1)


def reduction_coefficient(tree):
    """Same static axis proof as Rust: only literal multipliers on the axis.

    Runtime coefficients that happen to equal zero/one are deliberately excluded
    until both compile-time lowering and runtime proof can express them. This
    also decides scalar broadcast vs vector memory, including syntactic r*0.
    """
    op = tree["op"]
    if op == "reduction":
        return 1
    if op in ("constant", "scalar", "linear"):
        return 0
    values = [reduction_coefficient(a) for a in tree.get("args", ())]
    if any(v is None for v in values):
        return None
    if op == "add":
        result = 0
        for value in values:
            result += value
            if not -(1 << 63) <= result < (1 << 63):
                return None
        return result
    if op == "mul":
        terms = [(i, c) for i, c in enumerate(values) if c != 0]
        if not terms:
            return 0
        if len(terms) != 1:
            return None
        i, result = terms[0]
        for j, a in enumerate(tree["args"]):
            if j != i:
                if a["op"] != "constant":
                    return None
                value = a["value"]
                if type(value) is not int or not -(1 << 63) <= value < (1 << 63):
                    return None
                result *= value
                if not -(1 << 63) <= result < (1 << 63):
                    return None
        return result
    return None
