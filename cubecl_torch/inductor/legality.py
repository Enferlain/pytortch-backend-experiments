"""Bounded runtime proofs. Failure excludes a physical candidate, not semantics."""

from cubecl_torch.compiler.final_plan import STRATEGY as FINAL9
from cubecl_torch.compiler.final_plan import semantic_reason
from cubecl_torch.compiler.indexing import evaluate
from cubecl_torch.compiler.plan import catalog, welford_catalog

# Experimental continuity policy, not an accepted final default. Qualification
# and timing are separate from structural legality and from input tensor values.
LONG_REDUCTION_THRESHOLD = 4096

# Reduction types the compiler has a combiner for, and the plans whose
# arithmetic is specific to sum. Mirrors Combiner in native/src/descriptor3.rs.
REDUCTION_TYPES = frozenset({"sum", "max", "min", "welford_reduce", "welford_combine"})
EXACT_REDUCTION_TYPES = frozenset({"max", "min"})
SUM_ONLY_STRATEGIES = frozenset(
    {"shared_tree_compensated", "plane_shared_reference512", FINAL9}
)


def exact_reduction(semantic):
    """A reduction kernel whose every reduction is exact under any order."""
    kinds = [op["reduction_type"] for op in semantic["ops"] if op["op"] == "reduction"]
    return (
        semantic["schema"] == 3
        and bool(kinds)
        and all(kind in EXACT_REDUCTION_TYPES for kind in kinds)
    )


def welford_reduction(semantic):
    """A reduction kernel with a Welford reduction."""
    return semantic["schema"] == 3 and any(
        op["op"] == "reduction" and op["reduction_type"].startswith("welford")
        for op in semantic["ops"]
    )


def welford_weight_output(semantic):
    """A Welford kernel storing its weight: an element count, exact in any order."""
    ops = {op.get("id"): op for op in semantic["ops"]}
    for store in (op for op in semantic["ops"] if op["op"] == "store"):
        value = ops.get(store["args"][0])
        while value is not None and value["op"] == "cast":
            value = ops.get(value["args"][0])
        if value is not None and value["op"] == "reduction_component":
            if value["index"] == 2:
                return True
    return False


def combiner_reason(plan, semantic):
    if semantic["schema"] != 3:
        return None
    for op in semantic["ops"]:
        if op["op"] != "reduction":
            continue
        kind = op["reduction_type"]
        if kind not in REDUCTION_TYPES:
            return f"reduction type {kind} unsupported"
        if kind != "sum" and plan.strategy in SUM_ONLY_STRATEGIES:
            return f"strategy {plan.strategy} does not support reduction type {kind}"
    return None


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


def structural_reason(plan, semantic, scalars, tensors, target):
    if why := combiner_reason(plan, semantic):
        return why
    if plan.strategy == "bounded_column_pointwise":
        from cubecl_torch.compiler.column_plan import runtime_reason

        why = runtime_reason(semantic, scalars)
        if why:
            return why
    if plan.strategy == FINAL9:
        why = semantic_reason(semantic)
        if why:
            return why
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


def numerical_reason(plan, semantic, scalars):
    if semantic["schema"] != 3:
        return None
    # The policy below qualifies sum arithmetic. Max/min are exact under any
    # order (NaN-propagating, ordered zeros), so every eligible plan is legal.
    welford = welford_reduction(semantic)
    if not welford and plan.vector_width == 8:
        # Triton's lane count for Welford rows; sum, max and min keep their
        # qualified plan sets.
        return "eight-wide reduction vectors are Welford-only"
    if exact_reduction(semantic):
        return None
    # The diagnostic rules below record sum qualifications. Welford kernels
    # are judged by their own band test (welford-reductions 5).
    if plan.strategy == "plane_shared_reference512":
        return (
            "reference512 topology experiment is diagnostic only; no default admission"
        )
    if not welford and plan.strategy == "plane_shared_load_batch":
        return (
            "load-batch mechanism experiment is diagnostic only; not default-qualified"
        )
    if (
        not welford
        and plan.strategy == "plane_shared"
        and plan.private_accumulators == 4
    ):
        return "combined candidate failed per-element matched qualification; diagnostic only"
    cooperation = (
        plan.workgroup[1] if plan.strategy == "coalesced_columns" else plan.workgroup[0]
    )
    extent = evaluate(semantic["reduction_extent"], scalars)
    if not welford and plan.strategy == "shared_tree_multi_accumulator":
        return "diagnostic independent chains did not satisfy held-out accuracy"
    # Compensation exists only for sum. A kernel that also reduces max/min
    # (softmax) cannot use it and keeps the ordinary plans, which sit in
    # Triton's error band on long rows (4z8.17).
    compensable = all(
        op["reduction_type"] == "sum"
        for op in semantic["ops"]
        if op["op"] == "reduction"
    )
    if extent > LONG_REDUCTION_THRESHOLD and compensable:
        if plan.strategy != "shared_tree_compensated":
            return "experimental long-row continuity policy requires compensated plan"
    elif plan.strategy == "shared_tree_compensated":
        return "short reduction retains existing physical candidates"
    if cooperation < 256 and (extent + cooperation - 1) // cooperation > 256:
        return "bounded private FP32 chain exceeds executed qualification"
    return None


def welford_plan_reason(plan, semantic, tensors):
    """Welford rows use Triton-shaped plans: many lanes, balanced merges.

    Measured against Triton's band (welford-reductions 5): plane plans with
    the widest vector the storage allows (contiguous lanes combined as a pair
    tree) or with four private accumulators are in band; single-lane-per-unit
    trees and per-row warps are not. Column plans keep their own layout.
    """
    if not welford_reduction(semantic) or plan.strategy == "coalesced_columns":
        return None
    if plan.strategy in ("plane_shared", "plane_shared_load_batch"):
        if plan.private_accumulators == 4 and plan.vector_width == 1:
            return None
        loads = [tensors[op["buffer"]] for op in semantic["ops"] if op["op"] == "load"]
        widest = min((16 // t.element_size() for t in loads), default=4)
        if plan.strategy == "plane_shared" and plan.vector_width == min(widest, 8):
            return None
    return "Welford rows need the widest vector lanes or four private accumulators"


def reason(plan, semantic, scalars, tensors, target):
    return (
        structural_reason(plan, semantic, scalars, tensors, target)
        or numerical_reason(plan, semantic, scalars)
        or welford_plan_reason(plan, semantic, tensors)
    )


def candidates(semantic, scalars, tensors, target, *, reduction_context=None):
    from cubecl_torch.inductor.reference512_oracle import fenv
    from cubecl_torch.inductor.reference512_plan import STRATEGY, admits

    accepted, rejected = [], []
    plans = catalog()
    if welford_reduction(semantic):
        plans = (*plans, *welford_catalog())
    for p in plans:
        if (
            p.strategy == STRATEGY
            and admits(reduction_context, scalars)
            and fenv() is not None
        ):
            why = structural_reason(p, semantic, scalars, tensors, target)
        elif (
            reduction_context is not None
            and reduction_context["role"] == "final"
            and p.strategy not in ("shared_tree_reduction_baseline", FINAL9)
        ):
            why = "qualified reference512 pair retains the frozen final arithmetic"
        else:
            why = reason(p, semantic, scalars, tensors, target)
        if why:
            rejected.append({"plan": p.to_dict(), "reason": why})
        else:
            accepted.append(p)
    if not accepted:
        raise ValueError(
            "unsupported physical schedule: no legal candidate (numerical policy / queried target topology)"
        )
    return accepted, rejected


def signature(semantic, scalars, tensors, *, column_template=None):
    """Finite classes for every runtime fact used by reason(), never raw sizes.

    Affine coefficients are evaluated once, then classified by the actual
    vector/coalescing predicates. This caches proofs without specializing kernels
    to every shape. Non-affine descriptors have a fixed rejection class.
    """
    # Each current pointer is queried once; no pointer-keyed proof cache.
    pointers = tuple(t.data_ptr() for t in tensors)
    alignment = tuple(16 if not p else min(16, p & -p) for p in pointers)
    if semantic["schema"] != 3:
        from cubecl_torch.compiler.column_plan import extent_reason, semantic_reason

        if column_template is None:
            column_template = semantic_reason(semantic) is None
        if column_template:
            return (alignment, extent_reason(scalars) is None)
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
    return (
        alignment,
        tuple(access),
        bounded,
        outer <= 1,
        extent > LONG_REDUCTION_THRESHOLD,
    )


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
