"""Bounded runtime proofs. Failure excludes a physical candidate, not semantics."""

from cubecl_torch.compiler.final_plan import STRATEGY as FINAL9
from cubecl_torch.compiler.final_plan import semantic_reason
from cubecl_torch.compiler.indexing import evaluate
from cubecl_torch.compiler.plan import catalog, welford_catalog

# Experimental continuity policy, not an accepted final default. Qualification
# and timing are separate from structural legality and from input tensor values.
LONG_REDUCTION_THRESHOLD = 4096

# Reduction types the compiler has a combiner for, and the plans whose
# arithmetic is specific to sum. Mirrors Combiner in native/src/reduction/combiner.rs.
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


def welford_only(semantic):
    """Every reduction of the kernel is a Welford reduction: Welford-catalog
    plans would give any other reduction (a fused sum) unqualified arithmetic."""
    kinds = [op["reduction_type"] for op in semantic["ops"] if op["op"] == "reduction"]
    return bool(kinds) and all(k.startswith("welford") for k in kinds)


def stored_components(semantic):
    """The Welford component each store writes (None for any other value)."""
    ops = {op.get("id"): op for op in semantic["ops"]}
    found = []
    for store in (op for op in semantic["ops"] if op["op"] == "store"):
        value = ops.get(store["args"][0])
        while value is not None and value["op"] == "cast":
            value = ops.get(value["args"][0])
        found.append(
            value["index"]
            if value is not None and value["op"] == "reduction_component"
            else None
        )
    return found


def welford_weight_output(semantic):
    """A Welford kernel storing its weight: an element count, exact in any order."""
    return 2 in stored_components(semantic)


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
    floats = ("float32", "float16", "bfloat16")
    for op in semantic["ops"]:
        # Per-row stores stay scalar. The per-element pass stores vectors when
        # every one of its stores can (reduction emit `vector_stores`: a float
        # store along the row, split into 16-byte transactions), aligned as
        # the loads; otherwise it runs scalar and its stores need nothing.
        if op["op"] not in ("load", "store") or (
            reduction
            and op["op"] == "store"
            and (
                op.get("iteration_scope") == "outer"
                or not vector_element_stores(semantic)
            )
        ):
            continue
        tensor = tensors[op["buffer"]]
        # 16-byte transactions at most; int64 (two lanes, as Triton's 16-byte
        # loads) in pointwise plans only, as native plan.rs.
        if op["dtype"] not in (floats if reduction else (*floats, "int64")) or (
            tensor.element_size() * width > 16
            and not (reduction and op["op"] == "store")
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
                    if op["op"] == "store":
                        return "reduction vector store requires the row axis"
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
            if tensor.data_ptr() % min(16, width * tensor.element_size()):
                return "runtime pointer is misaligned"
        except ValueError:
            return "vector access not proven affine"
    return None


def memory_ops(node, kind):
    """Every ``kind`` op (load, store) in ``node``, including masked bodies."""
    if isinstance(node, dict):
        if node.get("op") == kind:
            yield node
        for value in node.values():
            yield from memory_ops(value, kind)
    elif isinstance(node, list):
        for value in node:
            yield from memory_ops(value, kind)


def sixteen_bit_rows(semantic):
    """The kernel reads and writes only 16-bit floats: the long-row sums
    measured in Triton's band without compensation (wide-reduction-workgroups
    5.3). A float32 load or output (``x.float().sum()``) keeps the float32 rule."""
    sixteen = ("float16", "bfloat16")
    loads = list(memory_ops(semantic["ops"], "load"))
    return bool(loads) and all(
        op["dtype"] in sixteen for op in (*loads, *memory_ops(semantic["ops"], "store"))
    )


def vector_element_stores(semantic):
    """A vector plan's per-element pass stores vectors: every per-element
    store is a float store along the row (native `vector_stores`)."""
    stores = [
        op
        for op in semantic["ops"]
        if op["op"] == "store" and op.get("iteration_scope") != "outer"
    ]
    return all(
        op["dtype"] in ("float32", "float16", "bfloat16")
        and reduction_coefficient(op["index"]) == 1
        for op in stores
    )


def numerical_reason(plan, semantic, scalars):
    if semantic["schema"] != 3:
        return None
    # The policy below qualifies sum arithmetic. Max/min are exact under any
    # order (NaN-propagating, ordered zeros), so every eligible plan is legal.
    # A kernel that also reduces a sum keeps the sum qualifications.
    welford = welford_only(semantic)
    if not welford and (plan.vector_width == 8 or plan in welford_catalog()):
        # Triton's lane counts for Welford rows; sum, max and min keep their
        # qualified plan sets.
        return "Welford-catalog plans are Welford-only"
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
    # Triton gives a sum fused with a Welford reduction the Welford kernel's
    # lanes; the four-accumulator plane plan puts it in Triton's band there
    # (bead 4z8.32).
    if (
        not welford_reduction(semantic)
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
    if extent > LONG_REDUCTION_THRESHOLD and compensable and sixteen_bit_rows(semantic):
        # Measured: on 16-bit rows (row sums, the sum consumer, the two-step
        # variance at 2048x8192) every plan is in Triton's band, and
        # compensation only costs time (wide-reduction-workgroups 5.3).
        if plan.strategy == "shared_tree_compensated":
            return "16-bit long-row sums are in Triton's band without compensation"
    elif extent > LONG_REDUCTION_THRESHOLD and compensable:
        if plan.strategy != "shared_tree_compensated":
            # Measured: every uncompensated plan, 64 to 512 units, was outside
            # Triton's band on float32 sums over 2048x8192 and 256x32768 rows
            # (wide-reduction-workgroups 3.2).
            return "long-row sums outside Triton's band without compensation"
    elif plan.strategy == "shared_tree_compensated":
        return "short reduction retains existing physical candidates"
    if cooperation < 256 and (extent + cooperation - 1) // cooperation > 256:
        return "bounded private FP32 chain exceeds executed qualification"
    return None


# Triton's largest reduction block: rows up to this many elements reduce in
# one pass, every lane holding at most one element.
TRITON_LANES = 2048

# Triton runs 16 warps (num_warps = R0_BLOCK // 128) from rows of this many
# elements; 512-unit plans are offered from here (wide-reduction-workgroups).
WIDE_MIN_EXTENT = 2048


def wide(plan):
    """A 512-unit plan of the general catalog (not the reference512 probe)."""
    return plan.units == 512 and plan.strategy != "plane_shared_reference512"


def wide_plan_reason(plan, semantic, scalars):
    if wide(plan) and evaluate(semantic["reduction_extent"], scalars) < WIDE_MIN_EXTENT:
        return (
            "512-unit plans only for rows of at least 2048 elements (Triton's 16 warps)"
        )
    return None


def lanes(plan):
    """Elements a plan reduces per row in one sweep, before any combine."""
    return plan.workgroup[0] * plan.vector_width * plan.private_accumulators


def welford_plan_reason(plan, semantic, scalars, admitted):
    """Welford rows use Triton-shaped plans: many lanes, balanced merges.

    A selection policy over ``admitted``, the plans the structural and
    numerical rules accept for these tensors, so it never empties the set by
    assuming a plan the storage cannot run (unaligned 16-bit rows lose the
    eight-wide plans). Measured against Triton's band (welford-reductions 5,
    7): plane plans with the widest admitted vector (contiguous lanes combined
    as a pair tree) or with four private accumulators are in band; single-lane
    trees and per-row warps are not. A row Triton reduces in one pass (at most
    ``TRITON_LANES`` elements) is a pure combine tree there: a plan whose lanes
    cover the row at the widest vector (or scalar lanes) is one too, and
    otherwise a Triton-shaped plan must have as many lanes as the largest
    admitted Triton-shaped plane plan. Vector chains count as Triton-shaped only
    there, the rows they were measured on. A kernel storing only weights
    (element counts, exact in any order) and column plans keep every plan.

    A kernel that also reduces a sum follows the same policy with scalar lanes
    only: Triton gives the fused sum the Welford kernel's lanes, but a vector
    plan adds a sum's lanes into one chain (bead 4z8.32).
    """
    if not welford_reduction(semantic) or plan.strategy == "coalesced_columns":
        return None
    if set(stored_components(semantic)) == {2}:
        return None
    widest = max(
        (
            q.vector_width
            for q in admitted
            if q.strategy in ("plane_shared", "independent_planes")
            and welford_only(semantic)
        ),
        default=1,
    )
    extent = evaluate(semantic["reduction_extent"], scalars)

    # One accumulator per vector lane is Triton's order on float32 rows longer
    # than one pass (each of its 2048 lanes runs over the row); there the
    # folding plans of the same shape were outside the band, and elsewhere the
    # lane plans were (wide-reduction-workgroups 2.4).
    # The row's storage dtype: a 16-bit row is upcast before the reduction.
    long_float32 = extent > TRITON_LANES and all(
        op["dtype"] == "float32" for op in semantic["ops"] if op["op"] == "load"
    )
    if plan.private_merge == "lanes" and not long_float32:
        return "Welford lane accumulators only on float32 rows longer than one pass"
    if (
        long_float32
        and plan.private_merge != "lanes"
        and plan.vector_width > 1
        and any(
            q.private_merge == "lanes"
            and q.vector_width == plan.vector_width
            and q.workgroup == plan.workgroup
            for q in admitted
        )
    ):
        return "long float32 Welford rows keep one accumulator per vector lane"

    # Four scalar accumulators that do not cover a one-pass row (float32
    # LayerNorm at 1280 against 1024 lanes) lose to Triton where a vector plane
    # plan (tuning's Welford fallback) exists; they stay for long rows and for
    # rows without one.
    vector_plane = any(
        q.strategy == "plane_shared" and q.vector_width == widest > 1 for q in admitted
    )

    def shaped(q):
        return (
            q.strategy in ("plane_shared", "plane_shared_load_batch")
            and q.private_accumulators == 4
            and q.vector_width == 1
            and (not vector_plane or extent > TRITON_LANES)
        ) or (
            (
                q.strategy == "plane_shared"
                or (q.strategy == "independent_planes" and extent <= TRITON_LANES)
            )
            and q.vector_width == widest > 1
        )

    most = max(
        (
            lanes(q)
            for q in admitted
            if shaped(q) and q.strategy != "independent_planes"
        ),
        default=0,
    )
    if extent <= TRITON_LANES:
        # Covering lanes at a narrower vector than the widest split the row
        # into blocks unlike Triton's warps (measured outside on bfloat16).
        covered = extent <= lanes(plan) and plan.vector_width in (1, widest)
        if covered or (shaped(plan) and lanes(plan) >= min(extent, most)):
            return None
        return "Welford rows Triton reduces in one pass need lanes covering the row"
    # Longer rows: the shaped plans with the most lanes. On 2048x8192 16-bit
    # rows only 512 x v8 was in Triton's band (wide-reduction-workgroups 2.2).
    if shaped(plan) and lanes(plan) >= most:
        return None
    return "Welford rows need the widest vector lanes or four private accumulators"


def reason(plan, semantic, scalars, tensors, target):
    return structural_reason(
        plan, semantic, scalars, tensors, target
    ) or numerical_reason(plan, semantic, scalars)


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
    if semantic["schema"] == 3:
        # Selection policy, before the Welford lanes so short rows keep theirs.
        for p in [p for p in accepted if wide_plan_reason(p, semantic, scalars)]:
            accepted.remove(p)
            rejected.append(
                {"plan": p.to_dict(), "reason": wide_plan_reason(p, semantic, scalars)}
            )
    if welford_reduction(semantic):
        # Selection policy over the legal set; reason() stays the launch
        # contract (memory and numerical qualification only).
        policy = {
            p: welford_plan_reason(p, semantic, scalars, accepted) for p in accepted
        }
        if any(why is None for why in policy.values()):
            rejected.extend(
                {"plan": p.to_dict(), "reason": why} for p, why in policy.items() if why
            )
            accepted = [p for p in accepted if policy[p] is None]
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
    facts = (
        alignment,
        tuple(access),
        bounded,
        outer <= 1,
        extent > LONG_REDUCTION_THRESHOLD,
        extent >= WIDE_MIN_EXTENT,
    )
    if welford_reduction(semantic):
        # The Welford plan policy compares the extent with plan lane counts.
        facts += (tuple(extent <= n for n in (32, 64, 128, 256, 512, 1024, 2048)),)
    return facts


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
