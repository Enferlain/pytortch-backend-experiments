"""What the backend admits, checked at Inductor's boundaries before code emission.

Three boundaries: backend selection (``check_graph``: graph inputs and
user-visible mutation), pre-fusion (``pre_fusion``: every scheduler node) and
fusion (``fusion_allowed``). The value-op table ``OP_RULES`` is shared with
code generation (``codegen``) and mirrored by the descriptor compiler
(``native/src/semantic.rs`` ``VALUE_OPS``).
"""

import functools
import operator
from dataclasses import dataclass

import sympy
import torch
from torch._inductor import config
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.virtualized import V

from cubecl_torch.compiler import descriptor
from cubecl_torch.compiler.descriptor import MAX_RANK
from cubecl_torch.compiler.errors import UnsupportedCubeCL
from cubecl_torch.compiler.indexing import encode
from cubecl_torch.inductor import diagnostics, rejections
from cubecl_torch.inductor.diagnostics import TraceLog

TRACE = TraceLog()
FLOATS = (torch.float32, torch.float16, torch.bfloat16)
# Inductor upcasts Welford inputs to float32 before reducing.
WELFORD = ("welford_reduce", "welford_combine")
REDUCTIONS = ("sum", "max", "min", *WELFORD)


def welford_component(node):
    """``getitem(reduction(...), k)``: one of a Welford reduction's three results."""
    source = node.args[0] if node.args else None
    return (
        node.target is operator.getitem
        and getattr(source, "op", None) == "call_method"
        and source.target == "reduction"
        and source.args[3] in WELFORD
        and node.args[1] in (0, 1, 2)
    )


# Integer and bool value dtypes, (signed, width), from the compiler's table.
INTEGERS = {getattr(torch, name): v for name, v in descriptor.INTEGERS.items()}
VALUES = (*FLOATS, *INTEGERS)
# Tensor storage the generated kernels read and write.
STORAGE = (*FLOATS, torch.bool, torch.int64)
COMPARISONS = ("eq", "ne", "lt", "le", "gt", "ge")
NUMBERS = (*FLOATS, *(t for t in INTEGERS if t != torch.bool))
# Value-op typing: op -> (operand dtypes, result dtype or "same" for the
# operands' promoted dtype, reason when an operand is outside them). Admission
# is separate: OPS plus each path's extra_ops. An admitted op without a row
# types as float arithmetic.
ARITHMETIC = (FLOATS, "same", "integer/bool arithmetic unsupported; indices only")
OP_RULES = {
    **dict.fromkeys(
        ("sin", "cos", "exp", "sqrt", "rsqrt", "reciprocal", "truediv"), ARITHMETIC
    ),
    # Integers wrap modulo 2**width, as eager CUDA and Triton.
    **dict.fromkeys(
        ("neg", "add", "sub", "mul"),
        (NUMBERS, "same", "bool add/sub/mul/neg unsupported"),
    ),
    **dict.fromkeys(
        ("maximum", "minimum"), (VALUES, "same", "unsupported extrema operands")
    ),
    **dict.fromkeys(
        ("bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not"),
        (tuple(INTEGERS), "same", "bitwise operands must be integer or bool values"),
    ),
    # Integer quotients and remainders; float mod/fmod/remainder is the
    # extrema family (x3c).
    **dict.fromkeys(
        ("truncdiv", "floordiv"),
        (
            tuple(t for t in INTEGERS if t != torch.bool),
            "same",
            "integer division operands must be integers",
        ),
    ),
    **dict.fromkeys(
        ("mod", "remainder"),
        (
            tuple(t for t in INTEGERS if t != torch.bool),
            "same",
            "float mod/fmod unsupported; integer operands only",
        ),
    ),
    "int_truediv": (
        tuple(t for t in INTEGERS if t != torch.bool),
        torch.float32,
        "int_truediv operands must be integers",
    ),
    # Out-of-range amounts follow eager: 0, or the sign fill shifting right.
    **dict.fromkeys(
        ("bitwise_left_shift", "bitwise_right_shift"),
        (
            tuple(t for t in INTEGERS if t != torch.bool),
            "same",
            "shift operands must be integer values",
        ),
    ),
    # Inductor converts the aten logical ops' operands to bool.
    **dict.fromkeys(
        ("logical_and", "logical_or", "logical_xor"),
        ((torch.bool,), "same", "logical operands must be bool"),
    ),
    "logical_not": (VALUES, torch.bool, "unsupported logical_not operand"),
    "isnan": (FLOATS, torch.bool, "isnan requires a floating operand"),
    **dict.fromkeys(
        COMPARISONS,
        (
            VALUES,
            torch.bool,
            "comparison operands require floating, integer or bool values",
        ),
    ),
}
# Inductor op names emitted as one descriptor op: the logical forms on bool and
# the operator spellings Inductor's lowerings use (``ops.and_`` for masks).
ALIASES = {
    "logical_and": "bitwise_and",
    "logical_or": "bitwise_or",
    "logical_xor": "bitwise_xor",
    "and_": "bitwise_and",
    "or_": "bitwise_or",
    "xor": "bitwise_xor",
    "lshift": "bitwise_left_shift",
    "rshift": "bitwise_right_shift",
}
OP_RULES.update({a: OP_RULES[t] for a, t in ALIASES.items() if a not in OP_RULES})
# Ops with their own typing in node_reasons: memory, conversions, selection, regions.
# Ops whose result dtype is their third argument.
CONVERSIONS = ("constant", "to_dtype", "to_dtype_bitcast", "index_expr", "value_expr")
STRUCTURAL = (
    *CONVERSIONS,
    "load",
    "store",
    "where",
    "masked",
    "reduction",
    "store_reduction",
)
OPS = frozenset((*STRUCTURAL, *OP_RULES))
RNG_OPS = frozenset(("load_seed", "rand", "randn", "rand_eager", "randint64"))


@dataclass(frozen=True)
class Path:
    """What a node may contain on one generation path: the limits of the
    kernel kind it becomes. The paths differ only here."""

    # Ops admitted beyond OPS.
    extra_ops: tuple = ()
    # Tensor-derived indices (set_indirect subblocks).
    indirect: bool = False
    # atomic_add stores.
    atomic: bool = False
    # Reduction kernels read or write no int64 buffers yet (4z8.25).
    storage: tuple = STORAGE
    max_inputs: int = 4


PATHS = {
    "rng": Path(extra_ops=("rand", "load_seed")),
    "indirect": Path(indirect=True, atomic=True),
    "reduction": Path(storage=(*FLOATS, torch.bool)),
    "pointwise": Path(max_inputs=5),
}


def cpu_scalar(x):
    """Dynamo's 0-d CPU float64 tensor for a lifted Python float (dynamic=True)."""
    return (
        isinstance(x, torch.Tensor)
        and x.device.type == "cpu"
        and x.ndim == 0
        and x.dtype == torch.float64
    )


def scalar_input(name):
    """A graph buffer that GPU kernels read as a float64 runtime scalar value."""
    return V.graph.is_unspec_arg(name) and V.graph.get_dtype(name) == torch.float64


CUDA_INPUTS = "CUDA tensor inputs required"


def input_reasons(inputs, storage_dtypes):
    """Why the region's inputs are outside the backend's contract, in check order."""
    # Lifted float scalars become kernel scalar values, not tensor arguments.
    tensors = [x for x in inputs if isinstance(x, torch.Tensor) and not cpu_scalar(x)]
    if not tensors or any(x.device.type != "cuda" for x in tensors):
        yield CUDA_INPUTS
        return
    if any(not isinstance(x, (torch.Tensor, torch.SymInt, int)) for x in inputs):
        yield "only tensor and integer shape/stride arguments supported"
    if bad := sorted({str(x.dtype) for x in tensors if x.dtype not in storage_dtypes}):
        yield (
            f"storage dtype {', '.join(bad)} unsupported ("
            + "/".join(str(t).removeprefix("torch.") for t in storage_dtypes)
            + " only)"
        )
    if any(x.layout != torch.strided or x.ndim > MAX_RANK for x in tensors):
        yield f"direct strided rank 0..{MAX_RANK} inputs required"
    # SymInt comparisons deliberately retain nonnegative-stride guards.
    if any(
        s.is_negative is True if isinstance(s, sympy.Expr) else s < 0
        for x in tensors
        if x.layout == torch.strided
        for s in x.stride()
    ):
        yield "negative strides unsupported"


def check_inputs(inputs, *, storage_dtypes=STORAGE):
    reasons = rejections.scan(input_reasons(inputs, storage_dtypes))
    TRACE.append(
        {
            "event": "capability",
            "boundary": "backend_selection",
            "decision": "reject" if reasons else "accept",
            "reason": reasons[0] if reasons else None,
        }
    )
    if reasons:
        # Without CUDA tensors there is no CubeCL graph to scan further.
        rejections.reject(
            *("CubeCL unsupported input contract: " + r for r in reasons),
            stop=CUDA_INPUTS in reasons,
        )


def check_graph(gm, inputs):
    """Backend selection: the graph's inputs, mutation and RNG ordering contract."""
    if diagnostics.tracing():
        TRACE.append(
            {
                "event": "input_contract",
                "inputs": [
                    {
                        "type": type(x).__name__,
                        "device": str(getattr(x, "device", None)),
                        "dtype": str(getattr(x, "dtype", None)),
                        "shape": str(getattr(x, "shape", None)),
                    }
                    for x in inputs
                ],
                "graph": str(gm.graph),
            }
        )
    check_inputs(inputs)
    for n in gm.graph.nodes:
        schema = getattr(n.target, "_schema", None)
        if (
            (n.op == "call_method" and str(n.target).endswith("_"))
            or (schema and schema.is_mutable)
            or n.kwargs.get("out") is not None
            or n.kwargs.get("inplace") is True
        ):
            TRACE.append(
                {
                    "event": "capability",
                    "boundary": "backend_selection",
                    "decision": "reject",
                    "reason": "unsupported user-visible mutation/alias pattern",
                }
            )
            rejections.reject(
                "unsupported user-visible mutation/alias pattern; functional atomic destinations only"
            )
            break
    if config.align_random_eager or config.numerics == "strict":
        rejections.reject("unsupported RNG alignment/exact ordering contract")


def call_methods(node):
    return {
        x.target for x in node._body.root_block.graph.nodes if x.op == "call_method"
    }


def bounds_requested(op):
    """``(lower, upper)`` of a ``check_bounds`` call, positional or keyword."""
    args = op.args[3:5]
    return (
        op.kwargs.get("lower", args[0] if args else True),
        op.kwargs.get("upper", args[1] if len(args) > 1 else True),
    )


def node_reason(node):
    """The first reason ``node`` cannot be generated, or None."""
    return next(node_reasons(node), None)


def node_reasons(node):
    """Every reason ``node`` cannot be generated, the first being the fail-fast one."""
    if not isinstance(node, SchedulerNode):
        return
    if call_methods(node) & RNG_OPS:
        yield from rng_reasons(node)
    elif node._body.indirect_vars:
        yield from indirect_reasons(node)
    elif node.is_reduction():
        yield from reduction_reasons(node)
    else:
        yield from body_reasons(node, PATHS["pointwise"])


def rng_reasons(node):
    if node.is_reduction() or node._body.subblocks:
        yield "RNG in reduction/lazy subblock unsupported"
        return
    # This lowering explicitly asks for i32 counters; it does not prove a
    # global fits-in-i32 property for us. Add a normal ShapeEnv guard here.
    extent = sympy.prod(node.get_ranges()[0])
    if not V.graph.sizevars.guard_or_false(sympy.Le(extent, (1 << 31) - 1)):
        yield "unsupported RNG physical extent: signed-i32 logical counter domain"
    try:
        yield from body_reasons(node, PATHS["rng"])
    except UnsupportedCubeCL as e:
        yield str(e)


def indirect_reasons(node):
    for expr in node._body.indexing_exprs.values():
        if any(
            set(node._body.indirect_vars) & term.free_symbols
            for term in expr.atoms(sympy.Function)
        ):
            yield "unsupported indirect quotient/modulo/function addressing"
            break
    if node.is_reduction() or node._body.subblocks:
        # The body checks would only restate this.
        yield "indirect reduction/lazy subblock scheduling unsupported"
        return
    for d in node.read_writes.writes:
        if hasattr(d, "name") and V.graph.get_dtype(d.name) not in FLOATS:
            yield "indirect writes require floating storage"
            break
    atomic = any(
        op.target == "store" and op.args[-1] == "atomic_add"
        for op in node._body.root_block.graph.nodes
        if op.op == "call_method"
    )
    if atomic:
        writes = list(node.read_writes.writes)
        if len(writes) != 1 or not hasattr(writes[0], "name"):
            yield "unsupported atomic destination dependency"
        elif (
            torch.are_deterministic_algorithms_enabled()
            or V.graph.get_dtype(writes[0].name) != torch.float32
        ):
            yield "unsupported atomic dtype or deterministic-mode incompatibility"
    yield from body_reasons(node, PATHS["indirect"])


def reduction_reasons(node):
    found = False
    for reason in body_reasons(node, PATHS["reduction"]):
        found = True
        yield reason
    if not found and isinstance(node, SchedulerNode) and node.is_reduction():
        extent = sympy.sympify(sympy.prod(node.get_ranges()[1]))
        if extent.is_number and extent > 1048576:
            yield "unsupported physical reduction extent > 1048576; split plan not implemented"
            return
        # A guard, so only for a node that is otherwise admitted.
        V.graph.sizevars.check_leq(extent, 1048576)


def body_reasons(n, path):
    """The checks every path shares: ranges, indexing, operations and their types,
    and buffer layouts. Checks after a failure run only where the failure leaves
    them meaningful."""
    if not isinstance(n, SchedulerNode):
        return
    if n.is_template():
        yield "templates unsupported"
        return
    if n._body.indirect_vars and not path.indirect:
        # Index checks below would only restate this.
        yield "indirect tensor-derived indexing unsupported"
        return
    ranges, reductions = n.get_ranges()
    if len(ranges) > MAX_RANK:
        yield f"pointwise rank 0..{MAX_RANK} required"
    if len(reductions) > 1:
        yield "one logical reduction dimension required"
    body = n._body
    reads = {
        d.name
        for d in n.read_writes.reads
        if hasattr(d, "name") and not scalar_input(d.name)
    }
    writes = {d.name for d in n.read_writes.writes if hasattr(d, "name")}
    inputs = len(reads - writes)
    if inputs > path.max_inputs:
        yield f"more than {path.max_inputs} direct inputs per kernel unsupported"
    if len(writes) != 1:
        yield "one output per node required"
    variables = {v: i for i, v in enumerate(body.iter_vars)}
    for expr in [
        *body.indexing_exprs.values(),
        *ranges,
        *reductions,
        sympy.prod(ranges),
    ]:
        try:
            encode(expr, lambda s: variables.get(s, 0))
        except UnsupportedCubeCL as error:
            yield str(error)
    admitted = OPS | frozenset(path.extra_ops)

    def check_operation(op):
        if op.op == "call_method":
            if op.target == "check_bounds":
                # A direct index Inductor could not prove in range. Unrequested
                # bounds emit nothing, as in Triton; requested ones would need a
                # device assertion (the indirect op checks its own bounds).
                if any(bounds_requested(op)):
                    return "check_bounds with requested bounds unsupported"
                return None
            if op.target not in admitted:
                return f"semantic operation {op.target} unsupported"
            if op.target in CONVERSIONS and op.args[2] not in VALUES:
                return f"dtype for {op.target} unsupported: {op.args[2]}"
            if op.target == "reduction" and (
                op.args[1] not in FLOATS
                or op.args[2] not in FLOATS
                or op.args[3] not in REDUCTIONS
                or (
                    op.args[3] in WELFORD
                    and (op.args[1], op.args[2]) != (torch.float32, torch.float32)
                )
            ):
                return (
                    f"unsupported reduction {op.args[3]} over {op.args[2]}: "
                    "sum/max/min with floating accumulation, Welford over float32 only"
                )
            if (
                op.target == "store"
                and op.args[-1] is not None
                and not (path.atomic and op.args[-1] == "atomic_add")
            ):
                return "atomic/mutation store mode unsupported"
        elif op.op == "call_module":
            if (
                op.target != "get_index"
                and not str(op.target).startswith("masked_subblock")
                and not (path.indirect and str(op.target).startswith("set_indirect"))
            ):
                return f"operation subblock {op.target} unsupported"
        elif op.op == "call_function" and welford_component(op):
            pass
        elif op.op not in ("placeholder", "output", "get_attr"):
            return f"operation graph {op.op} unsupported"

    types = {}
    found = []
    # Types are inferred until the first failure; after it only the
    # per-operation checks run, since later types would be unknown.
    typing = True

    def block_type(block):
        nonlocal typing
        for operation in block.graph.nodes:
            if reason := check_operation(operation):
                found.append(reason)
                typing = False
            if operation.op == "call_module" and str(operation.target).startswith(
                "masked_subblock"
            ):
                subblock = block_type(body.subblocks[operation.target])
                if typing:
                    types[operation] = subblock
            elif typing:
                try:
                    result = operation_result(operation)
                except UnsupportedCubeCL as error:
                    found.append(str(error))
                    typing = False
                else:
                    if operation.op == "output":
                        return result
            if found and not rejections.recording():
                return None
        return None

    def operation_result(operation):
        if operation.op == "call_method":
            target = operation.target
            if target == "load_seed":
                if V.graph.get_dtype(operation.args[1]) != torch.int64:
                    raise UnsupportedCubeCL("unsupported RNG seed storage")
                types[operation] = torch.int64
            elif target == "rand":
                if (
                    types.get(operation.args[1]) != torch.int64
                    or types.get(operation.args[2]) != torch.int32
                ):
                    raise UnsupportedCubeCL("unsupported RNG operands")
                types[operation] = torch.float32
            elif target == "load":
                types[operation] = V.graph.get_dtype(operation.args[1])
            elif target in CONVERSIONS:
                if target == "to_dtype_bitcast" and (
                    operation.args[2].itemsize != operation.args[3].itemsize
                    or torch.bool in operation.args[2:4]
                    or types.get(operation.args[1]) in (torch.float64, torch.bool)
                ):
                    raise UnsupportedCubeCL(
                        "to_dtype_bitcast requires equal sizes and non-bool tensor values"
                    )
                if (
                    target == "to_dtype"
                    and types.get(operation.args[1]) == torch.float64
                    and operation.args[2] not in FLOATS
                ):
                    raise UnsupportedCubeCL(
                        "float64 scalar values cast only to floating compute"
                    )
                types[operation] = operation.args[2]
            elif target == "reduction":
                values = operation.args[4]
                if not isinstance(values, tuple):
                    values = (values,)
                if any(types.get(v) not in FLOATS for v in values):
                    raise UnsupportedCubeCL("unsupported reduction input dtype")
                types[operation] = operation.args[1]
            elif target == "where":
                if types.get(operation.args[1]) != torch.bool:
                    raise UnsupportedCubeCL("where requires bool predicate")
                if torch.float64 in (
                    types.get(operation.args[2]),
                    types.get(operation.args[3]),
                ):
                    raise UnsupportedCubeCL(
                        "float64 scalar values require an explicit cast to floating compute"
                    )
                types[operation] = torch.promote_types(
                    types[operation.args[2]], types[operation.args[3]]
                )
            elif target not in ("store", "store_reduction", "check_bounds"):
                allowed, result, reason = OP_RULES.get(target, ARITHMETIC)
                operands = [types.get(a) for a in operation.args[1:]]
                if result == "same" and torch.float64 in operands:
                    raise UnsupportedCubeCL(
                        "float64 scalar values require an explicit cast to floating compute"
                    )
                if any(t not in allowed for t in operands):
                    raise UnsupportedCubeCL(reason)
                types[operation] = (
                    functools.reduce(torch.promote_types, operands)
                    if result == "same"
                    else result
                )
        elif operation.op == "call_function":
            # A Welford component (mean, m2 or weight): float32.
            types[operation] = torch.float32
        elif operation.op == "output":
            return types.get(operation.args[0])
        return None

    block_type(body.root_block)
    yield from found
    for dep in n.read_writes.reads | n.read_writes.writes:
        if not hasattr(dep, "name"):
            continue
        if V.graph.is_unspec_arg(dep.name):
            if dep in n.read_writes.writes:
                yield "writes to 0-d CPU scalar buffers unsupported"
            elif not scalar_input(dep.name):
                yield "0-d CPU scalar inputs other than float64 unsupported"
            continue
        b = V.graph.get_buffer(dep.name)
        l = b.get_layout()
        if b.get_dtype() not in path.storage:
            yield f"tensor storage dtype {b.get_dtype()} unsupported"
        if len(l.size) > MAX_RANK or any(
            s.is_negative is True for s in map(sympy.sympify, l.stride)
        ):
            yield "layout/stride unsupported"
            continue
        for expr in [*l.size, *l.stride, l.offset]:
            try:
                encode(expr, lambda symbol: 0)
            except UnsupportedCubeCL as error:
                yield "layout/index expression unsupported: " + str(error)
                break
    if any(
        op.target in ("reduction", "store", "store_reduction")
        for block in body.subblocks.values()
        for op in block.graph.nodes
        if op.op == "call_method"
    ):
        yield "masked reduction/effect unsupported; barriers require uniform physical phases"


def pre_fusion(nodes):
    if diagnostics.tracing():
        TRACE.append(
            {
                "event": "capability_graph",
                "is_backward": bool(getattr(V.graph, "is_backward", False)),
                "nodes": [n.get_name() for n in nodes],
            }
        )
    rejected = False
    for n in nodes:
        reasons = rejections.scan(node_reasons(n))
        reason = reasons[0] if reasons else None
        if diagnostics.tracing():
            TRACE.append(
                {
                    "event": "capability",
                    "boundary": "pre_fusion",
                    "node": n.get_name(),
                    "scheduler_type": type(n).__name__,
                    "ir_type": type(n.node).__name__,
                    "is_reduction": isinstance(n, SchedulerNode) and n.is_reduction(),
                    "semantic_ops": [
                        str(op.target)
                        for op in n._body.root_block.graph.nodes
                        if op.op == "call_method"
                    ]
                    if isinstance(n, SchedulerNode)
                    else [],
                    "decision": "reject" if reason else "accept",
                    "reason": reason,
                    "operation_graph": str(n._body.root_block.graph)
                    if isinstance(n, SchedulerNode)
                    else None,
                    "dependencies": str(n.read_writes),
                    "execution_path": "cubecl_generated"
                    if isinstance(n, SchedulerNode)
                    else "extern",
                    "kernel": getattr(n.node, "python_kernel_name", None),
                    "target": str(getattr(n.node, "op_overload", None)),
                }
            )
        if reasons:
            rejected = True
            rejections.reject(
                *("CubeCL unsupported before fusion: " + r for r in reasons)
            )
    if rejected:
        # Record mode: every node was scanned; rejected nodes are not scheduled.
        raise rejections.Stop("graph rejected before fusion")
    return nodes


def fusion_allowed(a, b):
    nodes = [*a.get_nodes(), *b.get_nodes()]
    # Keep RNG generation separate until effect-aware duplicate replay is proved.
    if any(
        isinstance(n, SchedulerNode)
        and (n._body.indirect_vars or call_methods(n) & {"rand", "load_seed"})
        for n in nodes
    ):
        TRACE.append(
            {
                "event": "capability",
                "boundary": "fusion",
                "decision": "reject",
                "nodes": [a.get_name(), b.get_name()],
                "reason": "RNG/indirect fusion requires separate effect-safe proof",
            }
        )
        return False
    reason = next((r for n in nodes if (r := node_reason(n))), None)
    if reason is None and any(n.is_reduction() for n in nodes):
        # The fused kernel is a reduction kernel: its buffer limits hold for every node.
        storage = PATHS["reduction"].storage
        reason = next(
            (
                f"tensor storage dtype {dtype} unsupported in reduction kernels"
                for n in nodes
                for d in n.read_writes.reads | n.read_writes.writes
                if hasattr(d, "name") and not V.graph.is_unspec_arg(d.name)
                if (dtype := V.graph.get_dtype(d.name)) not in storage
            ),
            None,
        )
    if any(not isinstance(n, SchedulerNode) for n in nodes):
        reason = "extern fusion unsupported"
    writes = {d.name for n in nodes for d in n.read_writes.writes if hasattr(d, "name")}
    reads = {
        d.name
        for n in nodes
        for d in n.read_writes.reads
        if hasattr(d, "name") and not scalar_input(d.name)
    }
    members = {n.get_name() for n in nodes} | {a.get_name(), b.get_name()}
    outputs = (writes - reads) | (writes & set(V.graph.get_output_names()))
    for n in nodes:
        for buffer in n.get_outputs():
            if any(
                not u.is_weak and u.node.get_name() not in members for u in buffer.users
            ):
                outputs.add(buffer.get_name())
    if reason is None and len(outputs) != 1:
        reason = "one disjoint external output required"
    if reason is None and len(reads - writes) not in (1, 2, 3, 4):
        reason = "one to four external inputs required"
    reductions = {n.group[1] for n in nodes if n.is_reduction()}
    if reason is None and len(reductions) > 1:
        reason = "different reduction domains require separate physical schedules"
    if (
        reason is None
        and sum(
            op.target == "reduction"
            for n in nodes
            if isinstance(n, SchedulerNode)
            for op in n._body.root_block.graph.nodes
            if op.op == "call_method"
        )
        > 4
    ):
        reason = "one to four reductions per cooperative schedule required"
    if reason is None and reductions:
        outer, extent = next(iter(reductions))
        if any(
            n.group[1] not in ((outer, extent), (outer, 1), (outer * extent, 1))
            for n in nodes
        ):
            reason = "unsupported reduction fusion iteration domain"
    elif reason is None and a.group != b.group:
        reason = "equal logical pointwise ranges required"
    TRACE.append(
        {
            "event": "capability",
            "boundary": "fusion",
            "nodes": [a.get_name(), b.get_name()],
            "decision": "reject" if reason else "accept",
            "reason": reason,
        }
    )
    return reason is None
