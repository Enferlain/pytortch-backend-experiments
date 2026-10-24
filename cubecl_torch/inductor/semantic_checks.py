"""Executed Stage 2 surface; checks precede scheduler fusion and code emission."""

import operator

import sympy
import torch
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.virtualized import V

from cubecl_torch.compiler.descriptor import MAX_RANK
from cubecl_torch.compiler.indexing import encode
from cubecl_torch.inductor.capabilities import UnsupportedCubeCL

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


VALUES = (*FLOATS, torch.bool, torch.int64)
OPS = frozenset(
    (
        "load",
        "store",
        "constant",
        "sin",
        "cos",
        "neg",
        "exp",
        "sqrt",
        "reciprocal",
        "add",
        "sub",
        "mul",
        "truediv",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "where",
        "maximum",
        "minimum",
        "isnan",
        "to_dtype",
        "index_expr",
        "value_expr",
        "masked",
    )
)


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


def check_inputs(inputs, trace, *, storage_dtypes=(*FLOATS, torch.bool)):
    # Lifted float scalars become kernel scalar values, not tensor arguments.
    tensors = [x for x in inputs if isinstance(x, torch.Tensor) and not cpu_scalar(x)]
    reason = None
    if not tensors or any(x.device.type != "cuda" for x in tensors):
        reason = "CUDA tensor inputs required"
    elif any(not isinstance(x, (torch.Tensor, torch.SymInt, int)) for x in inputs):
        reason = "only tensor and integer shape/stride arguments supported"
    elif bad := sorted(
        {str(x.dtype) for x in tensors if x.dtype not in storage_dtypes}
    ):
        reason = (
            f"storage dtype {', '.join(bad)} unsupported ("
            + "/".join(str(t).removeprefix("torch.") for t in storage_dtypes)
            + " only)"
        )
    elif any(x.layout != torch.strided or x.ndim > MAX_RANK for x in tensors):
        reason = f"direct strided rank 0..{MAX_RANK} inputs required"
    # SymInt comparisons deliberately retain nonnegative-stride guards.
    elif any(
        s.is_negative is True if isinstance(s, sympy.Expr) else s < 0
        for x in tensors
        for s in x.stride()
    ):
        reason = "negative strides unsupported"
    trace.append(
        {
            "event": "capability",
            "boundary": "backend_selection",
            "decision": "reject" if reason else "accept",
            "reason": reason,
        }
    )
    if reason:
        raise UnsupportedCubeCL("CubeCL unsupported input contract: " + reason)


def node_reason(
    n,
    *,
    allow_reductions=False,
    float_index=False,
    extra_ops=(),
    allow_empty_input=False,
    value_dtypes=VALUES,
    storage_dtypes=(*FLOATS, torch.bool),
    operation_type=None,
    allow_indirect=False,
    allow_atomic=False,
    max_inputs=4,
):
    if not isinstance(n, SchedulerNode):
        return None
    if n.is_template():
        return "templates unsupported"
    if n.is_reduction() and not allow_reductions:
        return "reductions unsupported"
    if n._body.indirect_vars and not allow_indirect:
        return "indirect tensor-derived indexing unsupported"
    ranges, reductions = n.get_ranges()
    if len(ranges) > MAX_RANK:
        return f"pointwise rank 0..{MAX_RANK} required"
    if allow_reductions and len(reductions) > 1:
        return "one logical reduction dimension required"
    body = n._body
    reads = {
        d.name
        for d in n.read_writes.reads
        if hasattr(d, "name") and not scalar_input(d.name)
    }
    writes = {d.name for d in n.read_writes.writes if hasattr(d, "name")}
    if (
        len(reads - writes) not in range(0 if allow_empty_input else 1, max_inputs + 1)
        or len(writes) != 1
    ):
        return "one to four direct inputs and one output required"
    try:
        variables = {v: i for i, v in enumerate(body.iter_vars)}
        for expr in [
            *body.indexing_exprs.values(),
            *ranges,
            *reductions,
            sympy.prod(ranges),
        ]:
            encode(expr, lambda s: variables.get(s, 0))
    except UnsupportedCubeCL as error:
        return str(error)

    def check_operation(op):
        if op.op == "call_method":
            if op.target not in OPS | frozenset(extra_ops):
                return f"semantic operation {op.target} unsupported"
            if (
                op.target in ("constant", "to_dtype", "index_expr", "value_expr")
                and op.args[2] not in value_dtypes
            ):
                return f"dtype for {op.target} unsupported: {op.args[2]}"
            if (
                op.target in ("index_expr", "value_expr")
                and op.args[2] != torch.int64
                and not float_index
            ):
                return "non-int64 symbolic value expression unsupported; integer shape/stride ABI only"
            if op.target == "reduction" and (
                op.args[1] not in FLOATS
                or op.args[2] not in FLOATS
                or op.args[3] not in REDUCTIONS
                or op.args[3] in WELFORD
                and (op.args[1], op.args[2]) != (torch.float32, torch.float32)
            ):
                return (
                    f"unsupported reduction {op.args[3]} over {op.args[2]}: "
                    "sum/max/min with floating accumulation, Welford over float32 only"
                )
            if (
                op.target == "store"
                and op.args[-1] is not None
                and not (allow_atomic and op.args[-1] == "atomic_add")
            ):
                return "atomic/mutation store mode unsupported"
        elif op.op == "call_module":
            if (
                op.target != "get_index"
                and not str(op.target).startswith("masked_subblock")
                and not (allow_indirect and str(op.target).startswith("set_indirect"))
            ):
                return f"operation subblock {op.target} unsupported"
        elif op.op == "call_function" and welford_component(op):
            pass
        elif op.op not in ("placeholder", "output", "get_attr"):
            return f"operation graph {op.op} unsupported"

    types = {}

    def block_type(block):
        for operation in block.graph.nodes:
            if reason := check_operation(operation):
                raise UnsupportedCubeCL(reason)
            if operation.op == "call_module" and str(operation.target).startswith(
                "masked_subblock"
            ):
                types[operation] = block_type(body.subblocks[operation.target])
            elif operation.op == "call_method":
                target = operation.target
                custom_type = (
                    operation_type(operation, types) if operation_type else None
                )
                if custom_type is not None:
                    types[operation] = custom_type
                elif target == "load":
                    types[operation] = V.graph.get_dtype(operation.args[1])
                elif target in ("constant", "to_dtype", "index_expr", "value_expr"):
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
                elif target == "isnan":
                    if types.get(operation.args[1]) not in FLOATS:
                        raise UnsupportedCubeCL("isnan requires a floating operand")
                    types[operation] = torch.bool
                elif target in ("eq", "ne", "lt", "le", "gt", "ge"):
                    operands = [types.get(a) for a in operation.args[1:]]
                    if any(t not in (*FLOATS, torch.int64) for t in operands):
                        raise UnsupportedCubeCL(
                            "comparison operands require floating values or signed index values"
                        )
                    types[operation] = torch.bool
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
                elif target not in ("store", "store_reduction"):
                    operands = [types.get(a) for a in operation.args[1:]]
                    if torch.float64 in operands:
                        raise UnsupportedCubeCL(
                            "float64 scalar values require an explicit cast to floating compute"
                        )
                    if any(t not in FLOATS for t in operands):
                        raise UnsupportedCubeCL(
                            "integer/bool arithmetic unsupported; indices only"
                        )
                    types[operation] = operands[0]
            elif operation.op == "call_function":
                # A Welford component (mean, m2 or weight): float32.
                types[operation] = torch.float32
            elif operation.op == "output":
                return types.get(operation.args[0])

    try:
        block_type(body.root_block)
    except UnsupportedCubeCL as error:
        return str(error)
    for dep in n.read_writes.reads | n.read_writes.writes:
        if not hasattr(dep, "name"):
            continue
        if V.graph.is_unspec_arg(dep.name):
            if dep in n.read_writes.writes:
                return "writes to 0-d CPU scalar buffers unsupported"
            if not scalar_input(dep.name):
                return "0-d CPU scalar inputs other than float64 unsupported"
            continue
        b = V.graph.get_buffer(dep.name)
        l = b.get_layout()
        if b.get_dtype() not in storage_dtypes:
            return "tensor storage dtype unsupported"
        if len(l.size) > MAX_RANK or any(
            s.is_negative is True for s in map(sympy.sympify, l.stride)
        ):
            return "layout/stride unsupported"
        try:
            for expr in [*l.size, *l.stride, l.offset]:
                encode(expr, lambda symbol: 0)
        except UnsupportedCubeCL as error:
            return "layout/index expression unsupported: " + str(error)
    return None


def pre_fusion(nodes, trace):
    for n in nodes:
        reason = node_reason(n)
        trace.append(
            {
                "event": "capability",
                "boundary": "pre_fusion",
                "node": n.get_name(),
                "decision": "reject" if reason else "accept",
                "reason": reason,
                "ranges": str(n.get_ranges()) if isinstance(n, SchedulerNode) else None,
            }
        )
        if reason:
            raise UnsupportedCubeCL("CubeCL unsupported before fusion: " + reason)
    return nodes


def fusion_allowed(a, b, trace):
    nodes = [*a.get_nodes(), *b.get_nodes()]
    reason = next((r for n in nodes if (r := node_reason(n))), None)
    writes = {d.name for n in nodes for d in n.read_writes.writes if hasattr(d, "name")}
    reads = {
        d.name
        for n in nodes
        for d in n.read_writes.reads
        if hasattr(d, "name") and not scalar_input(d.name)
    }
    if reason is None and any(not isinstance(n, SchedulerNode) for n in nodes):
        reason = "extern fusion unsupported"
    members = {n.get_name() for n in nodes} | {a.get_name(), b.get_name()}
    external_outputs = (writes - reads) | (writes & set(V.graph.get_output_names()))
    for n in nodes:
        for buffer in n.get_outputs():
            if any(
                not user.is_weak and user.node.get_name() not in members
                for user in buffer.users
            ):
                external_outputs.add(buffer.get_name())
    if reason is None and len(external_outputs) != 1:
        reason = "one disjoint external output required"
    if reason is None and len(reads - writes) not in (1, 2, 3, 4):
        reason = "one to four external inputs required"
    if reason is None and a.group != b.group:
        reason = "equal logical pointwise ranges required"
    trace.append(
        {
            "event": "capability",
            "boundary": "fusion",
            "nodes": [a.get_name(), b.get_name()],
            "decision": "reject" if reason else "accept",
            "reason": reason,
        }
    )
    return reason is None
