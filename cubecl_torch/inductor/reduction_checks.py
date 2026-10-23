"""Stage 3 uses existing pre-fusion and legality hooks; no fallback retry."""

import sympy
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.virtualized import V

from cubecl_torch.inductor.capabilities import UnsupportedCubeCL
from cubecl_torch.inductor.semantic_checks import node_reason as pointwise_reason
from cubecl_torch.inductor.semantic_checks import scalar_input


def node_reason(node):
    reason = pointwise_reason(
        node,
        allow_reductions=True,
        float_index=True,
        extra_ops=("reduction", "store_reduction", "rsqrt"),
        allow_empty_input=True,
    )
    if reason is None and isinstance(node, SchedulerNode):
        if any(
            op.target in ("reduction", "store", "store_reduction")
            for block in node._body.subblocks.values()
            for op in block.graph.nodes
            if op.op == "call_method"
        ):
            return "masked reduction/effect unsupported; barriers require uniform physical phases"
    if reason is None and isinstance(node, SchedulerNode) and node.is_reduction():
        extent = sympy.sympify(sympy.prod(node.get_ranges()[1]))
        if extent.is_number and extent > 1048576:
            return "unsupported physical reduction extent > 1048576; split plan not implemented"
        V.graph.sizevars.check_leq(extent, 1048576)
    return reason


def pre_fusion(nodes, trace):
    for node in nodes:
        reason = node_reason(node)
        trace.append(
            {
                "event": "capability",
                "boundary": "pre_fusion",
                "node": node.get_name(),
                "decision": "reject" if reason else "accept",
                "reason": reason,
                "group": str(getattr(node, "group", None)),
                "ranges": str(node.get_ranges())
                if isinstance(node, SchedulerNode)
                else None,
                "operation_graph": str(node._body.root_block.graph)
                if isinstance(node, SchedulerNode)
                else None,
            }
        )
        if reason:
            raise UnsupportedCubeCL(
                "CubeCL Stage 3 unsupported before fusion: " + reason
            )
    return nodes


def fusion_allowed(a, b, trace):
    nodes = [*a.get_nodes(), *b.get_nodes()]
    reason = next((r for n in nodes if (r := node_reason(n))), None)
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
