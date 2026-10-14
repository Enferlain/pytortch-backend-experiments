"""Proven widening of int32 comparison-only indices; no new tensor dtype."""

import torch
from torch._inductor.optimize_indexing import range_expressable_in_32_bits
from torch._inductor.scheduler import SchedulerNode

EVENTS = []


def widen_mask_indices(nodes):
    for scheduled in nodes:
        if not isinstance(scheduled, SchedulerNode):
            continue
        body = scheduled._body
        bounds = None
        for node in body.get_nodes():
            if (
                node.op != "call_method"
                or node.target != "index_expr"
                or node.args[2] != torch.int32
            ):
                continue
            if not node.users or any(
                u.op != "call_method"
                or u.target not in ("eq", "ne", "lt", "le", "gt", "ge")
                for u in node.users
            ):
                continue
            if bounds is None:
                bounds = body.bounds().get_bounds()
            interval = bounds.get(node.args[1])
            if interval is None or not range_expressable_in_32_bits(interval):
                continue
            node.args = (*node.args[:2], torch.int64)
            EVENTS.append(
                {
                    "event": "widen_mask_index",
                    "node": scheduled.get_name(),
                    "expression": str(node.args[1]),
                    "lower": str(interval.lower),
                    "upper": str(interval.upper),
                    "from": "int32",
                    "to": "int64",
                    "proof": "raw expression fits int32; comparison-only users",
                }
            )
    return nodes
