"""Bounded capabilities, checked before fusion; no fallback policy is implied."""
from dataclasses import dataclass
import math
import sympy
import torch
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.virtualized import V


class UnsupportedCubeCL(NotImplementedError):
    pass


@dataclass(frozen=True)
class ScalarFloat32Policy:
    # Semantic dtype, storage dtype and compute dtype coincide only in this scope.
    semantic_dtype: torch.dtype = torch.float32
    storage_dtype: torch.dtype = torch.float32
    compute_dtype: torch.dtype = torch.float32
    value_shape: tuple = ()

    def check_storage(self, dtype):
        if dtype != self.storage_dtype:
            raise UnsupportedCubeCL(f'CubeCL unsupported graph/operator: storage dtype {dtype}; float32 required')


TYPE_POLICY = ScalarFloat32Policy()
SEMANTIC_OPS = frozenset(('load', 'store', 'constant', 'sin', 'cos', 'add', 'mul', 'to_dtype'))


def check_inputs(inputs, trace):
    tensors = [value for value in inputs if isinstance(value, torch.Tensor)]
    reason = None
    if not tensors or any(t.device.type != 'cuda' for t in tensors):
        reason = 'CUDA tensor inputs required'
    elif any(t.dtype != TYPE_POLICY.storage_dtype for t in tensors):
        reason = 'float32 required'
    elif any(t.ndim != 1 or t.stride() != (1,) or t.storage_offset() != 0 for t in tensors):
        reason = '1D contiguous zero-offset input required'
    elif any(not isinstance(t.shape[0], int) for t in tensors):
        reason = 'static shapes required'
    elif any(t.shape != tensors[0].shape for t in tensors):
        reason = 'equal input shapes required'
    trace.append({'event': 'capability', 'boundary': 'backend_selection',
                  'decision': 'reject' if reason else 'accept', 'reason': reason})
    if reason:
        raise UnsupportedCubeCL('CubeCL unsupported graph/operator at backend selection: ' + reason)


def node_reason(node):
    if not isinstance(node, SchedulerNode):
        # Existing extern/nop nodes retain their natural Inductor scheduling.
        return None
    if node.is_template():
        return 'template schedules unsupported'
    if node.is_reduction():
        return 'reductions unsupported'
    ranges, reductions = node.get_ranges()
    if reductions or len(ranges) > 1 or any(not isinstance(n, sympy.Integer) for n in ranges):
        return 'static flattened pointwise schedule required'
    body = node._body
    if body.indirect_vars or body.subblocks:
        return 'indirect indexing / nested operation bodies unsupported'
    numel = sympy.prod(ranges)
    for index in body.indexing_exprs.values():
        if not ((len(body.iter_vars) == 1 and index == body.iter_vars[0])
                or (numel == 1 and index == 0)):
            return 'only linear pointwise indexing supported'
    for operation in body.get_nodes():
        if operation.op == 'call_method':
            if operation.target not in SEMANTIC_OPS:
                return f'operation {operation.target} unsupported'
            if operation.target == 'constant':
                if operation.args[2] != torch.float32 or not math.isfinite(float(operation.args[1])):
                    return 'non-finite or non-float32 constant unsupported'
            if operation.target == 'to_dtype' and operation.args[2] != torch.float32:
                return 'non-float32 cast unsupported'
        elif operation.op == 'call_module' and operation.target == 'get_index':
            # LoopBody's symbolic indexing lookup, not a tensor/value operation.
            pass
        elif operation.op not in ('placeholder', 'output', 'get_attr'):
            return f'operation graph node {operation.op} unsupported'
    for dep in node.read_writes.reads | node.read_writes.writes:
        name = getattr(dep, 'name', None)
        if name is None:
            continue
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if buf.get_dtype() != TYPE_POLICY.storage_dtype:
            return 'float32 buffer required'
        if len(layout.size) != 1 or layout.stride != [1] or layout.offset != 0:
            return '1D contiguous zero-offset buffer required'
        if layout.size[0] != numel:
            return 'broadcasting unsupported'
    return None


def pre_fusion(nodes, trace):
    for node in nodes:
        reason = node_reason(node)
        trace.append({'event': 'capability', 'boundary': 'pre_fusion', 'node': node.get_name(),
                      'decision': 'reject' if reason else 'accept', 'reason': reason,
                      'semantic_ops': [str(op.target) for op in node._body.get_nodes()
                                       if op.op == 'call_method'] if isinstance(node, SchedulerNode) else []})
        if reason:
            raise UnsupportedCubeCL('CubeCL unsupported graph/operator before fusion: ' + reason)
    return nodes


def fusion_allowed(node1, node2, trace):
    nodes = [*node1.get_nodes(), *node2.get_nodes()]
    reason = next((r for n in nodes if (r := node_reason(n))), None)
    if reason is None and any(not isinstance(n, SchedulerNode) for n in nodes):
        reason = 'extern fusion unsupported; retain natural extern scheduling'
    # A disjoint multi-output fused body exceeds the existing compiler ABI.
    writes = {dep.name for n in nodes for dep in n.read_writes.writes if hasattr(dep, 'name')}
    reads = {dep.name for n in nodes for dep in n.read_writes.reads if hasattr(dep, 'name')}
    if reason is None and len(writes - reads) != 1:
        reason = 'one disjoint output required'
    if reason is None and len(reads - writes) not in (1, 2):
        reason = 'one/two external inputs required'
    if reason is None and node1.group != node2.group:
        reason = 'equal pointwise ranges required'
    trace.append({'event': 'capability', 'boundary': 'fusion',
                  'nodes': [node1.get_name(), node2.get_name()],
                  'decision': 'reject' if reason else 'accept', 'reason': reason})
    return reason is None
