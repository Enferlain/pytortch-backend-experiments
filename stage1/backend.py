"""Experimental, bounded Inductor CUDA generated-kernel backend (not production)."""
import json
import math
import traceback
import sympy
import torch
from torch._inductor import config
from torch._inductor.codegen.common import CSEVariable, register_backend_for_device, device_codegens
from torch._inductor.codegen.simd import SIMDKernel, SIMDScheduling
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from stage1.capabilities import (UnsupportedCubeCL, TYPE_POLICY, check_inputs,
                                pre_fusion, fusion_allowed)
from stage1.plan import plan_pointwise

TRACE = []

class CubeCLValue(CSEVariable):
    """SSA identity is the CSE name; values represent one scalar per CUDA unit."""
    def __init__(self, name, bounds, dtype=torch.float32, shape=()):
        super().__init__(name, bounds, dtype=dtype, shape=shape)
        self.ssa = name
        self.local_shape = TYPE_POLICY.value_shape
        self.mask_provenance = 'logical_in_bounds'

class CubeCLOps(DefaultHandler):
    def _default(self, name, args, kwargs):
        raise UnsupportedCubeCL(f'CubeCL Stage 1 unsupported operation: {name}')

    @staticmethod
    def constant(value, dtype):
        if dtype != torch.float32:
            raise UnsupportedCubeCL(f'constant dtype {dtype}')
        if not math.isfinite(float(value)):
            raise UnsupportedCubeCL('Stage 1 rejects non-finite constants')
        return V.kernel.emit('constant', value=float(value))

    @staticmethod
    def sin(x): return V.kernel.emit('sin', x)
    @staticmethod
    def cos(x): return V.kernel.emit('cos', x)
    @staticmethod
    def add(x,y): return V.kernel.emit('add', x, y)
    @staticmethod
    def mul(x,y): return V.kernel.emit('mul', x, y)

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None, use_compute_types=True):
        if dtype != torch.float32 or x.dtype != torch.float32:
            raise UnsupportedCubeCL(f'cast {x.dtype} to {dtype}')
        return x

class CubeCLKernel(SIMDKernel):
    overrides = CubeCLOps

    def __init__(self, *args, **kwargs):
        self.operations = []
        self.replay = []
        self.replay_stack = []
        self.semantic_cache = {}
        super().__init__(*args, **kwargs)
        if self.features.is_reduction() or self.features.reduction_numel != 1:
            raise UnsupportedCubeCL('Stage 1 supports pointwise schedules only')
        if len(self.numels) != 1 or 'x' not in self.numels or not isinstance(self.numels['x'], sympy.Integer):
            raise UnsupportedCubeCL('Stage 1 requires one static flattened pointwise range')
        self.physical_plan = plan_pointwise(self.numels['x'], self.features.reduction_numel)

    def emit(self, op, *args, **kwargs):
        if op not in ('load','constant','sin','cos','add','mul'):
            raise UnsupportedCubeCL(f'Stage 1 unsupported descriptor operation: {op}')
        if any(not isinstance(v, CubeCLValue) or v.dtype != torch.float32 for v in args):
            raise UnsupportedCubeCL(f'untyped CubeCL operands for {op}')
        key = (op, tuple(v.ssa for v in args), tuple(sorted(kwargs.items())))
        if key in self.semantic_cache:
            value = self.semantic_cache[key]
            value.use_count += 1
            return value
        value = self.cse.newvar(dtype=TYPE_POLICY.compute_dtype, shape=TYPE_POLICY.value_shape)
        self.operations.append({'id':value.ssa, 'op':op, 'args':[v.ssa for v in args], 'dtype':'float32', **kwargs})
        self.semantic_cache[key] = value
        return value

    def create_cse_var(self, name, bounds, dtype=None, shape=None):
        if dtype != torch.float32 or shape not in (None, ()):
            raise UnsupportedCubeCL(f'unsupported CSE type/shape: {dtype}, {shape}')
        return CubeCLValue(name, bounds, dtype, ())

    def _reject_advanced(self, *args, **kwargs):
        raise UnsupportedCubeCL('Stage 1 rejects indirect indexing, reductions and advanced kernel operations')

    indirect_load = _reject_advanced
    reduction = _reject_advanced
    store_reduction = _reject_advanced
    partial_accumulate = _reject_advanced
    scan = _reject_advanced
    sort = _reject_advanced
    bucketize = _reject_advanced

    def dtype_to_str(self, dtype):
        if dtype in (torch.int32, torch.int64): return 'index'
        if dtype == torch.float32: return 'float32'
        raise UnsupportedCubeCL(f'dtype {dtype}')

    def check_index(self, index):
        expanded = self.simplify_indexing(index)
        substitutions = {sym:entry.expr for sym,entry in self.range_tree_nodes.items()}
        expanded = sympy.simplify(expanded.xreplace(substitutions))
        root = self.range_trees[0]
        if expanded == root.index_sym() or (self.numels['x'] == 1 and expanded == 0):
            return 'linear'
        raise UnsupportedCubeCL(f'only contiguous i indexing supported: {index} -> {expanded}')

    def check_buffer(self, name):
        TYPE_POLICY.check_storage(V.graph.get_dtype(name))
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if len(layout.size) != 1 or layout.stride != [1] or layout.offset != 0:
            raise UnsupportedCubeCL(f'Stage 1 requires 1D contiguous buffers: {layout}')
        if layout.size[0] != self.numels['x']:
            raise UnsupportedCubeCL('broadcasting is not supported')

    def load(self, name, index):
        if not self.replay_stack:
            self.replay_stack = [{'file':f.filename,'line':f.lineno,'function':f.name}
                for f in traceback.extract_stack()]
        self.check_buffer(name)
        self.args.input(name)
        return self.emit('load', buffer_name=name, index=self.check_index(index))

    def store(self, name, index, value, mode=None):
        if mode is not None: raise UnsupportedCubeCL(f'store mode {mode}')
        self.check_buffer(name)
        self.args.output(name)
        self.operations.append({'op':'store','args':[value.ssa], 'buffer_name':name, 'index':self.check_index(index), 'dtype':'float32'})

    def record_op_trace(self, name, args, kwargs, result=None):
        def display(v):
            return {'ssa':v.ssa, 'dtype':str(v.dtype), 'shape':v.local_shape, 'mask':v.mask_provenance} if isinstance(v,CubeCLValue) else str(v)
        self.replay.append({'op':name, 'args':[display(v) for v in args], 'kwargs':{k:display(v) for k,v in kwargs.items()}, 'result':display(result)})

    def codegen_kernel(self):
        self.input_names = [n for n in self.args.input_buffers if n not in self.removed_buffers]
        self.output_names = [n for n in self.args.output_buffers if n not in self.removed_buffers]
        if len(self.input_names) not in (1,2) or len(self.output_names) != 1 or self.args.inplace_buffers:
            raise UnsupportedCubeCL('one/two inputs, one disjoint output required')
        names = self.input_names + self.output_names
        operations = []
        for op in self.operations:
            op = dict(op)
            if 'buffer_name' in op:
                name = op.pop('buffer_name')
                if name in self.removed_buffers:
                    if op['op'] == 'store': continue
                    raise UnsupportedCubeCL('unresolved load of fused temporary')
                op['buffer'] = names.index(name)
            operations.append(op)
        descriptor = {'schema':1, 'dtype':'float32',
            'buffers':[{'access':'read' if i<len(self.input_names) else 'write'} for i in range(len(names))], 'ops':operations}
        self.descriptor = descriptor
        TRACE.append({'event':'kernel', 'nodes':[n.get_name() for n in self.features.scheduler_nodes()],
            'numel':str(self.numels['x']), 'descriptor':descriptor,
            'physical_plan':self.physical_plan.to_dict(),
            'replay':self.replay,'actual_replay_stack':self.replay_stack})
        return json.dumps(descriptor, sort_keys=True)

    def call_kernel(self, name, node=None, deallocate_ws=True):
        # Python wrapper's existing tensor allocation/lifetime code remains in charge.
        args = ', '.join(self.input_names + self.output_names)
        V.graph.wrapper_code.writeline(f'{name}({args})')

class CubeCLScheduling(SIMDScheduling):
    kernel_type = CubeCLKernel

    def check_backend_available(self, device):
        from stage1.runtime import check_available
        check_available(device)
        TRACE.append({'event':'availability', 'backend':type(self).__name__, 'device':str(device)})

    def can_fuse(self, node1, node2):
        return fusion_allowed(node1, node2, TRACE) and super().can_fuse(node1, node2)

    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse

    @classmethod
    def get_backend_features(cls, device): return OrderedSet()

    def get_tiling_and_scores(self, node_schedule, numel, reduction_numel, coalesce_analysis=None):
        if reduction_numel != 1 or any(n.is_reduction() for n in node_schedule):
            raise UnsupportedCubeCL('Stage 1 rejects reductions')
        TRACE.append({'event':'schedule','nodes':[n.get_name() for n in node_schedule], 'numel':str(numel)})
        return {'x':numel}, {}

    def define_kernel(self, src_code, node_schedule, kernel):
        wrapper = V.graph.wrapper_code
        name = 'cubecl_kernel_'+str(wrapper.next_kernel_suffix())
        wrapper.add_import_once('from stage1.runtime import compile_kernel as cubecl_compile_kernel')
        device = V.graph.get_current_device_or_throw().index
        wrapper.define_kernel(name, f'cubecl_compile_kernel({kernel.descriptor!r}, {device}, {kernel.physical_plan.to_dict()!r})')
        return name

    def codegen_combo_kernel(self, *args, **kwargs):
        raise UnsupportedCubeCL('combo kernels unsupported')
    def codegen_template(self, *args, **kwargs):
        raise UnsupportedCubeCL('template kernels unsupported')


_registered = False

def register():
    global _registered
    if _registered:
        return
    previous = device_codegens['cuda']
    def select(scheduler):
        if config.cuda_backend == 'cubecl':
            return CubeCLScheduling(scheduler)
        return previous.scheduling(scheduler)
    register_backend_for_device('cuda', select, previous.wrapper_codegen,
        previous.cpp_wrapper_codegen, previous.fx_wrapper_codegen)
    _registered = True


def compile_backend(gm, example_inputs):
    """Local experimental Dynamo spelling; delegates capture/compiler to Inductor."""
    check_inputs(example_inputs, TRACE)
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor.codegen.common import init_backend_registration
    init_backend_registration()
    register()
    previous_pass = config._pre_fusion_custom_pass
    def validate_before_fusion(nodes):
        if previous_pass is not None:
            nodes = previous_pass(nodes)
        return pre_fusion(nodes, TRACE)
    patches = {'cuda_backend':'cubecl', 'compile_threads':1, 'fx_graph_cache':False,
        '_pre_fusion_custom_pass':validate_before_fusion,
         'triton.cudagraphs':False, 'inplace_buffers':False,
        'benchmark_kernel':False, 'combo_kernels':False, 'triton.coalesce_tiling_analysis':False}
    return compile_fx(gm, example_inputs, config_patches=patches)
