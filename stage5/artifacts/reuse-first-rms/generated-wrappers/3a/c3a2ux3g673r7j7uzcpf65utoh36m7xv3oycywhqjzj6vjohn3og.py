# AOT ID: ['10_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
from stage5.rms_trial_runtime import compile_kernel as rms_trial_compile

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cubecl_rms_trial_0 = rms_trial_compile({'schema': 3, 'index_dtype': 'int64', 'numel': {'op': 'scalar', 'slot': 1}, 'scalars': [{'dtype': 'int64', 'id': 0, 'positive': True}, {'dtype': 'int64', 'id': 1, 'positive': True}], 'buffers': [{'access': 'read', 'dtype': 'bfloat16', 'size': [{'op': 'scalar', 'slot': 1}, {'op': 'scalar', 'slot': 0}], 'stride': [{'op': 'scalar', 'slot': 0}, {'op': 'constant', 'value': 1}], 'offset': {'op': 'constant', 'value': 0}}, {'access': 'write', 'dtype': 'bfloat16', 'size': [{'op': 'scalar', 'slot': 1}, {'op': 'scalar', 'slot': 0}], 'stride': [{'op': 'scalar', 'slot': 0}, {'op': 'constant', 'value': 1}], 'offset': {'op': 'constant', 'value': 0}}], 'ops': [{'id': 'tmp0', 'op': 'load', 'args': [], 'dtype': 'bfloat16', 'semantic_dtype': 'bfloat16', 'index': {'op': 'add', 'args': [{'op': 'reduction'}, {'op': 'mul', 'args': [{'op': 'linear'}, {'op': 'scalar', 'slot': 0}]}]}, 'buffer': 0}, {'id': 'tmp1', 'op': 'cast', 'args': ['tmp0'], 'dtype': 'float32', 'semantic_dtype': 'bfloat16'}, {'id': 'tmp2', 'op': 'cast', 'args': ['tmp1'], 'dtype': 'bfloat16', 'semantic_dtype': 'bfloat16'}, {'id': 'tmp3', 'op': 'cast', 'args': ['tmp2'], 'dtype': 'float32', 'semantic_dtype': 'bfloat16'}, {'id': 'tmp4', 'op': 'cast', 'args': ['tmp3'], 'dtype': 'float32', 'semantic_dtype': 'float32'}, {'id': 'tmp5', 'op': 'mul', 'args': ['tmp4', 'tmp4'], 'dtype': 'float32', 'semantic_dtype': 'float32'}, {'id': 'tmp6', 'op': 'reduction', 'args': ['tmp5'], 'dtype': 'float32', 'semantic_dtype': 'float32', 'reduction_type': 'sum', 'source_dtype': 'float32', 'accumulator_dtype': 'float32'}, {'id': 'tmp7', 'op': 'index', 'args': [], 'dtype': 'int64', 'semantic_dtype': 'int64', 'index': {'op': 'scalar', 'slot': 0}}, {'id': 'tmp8', 'op': 'cast', 'args': ['tmp7'], 'dtype': 'float32', 'semantic_dtype': 'float32'}, {'id': 'tmp9', 'op': 'truediv', 'args': ['tmp6', 'tmp8'], 'dtype': 'float32', 'semantic_dtype': 'float32'}, {'id': 'tmp10', 'op': 'constant', 'args': [], 'dtype': 'float32', 'semantic_dtype': 'float32', 'value': 1e-05}, {'id': 'tmp11', 'op': 'add', 'args': ['tmp9', 'tmp10'], 'dtype': 'float32', 'semantic_dtype': 'float32'}, {'id': 'tmp12', 'op': 'rsqrt', 'args': ['tmp11'], 'dtype': 'float32', 'semantic_dtype': 'float32'}, {'id': 'tmp13', 'op': 'mul', 'args': ['tmp4', 'tmp12'], 'dtype': 'float32', 'semantic_dtype': 'float32'}, {'id': 'tmp14', 'op': 'cast', 'args': ['tmp13'], 'dtype': 'bfloat16', 'semantic_dtype': 'bfloat16'}, {'id': 'tmp15', 'op': 'cast', 'args': ['tmp14'], 'dtype': 'float32', 'semantic_dtype': 'bfloat16'}, {'id': 'tmp16', 'op': 'cast', 'args': ['tmp15'], 'dtype': 'bfloat16', 'semantic_dtype': 'bfloat16'}, {'id': 'tmp17', 'op': 'cast', 'args': ['tmp16'], 'dtype': 'float32', 'semantic_dtype': 'bfloat16'}, {'id': 'tmp18', 'op': 'cast', 'args': ['tmp17'], 'dtype': 'bfloat16', 'semantic_dtype': 'bfloat16'}, {'op': 'store', 'args': ['tmp18'], 'dtype': 'bfloat16', 'semantic_dtype': 'bfloat16', 'index': {'op': 'add', 'args': [{'op': 'reduction'}, {'op': 'mul', 'args': [{'op': 'linear'}, {'op': 'scalar', 'slot': 0}]}]}, 'iteration_scope': 'reduction', 'buffer': 1}], 'outer': {'op': 'scalar', 'slot': 1}, 'reduction_extent': {'op': 'scalar', 'slot': 0}}, 0)


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        s77 = arg0_1
        s27 = arg1_1
        assert_size_stride(arg2_1, (s77, s27), (s27, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf1 = empty_strided_cuda((s77, s27), (s27, 1), torch.bfloat16)
            cubecl_rms_trial_0((s27, s77), arg2_1, buf1)
            del arg2_1
        return (buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 16
    arg1_1 = 4099
    arg2_1 = rand_strided((16, 4099), (4099, 1), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
