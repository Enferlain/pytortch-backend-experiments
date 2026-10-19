# AOT ID: ['0_inference']
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
from stage5.runtime import compile_kernel as cubecl_stage5_compile

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


cubecl_stage5_0 = cubecl_stage5_compile({'schema': 4, 'index_dtype': 'int64', 'numel': {'op': 'mul', 'args': [{'op': 'scalar', 'slot': 0}, {'op': 'scalar', 'slot': 1}]}, 'scalars': [{'dtype': 'int64', 'id': 0, 'positive': True}, {'dtype': 'int64', 'id': 1, 'positive': True}], 'buffers': [{'access': 'read', 'dtype': 'float16', 'size': [{'op': 'scalar', 'slot': 1}, {'op': 'scalar', 'slot': 0}], 'stride': [{'op': 'scalar', 'slot': 0}, {'op': 'constant', 'value': 1}], 'offset': {'op': 'constant', 'value': 0}}, {'access': 'read', 'dtype': 'float16', 'size': [{'op': 'scalar', 'slot': 1}, {'op': 'constant', 'value': 1}], 'stride': [{'op': 'constant', 'value': 1}, {'op': 'constant', 'value': 1}], 'offset': {'op': 'constant', 'value': 0}}, {'access': 'write', 'dtype': 'float16', 'size': [{'op': 'scalar', 'slot': 1}, {'op': 'scalar', 'slot': 0}], 'stride': [{'op': 'scalar', 'slot': 0}, {'op': 'constant', 'value': 1}], 'offset': {'op': 'constant', 'value': 0}}], 'ops': [{'id': 'tmp0', 'op': 'load', 'args': [], 'dtype': 'float16', 'semantic_dtype': 'float16', 'index': {'op': 'linear'}, 'buffer': 0}, {'id': 'tmp1', 'op': 'cast', 'args': ['tmp0'], 'dtype': 'float32', 'semantic_dtype': 'float16'}, {'id': 'tmp2', 'op': 'cast', 'args': ['tmp1'], 'dtype': 'float16', 'semantic_dtype': 'float16'}, {'id': 'tmp3', 'op': 'cast', 'args': ['tmp2'], 'dtype': 'float32', 'semantic_dtype': 'float16'}, {'id': 'tmp4', 'op': 'load', 'args': [], 'dtype': 'float16', 'semantic_dtype': 'float16', 'index': {'op': 'floor_div', 'args': [{'op': 'linear'}, {'op': 'scalar', 'slot': 0}]}, 'buffer': 1}, {'id': 'tmp5', 'op': 'cast', 'args': ['tmp4'], 'dtype': 'float32', 'semantic_dtype': 'float16'}, {'id': 'tmp6', 'op': 'cast', 'args': ['tmp5'], 'dtype': 'float16', 'semantic_dtype': 'float16'}, {'id': 'tmp7', 'op': 'cast', 'args': ['tmp6'], 'dtype': 'float32', 'semantic_dtype': 'float16'}, {'id': 'tmp8', 'op': 'add', 'args': ['tmp3', 'tmp7'], 'dtype': 'float32', 'semantic_dtype': 'float16'}, {'id': 'tmp9', 'op': 'cast', 'args': ['tmp8'], 'dtype': 'float16', 'semantic_dtype': 'float16'}, {'id': 'tmp10', 'op': 'cast', 'args': ['tmp9'], 'dtype': 'float32', 'semantic_dtype': 'float16'}, {'id': 'tmp11', 'op': 'cast', 'args': ['tmp10'], 'dtype': 'float16', 'semantic_dtype': 'float16'}, {'op': 'store', 'args': ['tmp11'], 'dtype': 'float16', 'semantic_dtype': 'float16', 'index': {'op': 'linear'}, 'buffer': 2}], 'effects': {'rng': None, 'mutation': False}}, 0, policy='first_use')


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
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        s77 = arg0_1
        s27 = arg1_1
        assert_size_stride_grouped((arg2_1, arg3_1), ((s77, s27), (s77, 1)), ((s27, 1), (1, 1)), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg2_1 = copy_if_misaligned(arg2_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            buf0 = empty_strided_cuda((s77, s27), (s27, 1), torch.float16)
            cubecl_stage5_0((s27, s77), arg2_1, arg3_1, buf0)
            del arg2_1
            del arg3_1
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 3
    arg1_1 = 255
    arg2_1 = rand_strided((3, 255), (255, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((3, 1), (1, 1), device='cuda:0', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
