
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr1': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 393216}, 'backend_hash': 'D211EA373091881AD6035AEE7CA9024CEC3FFA3259319C745F96FEF3A7197C29', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_0(in_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float16)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tmp8.to(tl.float16)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = (ks0).to(tl.float32)
        tmp13 = (tmp12).to(tl.float32)
        tmp14 = triton.language.div_rn(tmp6, tmp13)
        tmp15 = tl.full([1, 1], 1e-05, tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp11 * tmp17
        tmp19 = tmp18.to(tl.float16)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20.to(tl.float16)
        tmp22 = tmp21.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + ks0*x0), tmp22, r0_mask & xmask)
