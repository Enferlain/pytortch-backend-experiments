# Streaming loads in the per-element pass (bead 4z8.33)

## The gap

Float32 LayerNorm 4096×1280 ran 69.4 µs on the RTX 3080, against Triton's 66.1 (+5%).
Nsight Compute at base clocks on the selected plan, `plane_shared` 256×1 v4:

| | CubeCL | Triton |
|---|---|---|
| Time | 70.1–70.6 µs | 64.8 µs |
| L1 global-load misses | ~928k sectors | 683k (≈ `x` once) |
| L2 reads | 922–930k sectors | 683k |
| DRAM reads | 23.2–23.6 MB | 21.1 MB |

About 38% of our second-pass re-reads of `x` miss L1.

Triton's program (`XBLOCK 2, R0_BLOCK 2048`, 16 warps) also re-reads `x` in its second loop, but:
- it loads `x` there with `eviction_policy='evict_first'`;
- its first-pass `x` and its `weight`/`bias` loads use `evict_last`.

It lowers these to `ld.global.L1::evict_*` with an `L2::cache_hint` policy.

## Experiments (patched PTX of our kernel, ncu, 3080)

| Change | µs | L2 reads |
|---|---|---|
| None | 70.1–70.6 | 922–930k |
| Second-pass `x`: `ld.global.nc.L1::evict_first` | 64.4–64.5 | 729k |
| Second-pass `x`: `ld.global.cs` (CUDA `__ldcs`) | 65.1–65.3 | 726–734k |
| First-pass `x`: `evict_last` only | 65.1 | 739k |
| All of Triton's L1 hints | 63.9 | 705k |

- **Scalar or vector second pass:** the same with the hint. A vectorized per-element pass on its
  own gave nothing (70.6 µs), so it was not kept.
- **CubeCL and CubeK** have no load cache hints.

## Fix

**CubeCL patch** (`cubecl-fp.patch`):
- `LoadOp` gains an optional `streaming_load` attribute (`set_streaming` / `is_streaming`).
- The CUDA C++ backend lowers it to `ld.global.nc.L1::evict_first`. This is inline PTX, not
  `volatile`, reinterpreted from 16/32-bit words.
- `__ldcs` made NVRTC drop `.nc` from every other load of the buffer, about 1 µs.
- Unflagged loads, unsized types and other targets load as before.
- A first version sized every load before checking the flag and panicked on `bool`. The suite
  caught it.

**Emitter:** the per-element pass (`reduction::emit` `stores`) marks its loads of buffers that
the reduction pass also loaded.

Result: float32 LayerNorm 4096×1280 takes 64.8–65.2 µs under ncu (Triton 64.8). Traffic is at
Triton's level: 727k L2 read sectors, 21.4 MB from DRAM.

## Per-plan effect (ncu, 3080, each admitted plan pinned, with and without the hint)

| Case | Best with hint | Best without |
|---|---|---|
| LayerNorm float32 300×4096 | 15.14 | 16.16 |
| sum consumer float32 300×4096 | 14.21 | 15.23 |
| softmax float32 300×4096 | 14.59 | 16.06 |
| LayerNorm float32 4096×1280 | 62.98 | 68.10 |
| softmax bfloat16 300×4096 | 11.01 | 11.23 |
| LayerNorm bfloat16 2048×8192 | 122.78 | 132.10 |

- Most plans gain 2–13%.
- Some are slower with it: the 64-thread plans (about +5%) and `independent_planes` 32×1 on the
  sum consumer (+17–30%).
- First-use selection times every plan with the hint, so those simply lose selection.

## Manifest

38 cases changed: RMSNorm inference and backward (18 each) and 2 training cases.
- Only `cubecl.ir`, `kernel.cu` and `kernel.ptx` of candidate and chosen artifacts changed.
- No plan, candidate-list, descriptor or contract changed.
- A spot check (bfloat16 RMSNorm 64×1024) shows one streaming load, the input re-read in the
  per-element loop, with output equal to eager. Re-recorded.

## Not addressed here (bead `4z8.35`)

An ad-hoc CUPTI comparison against Triton (softmax, `x * x.sum(-1, keepdim=True)`, LayerNorm
without affine) found gaps that the hint does not touch:
- 2048×8192 rows: about 1.4× in float32, 2.3× in bfloat16;
- bfloat16 37×129: 3–4×;
- bfloat16 300×4096 LayerNorm and softmax: about 1.2–1.9×.

None of these shapes is in the timing tools.
