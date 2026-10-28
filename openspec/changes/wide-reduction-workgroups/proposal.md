# Proposal

## Why

Reduction kernels that read a row more than once (softmax, `x * x.sum(-1, keepdim=True)`,
LayerNorm) are 1.2 to 1.7 times slower than Triton on long rows. The figures are from Nsight
Compute on the RTX 3080, bead `4z8.35`, at 2048×8192:

| Case | CubeCL | Triton |
|---|---|---|
| float32 softmax | 351 µs | 203 µs |
| float32 sum consumer | 289 µs | 200 µs |
| float32 LayerNorm | 291 µs | 202 µs |
| bfloat16 softmax | 125 µs | 103 µs |

The cause is DRAM re-reads. For float32 softmax:
- CubeCL reads 174 MB from DRAM, Triton 67 MB.
- Our L1 hit rate is 13%, against Triton's 50% (Triton's second pass hits entirely).

Every reduction plan we have uses at most 256 threads per workgroup:
- With a 256-thread plan, 6 to 8 workgroups fit per SM, so 192 to 256 KB of float32 rows are in
  flight. That is more than L1 holds, so each row is evicted before its later passes.
- Triton's inner reductions run 16 warps (512 threads) for rows of 2048 elements and more, which
  keeps 2 to 3 rows per SM.

With a 512-thread `plane_shared` plan pinned in an experiment:
- float32 softmax: 214 µs, 77 MB read;
- sum consumer: 209 µs;
- bfloat16 softmax: 105 µs.

These are within about 5% of Triton.

## What Changes

- **512-thread reduction plans.**
  - `plane_shared` gains a 512-unit workgroup at vector widths 1, 2 and 4, in the shared plan
    catalog. Every reduction type can select it at any extent, as Triton applies 16 warps from
    2048 elements on.
  - Both legality layers admit it (Python `legality.py`, Rust `plan.rs`).
  - Welford kernels get it through the existing lane policy. A 512-unit plan at the widest
    vector covers Triton's 2048 lanes for float32.
- **Launch geometry.**
  - Today the launcher (`launcher.c`) accepts more than 256 threads only for the reference512
    plan, with exactly 64 bytes of shared memory.
  - It will accept 512-unit workgroups with up to 12 bytes of shared memory per unit, so Welford
    (16 planes × 12 bytes) and a compensated sum fit.
- **Long-row sums judged against Triton.**
  - Today sums longer than 4096 elements must use `shared_tree_compensated` at 256 units, a
    Stage 5 continuity policy that never compared against Triton.
  - The change measures long-row sums against Triton's accuracy band (the band test of
    `mean-variance-reductions`, extended to float32 sums at 8192 and longer), at 256 and 512
    units, compensated and not.
  - The policy becomes the plans that stay in Triton's band.
  - A 512-unit compensated plan is added if only compensated plans stay in band.
- **Timing coverage.**
  - Softmax, the sum consumer and LayerNorm (with and without affine) at 300×4096 and 2048×8192,
    float32 and bfloat16, join the timing tool. They are timed against Triton on both GPUs.
  - None of these shapes is timed today.
- **Manifest.** Every reduction case's candidate list changes, because the 512-unit plans are
  added or rejected with a reason. The changes are reviewed by content and re-recorded.

Out of scope:

- **Online softmax.** Triton's softmax makes two passes (online max and sum, then the output);
  ours makes three. That is a separate change (a follow-up bead).
- **1024-unit workgroups.** Triton caps rows of up to 8192 at 16 warps. Wider workgroups are not
  added without a measured case.
- **Short rows** (`6cl`) and the bfloat16 GroupNorm epilogue (`9hg`).

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `reduction-kernels`: wide workgroups as plan candidates for every reduction type; long-row sum
  accuracy judged against Triton's band instead of the fixed compensation rule; a performance
  floor for reductions whose kernels read the row more than once (softmax, sum consumer,
  LayerNorm) at medium and long rows.

## Impact

- **Plans:** `cubecl_torch/compiler/plan.py` (catalog), `cubecl_torch/inductor/legality.py`
  (structural, numerical and Welford lane policies, the long-row sum rule),
  `cubecl_torch/native/src/plan.rs`.
- **Native:**
  - `cubecl_torch/native/launcher.c` (launch geometry);
  - `cubecl_torch/native/src/reduction/emit.rs`, only if 512 units expose an emitter limit (the
    experiment compiled and ran without one).
- **Tests and tools:**
  - `testing/welford_reference.py` and `welford_band.py` (float32 sum cases, long rows; a new
    Triton reference for added cases only);
  - `testing/welford_timing.py` (new kinds and shapes);
  - plan and legality tests, and the manifest.
- **Runtime contract:** launch geometry only; no artifact or descriptor format change.
