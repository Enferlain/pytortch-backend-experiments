# 512-unit reduction workgroups (OpenSpec wide-reduction-workgroups)

## 1.4 Catalog plans, pinned, against Triton (2048×8192)

Setup:
- Nsight Compute on the RTX 3080 at base clocks.
- One kernel per case on both sides.
- Each CubeCL plan pinned by replacing `tuning.AutotunedKernel.select`, with the numerical policy
  lifted for the sum consumer (the long-row sum rule is group 3).

Cells are DRAM read in MB / kernel time in µs.

| Plan | softmax f32 | sum consumer f32 | softmax bf16 | sum consumer bf16 |
|---|---|---|---|---|
| `plane_shared` 256×v1 | 175 / 348 | 128 / 284 | 50 / 125 | 45 / 114 |
| `plane_shared` 256×v2 | 194 / 374 | 132 / 289 | 80 / 169 | 60 / 138 |
| `plane_shared` 256×v4 | 198 / 379 | 132 / 288 | 82 / 174 | 60 / 139 |
| `shared_tree_compensated` 256 (sum-only) | | 132 / 288 | | 53 / 128 |
| `plane_shared` 512×v1 | 78 / 213 | 80 / 215 | 34 / 105 | 34 / 101 |
| `plane_shared` 512×v2 | 79 / 213 | 78 / 212 | 36 / 108 | 38 / 109 |
| `plane_shared` 512×v4 | 78 / 213 | 78 / 212 | 36 / 108 | 38 / 108 |
| Triton | 67 / 203 | 67 / 202 | 34 / 104 | 34 / 99 |

- **The 512-unit plans** keep rows in L1 across passes. DRAM reads fall to within 16% of
  Triton's in float32 and match it in bfloat16.
- **bfloat16** is level with Triton.
- **float32 remains about 5% slower** (213 against 203 µs), with 11 MB more read:
  - for softmax, part of it is the third pass (Triton's softmax makes two; online softmax is a
    follow-up);
  - for the sum consumer, which makes two passes like Triton, it remains to be explained (task
    4.2 records or fixes it).
- **The sum consumer's current plan** (long-row rule: `shared_tree_compensated` 256) takes
  288 µs in float32 and 128 µs in bfloat16, so the rule decides most of its gap (group 3).

## 1.5 Manifest review (group 1)

`manifest_check check`: 162 cases changed. The records were compared on content: the old
manifest against a fresh record, every entry parsed.

- **`rejected` (198 regions):** each gained only the 512-unit `plane_shared` entries:
  - "wrong semantic family" for pointwise regions;
  - the extent rule for rows under 2048;
  - the long-row sum rule for the corpus's sum rows over 4096.
- **`selections` (79 regions):** each tuning key gained only the 512-unit catalog entries and the
  new runtime fact `extent >= 2048`.
- **Unchanged:** no winner; no `candidates` list (no corpus case admits a 512-unit plan yet); no
  compiled or chosen artifact; no descriptor.

Re-recorded.

## 2.2 Welford band with each admitted plan pinned (both GPUs, identical verdicts)

New band cases (Triton reference via `welford_reference --new`; the 93 stored cases are
unchanged): LayerNorm with and without affine at 4096×2048 and 2048×8192, in float32, float16
and bfloat16, 2 inputs each. In Triton, every one is a single looped Welford kernel.

- **4096×2048:** every admitted plan is in band in every dtype (float32: 512×v4 and the 32×16×v4
  chains; 16-bit: 256×v8, 512×v8 and the 32×8×v8 chains).
- **16-bit, 2048×8192:** `plane_shared` 512×v8 is in band in all four cases (float16 and
  bfloat16, affine and plain). Every plan admitted before this change is outside the band on
  `mean` and/or `rstd`: 256×4 and load-batch 256×4 (`mean`, p ≈ 0), 128×v8, and 256×v8
  (bfloat16 `rstd` p 0.0007, float16 plain `mean` p 0.006). This was an existing accuracy gap on
  long 16-bit rows.
- **float32, 2048×8192:**
  - With affine (different inputs), no plan is in band:

    | Plan | Outside on |
    |---|---|
    | 512×v4 | `rstd` 181/256, `out` |
    | 256×4 | `mean` 1623/1733 |
    | 256×v4 | `mean`, `rstd` |
    | load-batch | `rstd`, `out` |

  - Without affine, 256×4, 256×v4 and 512×v4 are in band.
  - This is also an existing gap: before this change the admitted plans were 256×4 and 256×v4.
  - Likely cause: reduction order. Triton keeps 2048 independent Welford lanes, each sequential
    over the row's 4 chunks, merged by a tree at the end. Our vector plans merge a vector's
    lanes at every step (the lane folding of `4z8.32`, here for Welford).

## 2.4 Lane accumulators (design decision 7): pinned band on both GPUs, identical verdicts

`plane_shared` 512×v4 and 512×v8 with `private_merge = "lanes"` keep one Welford accumulator per
vector lane and merge the lanes in order after the loop. Pinned against the folding plans of
the same shape:

| Rows | Folding 512 | Lanes 512 |
|---|---|---|
| float32 2048×8192, affine | outside (`rstd` 181/256, `out`) | in band (`rstd` 158/180) |
| float32 2048×8192, plain | in band | in band |
| float32 4096×2048 (one pass) | in band | outside (`rstd` 434/563, p ≈ 0) |
| 16-bit 2048×8192 and 4096×2048 | in band | outside (`mean`, p ≈ 0) |
| `var_mean` float32 300×4096, 16-bit 300×4096 | in band | in band |

The variant reproduces Triton's order where Triton's lanes each run over the row (float32 rows
longer than 2048). On a one-pass row Triton is a pure tree, which the folding plans match.

Policy (`welford_plan_reason`):
- **Lanes plans** only on rows longer than `TRITON_LANES` whose loads are all float32. There they
  replace the folding plan of the same shape.
- **Longer rows** keep the shaped plans with the most lanes (2.2).
- **Bug caught by the band:** the first version keyed the rule on the reduction's
  `source_dtype`, which is float32 after the upcast of 16-bit rows. That selected lanes for 16-bit
  rows (4 cases outside on `mean`). It now uses the loads' storage dtype.

Full band with first-use selection, both GPUs: **0 of 96 cases outside**. The 6 GroupNorm cases
that do not compile are unchanged. Selected plans: float32 LayerNorm 2048×8192 512×v4 lanes;
16-bit 512×v8 folding. Results are in `welford-band-3080.json` and `-3090.json`.

## 3.1–3.2 Long-row float32 sums against Triton's band

New band cases (Triton reference via `--new`; the stored cases and the file's torch label are
unchanged):
- `x.sum(-1)` and `x * x.sum(-1, keepdim=True)` in float32;
- at 2048×8192 and 256×32768, zero-mean and offset data, 2 inputs each.

Each is one unsplit kernel in both backends (Triton `two_pass`). Under first-use selection
(3080), all 8 cases are in band, on today's `shared_tree_compensated` 256.

**Forced plans, both GPUs (identical verdicts).** Every uncompensated plan is outside the band on
the zero-mean cases (`sum`, and `out` through it), and in band only on some offset cases:

| Plans | In band (of 16 runs) |
|---|---|
| `plane_shared` 64–512 at v1/v2/v4, `shared_tree` 64/128 | 0–8 |
| `independent_planes` 32 | 0 |
| the 256-unit baseline and multi-accumulator plans | 8 |
| `plane_shared` 512×v1 / v2 | 8 |
| `shared_tree_compensated` 256 | **16** |
| `shared_tree_compensated` 512 (added in 3.3) | **16** |

So the long-row rule stays "compensated only", now as a measured result. Its rejection reason
names Triton's band, and `shared_tree_compensated` gains a 512-unit workgroup (8 bytes of
shared memory per unit, inside the launcher's limit).

**ncu, 3080** (DRAM read MB / µs):

| Case | compensated 256 | compensated 512 | Triton |
|---|---|---|---|
| sum consumer 2048×8192 | 132 / 289 | 86 / 224 | 67 / 202 |
| `x.sum(-1)` 2048×8192 | 67 / 97.1 | 67 / 97.1 | 67 / 95.2 |
| sum consumer 256×32768 | 67 / 145.4 | 67 / 146.0 | (not measured) |

The sum consumer at 2048×8192 stays about 11% slower than Triton (uncompensated `plane_shared`
512 reaches 212 µs but is outside the band); group 4 times it and records or fixes it.

## 3.3 Long-row rule, fallback, band and manifest

- **Rule.** `numerical_reason` keeps "compensated only" for sum-only kernels over 4096 elements,
  with the reason "long-row sums outside Triton's band without compensation". The catalog
  gains `shared_tree_compensated` 512, and `plan.rs` admits it.
- **Fallback.** With two compensated plans admitted, first-use selection runs. The frozen sum
  baseline is not a candidate there, so the band and 4 suite tests failed with "no qualified
  fallback candidate". `tuning.select` now falls back to `shared_tree_compensated` 256 when the
  frozen plan is not a candidate but it is, as the reference512 branch already did.
- **Band**, first-use, both GPUs: **0 of 104 cases outside**. The 11 cases that had failed on the
  fallback (the long-row sums, and the split 8×300000 sum-consumer and mixed cases, whose
  final stage reduces a long row) were rerun after the fix and merged into
  `welford-band-*.json`. Selection takes compensated 512 for
  the 2048×8192 sum consumer on both GPUs, and for the 2048×8192 row sum on the 3090.
- **Manifest:** 162 cases changed. Re-recorded and compared on content:
  - `rejected` in 198 regions: the 512-unit compensated plan's entries, and the renamed reason;
  - `candidates` and `selections` in 21 regions: the new plan added to the corpus's long-row sum
    kernels (64×4099 mean epilogue, 16×4099 RMSNorm forward and backward, row sums over 4099
    and 65539);
  - in those 21 first-use regions, the chosen plan became `shared_tree_compensated` 512, and the
    compiled list gained exactly one entry (none removed or changed);
  - no descriptor changed.
- **Suite:** 412 passed.

## 4.2 Timing against Triton, and the 300×4096 investigation

`welford_timing --rounds 3`, CUPTI per-kernel medians in µs (CubeCL / Triton), first-use
selection. Full data in `welford-timing-3080.json` and `-3090.json`. Part of the 3090 run shared
the GPU with video playback (it drives the display); the pinned ncu measurements below are on
the 3080 at fixed clocks.

| Case | 3080 | 3090 | Selected (3080) |
|---|---|---|---|
| `var_mean` f32 300×4096 | 8.87 / 9.65 | 8.40 / 8.63 | 512×v4 lanes |
| `var_mean` bf16 300×4096 | 6.33 / 5.12 | 5.82 / 4.97 | 512×v8 |
| softmax f32 300×4096 | 16.47 / 16.92 | 13.34 / 14.40 | 512×v2 |
| softmax f32 2048×8192 | 215.6 / 206.9 | 178.6 / 181.4 | 512×v4 |
| softmax bf16 300×4096 | 8.55 / 9.85 | 8.46 / 9.77 | 128×v1 |
| softmax bf16 2048×8192 | 109.5 / 105.6 | 91.6 / 87.5 | 512×v1 |
| sum consumer f32 300×4096 | 16.20 / 15.43 | 11.87 / 10.42 | 512×v2 |
| sum consumer f32 2048×8192 | 226.5 / 205.1 | 193.0 / 171.2 | compensated 512 |
| sum consumer bf16 300×4096 | 5.58 / 4.92 | 5.96 / 4.87 | 128×v1 |
| sum consumer bf16 2048×8192 | 102.5 / 103.6 | 85.6 / 86.3 | compensated 512 |
| LayerNorm f32 300×4096 | 17.38 / 15.34 | 14.90 / 11.07 | 512×v4 lanes |
| LayerNorm f32 2048×8192 | 216.6 / 205.3 | 179.5 / 170.6 | 512×v4 lanes |
| LayerNorm bf16 300×4096 | 8.82 / 8.01 | 8.19 / 7.11 | 512×v8 |
| LayerNorm bf16 2048×8192 | 108.7 / 103.8 | 91.0 / 86.4 | 512×v8 |
| LayerNorm affine f32 300×4096 | 18.66 / 17.29 | 16.92 / 14.94 | 512×v4 lanes |
| LayerNorm affine f32 2048×8192 | 280.7 / 225.3 | 233.0 / 209.1 | 512×v4 lanes |
| LayerNorm affine bf16 300×4096 | 10.14 / 8.65 | 10.89 / 9.52 | 512×v8 |
| LayerNorm affine bf16 2048×8192 | 109.5 / 107.5 | 92.2 / 89.3 | 512×v8 |

### Each plan pinned at 300×4096 (ncu, 3080, µs), with the "most lanes" rule lifted

| Case | Triton | Selected today | Fastest plan |
|---|---|---|---|
| `var_mean` bf16 | 6.82 | 512×v8 7.94 | 128×v8 7.39 |
| LayerNorm bf16 | 8.48 | 512×v8 9.86 | 128×v8 8.86 |
| LayerNorm affine bf16 | 9.54 | 512×v8 11.17 | 128×v8 10.85 |
| `var_mean` f32 | 10.43 | 512×v4 lanes 11.62 | 128×v4 10.56 |
| LayerNorm f32 | 12.19 | 512×v4 lanes 14.14 | the same |
| LayerNorm affine f32 | 14.72 | 512×v4 lanes 15.55 | the same |
| sum consumer f32 | 12.10 | (selection) | 512×v2 12.93 |

### Band with each plan pinned at 300×4096 (both GPUs, identical verdicts)

New band cases: LayerNorm with and without affine at 300×4096 in every dtype (Triton reference
via `--new`; the stored cases are unchanged).

- **float32:** every shaped plan is in band (`var_mean` normal and offset, LayerNorm).
- **16-bit LayerNorm:** only 512×v8 is in band in all four cases. 128×v8 is outside on `mean`
  (bfloat16 p 0.0007), 256×v8 on `rstd` (bfloat16 plain p 0.006), and the scalar and
  load-batch plans on `mean`.
- **16-bit `var_mean`:** every plan is in band. Its 16-bit outputs hide the differences the
  LayerNorm cases show on their float32 `mean` and `rstd`.

### Causes

- **16-bit `var_mean`: a different algorithm.** Triton's kernel for bfloat16 `var_mean` at
  300×4096 is a sum for the mean and a sum of squared deviations, not Welford:
  `lowering.use_two_step_variance` applies to CUDA 16-bit inputs of 1024 to 32768 elements,
  unsplit, under `is_triton()`. Our backend sets `cuda_backend` to its own name, so it lowers
  these rows to Welford. LayerNorm and GroupNorm upcast to float32 before `var_mean` and keep
  Welford in both backends. → Group 5 (design decision 8).
- **LayerNorm, every dtype: Welford per-block cost.** At 300×4096 both backends launch 300 blocks
  of 512 threads and read the same DRAM bytes (float32 LayerNorm 4.9 MB). ncu (float32
  LayerNorm, 512×v4 lanes against Triton):

  | | Triton | CubeCL |
  |---|---|---|
  | Instructions executed | 1.67 M | 2.58 M |
  | Barrier stall | 9.7% | 17.1% |
  | Long-scoreboard stall | 55.1% | 44.8% |
  | DRAM throughput | 84.7% | 76.4% |

  On LayerNorm with affine at 2048×8192 (float32), our kernel also keeps 3 blocks per SM (40
  registers) against Triton's 2 (50). Three rows plus the weights exceed L1: 125 MB read against
  75 MB. The 16-bit LayerNorm cases (300×4096, where only 512×v8 is in band, and 2048×8192) and
  float32 LayerNorm at 2048×8192 are the same Welford-against-Welford comparison. → Group 6
  (design decision 9).
- **Sum consumer** (float32 at both shapes, bfloat16 at 300×4096): bead `4z8.37`. On long rows
  the band admits only the compensated plans.
- **Softmax at 2048×8192** (float32 and bfloat16): the third pass. Bead `4x9.10` (online
  softmax); the pass counts and DRAM traffic were measured under the closed duplicate `4z8.36`.

Note on the JSON: `cubecl_fastest_plan` times every catalog plan, including plans the Welford
policy excludes (for example `shared_tree` 128 on float32 `var_mean` at 300×4096, 5.9 µs). Those
plans are outside Triton's band (`welford-reductions` 5), so they are not candidates. The 3090
selected different plans for some cases; the table lists the 3080's.
