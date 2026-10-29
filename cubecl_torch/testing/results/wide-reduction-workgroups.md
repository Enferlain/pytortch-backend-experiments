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

## 5.1–5.3 Two-step variance for direct 16-bit variances

**Lowering (5.1).** The backend evaluates Inductor's `use_two_step_variance` as for Triton (a
context-scoped `is_triton`). Kernel structure, checked against Inductor+Triton's generated code
on each shape:

| Case | Both backends |
|---|---|
| bf16/f16 `var_mean`, 300×1024, 300×4096, 37×4099, 256×32768 | two sums, one unsplit kernel |
| bf16 300×1023 | two sums (the rule declines; persistent-row fallback) |
| bf16 8×32768, 64×32768, 4×32769, 8×300000 | split Welford |
| bf16 1×4096 (one output), float32 300×4096, bf16 LayerNorm 300×4096 | Welford |

**Band (5.2), both GPUs, identical verdicts.** New cases: 16-bit `var_mean` at 2048×8192 (normal,
offset), and 16-bit `x.sum(-1)` and the sum consumer at 2048×8192 (Triton reference via `--new`;
stored cases unchanged).
- Every admitted plan pinned on the two-step cases (16-bit `var_mean` at 300×4096 and 2048×8192,
  mixed bf16 300×4096): 140 of 140 runs in band.
- Uncompensated plans forced on the 16-bit long rows (two-step `var_mean`, row sums, sum
  consumer; 13 plans from 64 to 512 units, v1 to v4, compensated 256/512): every run in band.
  Unlike float32 sums (3.2), compensation is not needed on 16-bit rows.

**ncu, 3080, bf16 `var_mean` (µs):**

| Plan | 300×4096 | 2048×8192 |
|---|---|---|
| Triton (two sums) | 6.88 | 50.8 |
| compensated 512 (the long-row rule before 5.3) | 12.67 | 79.9 |
| compensated 256 | 10.34 | 70.3 |
| `plane_shared` 512×v1 / v2 / v4 | 9.63 / – / 8.99 | 59.4 / 53.8 / 55.1 |
| `plane_shared` 256×v4 | 7.74 | 85.9 |
| `shared_tree` 64 | 7.78 | 97.9 |
| Welford 128×v8 before this group | 7.39 | |

**Policy (5.3).** The long-row rule (compensated only above 4096) keeps applying to sum-only
kernels unless every load and every store is 16-bit (`legality.sixteen_bit_rows`, loads found in
masked bodies too). Those kernels, the measured case, keep the plans of a short row, and the
compensated plans are rejected there ("16-bit long-row sums are in Triton's band without
compensation"). A float32 output (`x.float().sum(-1)`) or a float32 load keeps the float32 rule;
so does the first stage of a split 16-bit sum, which stores float32 partials. Review finding:
the first version tested only top-level loads, which dropped compensation from float32 split row
sums (their first stage reads inside a masked op); the manifest check caught it. Full band, first-use selection, both GPUs: **0 of 122 cases outside** (116
judged; the 6 GroupNorm cases that do not compile are unchanged). Selected
plans at 2048×8192: `var_mean` 512×v2, row sums 512×v2 (3090; 128×v2 on the 3080 for float16),
sum consumer 512×v1.

At 300×4096 the two-step kernel (7.74 µs at best) gains little over the Welford plan selection
used before (512×v8, 7.94; the faster 128×v8 at 7.39 was outside the LayerNorm band) and stays
behind Triton (6.88). Sum plans go up to 4-wide vectors for 16-bit rows, Welford's to 8-wide.
→ Group 6 (per-block cost).

## 5.4 Timing, 16-bit two-step kernels

`welford_timing --rounds 3 --only var_mean-bfloat16` and `--only sum_consumer-bfloat16`, CUPTI
medians in µs (CubeCL / Triton), first-use selection; merged into `welford-timing-*.json`.

| Case | 3080 | 3090 | Selected |
|---|---|---|---|
| `var_mean` bf16 300×4096 | 4.58 / 4.55 (was 6.33 / 5.12) | 4.60 / 4.61 (was 5.82 / 4.97) | `shared_tree` 64 |
| `var_mean` bf16 2048×8192 (new) | 54.98 / 49.62 | 48.69 / 41.39 | 512×v2 |
| sum consumer bf16 300×4096 | 5.02 / 4.35 | 4.98 / 4.40 | 128×v1 / `shared_tree` 128 |
| sum consumer bf16 2048×8192 | 102.4 / 103.8 | 85.0 / 86.1 | 512×v1 (was compensated 512) |

- `var_mean` at 300×4096 is level with Triton on both GPUs.
- At 2048×8192 the two-step kernel is a plain sum kernel, 11 to 18% behind Triton (ncu, 3080:
  53.8 against 50.8 µs). It joins the sum consumer on bead `4z8.37` (per-block sum cost).
- The bf16 sum consumer at 300×4096 stays 13 to 15% behind; it is on bead `4z8.37` from 4.2.

## 5.5 Manifest (group 5)

`manifest_check check`: 6 cases changed, all unsplit 16-bit rows of 4099 elements whose loads
and stores are 16-bit (`mean_epilogue` 64×4099, `rms_inference` 16×4099, `row_sum` 16×4099; float16
and bfloat16). In each, the compensated plans left the candidates, the short-row plans joined
them, and the chosen plan and artifact changed. No descriptor changed. Re-recorded. Suite: 415
passed.

The band measured 16-bit long rows at 8192 elements; shorter rows have shorter accumulation
chains. Unsplit 16-bit rows longer than 8192 are admitted on the same evidence and are not yet
measured.

## 6.1 Welford per-block cost: attribution

Float32 LayerNorm at 300×4096, 3080. Both backends launch 300 blocks of 512 threads.

**PTX instruction mix** (`plane_shared` 512×v4 lanes against Triton's `triton_red_fused_native_layer_norm_0`):

| | Triton | CubeCL |
|---|---|---|
| Welford-pass loads | 128-bit (`v4`, `evict_last`) | 128-bit (`v4`) plus scalar tail loads |
| Normalize-pass loads | 128-bit (`v4`, `evict_first`) | scalar 32-bit (`ld.global.nc.L1::evict_first`) |
| Normalize-pass stores | 128-bit (`v4`) | scalar 32-bit |
| `div.full.f32` | 16 | 20 |
| Warp shuffles | 33 | 30 |

Our per-element pass is scalar: one element per iteration, step one workgroup. It issues four
times Triton's memory instructions and index arithmetic there. A vector per-element pass was
measured in bead `4z8.33` on 4096×1280 (DRAM-bound, 256-unit blocks) with no gain and was not
kept. At 300×4096 the kernel is issue-bound (1.5 waves, 54% more instructions than Triton).

**Split by pass** (ncu µs, plans pinned):

| | Triton | 512×v4 lanes | 128×v4 |
|---|---|---|---|
| `var_mean` (Welford pass) | 10.43 | 11.62 | 10.56 |
| LayerNorm (Welford + normalize) | 12.19 | 14.14 | 15.65 |
| Normalize pass (difference) | 1.76 | 2.52 | 5.09 |

The gap is about 1.2 µs in the 512-lane Welford pass and about 0.75 µs in the scalar normalize
pass. On 128 units the normalize pass loses L1 reuse instead (more rows in flight per SM, as in
the proposal).

## 6.2 Vector per-element pass

For vector row plans the per-element pass now runs on the reduction pass's vectors
(`reduction::emit` `vector_elements`): one 128-bit streaming load per buffer and one 128-bit store
per output, with guarded scalar lanes for the ragged tail. Python legality checks per-element
stores as it checks loads (row stride 1, aligned base and pointer). Combine orders are unchanged.

Each plan pinned under ncu, 3080 (µs). Triton's time moved by up to 4% between the runs, so the
comparison is the ratio to Triton in each run:

| Case | Plan | Before (µs, ratio) | After (µs, ratio) | Triton now |
|---|---|---|---|---|
| LayerNorm bf16 300×4096 | 512×v8 | 9.86, 1.16 | 8.48, 0.98 | 8.61 |
| LayerNorm affine bf16 300×4096 | 512×v8 | 11.17, 1.17 | 9.63, 1.03 | 9.31 |
| LayerNorm f32 300×4096 | 512×v4 lanes | 14.14, 1.16 | 13.66, 1.08 | 12.70 |
| LayerNorm affine f32 300×4096 | 512×v4 lanes | 15.55, 1.06 | 15.94, 1.07 | 14.85 |
| sum consumer f32 300×4096 | best (512×v4) | 12.93, 1.07 | 12.64, 1.06 | 11.90 |
| LayerNorm bf16 2048×8192 | 512×v8 | | 98.66, 1.00 | 98.85 |
| LayerNorm f32 2048×8192 | 512×v4 lanes | | 218.9, 1.08 | 202.5 |
| LayerNorm affine f32 2048×8192 | 512×v4 lanes | | 276.5, 1.12 | 247.7 |

16-bit LayerNorm is now level with Triton at both shapes. Float32 keeps about 8% (the 512-lane
Welford pass, 6.1) and the affine long row (occupancy, 6.3).

### 6.2 (continued) Welford pass of the 512-lane plan

ncu, float32 `var_mean` 300×4096 (Welford pass only), 3080:

| | Triton | 512×v4 lanes | 128×v4 |
|---|---|---|---|
| µs | 10.40–10.59 | 11.58–11.65 | 10.91 |
| Instructions | 1.53 M | 1.98 M | 0.85 M |
| Branch instructions | 5,100 | 63,300 | 41,700 |
| XU (special function) | 57,600 | 78,300 | 17,100 |
| Barrier stall | 13.1% | 17.3% | 7.8% |

- **Ragged-tail branch.** Every vector step branched between the vector body and a guarded scalar
  tail. When the extent is a constant multiple of the vector width, the condition is now the
  constant `true` and the tail folds away (`Kernel::exact_vectors`). Branches 63,300 → 44,100;
  time unchanged (11.58 µs). Kept: fewer instructions, same order.
- **SASS** (static, `ptxas -O3 sm_86`): ours 688 instructions, Triton 336; `BRA` 40 against 1,
  `MUFU.RCP` 17 against 12, `FSEL` 86 against 54.
- **First element.** Triton's `welford_reduce` special-cases each accumulator's first element
  (`mean = x, m2 = 0, weight = 1`). We combine it into the empty accumulator, one division that
  gives exactly the same values. At two vectors per thread that is half of the element divisions.
- **First element peeled** (lanes plans, one chain, every unit's first vector inside the row):
  the first vector initializes the lane accumulators as (x, 0, 1), Triton's `is_first` update,
  and the loop starts one step later. Identical values for finite inputs; for an infinite first
  element m2 starts at 0, as in Triton, instead of NaN.

  | ncu µs, 3080 (two runs) | Triton | 512×v4 lanes | before 6.2 |
  |---|---|---|---|
  | `var_mean` f32 300×4096 | 10.34 / 10.43 | 10.34 / 10.46 | 11.62 |
  | LayerNorm f32 300×4096 | 12.61 / 12.26 | 12.58 / 12.38 | 14.14 |
  | LayerNorm f32 2048×8192 | 200.9 / 202.8 | 225.8 / 226.4 | |

  XU instructions 78,300 → 39,900 on `var_mean`. Float32 at 300×4096 was level with Triton at this
  point; with retained loads added in 6.3 the pair costs about 1 µs there again (bead `4z8.38`).
- **Exact vectors bug caught by the every-plan tests:** `independent_planes` chains beyond the
  first read past their loop coordinate, so exact mode applies to one-chain plans only.

**LayerNorm f32 2048×8192 is L1-bound, not instruction-bound.** Same registers (40) and blocks per
SM (3) as Triton, but L1 hit rate 22.6% against 33.3% and DRAM reads 88 against 67 MB. Triton
also loads the first pass `evict_last` (bead `4z8.33` measured that hint at 4096×1280 only). → 6.3.

## 6.3 Retained first-pass loads (Triton's `evict_last`)

**PTX experiment** (our kernel's PTX patched, plan pinned, ncu, 3080; µs, DRAM MB):

| Float32, 2048×8192 | Triton | ours | + L1 `evict_last` on first-pass loads | + Triton's L2 cache hints too |
|---|---|---|---|---|
| LayerNorm | 201.0; 67 | 226.4; 88 | 197.4; 68 | 204.5; 68 |
| LayerNorm affine | 252.5; 100 | 234.6; 91 | 205.9; 72 | 217.0; 73 |

The remaining long-row gap was the missing cache hint, not the approach. Triton's L2 hints cost
7 to 11 µs here, so only the L1 hint is adopted.

**Implementation.** The CubeCL patch gains `retained_load` (`LoadOp::set_retained`), lowered on
CUDA to `ld.global.L1::evict_last` (inline PTX, coherent path) beside the streaming
`evict_first`. The emitter (`retain_reread`) marks the reduction pass's loads of every buffer the
per-element pass reads again. Measured with the change: LayerNorm f32 2048×8192 197.3 µs (Triton
201.5), affine 205.9 (254.6), bf16 2048×8192 98.0 (99.9).

**Peel exactness.** The same pinned plan compiled static (constant extent: peeled, tail folded)
and dynamic (runtime extent: neither) gives bit-identical `out`, `mean` and `rstd` at 300×4096 and
2048×8192, zero-mean and offset data, with an infinite, a NaN and a constant row.

**Exact vectors × retained loads, synthetic shapes** (ncu µs, mean of 3):

| Case | Triton | both | retained only | exact only | neither |
|---|---|---|---|---|---|
| LayerNorm f32 300×4096 | ~12.5 | 13.41 | 12.07 | 12.49 | 12.63 |
| LayerNorm f32 2048×8192 | 201.5 | 196.5 | 197.9 | 226.3 | 218.1 |
| LayerNorm affine f32 2048×8192 | 254.6 | 205.8 | 236.0 | 235.1 | 278.4 |
| `var_mean` f32 300×4096 | ~10.3 | 10.43 | 10.90 | 10.41 | 10.85 |
| LayerNorm bf16 300×4096 | 8.58 | 8.51 | 8.70 | 8.59 | 8.64 |

**Workload shapes.** One eager training step of each workload under a dispatch-mode census
(LayerNorm inputs are float32 there: autocast upcasts before the norm):

| Workload | LayerNorm rows × extent | Calls per step (fwd + bwd) |
|---|---|---|
| SDXL | 1024 × 1280; 4096 × 640 | 180 + 180; 30 + 30 |
| Anima | 2304 × 2048 | 85 + 85 |
| Qwen-Image | 4352 × 4096 | 5 + 5 |

(Krea 2 is not in the census: the census script failed on one of its ops.)

| Case | Triton | both | retained only | exact only | neither |
|---|---|---|---|---|---|
| f32 1024×1280 | 15.00 | 15.74 | 15.97 | 15.80 | 16.09 |
| f32 4096×640 | 28.35 | 29.50 | 31.82 | 30.05 | 32.17 |
| f32 2304×2048 | 54.62 | 54.72 | 54.82 | 54.33 | 54.81 |
| f32 4352×4096 | 215.89 | 206.47 | 205.36 | 209.78 | 212.29 |
| bf16→f32 1024×1280 | 12.75 | 12.93 | 12.88 | 12.64 | 13.22 |
| bf16→f32 2304×2048 | 41.62 | 42.66 | 42.64 | 42.72 | 42.78 |
| bf16→f32 4352×4096 | 157.66 | 157.78 | 156.88 | 156.90 | 156.98 |

- At workload shapes "both" is best or tied, so exact vectors and retained loads stay
  unconditional. The penalty of the pair at 300×4096 (about 1.5 waves) is bead `4z8.38`.
- The remaining workload gap is SDXL's two shapes in float32, 4 to 5%. Their plans (256×v4,
  128×v4) fold lanes and do not take the first-element peel. → next.

## 6.1 (continued) SDXL LayerNorm shapes

The workload shapes still behind Triton after 6.2–6.3 are SDXL's float32 LayerNorms, 3.5 to 5%.
ncu, 3080, the selected plan pinned (times inflated by the multi-metric replay; compare within a
row):

| | Triton | CubeCL |
|---|---|---|
| **1024×1280** | `triton_red_`, looped Welford, 2 rows per block (512 blocks × 512 threads) | `plane_shared` 256×v4, 1 row per block (1024 × 256) |
| Instructions / branches | 4.07 M / 13k | 2.85 M / 133k |
| Barrier / long-scoreboard stall | 17% / 11% | 31% / 30% |
| **4096×640** | `triton_per_`, persistent: one warp per row, row in registers (`R0_BLOCK` 1024) | `plane_shared` 128×v4, two sums (`welford_two_pass`), 128 threads per row |
| Global load requests | 20,480 (x once) | 61,440 (x three times) |
| Instructions / barrier stall | 0.61 M / 0% | 4.60 M / 19% |

- **1280:** 320 vectors over 256 units: a quarter of the units run a second iteration while the
  rest wait at the cross-plane barrier. Triton's loop is masked, so all threads run the same
  iterations, and each block reduces two rows. → Bead `4z8.40`.
- **640:** Triton keeps the row in registers and needs no barrier; ours reads it three times
  and synchronizes across planes. → Bead `4z8.39` (register-resident persistent rows).

Both need new plan topologies (design non-goal), so they are recorded on beads, not fixed here.

## 6.4 The "most lanes" rule for float32 rows of 2049–4096

Each in-band plan pinned with the rule lifted, LayerNorm f32 at Qwen-Image's 4352×4096 (ncu,
3080, µs): Triton 216.9; `plane_shared` 512×v4 lanes (the plan the rule keeps) **206.9**; 256×v4
224.1; 256×4 and the load-batch plans 228.1–228.6; 128×v4 279.3; 128×v2 283.3. At 300×4096 the
lanes plan is level with Triton since the peel (6.2). No measured gain from lifting the rule, so
the policy is unchanged.

## 6.5 Timing after group 6

`welford_timing --rounds 3`, both GPUs (CUPTI medians in µs, first-use selection; full data in
`welford-timing-*.json`). Ratio = CubeCL / Triton; * = slower beyond the run-to-run spread.

| Case | 3080 | 3090 | Owner |
|---|---|---|---|
| var_mean-float32-37x129 | 1.93 / 1.53 (1.26*) | 2.05 / 1.66 (1.23*) | 6cl |
| var_mean-float32-300x4096 | 6.34 / 9.68 (0.65) | 6.88 / 8.74 (0.79) |  |
| var_mean-bfloat16-300x4096 | 4.65 / 5.22 (0.89) | 4.79 / 5.02 (0.95) |  |
| var_mean-bfloat16-2048x8192 | 50.32 / 49.6 (1.01*) | 42.58 / 41.5 (1.03*) | 4z8.37 (two-step sum kernel) |
| var_mean-float32-8x300000 | 20.89 / 19.95 (1.05*) | 18.5 / 19.46 (0.95) | spread (1.05 / 0.95) |
| layer_norm-bfloat16-4096x1280 | 33.36 / 40.32 (0.83) | 28.16 / 36.74 (0.77) |  |
| layer_norm-float32-4096x1280 | 64.56 / 66.1 (0.98) | 53.89 / 54.68 (0.99) |  |
| layer_norm_mean-bfloat16-4096x1280 | 22.05 / 31.97 (0.69) | 21.07 / 29.64 (0.71) |  |
| layer_norm_mean-float32-4096x1280 | 35.13 / 41.88 (0.84) | 31.1 / 39.03 (0.80) |  |
| group_norm-float32-1x320x64x64 | 28.47 / 26.99 (1.05*) | 24.01 / 23.24 (1.03*) | 9hg |
| group_norm-bfloat16-1x320x64x64 | 14.9 / 12.89 (1.16*) | 13.88 / 11.27 (1.23*) | 9hg |
| softmax-float32-300x4096 | 15.94 / 16.82 (0.95) | 12.9 / 14.29 (0.90) |  |
| softmax-float32-2048x8192 | 202.75 / 206.82 (0.98) | 167.29 / 171.97 (0.97) |  |
| softmax-bfloat16-300x4096 | 7.04 / 9.76 (0.72) | 7.18 / 9.99 (0.72) |  |
| softmax-bfloat16-2048x8192 | 101.38 / 105.87 (0.96) | 84.52 / 87.23 (0.97) |  |
| sum_consumer-float32-300x4096 | 16.11 / 15.45 (1.04*) | 11.68 / 10.35 (1.13*) | 4z8.37 |
| sum_consumer-float32-2048x8192 | 207.05 / 205.19 (1.01*) | 170.94 / 171.63 (1.00) | 4z8.37 |
| sum_consumer-bfloat16-300x4096 | 6.76 / 4.96 (1.36*) | 4.79 / 4.96 (0.97) | 4z8.37 |
| sum_consumer-bfloat16-2048x8192 | 99.72 / 103.7 (0.96) | 82.5 / 86.09 (0.96) |  |
| layer_norm-float32-300x4096 | 15.71 / 15.32 (1.03*) | 11.93 / 11.12 (1.07*) | 4z8.38 |
| layer_norm-float32-2048x8192 | 199.58 / 205.48 (0.97) | 165.78 / 170.35 (0.97) |  |
| layer_norm-bfloat16-300x4096 | 7.33 / 7.87 (0.93) | 7.19 / 7.37 (0.98) |  |
| layer_norm-bfloat16-2048x8192 | 100.58 / 103.86 (0.97) | 83.84 / 86.52 (0.97) |  |
| layer_norm_affine-float32-300x4096 | 17.33 / 17.19 (1.01) | 15.05 / 14.9 (1.01) |  |
| layer_norm_affine-float32-2048x8192 | 207.56 / 224.56 (0.92) | 173.22 / 209.62 (0.83) |  |
| layer_norm_affine-bfloat16-300x4096 | 8.23 / 8.69 (0.95) | 8.67 / 9.28 (0.93) |  |
| layer_norm_affine-bfloat16-2048x8192 | 104.21 / 107.8 (0.97) | 87.05 / 89.5 (0.97) |  |
| layer_norm-float32-1024x1280 | 18.43 / 17.12 (1.08*) | 15.63 / 15.83 (0.99) | 4z8.40 |
| layer_norm-float32-4096x640 | 33.13 / 31.72 (1.04*) | 27.48 / 26.22 (1.05*) | 4z8.39 |
| layer_norm-float32-2304x2048 | 57.46 / 58.92 (0.98) | 48.18 / 49.27 (0.98) |  |
| layer_norm-float32-4352x4096 | 209.34 / 218.91 (0.96) | 173.0 / 181.22 (0.95) |  |
| layer_norm_upcast-bfloat16-1024x1280 | 14.66 / 14.11 (1.04*) | 12.86 / 13.14 (0.98) | 4z8.40 |
| layer_norm_upcast-bfloat16-4096x640 | 25.33 / 24.13 (1.05*) | 21.51 / 19.95 (1.08*) | 4z8.39 |
| layer_norm_upcast-bfloat16-2304x2048 | 44.0 / 43.74 (1.01*) | 37.27 / 36.78 (1.01*) | spread (1.01 on both) |
| layer_norm_upcast-bfloat16-4352x4096 | 159.3 / 159.89 (1.00) | 132.31 / 133.34 (0.99) |  |

Every case slower than Triton on either GPU has an owner ("spread": slower on one GPU or by
0.01 only, within what repeated runs move): short rows `6cl`, GroupNorm `9hg`, the
sum consumer and two-step sums `4z8.37`, the low-wave pair `4z8.38`, SDXL's LayerNorm shapes
`4z8.39` (640, persistent rows) and `4z8.40` (1280, multi-row blocks). The long float32 rows that
were 8 to 25% behind before group 6 are ahead on both GPUs.

## 6.6 Review fixes, checks and manifest (group 6)

**Review findings fixed.**
- The per-element store check first rejected whole vector plans:
  - bf16-to-float32 LayerNorm (autocast's form) lost its eight-lane plans, because a float32
    store of eight lanes is 32 bytes;
  - bool or integer outputs, permuted stores and broadcast stores rejected every vector plan.
- Now:
  - the pass stores vectors only when every per-element store is a float store along the row
    (Rust `vector_stores`, Python `vector_element_stores`), and runs scalar otherwise;
  - float32 stores of eight lanes are split into two 16-byte transactions.
- Retained loads are emitted for row plans only.
- On a scalar plan (`plane_shared` 256×4, ragged rows) the hint is neutral, ncu µs with/without:
  - f32 1024×1283: 17.72 / 17.78;
  - f32 4096×2051: 102.6 / 102.8;
  - bf16→f32 4096×2051: 75.6 / 76.0.
- New tests: retained loads before the per-element loop in the PTX; the per-element pass over
  bf16, upcast-float32 and bool outputs; the peel against the unpeeled (dynamic) kernel, bit for
  bit, with an infinite, a NaN and a constant row.

**bf16→f32 LayerNorm re-timed** (supersedes those rows of 6.5; CubeCL / Triton µs):

| Shape | 3080 | 3090 | Plan |
|---|---|---|---|
| 1024×1280 | 12.98 / 13.10 (0.99) | 10.44 / 11.88 (0.88) | 256×v8 |
| 4096×640 | 25.42 / 24.10 (1.05*) | 21.54 / 19.92 (1.08*) | 128×v4 (bead `4z8.39`) |
| 2304×2048 | 42.83 / 43.75 (0.98) | 36.06 / 36.69 (0.98) | 256×v8 |
| 4352×4096 | 158.1 / 159.7 (0.99) | 131.2 / 133.5 (0.98) | 512×v8 |

**Checks.**
- Suite: 417 passed.
- Full band, first-use: 0 of 122 cases outside on both GPUs.
- Pinned LayerNorm band: every plan in every dtype at 300×4096, 2048×8192 and 4096×2048, 56 of
  56 runs in band.
- Manifest: 38 cases changed (RMSNorm forward and backward, training), all of them kernels with a
  per-element pass. The changes are in compiled code only: 264 candidate artifacts and 58 chosen
  artifacts (CubeCL IR, CUDA, PTX). No candidate was added or removed, and no selection, plan or
  descriptor changed. Re-recorded.
