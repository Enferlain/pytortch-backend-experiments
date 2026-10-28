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
