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
