# Manifest changes from multi-output fusion (multi-output-kernels 4.3)

Group 4.2 dropped the fusion rule "one disjoint external output required". The manifest check
against the group-3 manifest (2026-10-28, PASS after groups 1–3) reported 32 changed entries:
16 corpus cases × 2 policies, in three families.

| Family | Cases | Regions before → after | Old regions without a match | New regions without a match |
|---|---|---|---|---|
| `dropout_backward` (float32/16, bfloat16 × 257, 65539) | 6 | 4 → 3 | 2 | 1 |
| `rms_backward` (float32/16, bfloat16 × 16×4099, 64×1024, 64×255) | 9 | 3 → 2 | 2 | 1 |
| `training-float32-5x9` | 1 | 8 → 7 | 2 | 1 |

Every entry has exactly the same shape: two old kernels are gone and one new kernel replaces
them. Each region's full record must match exactly (descriptor digest, chosen plan, every
candidate's IR/CUDA/PTX digests, candidate sets and selections), and every other region of
every changed case does. The check's own report lists later regions as changed only because it
compares by region index, and the merge shifts the indices.

## Method

`manifest_review.py` (scratch, not kept) recompiled the 16 cases under both policies with the
package only, as `manifest_check check` does. For each entry it then matched the recorded and
the new region lists by the canonical JSON of each whole region record, not by index. No
compile errors.

## Why these are the multi-output fusions

The only admission change since the last PASS (group 3) is lifting the single-output fusion
rule. Group 1 raised the input limit with the manifest unchanged, and group 3's masked-region
key also left it unchanged. So a new kernel can only be a fusion that the rule used to split,
replacing the kernels it was split into:
- `dropout_backward`: the backward graph's two pointwise outputs;
- `rms_backward`: the gradient and its companion output;
- `training-float32-5x9`: one fused pair in its backward graph.

Every other kernel in these cases, and every kernel in the other 152 corpus cases, is
byte-identical.

The manifest was re-recorded after this review (`manifest_check record --jobs 6`), and
`check` passes against it.
