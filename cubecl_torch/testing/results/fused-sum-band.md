# Fused sums and two-pass division in Triton's band (bead 4z8.32)

Band runs after the fix: `welford-band-3080.json` and `welford-band-3090.json`.
- 0 of 84 cases outside on either GPU.
- The 6 GroupNorm cases that did not compile before (`4z8.22`, `4z8.7`) still don't.
- Manifest check: PASS, unchanged. The Stage 5 corpus has no two-pass Welford kernel and no
  Welford-plus-sum kernel.

Diagnosis (3080) forced each reduction plan through `tuning.candidates` and ran the band on the
cases that were outside.

## 1. A sum fused with a Welford reduction (300×4096)

Triton's kernel is a looped Welford reduction, `R0_BLOCK 2048`, 16 warps. Its plain sum is a
2048-lane accumulator with 2-element chains per lane, then `tl.sum`.

Our kernel was not Welford-only, so it got the sum plan set:
- 16- to 32-element chains per lane;
- vector plans don't shorten them, because the emitter adds a sum's vector lanes into one
  accumulator each step (CubeK instead keeps one accumulator per vector lane).

Forced plans, float32, normal data (sum, wins/losses):

| Plan | Sum | Verdict |
|---|---|---|
| `plane_shared` 256×4 | 930/902 | in band |
| `shared_tree_multi_accumulator` 256×4 | 916/964 | in band |
| `plane_shared_load_batch` 256×4 | 930/902 | in band |
| `plane_shared` 256×1, v2, v4 | 808–841 / 1039–1076 | out |
| `shared_tree_reduction_baseline` | 828/1051 | out |
| `independent_planes` 32×1 | 427/1635 | out |

**Fix.** A kernel with a Welford reduction and a sum follows the Welford plan policy with
scalar lanes only:
- the fallback is `plane_shared` 256×4;
- the four-accumulator plane plan is admitted for the fused sum;
- Welford-catalog and vector plans are not Triton-shaped there.

## 2. Kernel-wide correctly rounded division in two-pass kernels (37×129 offset)

The short-row kernel is Inductor's two-pass fallback (three sums, no Welford reduction).
- **The old rule.** To keep constant rows at variance exactly 0, the whole kernel was compiled
  with `--prec-div=true`. Every fused division was therefore `div.rn`, while Triton uses
  `div.full` everywhere (its PTX).
- **Effect.** `out` (offset data) lost to Triton under nearly every plan: 132/164.
- **With `div.full` patched in:** level (69/56).

**Fix.** Only the two-pass mean is correctly rounded (`div_rn`, `codegen.welford_two_pass`).
Other divisions follow the division policy, so constant rows still give variance 0.

## 3. Four scalar accumulators on float32 one-pass rows (LayerNorm 4096×1280)

Found while re-running the band: a selection change picked the policy's fallback,
`plane_shared` 256×4 (1024 lanes), for the dynamic case.

| Plan | rstd | out |
|---|---|---|
| `plane_shared` 256×4 | 5405/6089 | out (p ≈ 0) |
| `plane_shared` 256×1 v4 | 2145/2160 | in band |
| `independent_planes` v4 chains | 639/642 | in band |

The result was the same in the static case and in a second run.

**Fix.** On rows of at most 2048 elements (Triton's one pass), four scalar accumulators are
Triton-shaped only when no vector plane plan is admitted. Rows that cover the lanes, long
rows, unaligned rows and fused-sum kernels keep the plan.
