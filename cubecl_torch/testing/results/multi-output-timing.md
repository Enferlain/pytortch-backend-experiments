# Multi-output kernel timing (multi-output-kernels 6.1)

Re-run on 2026-10-28 after groups 1–5:
- `welford_timing.py`, `float_arithmetic_timing.py` and `rounding_remainder_timing.py`;
- `--rounds 3`, alternating backends, warm clocks;
- RTX 3080, then RTX 3090.

Raw data: the `welford-timing-*`, `float-arithmetic-timing-*` and `rounding-remainder-timing-*`
JSON files in this directory.

The table gives medians as CubeCL / Triton, in µs.

| Case | RTX 3080 | RTX 3090 | Before this change (3080) | Verdict |
|---|---|---|---|---|
| Adam step, 4M float32, 3 outputs | 196.7 / 196.8 | 162.5 / 166.0 | 248.2 / 196.7 | at or better than Triton |
| frexp, both outputs, 4M | 100.1 / 100.1 | 83.2 / 83.2 | 126.4 / 100.2 | level |
| var_mean float32 300×4096 | 6.2 / 9.6 | 6.1 / 8.6 | | faster |
| var_mean bfloat16 300×4096 | 5.4 / 5.2 | 4.8 / 5.1 | | within spread (3080), faster (3090) |
| var_mean float32 8×300000 (split) | 20.8 / 19.8 | 18.7 / 19.1 | | first stage ~3% slower per kernel (`4z8.34`); second stage short (`6cl`) |
| var_mean float32 37×129 | 1.9 / 1.6 | 2.0 / 1.7 | | short-kernel exception (`6cl`) |
| LayerNorm bfloat16 4096×1280 | 36.6 / 40.2 | 33.0 / 36.0 | 76 / 41 (welford-reductions) | faster |
| LayerNorm float32 4096×1280 | 69.3 / 66.0 | 58.6 / 54.8 | 129 / 66 | 5–7% slower (`4z8.33`) |
| GroupNorm float32 1×320×64×64 | 25.4 / 26.9 | 21.7 / 23.0 | 47 / 27 | faster |
| GroupNorm bfloat16 1×320×64×64 | 14.3 / 13.0 | 13.1 / 11.2 | 30 / 12 | slower: epilogue 7.6 / 5.1 and 7.0 / 5.1, first stage 5.16 / 4.81 (+7%) and 4.76 / 4.61 (+3%) (`9hg`) |

The table is a subset; the JSON files also hold the `layer_norm_mean` cases, which are faster
than Triton on both GPUs. The norms now run Triton's kernel count, each kernel computing its
reduction once.

**The Welford plan rule's cost, revisited** (as asked in the welford-reductions notes). On the
now single-kernel split `var_mean` 8×300000, the default selection takes 20.8 µs on the 3080 and
18.7 on the 3090. The unrestricted selection takes 19.3 and 17.2: about 7–8% faster, through
plans outside Triton's accuracy band.
- On LayerNorm mean only, the unrestricted selection is also 10–16% faster: 3080 bfloat16
  18.8 vs 22.4 µs, float32 32.0 vs 35.5; 3090 18.6 vs 21.1 and 28.0 vs 30.9. The default
  selection is still faster than Triton there.
- On the multi-output LayerNorm, GroupNorm and unsplit `var_mean` cases, the two selections are
  within about 3% of each other.

The split first stage's gap stays on `4z8.34`.

## Remaining gaps

- **LayerNorm float32 (`4z8.33`).** Nsight Compute on the 3080 at base clocks: 70.9 vs 65.2 µs.
  - Our kernel reads 23.5 MB from DRAM against Triton's 21.0 MB, the input size. Writes are equal.
  - The per-element pass re-reads about 12% of rows from DRAM. Triton keeps rows in L2 with
    `evict_last` on its first-pass loads and `evict_first` on the second.
  - Closing it needs cache hints or register-resident rows in the emitter.
- **Split Welford first stage (`4z8.34`).** About 3% slower per kernel on both GPUs, the
  occupancy gap already recorded for this kernel. The whole `var_mean` is faster on the 3090.
- **Short kernels (`6cl`)** are unchanged.
- **bfloat16 GroupNorm (`9hg`).** The epilogue gap is unchanged. Its first stage is also 3–7% slower
  per kernel, now recorded on `9hg`.

The float-arithmetic and rounding/remainder cases outside the multi-output ones show the same
picture as their earlier records.
- Small cases (4096 elements, the Adam step included: 1.62 / 1.50 and 1.71 / 1.63) are inside
  the short-kernel exception.
- Several large pointwise cases are marked slower by under 1%: `copysign`, `floor`/`round`,
  `abs_relu`, the separate Adam updates, and `fmod_small_q` on the 3090. These are at
  noise level, as in their earlier records.
- The large-quotient `fmod` and `remainder` cases (2–5%) are on bead `njp`.
