# Basic float kernel timing against Inductor+Triton (basic-float-arithmetic 5.2)

The data is in `float-arithmetic-timing-3080.json` and `float-arithmetic-timing-3090.json`,
produced by `python -m cubecl_torch.testing.float_arithmetic_timing run --rounds 3 --gpu <gpu>`
(2026-10-27). Each value is the generated kernels' summed device time per call, in µs: per-round
values, then the median gap. As in `integer-timing.md`, a case is slower when every CubeCL round
exceeds every Triton round, faster in the reverse case, and otherwise the ranges overlap.

## 4M elements

| Case | GPU | CubeCL rounds | Triton rounds | Median gap | Verdict |
|---|---|---|---|---|---|
| abs/relu | 3080 | 50.22, 50.26, 50.33 | 50.61, 50.63, 50.72 | −0.7% | faster |
| abs/relu | 3090 | 42.10, 42.15, 41.99 | 41.94, 41.93, 41.98 | +0.4% | slower |
| addcdiv, value 0.3 | 3080 | 96.98, 96.99, 96.99 | 96.87, 96.91, 96.94 | +0.1% | slower |
| addcdiv, value 0.3 | 3090 | 80.73, 80.23, 80.57 | 80.27, 80.11, 80.27 | +0.4% | ranges overlap |
| addcdiv, value 1 | 3080 | 96.90, 96.94, 97.05 | 96.76, 96.79, 96.73 | +0.2% | slower |
| addcdiv, value 1 | 3090 | 80.50, 80.42, 80.25 | 79.63, 80.13, 79.86 | +0.7% | slower |
| Adam update: addcdiv alone | 3080 | 96.97, 96.97, 96.99 | 96.90, 96.97, 96.97 | +0.0% | ranges overlap |
| Adam update: addcdiv alone | 3090 | 80.06, 80.34, 80.79 | 80.30, 80.17, 80.11 | +0.2% | ranges overlap |
| Adam update: lerp alone | 3080 | 75.24, 75.35, 75.42 | 74.45, 74.48, 74.48 | +1.2% | slower |
| Adam update: lerp alone | 3090 | 62.85, 63.05, 62.96 | 62.11, 61.25, 61.54 | +2.3% | slower |
| Adam update: addcmul alone | 3080 | 75.40, 75.29, 75.31 | 74.38, 74.45, 74.54 | +1.2% | slower |
| Adam update: addcmul alone | 3090 | 63.02, 63.03, 63.05 | 61.60, 61.45, 61.64 | +2.3% | slower |
| Adam step, whole | 3080 | 248.15, 248.33, 248.15 | 196.63, 196.71, 196.73 | +26.2% | slower |
| Adam step, whole | 3090 | 205.73, 205.79, 205.67 | 167.93, 162.50, 190.20 | +22.5% | slower |

None of the gaps comes from the new ops:

- **The division in `addcdiv`.** Both backends run one IEEE division routine per element for a
  value other than 1 (design.md, generated code), and the kernels are memory-bound: level or within
  0.1–0.4%. With value 1 ours still divides with `div_rn` while Triton uses `div.full` (design
  Risks: kept, eager-exact). That costs 0.2–0.7%, of which about 0.1 µs is the division itself.
- **Two-input kernels (`lerp`, `addcmul` updates).** A plain `a * 0.9 + b * 0.1`, with none of
  these ops, shows the same gap on the 3080: 75.3 against 74.3 µs, while a one-input kernel is
  level (50.4 against 50.5 µs). It is the scalar baseline plan against Triton's wider loads, the
  gap bead `4z8.28` already tracks for the integer kernels, where it is recorded.
- **Margins of about 0.1% or less.** `abs`/`relu` on the 3090 (+0.4% in medians, but 0.01 µs
  between CubeCL's fastest and Triton's slowest round) and `addcdiv` with value 0.3 on the 3080
  (0.04 µs between the ranges) are slower only by the strict range convention. Both are single
  memory-bound kernels moving the same bytes as Triton's, and are recorded on `4z8.28` as
  noise-level.
- **The whole Adam step.** Triton fuses the three updates into one kernel with three outputs;
  CubeCL emits three single-output kernels (74 + 97 + 75 µs on the 3080) and reads the shared
  inputs again in each. Each update alone is at the level above. Multi-output fusion is bead
  `4z8.21`, widened to pointwise kernels for this (decided with the user). Triton's whole step
  on the 3090 varied between rounds (162.5–190.2 µs); every CubeCL round (205.7–205.8 µs) is
  still above it.

## 4K elements

Every single-kernel case is launch-bound, at about 1.3–1.9 µs. CubeCL's medians are 0.07–0.30 µs
slower per kernel (the largest: the Adam addcdiv update on the 3090, 1.83 against 1.55 µs), within
the recorded short-kernel exception (bead `6cl`). The whole Adam step at 4K is three such kernels
against one (4.6 against 1.5 µs), which is `4z8.21` again.
