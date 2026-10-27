# Extrema-family kernel timing against Inductor+Triton (float-extrema-rounding-remainder 6.2)

The data is in `rounding-remainder-timing-3080.json` and `rounding-remainder-timing-3090.json`,
produced by `python -m cubecl_torch.testing.rounding_remainder_timing run --rounds 3 --gpu <gpu>`
(2026-10-27). Each value is the generated kernels' summed device time per call, in µs. The tables
list the per-round values, then the median gap. As in `float-arithmetic-timing.md`, a case is
slower when every CubeCL round exceeds every Triton round, faster in the reverse case, and
otherwise the ranges overlap.

- "Small quotients" means |a/b| up to about 100 (dividends `randn * 10`, divisors in [0.1, 3.1]).
- "Large" means up to about 1e10 (dividends `randn * 1e9`).
- Float floor division is measured at the small scale only. Its lowering, `floor(div_rn(a, b))`
  with selects, does the same work for any quotient. `fmod`'s reduction does not, which is why
  `fmod` and `remainder` are measured at both scales.

## 4M elements

| Case | GPU | CubeCL rounds | Triton rounds | Median gap | Verdict |
|---|---|---|---|---|---|
| fmod, small quotients | 3080 | 74.96, 74.81, 74.70 | 75.05, 75.08, 74.94 | -0.3% | ranges overlap |
| fmod, small quotients | 3090 | 63.16, 62.09, 61.87 | 62.38, 62.47, 62.31 | -0.5% | ranges overlap |
| fmod, large quotients | 3080 | 77.53, 76.35, 78.14 | 76.07, 76.12, 76.18 | +1.9% | slower |
| fmod, large quotients | 3090 | 64.91, 65.58, 65.58 | 63.22, 63.69, 63.49 | +3.3% | slower |
| remainder, small quotients | 3080 | 75.43, 76.07, 75.78 | 74.82, 74.87, 74.83 | +1.3% | slower |
| remainder, small quotients | 3090 | 63.70, 63.26, 63.32 | 61.86, 62.35, 61.80 | +2.4% | slower |
| remainder, large quotients | 3080 | 77.62, 77.91, 75.13 | 74.84, 74.83, 74.84 | +3.7% | slower |
| remainder, large quotients | 3090 | 66.68, 65.35, 66.55 | 61.94, 62.40, 61.71 | +7.4% | slower |
| float floor division | 3080 | 74.90, 74.95, 74.90 | 74.90, 74.84, 74.91 | +0.0% | ranges overlap |
| float floor division | 3090 | 62.14, 62.23, 62.48 | 62.08, 62.22, 61.85 | +0.2% | ranges overlap |
| floor, float32 | 3080 | 51.00, 51.13, 51.02 | 50.62, 50.60, 50.59 | +0.8% | slower |
| floor, float32 | 3090 | 41.90, 41.79, 42.11 | 41.83, 42.24, 41.81 | +0.2% | ranges overlap |
| round, float32 | 3080 | 50.98, 50.86, 51.01 | 50.71, 50.69, 50.64 | +0.6% | slower |
| round, float32 | 3090 | 42.29, 42.09, 41.92 | 42.47, 42.78, 42.63 | -1.3% | faster |
| floor, bfloat16 | 3080 | 26.48, 26.54, 26.55 | 26.09, 26.22, 26.13 | +1.6% | slower |
| floor, bfloat16 | 3090 | 22.08, 22.16, 22.19 | 21.70, 21.99, 21.95 | +1.0% | slower |
| round, bfloat16 | 3080 | 26.43, 26.50, 26.56 | 26.23, 26.28, 26.22 | +1.0% | slower |
| round, bfloat16 | 3090 | 22.09, 22.17, 21.88 | 21.74, 21.77, 21.91 | +1.5% | ranges overlap |
| copysign | 3080 | 74.79, 74.70, 74.75 | 74.51, 74.50, 74.47 | +0.3% | slower |
| copysign | 3090 | 62.01, 61.93, 61.75 | 61.37, 61.59, 61.45 | +0.8% | slower |
| nextafter | 3080 | 74.84, 74.93, 74.83 | 76.05, 76.00, 75.97 | -1.5% | faster |
| nextafter | 3090 | 61.97, 61.84, 61.75 | 62.33, 63.02, 62.34 | -0.8% | faster |
| ldexp | 3080 | 97.00, 97.07, 97.06 | 96.65, 96.33, 96.67 | +0.4% | slower |
| ldexp | 3090 | 81.04, 80.34, 80.72 | 80.31, 80.25, 80.43 | +0.5% | ranges overlap |
| frexp (both outputs) | 3080 | 126.42, 126.43, 126.55 | 100.14, 100.15, 100.19 | +26.2% | slower |
| frexp (both outputs) | 3090 | 105.31, 104.82, 104.90 | 82.99, 83.93, 83.61 | +25.5% | slower |

What the gaps come from. Same-process controls on the RTX 3080 (5 interleaved rounds each, the
op against a plain kernel of the same shape: `-x` for one input, `x + y` for two) separate op
cost from plan, placement and launch effects:

- **No op cost:** `floor`, `round`, `copysign`, `ldexp`, `nextafter` and floor division.
  - `copysign`, `ldexp` and `nextafter` sit on their control in both backends: 74.91 against
    74.84 µs for `copysign` here, 74.57 against 74.39 µs in Triton.
  - `copysign` and `ldexp` are marked slower above by 0.3–0.8%, against a same-process control
    difference of 0.07–0.2 µs. Those verdicts are buffer placement between processes (see
    below), not the ops.
  - `floor` costs the same as `abs`/`relu` under every pinned plan (50.4–51.0 µs). Its 3080 gap is
    the plan first-use selection settled on (the scalar baseline; `abs`/`relu` gets the 4-wide
    vector plan). That is a selection-rule matter (`6cl`, `4z8.28`), and the 3090 is level.
  - In bfloat16, our `floor` and `round` are within 0.15 µs of our bfloat16 `-x` (26.39 and 26.47
    against 26.32 µs), as Triton's are of its own (26.19 and 26.22 against 26.17 µs). The
    remaining ~0.6% is the plain bfloat16 one-input kernel itself, not the ops. That gap is
    open as bead `b1h`.
  - Measured across processes, the same CubeCL kernel moves by up to about 1 µs at 4M (buffer
    placement: memory `timing-placement-and-clock`). Sub-1% verdicts above are inside that.
- **`frexp` (+26%): multi-output fusion, bead `4z8.21`.** CubeCL runs two kernels (mantissa
  50.2 µs, int64 exponent 74.9 µs, each at its control's speed). Triton fuses both outputs into
  one 100 µs kernel. This is the same cause as the Adam step in `float-arithmetic-timing.md`.
- **`fmod`/`remainder` with large quotients (+2–7%), and `remainder` on the 3090 with small
  quotients (+2.4%): bead `njp`.**
  - Our exact `fmodf` loop costs more than Triton's kernels: ours is 77.95 µs against its 74.84 µs
    control, Triton's `remainder` 74.81 against 74.39 µs.
  - Triton's `remainder` is wrong for these inputs (`1e10 % 3 = 256`). Its exact `fmod` (libdevice,
    which flushes subnormals) is still 2–3% faster than ours.
  - With small quotients our `fmod` is level with Triton on both GPUs.
  - A same-process control on the 3090 settles the small-quotient `remainder` row:
    - our `remainder` and `fmod` are equal (63.13 and 63.20 µs), so the sign fix-up costs
      nothing;
    - our plain `x + y` is already 0.56 µs behind Triton's (61.98 against 61.42 µs);
    - exact `fmodf` costs 1.15 µs over that control, as Triton's exact `fmod` does (1.05 µs);
    - Triton's inexact `frem` remainder costs 0.5 µs.
  - So about 1% is the price of exactness, and closing it needs a fast exact path. The results
    are exact, and the speed gap stays open until `njp` closes. The bead holds the numbers and
    candidates.

## 4096 elements

| Case | GPU | CubeCL rounds | Triton rounds | Median gap | Verdict |
|---|---|---|---|---|---|
| fmod, small quotients | 3080 | 1.61, 1.72, 1.68 | 1.64, 1.63, 1.66 | +2.4% | ranges overlap |
| fmod, small quotients | 3090 | 1.71, 1.69, 1.76 | 1.43, 1.56, 1.56 | +9.6% | slower |
| fmod, large quotients | 3080 | 1.75, 1.84, 1.87 | 1.72, 1.89, 1.87 | -1.6% | ranges overlap |
| fmod, large quotients | 3090 | 1.89, 1.96, 1.92 | 1.76, 1.78, 1.77 | +8.5% | slower |
| remainder, small quotients | 3080 | 1.69, 1.69, 1.66 | 1.48, 1.50, 1.52 | +12.7% | slower |
| remainder, small quotients | 3090 | 1.80, 1.69, 1.83 | 1.52, 1.56, 1.54 | +16.9% | slower |
| remainder, large quotients | 3080 | 1.85, 1.88, 1.88 | 1.51, 1.50, 1.31 | +25.3% | slower |
| remainder, large quotients | 3090 | 1.92, 1.92, 1.95 | 1.56, 1.55, 1.48 | +23.9% | slower |
| float floor division | 3080 | 1.52, 1.63, 1.58 | 1.43, 1.46, 1.44 | +9.7% | slower |
| float floor division | 3090 | 1.68, 1.69, 1.69 | 1.59, 1.62, 1.58 | +6.3% | slower |
| floor, float32 | 3080 | 1.48, 1.52, 1.57 | 1.39, 1.41, 1.36 | +9.4% | slower |
| floor, float32 | 3090 | 1.59, 1.58, 1.43 | 1.45, 1.45, 1.46 | +9.0% | ranges overlap |
| round, float32 | 3080 | 1.48, 1.58, 1.54 | 1.36, 1.35, 1.42 | +13.2% | slower |
| round, float32 | 3090 | 1.60, 1.61, 1.57 | 1.47, 1.49, 1.49 | +7.4% | slower |
| floor, bfloat16 | 3080 | 1.37, 1.54, 1.49 | 1.41, 1.39, 1.37 | +7.2% | ranges overlap |
| floor, bfloat16 | 3090 | 1.63, 1.65, 1.67 | 1.49, 1.49, 1.48 | +10.7% | slower |
| round, bfloat16 | 3080 | 1.49, 1.50, 1.51 | 1.33, 1.42, 1.40 | +7.1% | slower |
| round, bfloat16 | 3090 | 1.59, 1.63, 1.65 | 1.48, 1.47, 1.48 | +10.1% | slower |
| copysign | 3080 | 1.49, 1.53, 1.54 | 1.43, 1.41, 1.42 | +7.7% | slower |
| copysign | 3090 | 1.57, 1.64, 1.66 | 1.46, 1.47, 1.48 | +11.6% | slower |
| nextafter | 3080 | 1.53, 1.61, 1.52 | 1.60, 1.61, 1.59 | -4.4% | ranges overlap |
| nextafter | 3090 | 1.58, 1.71, 1.64 | 1.71, 1.67, 1.62 | -1.8% | ranges overlap |
| ldexp | 3080 | 1.62, 1.62, 1.59 | 1.56, 1.61, 1.57 | +3.2% | ranges overlap |
| ldexp | 3090 | 1.66, 1.59, 1.64 | 1.70, 1.68, 1.63 | -2.4% | ranges overlap |
| frexp (both outputs) | 3080 | 2.79, 3.03, 3.02 | 1.66, 1.67, 1.65 | +81.9% | slower |
| frexp (both outputs) | 3090 | 3.15, 3.19, 3.07 | 1.62, 1.59, 1.57 | +98.1% | slower |

Every single-kernel case is launch-bound, at about 1.3–2 µs. `frexp` is two launches, about
3 µs, against Triton's one at 1.6 µs (`4z8.21`).
- The other CubeCL medians are 0.05–0.2 µs slower per kernel. That covers floor division,
  `floor`, `round`, `copysign` and `fmod`. It is within the recorded short-kernel exception (bead
  `6cl`), which applies only to sub-0.3 µs launch-level differences like these.
- `remainder` is 0.2–0.4 µs slower. That is not a launch effect but the same `fmodf`-plus-fix-up
  cost as at 4M, open as bead `njp`.
