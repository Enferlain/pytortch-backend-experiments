# Integer kernel timing against Inductor+Triton (integer-value-arithmetic 5.2)

The data is in `integer-timing-3080.json` and `integer-timing-3090.json`, produced by
`python -m cubecl_torch.testing.integer_timing run --rounds 3 --gpu <gpu>` (2026-10-25). Each
value is the generated kernels' summed device time per call, in µs: per-round values, then the
median. The tool's `slower` flag uses a spread margin that one outlier round can inflate. The
verdicts below compare the backends' per-round ranges instead: a case is slower when every CubeCL
round exceeds every Triton round.

| Case | GPU | CubeCL rounds | Triton rounds | Median gap | Verdict |
|---|---|---|---|---|---|
| pad (float32), 4M | 3080 | 51.86, 51.25, 50.85 | 50.87, 50.74, 50.96 | +0.7% | ranges overlap |
| pad (float32), 4M | 3090 | 43.91, 42.42, 42.70 | 42.06, 42.21, 42.20 | +1.2% | slower |
| int64 floordiv, 4M | 3080 | 148.98, 148.98, 149.14 | 147.93, 147.85, 148.06 | +0.7% | slower |
| int64 floordiv, 4M | 3090 | 123.71, 122.91, 123.50 | 122.74, 141.44, 122.73 | +0.6% | slower apart from one Triton outlier |
| shift/xor mix, 4M | 3080 | 148.67, 156.45, 149.64 | 147.35, 147.38, 147.01 | +1.6% | slower |
| shift/xor mix, 4M | 3090 | 122.79, 127.40, 123.36 | 121.67, 122.30, 121.93 | +1.2% | slower |
| int8 wrap, 4M | 3080 | 50.12, 50.29, 50.45 | 50.63, 50.75, 50.60 | −0.7% | faster |
| int8 wrap, 4M | 3090 | 42.17, 41.65, 42.28 | 42.13, 42.94, 42.82 | −1.5% | level or faster |
| all four cases, 4K | both | 1.32–2.06 | 1.17–2.02 | ≤ +0.25 µs | short-kernel exception (`6cl`) |

## The 4M gaps (bead `4z8.28`)

The three slower cases are memory-bound kernels where CubeCL runs the frozen scalar baseline
plan: one 4- or 8-byte access per thread. Triton's kernels use 16-byte accesses.

- **int64 floordiv.** The SASS is not the cause: there is one 64-bit division routine per element,
  as in Triton (design.md, SASS check). Pointwise vector plans now admit int64 storage at two
  lanes. A fresh first-use run on the 3080 measured these candidate medians (µs):
  - `vector_pointwise` 128×2: 147.79
  - `scalar_pointwise` 64: 147.98
  - `vector_pointwise` 256×2: 148.14
  - `vector_pointwise` 64×2: 148.40
  - `scalar_pointwise` 128: 148.94
  - `scalar_pointwise_baseline`: 148.94
  - `scalar_pointwise` 128×4 elements: 150.01

  Their MADs were 0.38–0.65 µs. The selection rule of the time kept the baseline unless the best
  candidate's median + 3 MAD lay below the baseline's median − 3 MAD, so the 128×2 plan, level with
  Triton, was not chosen (bead `4z8.28` replaced that rule). Those figures come from that run's `selection.json`, which was a scratch cache
  and is not kept.
- **shift/xor mix** (int64 in and out): the same plan family applies. It was not separately probed.
- **pad mask** (float32): its masked load makes vector plans ineligible ("vector lazy/effect region
  unsupported"), so only scalar plans compete. This is a float kernel, so the gap is not specific
  to integers.

Changing the selection rule, or admitting vector plans around masked loads, affects every
kernel's plan choice. That is outside this change, and `4z8.28` tracks it with these numbers.

## Short kernels

At 4096 elements every case is launch-bound, at about 1.2–2 µs, and CubeCL is up to 0.25 µs slower
(the 3090's floordiv: 2.05 vs 1.81 µs). The float pad mask behaves the same way. This is within
the recorded short-kernel exception: a gap of at most 1.2 µs in kernels of at most about 2.5 µs at
Triton's time (bead `6cl`).

## Later

The 4M gaps above that come from the selection rule were closed by the paired selection rule
(bead `4z8.28`, 2026-10-27); see `selection-rule/README.md` for the new figures.
