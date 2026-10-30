# Proposal

## Why

`cubecl_torch` builds the mapping of kernels onto threads by hand, plan by plan, and Triton's
mapping has been arriving piece by piece (16 warps, two-step variance, accumulation order). The
spike `sk8` (`docs/research/sk8`) showed that CubeCL's tile DSL supplies that mechanism: tile
kernels running our own emitter's loop bodies match or beat Triton on speed, with runtime extents
at no cost and compile time equal to ours. It also found gaps, including an out-of-bounds read, all
in code we can own once it is in our tree. Stage 1 takes the code and proves it on the one family
it can already express: row reductions that only sum.

## What Changes

- Copy `cubecl-tile` from the CubeCL pin into `cubecl_torch/native` as our own crate, recording the
  upstream revision and every local change.
- Offer tile kernels as new candidates beside today's plans for kernels with one row sum, carrying
  Inductor's fused per-element work (`x.sum`, `x.mean`, the sum consumer, RMSNorm-style norms).
  First-use selection and the band tests judge them. Today's plans stay, and stay the fallback.
- Change our copy where our kernels need it: bounds-safe and vector-valued recipes, and an explicit
  summation order setting that defaults to upstream's, with tile kernels asking for Triton's.
- Tile kernels use the existing launch contract. **Changes bead `f05.9`:** the launcher needs no new
  calling convention, because the grid-constant form the spike saw comes from CubeCL's own runtime,
  not from our compiler route (design decision 2).
- The first build changes the compiler binary's digest once, so every artifact key changes and all
  caches rebuild. Generated code for existing plans must not change.

Out of stage 1: Welford and argmax (bead `8im`), max/min combines across planes (bead `f05.8`,
nothing here needs them), mask regions, kernels with several reductions, split reductions, and
rewriting or removing any existing plan.

## Capabilities

### New Capabilities

- `tile-reductions`: row reductions compiled as tile kernels: which kernels may use them, how they
  are selected, and the bounds, accuracy, speed and launch contracts they meet.

### Modified Capabilities

- `descriptor-compiler`: the audited-source requirement extends to the copied crate, whose upstream
  revision and local changes are recorded.

## Impact

- **Native:** new crate under `cubecl_torch/native`; `Cargo.toml` and `Cargo.lock`; new kernel
  builder in `native/src/reduction`; `native/src/plan.rs`.
- **Python:** `compiler/plan.py` (catalog), `inductor/legality.py` (structural rules, numerical
  policy, signature facts), `inductor/tuning.py` if selection needs it.
- **Unchanged:** `launcher.c`, the metadata library and the runtime's launch contract.
- **Tests and records:** new tile tests, band and timing results under `testing/results`, and a
  re-recorded manifest, since candidate lists change and tile kernels may win selection.
- **Dependencies:** the crate's own (`hashbrown`, `half`, `serde`); no CUDA runtime in the compiler.
- **Later merges:** the copy diverges from a fast-moving upstream; we merge on purpose, not by pin.
