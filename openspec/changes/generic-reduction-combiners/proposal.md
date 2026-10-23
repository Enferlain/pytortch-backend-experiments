# Proposal

## Why

Every reduction the backend can compile is a floating-point sum. The sum is built into each layer:
the front end accepts only `reduction_type == "sum"` with float32 accumulation, and the Rust
emitter's strategies (shared tree, compensated, plane-shared, reference512, independent planes,
columns, multiple private accumulators) all start from `0.0`, combine with `Add` and reduce
across a warp with `plane::FSum`. Max/min, argmax/argmin, any/prod/xor_sum, online softmax and
Welford (beads `4x9.7`–`4x9.10`, `4z8.15`) are all blocked on this (bead `4x9.6`). Welford is
where SDXL and Anima stop first on the workload scoreboard.

A framework proven only on sum isn't proven, so this change brings in max/min reductions as its
first other client. They share their combine function with float `maximum`/`minimum`, which
the backend also rejects today. That rejection blocks 3 of the 4 optimizer regions on the
scoreboard (the clamp in sd-scripts' Newton-Schulz, bead `x3c`). The extrema contract (NaN, ±inf,
signed zero) is therefore defined once here, for both the reductions and the pointwise ops.

## What Changes

- **Reduction combiners.** A reduction type is described by a combiner: its identity value, an
  ordered two-operand combine, and a cross-lane (plane) combine. The emitter's strategies take
  the combiner as a parameter instead of hard-coding the sum. The model is CubeK's
  `ReduceInstruction` (reuse first).
  - Sum keeps `plane::FSum` and its current instruction sequence. Every existing sum kernel stays
    byte-identical (manifest check PASS, Stage 5 equivalence unchanged).
  - Combiners without a native plane op use a shuffle-XOR tree over their own combine.
  - Each strategy declares which combiners it supports. Compensated summation and the frozen,
    qualified sum plans (reference512 partial, final9 constant final) stay sum-only.
  - The accumulator is a tuple with a dtype per value. Strategies thread tuples, but only 1-tuples
    are admitted until the first multi-value client (Welford, `4z8.15`) arrives with its tests.
  - Split reductions combine their partials with the reduction's combiner. For max/min the
    second stage reuses the same type; tuple types bring their own combine type (for example
    `welford_combine`).
- **Max/min reductions.** `max` and `min` reduction types for float32/float16/bfloat16 inputs
  (`amax`, `amin`, `max`/`min` over all elements, `clamp` and the like inside fused kernels). This
  covers direct, fused and split plans, static and dynamic shapes.
- **Float extrema contract and pointwise ops.**
  - `maximum`/`minimum` on floats propagate NaN and order −0 below +0.
  - `isnan` is added, so Inductor's `fmax`/`fmin` decomposition (`isnan`, `where`, comparisons)
    compiles.
  - Max/min reductions follow the same contract. That makes their result independent of the
    reduction order, apart from NaN payload.
- **Coverage inventory.**
  - `max`, `min`, `maximum`, `minimum` and `isnan` become implemented, each with a domain note.
  - `TUPLE_REDUCTION` moves to the scan family: in the pinned Inductor it only gates multi-output
    scans.
  - `REDUCE_TO_SINGLE_ELEMENT` becomes n/a by default, as in Triton. Honouring
    `config.triton.cooperative_reductions` is tracked in `4x9.16`.
  - The `fmax` reduction type and the `fmaximum` op move to the online-softmax family: their only
    producer is Inductor's two-pass softmax fallback.
- **Bead split.** `x3c` keeps the remainder/fmod contract. Its extrema half is delivered here.

Out of scope, each becoming a client of the framework later:
- argmax/argmin and `max.dim`/`min.dim` values, which Inductor lowers through max-with-index (`4x9.8`)
- any, prod and xor_sum (`4x9.9`)
- online softmax and the `fmax` reduction type (`4x9.10`)
- Welford (`4z8.15`)
- integer and bool max/min, which need integer value arithmetic (`4x9.1`)
- `relu` (basic arithmetic family, `4x9.2`)

## Capabilities

### New Capabilities
- `reduction-kernels`: Which reductions the backend compiles and how. It covers reduction types as
  combiners, accumulator tuples, strategy support per combiner, split-partial combination, the
  descriptor fields that carry them, and the requirement that adding a combiner leaves existing
  kernels byte-identical.
- `float-extrema`: The numerical contract of float `maximum`/`minimum`, max/min reductions and
  `isnan`, measured against eager PyTorch and Inductor+Triton: NaN propagation, infinities and
  signed zero.

### Modified Capabilities
None. The existing `inductor-backend` requirements (structural equivalence with Stage 5,
observable rejection, manifest reproducibility) are unchanged and must keep holding.

## Impact

- **Python front end:**
  - `cubecl_torch/inductor/reductions.py`, `semantic_checks.py` and `reduction_checks.py`:
    reduction type and dtype admission, fusion limits phrased per reduction rather than per sum.
  - `semantics.py`: float `maximum`/`minimum`, `isnan`.
  - `legality.py`: strategy support per combiner.
  - `compiler/plan.py` and `compiler/final_plan.py`.
- **Rust compiler:**
  - `cubecl_torch/native/src/descriptor3.rs` (and where they share reduction code,
    `descriptor4.rs`/`descriptor5.rs`): a combiner abstraction threaded through every strategy,
    plus a generic plane-combine tree.
  - Descriptor validation accepts the new reduction types.
  - If no CubeCL op meets the contract at Triton's cost, a NaN-propagating max/min lowering may
    need an addition to the audited CubeCL patch stack (see design).
- **Tests:**
  - New contract tests: extrema edge cases, bitwise agreement with Triton on max/min results,
    split and dynamic plans.
  - `test_contracts_effects.py`: its `max` rejection case becomes an acceptance case.
  - The gate contract inventory entry for that assertion is updated.
- **Inventory and beads:**
  - `coverage_inventory.py`/`.md` updated as above.
  - `4x9.6` and `4x9.7` are closed by this change.
  - `x3c` is narrowed to remainder/fmod.
- **No change** to the corpus outcome, the runtime layer, the custom-op front end or the overlay.
