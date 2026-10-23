# Proposal

## Why

Inductor lowers `var`, `std`, `var_mean` and every normalization forward (GroupNorm, LayerNorm,
InstanceNorm, BatchNorm statistics) to the `welford_reduce` and `welford_combine` reduction
types. The backend rejects both, so every norm region is rejected. On the workload scoreboard,
all 18 SDXL regions and Anima's single graph stop at `unsupported reduction welford_reduce`
(bead `4z8.15`). That hides every gap behind it.

The reduction framework from `generic-reduction-combiners` (bead `4x9.6`) was shaped for this.
Accumulators and shared scratch are already per component, but validation admits only one
value. Welford is the first combiner with three values (mean, m2, weight) and the first with a
second-stage type of its own (`welford_combine`).

## What Changes

- **Welford combiners.** Two new reduction types in the compiler's combiner table:
  - `welford_reduce`: one input value, three-value accumulator.
  - `welford_combine`: three input values (mean, m2, weight). It is the second stage of a split
    reduction, and also the first stage when Inductor needs a mask on the split.

  The combine is the parallel Welford update Inductor and Triton use. It handles the empty
  partial (weight 0) and equal infinite means without producing NaN. Both types are eligible for
  every strategy except the sum-only plans (compensated, reference512, final9).
- **Three-value accumulators admitted.** Validation accepts arity 3. Every strategy carries the
  three components through private accumulation, shared-memory trees and plane steps. A plane
  step whose combine is not bitwise commutative ends with a broadcast from lane 0, so all units
  see the same result.
- **Tuple results in the descriptor.** A Welford reduction yields three values. The front end
  selects each one with a new `reduction_component` op. `welford_combine` takes three float32
  operands, and validation checks all of them.
- **Front-end admission.** `welford_reduce` and `welford_combine` with a float32 result are
  admitted in `reductions.py`, `semantic_checks.py` and the legality mirror. Inductor upcasts
  lower-precision inputs to float32 before the Welford reduction. Other result dtypes are
  rejected with an error naming the type and dtype.
- **Numerical contract.** Mean and variance are judged against an fp64 reference on the same
  inputs. They must stay within Inductor+Triton's error band at matched settings: the floor
  policy, not bitwise Triton equality. The weight (element count) is exact. The design records
  how to choose between one-pass Welford and Triton's two-pass formula for short rows.
- **Coverage inventory.** `welford_reduce` and `welford_combine` become implemented, with a
  domain note: float32 result, one reduction dimension, extent ≤ 1,048,576 per kernel, every
  non-sum-only strategy, split and dynamic.

Out of scope:
- **Kernels with more than one external output** (new bead). Norm kernels normally write mean,
  rstd and often the normalized output. The fusion rule "one disjoint external output required"
  still applies, so here each output is its own kernel, and each kernel repeats the reduction.
  Results are correct, but Triton's performance floor isn't expected to hold. A follow-up change
  lifts the single-output rule for reduction kernels and checks the floor for norms there.
- Welford in cooperative reductions (`4x9.16`) and online softmax (`4x9.10`).
- Norm backward. It is made of sums the backend already compiles.

## Capabilities

### New Capabilities
- `mean-variance-reductions`: The numerical contract of Welford reductions: mean, m2 and weight
  results against fp64 and against Inductor+Triton, special values (NaN, ±inf, all-equal rows,
  empty partials), split agreement, and agreement of every unit on the reduced values.

### Modified Capabilities
- `reduction-kernels`: The admitted set grows to `welford_reduce` and `welford_combine` with
  float32 results. Their split plan uses `welford_combine` as the second stage. Malformed tuple
  descriptors are rejected. The performance-floor requirement names Welford's deferred floor
  check.

## Impact

- **Rust compiler:** `cubecl_torch/native/src/descriptor3.rs`:
  - `Combiner::WelfordReduce` and `Combiner::WelfordCombine`, with arity 3, lift, combine, output
    and plane-with-broadcast.
  - Multi-operand reductions in validation and in the operand closure.
  - The `reduction_component` op.
  - Validation for arity 3.

  `descriptor5.rs` only if its shared plan checks assume one component.
- **Python front end:**
  - `inductor/reductions.py`: tuple results and `reduction_component`.
  - `semantic_checks.py`: admission.
  - `legality.py`: `REDUCTION_TYPES`, plus exactness and compensation rules for the new types.
  - `tuning.compare`: tolerance on mean and m2, exact on weight.
- **Tests:**
  - New `test_reduction_welford.py`.
  - A Triton reference data generator for var_mean, GroupNorm and LayerNorm statistics, beside
    `extrema_reference.py`.
  - The `welford_reduce` rejection scenario in existing tests becomes acceptance.
- **Inventory and beads:**
  - `coverage_inventory.py`/`.md`.
  - `4z8.15` is closed by this change.
  - New bead for multi-output reduction kernels, blocking `4z8.11` with `4z8.15`.
- **Native launcher:** `cubecl_torch/native/launcher.c` allows shared scratch above 1 KiB up
  to 12 bytes per unit (three float32 components). Before, it allowed only 2048 bytes for the
  256-unit compensated sum. A 256-unit Welford shared tree needs 3072 bytes.
- **No change** to sum, max or min kernels (manifest PASS), the rest of the runtime layer or the
  custom-op front end.
