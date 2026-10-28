# reduction-kernels Specification

## Purpose
Defines which Inductor reductions the CubeCL backend compiles and the guarantees each one carries:
which reduction types and dtypes are admitted, how they behave under direct, fused and split
plans, and how adding a reduction type leaves existing kernels unchanged.

## Requirements

### Requirement: Admitted reduction types

The backend SHALL compile a reduction only when its reduction type, input dtype and result dtype
are in the admitted set: `sum`, `max` and `min` over float32, float16 or bfloat16 inputs, and
`welford_reduce` and `welford_combine` with float32 inputs and results, one reduction dimension.
It SHALL reject any other reduction before fusion with an unsupported-capability error that
names the reduction type and dtype.

#### Scenario: Admitted max reduction

- **WHEN** `torch.compile` compiles `x.amax(-1)` for a float32, float16 or bfloat16 CUDA tensor
- **THEN** the reduction runs as a CubeCL kernel and no unsupported-capability error is raised

#### Scenario: Admitted Welford reduction

- **WHEN** `torch.compile` compiles `torch.var_mean(x, -1)`, `F.group_norm` or `F.layer_norm` for a float32, float16 or bfloat16 CUDA tensor
- **THEN** the Welford reductions run as CubeCL kernels and no unsupported-capability error is raised

#### Scenario: Reduction type not yet admitted

- **WHEN** a graph needs a `prod`, `argmax`, `any`, `xor_sum` or `online_softmax_reduce` reduction
- **THEN** compilation fails with the unsupported-capability error, and its message names that reduction type

#### Scenario: Integer max

- **WHEN** a graph needs a `max` reduction over an int32 or int64 tensor
- **THEN** compilation fails with the unsupported-capability error, and its message names the dtype

### Requirement: Existing sum kernels unchanged

Adding reduction types SHALL NOT change the generated code of any sum kernel. For every case in
the Stage 5 corpus, the structural records (descriptor, physical plan, IR, CUDA and PTX digests)
SHALL equal the recorded manifest.

#### Scenario: Manifest after the change

- **WHEN** the manifest check runs over the corpus after max/min support is added
- **THEN** it reports PASS with no changed case, region or file

#### Scenario: Manifest after Welford support

- **WHEN** the manifest check runs over the corpus after Welford support is added
- **THEN** it reports PASS with no changed case, region or file, and the max/min contract tests still pass

### Requirement: Plan coverage for every admitted type

Each admitted reduction type SHALL be compilable under every plan Inductor can request of it:
an unfused reduction, a reduction fused with pointwise prologue and epilogue work in the same
kernel, and a split reduction whose partial results are combined by a second kernel. Static and dynamic shapes are both covered.

#### Scenario: Split max with a ragged extent

- **WHEN** `x.amax(-1)` is compiled for an 8×65539 float32 tensor, with static and with dynamic shapes
- **THEN** both compile as two-stage split reductions and the result equals eager's

#### Scenario: Fused max epilogue

- **WHEN** `(x * 2).amax(-1) + 1` is compiled
- **THEN** the multiplication, reduction and addition run without an unsupported-capability error and the result equals eager's

#### Scenario: Split Welford with a masked split

- **WHEN** `torch.var_mean(x, -1)` is compiled for an 8×300000 float32 tensor (Inductor splits it with a mask, so the first stage is `welford_combine` over constant m2 and weight inputs), with static and with dynamic shapes
- **THEN** both stages compile, the second stage combines the partials with `welford_combine`, and the result meets the mean-variance accuracy contract

#### Scenario: Split Welford without a mask

- **WHEN** GroupNorm statistics are compiled for a 1×320×64×64 input (Inductor splits each 40960-element group into equal parts, so the first stage is `welford_reduce`)
- **THEN** both stages compile and the result meets the mean-variance accuracy contract

### Requirement: Strategy eligibility per reduction type

A physical plan strategy SHALL be eligible for a reduction only if it supports that reduction
type. Strategies whose numerics are specific to sum (compensated summation and the qualified sum
plans) SHALL be ineligible for other types, and first-use tuning SHALL never select an ineligible
strategy.

#### Scenario: Max under first-use tuning

- **WHEN** a max reduction is compiled under the `first_use` policy
- **THEN** the recorded candidate set contains no compensated or sum-only qualified strategy, and the selected strategy is one of the eligible candidates

#### Scenario: Baseline policy

- **WHEN** a max reduction is compiled under the `baseline` policy
- **THEN** it compiles with the baseline reduction strategy

### Requirement: Malformed reduction descriptors rejected

The descriptor compiler SHALL reject a reduction descriptor whose reduction type is unknown,
whose accumulator dtype does not match its type, whose physical plan is ineligible for its
type, whose operand count does not match its type, or that selects a component its type does
not have, with an error naming the problem and without producing an artifact.

#### Scenario: Hand-authored descriptor with an ineligible plan

- **WHEN** a max-reduction descriptor is compiled with the compensated strategy
- **THEN** compilation fails with an error naming the strategy and reduction type, and nothing is written to the cache

#### Scenario: Welford operand and component mismatch

- **WHEN** a hand-authored descriptor gives `welford_combine` one operand, gives `welford_reduce` three, or selects component 3 of a Welford reduction or any component of a sum
- **THEN** compilation fails with an error naming the reduction type and the mismatch, and nothing is written to the cache

### Requirement: Performance floor for new reduction types

On representative shapes, a max, min or Welford reduction compiled by the backend SHALL NOT be
measurably slower than the same graph compiled by Inductor with Triton, apart from recorded
short-kernel exceptions. "Measurably" means outside the run-to-run spread of the two
measurements under the project's GPU timing method. Welford reductions are judged on the
multi-output kernels that compute each statistic once (`multi-output-kernels`), at the plan
selection that meets the accuracy floor.

#### Scenario: Row max timing

- **WHEN** `x.amax(-1)` and `x.amin(-1)` are timed for float32 and bfloat16 inputs at short, medium and long rows (including a split shape) under both backends
- **THEN** each CubeCL time is within the run-to-run spread of the Triton time or faster, or the case is recorded as a short-kernel exception, and the measurements are recorded

#### Scenario: Short-kernel exception

- **WHEN** a case's median is slower than Triton's beyond both backends' relative spread, by at most 1.5 us, and the per-kernel breakdown places the gap in kernels whose Triton time is at most 2.5 us (a split's final stage, a short row)
- **THEN** the case, its per-kernel breakdown and its cause (first-use tuning kept the baseline plan, or the selected plan is marginally slower) are recorded on the bead that changes the selection rule for all reduction types (`6cl`); any larger gap is a failure

#### Scenario: Welford single-kernel timing recorded

- **WHEN** `torch.var_mean(x, -1)` is timed for float32 and bfloat16 inputs at short, medium and split shapes under both backends, comparing one CubeCL Welford kernel against Triton's kernel for the same statistic
- **THEN** the times are recorded with the multi-output bead, and any per-kernel gap beyond the short-kernel exception is investigated with SASS and recorded there, without blocking this change

#### Scenario: Welford timing

- **WHEN** `torch.var_mean(x, -1)` is timed for float32 and bfloat16 inputs at short, medium and split shapes, and LayerNorm and GroupNorm forward at the shapes of `multi-output-kernels`, under both backends
- **THEN** each CubeCL time is within the run-to-run spread of the Triton time or faster, or a short-kernel exception, or a gap fixed in the change or recorded on a bead with its per-kernel breakdown and SASS investigation if fixing it takes much more effort
