# reduction-kernels Specification

## Purpose
Defines which Inductor reductions the CubeCL backend compiles and the guarantees each one carries:
which reduction types and dtypes are admitted, how they behave under direct, fused and split
plans, and how adding a reduction type leaves existing kernels unchanged.

## Requirements

### Requirement: Admitted reduction types

The backend SHALL compile a reduction only when its reduction type, input dtype and result dtype
are in the admitted set: `sum`, `max` and `min` over float32, float16 or bfloat16 inputs, one
reduction dimension. It SHALL reject any other reduction before fusion with an unsupported-capability error
that names the reduction type and dtype.

#### Scenario: Admitted max reduction

- **WHEN** `torch.compile` compiles `x.amax(-1)` for a float32, float16 or bfloat16 CUDA tensor
- **THEN** the reduction runs as a CubeCL kernel and no unsupported-capability error is raised

#### Scenario: Reduction type not yet admitted

- **WHEN** a graph needs a `prod`, `argmax`, `any`, `xor_sum`, `welford_reduce` or `online_softmax_reduce` reduction
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
whose accumulator dtype does not match its type, or whose physical plan is ineligible for its
type, with an error naming the problem and without producing an artifact.

#### Scenario: Hand-authored descriptor with an ineligible plan

- **WHEN** a max-reduction descriptor is compiled with the compensated strategy
- **THEN** compilation fails with an error naming the strategy and reduction type, and nothing is written to the cache

### Requirement: Performance floor for new reduction types

On representative shapes, a max or min reduction compiled by the backend SHALL NOT be measurably
slower than the same graph compiled by Inductor with Triton, apart from recorded short-kernel
exceptions. "Measurably" means outside the run-to-run spread of the two measurements under the
project's GPU timing method.

#### Scenario: Row max timing

- **WHEN** `x.amax(-1)` and `x.amin(-1)` are timed for float32 and bfloat16 inputs at short, medium and long rows (including a split shape) under both backends
- **THEN** each CubeCL time is within the run-to-run spread of the Triton time or faster, or the case is recorded as a short-kernel exception, and the measurements are recorded

#### Scenario: Short-kernel exception

- **WHEN** a case's median is slower than Triton's beyond both backends' relative spread, by at most 1.5 us, and the per-kernel breakdown places the gap in kernels whose Triton time is at most 2.5 us (a split's final stage, a short row)
- **THEN** the case, its per-kernel breakdown and its cause (first-use tuning kept the baseline plan, or the selected plan is marginally slower) are recorded on the bead that changes the selection rule for all reduction types (`6cl`); any larger gap is a failure
