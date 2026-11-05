# Spec Delta

## Purpose

Defines how a reduction kernel Inductor schedules, with its fused per-element body, is lowered
through the CubeCL ecosystem's reduce instead of a native plan. It covers which kernels the
ecosystem path is offered for, what it must compute, and the Triton floor it is held to.

## ADDED Requirements

### Requirement: Fused body offered to the ecosystem reduce

For a sum reduction kernel whose body is loads and per-element ops feeding one reduction, followed
by per-output ops and stores, the backend SHALL offer a candidate that runs that body inside the
ecosystem reduce. It runs as one kernel launch and moves no intermediate tensor through memory.

#### Scenario: Split weight gradient

- **WHEN** RMSNorm's backward is compiled for an 8192×1280 float32 or bfloat16 input and Inductor emits the weight gradient's split first stage
- **THEN** the candidate set for that kernel includes the fused ecosystem reduce

#### Scenario: Unsplit column reduction

- **WHEN** RMSNorm's backward is compiled for a 64×1024 input and Inductor emits the weight gradient as one column reduction of `g·x·rstd`
- **THEN** the candidate set for that kernel includes the fused ecosystem reduce

### Requirement: Same result as the recorded body

The fused ecosystem reduce SHALL compute every output from the recorded body with the same
loads, per-element ops, casts and dtypes, under the backend's numerics policy. It SHALL also
reproduce the recorded store indices, including indices built from integer division and
remainder. Numerical acceptance comes from Triton's band against a high-precision reference,
never from agreement with a native plan.

#### Scenario: Non-affine store and load indices

- **WHEN** the split weight gradient runs through the fused ecosystem reduce on inputs whose sums are exact in every summation order (small integer values)
- **THEN** every output element equals the float64 reference exactly, and no element outside the outputs is written

#### Scenario: Numerical acceptance

- **WHEN** the fused ecosystem reduce runs the reference graphs on the band's random inputs
- **THEN** its verdict is taken from the band test against Triton, and a native plan's output is not used as the reference

#### Scenario: Low-precision loads

- **WHEN** a bfloat16 body casts its loads to float32 before the reduction
- **THEN** the reduction accumulates in the recorded accumulator dtype and each output is cast as recorded

### Requirement: Real layout given to the ecosystem

The backend SHALL describe the reduced operand to the ecosystem reduce with the shape and strides
of its memory as Inductor indexes it, so the reduce's own mapping and vectorization choices see
the actual memory order. The backend SHALL NOT impose a blueprint of its own on the new kind.

#### Scenario: Strided reduction axis

- **WHEN** the reduction axis of the operand is not the contiguous axis in memory, as in the weight gradient
- **THEN** the layout the ecosystem receives has that axis strided and the contiguous axis among the outputs

#### Scenario: Loads that disagree on layout

- **WHEN** two loads feeding the reduction operand span every output and reduction dimension but with different strides
- **THEN** no fused ecosystem candidate is offered and the rejection names the disagreeing layouts; the backend does not choose one of them by a rule of its own

### Requirement: Coverage refused with a reason

A kernel the fused ecosystem reduce cannot run exactly SHALL get no such candidate, and the
rejection SHALL name the part of the body or index that is not covered. Examples are per-element
stores in the reduction scope, several reductions, non-sum reductions and indices that cannot be
decoded.

#### Scenario: Per-element pass over the row

- **WHEN** RMSNorm's forward kernel (a per-row `rstd` store and a per-element normalized store) is compiled
- **THEN** no fused ecosystem candidate is offered, the rejection names the reduction-scope store, and the selected plan is one the kernel could select before this change

### Requirement: Triton floor for fused ecosystem reductions

On the first-proof cases, the fused ecosystem reduce SHALL be measured against Inductor+Triton:
the band; ncu kernel time on the RTX 3080, confirmed on the RTX 3090; and first-call cost. It is a
diagnostic gate. A gap is traced to its cause and fixed in the ecosystem sources or recorded on a
bead, never closed by forcing a blueprint or an admission rule in the backend.

#### Scenario: Accuracy band

- **WHEN** the band test pins the fused ecosystem reduce on the weight-gradient cases
- **THEN** it is inside Triton's band on both GPUs, or the failing case is recorded with its cause

#### Scenario: Kernel time

- **WHEN** the fused ecosystem reduce and the Triton kernel for the same weight-gradient case are timed under ncu at fixed clocks
- **THEN** the fused ecosystem reduce is no slower than Triton beyond noise, or the gap and its cause are recorded against a bead
