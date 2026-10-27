# Spec Delta

## MODIFIED Requirements

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
