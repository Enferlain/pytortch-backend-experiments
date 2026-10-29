## ADDED Requirements

### Requirement: Wide workgroups for reductions

Plan candidates for every admitted reduction type SHALL include workgroups of 512 units, as
Triton runs 16 warps on rows of 2048 elements and more. First-use selection SHALL choose among
them like any other candidate. A 512-unit plan SHALL meet the same accuracy contracts as the
other plans for its reduction type, and the runtime SHALL launch it.

#### Scenario: Long-row candidates

- **WHEN** softmax, `x * x.sum(-1, keepdim=True)` or LayerNorm is compiled for a 2048×8192 input under the `first_use` policy
- **THEN** the recorded candidate set includes a 512-unit plan, or records the rule that rejected it

#### Scenario: Welford at 512 units

- **WHEN** a Welford kernel (LayerNorm statistics with the normalized output) is compiled and launched with a 512-unit plan pinned
- **THEN** it launches, and its outputs meet the mean-variance accuracy contract

#### Scenario: Every reduction type at 512 units

- **WHEN** sum, max, min and Welford reductions are each compiled and run with a 512-unit plan pinned, at row lengths that leave lanes empty and ragged vector tails
- **THEN** each result matches the reference as the other plans for that type do

### Requirement: Long-row sums in Triton's accuracy band

For rows longer than 4096 elements, every plan first-use selection may choose for a kernel that
reduces only sums SHALL be no less accurate than Triton for the same case. This is judged by the
paired band test of `mean-variance-reductions`: the per-row sign test and mean-difference test,
both at p < 0.01, and non-finite rows, with no margin. Plans outside the band SHALL NOT be
candidates. Their rejection reason is recorded. Kernels that also reduce max or min keep their
current plan sets.

#### Scenario: Float32 long-row sums

- **WHEN** `x.sum(-1)` and `x * x.sum(-1, keepdim=True)` are compiled for float32 inputs at 2048×8192 and at least one longer unsplit row, with zero-mean and offset data
- **THEN** no output is less accurate than Triton's for the same inputs, under every plan the policy admits, and the measurements are recorded

#### Scenario: Policy recorded

- **WHEN** a long-row sum kernel is compiled under the `first_use` policy
- **THEN** every candidate is a plan measured in the band, and every other reduction plan appears among the rejected candidates with the long-row rule as its reason

#### Scenario: 16-bit long-row sums

- **WHEN** a sum-only kernel whose loads and stores are all 16-bit (a row sum, the sum consumer, a two-step variance) is compiled for a row longer than 4096 elements
- **THEN** its candidates are the plans measured in Triton's band on such rows, uncompensated included, and a kernel with a float32 load or a float32 output keeps the float32 rule

### Requirement: Performance floor for kernels that read the row again

On medium and long rows, a reduction kernel that reads its row after the reduction (softmax, a
sum consumer, LayerNorm) SHALL NOT be measurably slower than the same graph compiled by
Inductor with Triton. The only exceptions are recorded short-kernel exceptions and gaps fixed in
the change, or recorded on a bead with their per-kernel breakdown. "Measurably" means outside
the run-to-run spread of the two measurements under the project's GPU timing method.

#### Scenario: Row timing

- **WHEN** softmax, `x * x.sum(-1, keepdim=True)` and LayerNorm with and without affine parameters are timed for float32 and bfloat16 inputs at 300×4096 and 2048×8192 under both backends, on the RTX 3080 and the RTX 3090
- **THEN** each CubeCL time is within the run-to-run spread of the Triton time or faster, or the gap is recorded on a bead with its per-kernel breakdown, and the measurements are recorded

### Requirement: Variance lowered as Inductor lowers it for Triton

Where Inductor with Triton lowers a variance to the two-step algorithm (a sum for the mean, then
a sum of squared deviations), the CubeCL backend SHALL lower it the same way, under the pinned
PyTorch's conditions. Elsewhere it SHALL keep Welford. The two-step kernels SHALL meet the
mean-variance accuracy contract against Triton's band.

#### Scenario: 16-bit rows Triton reduces in two steps

- **WHEN** `var_mean` is compiled for bfloat16 and float16 inputs at 300×4096 and 2048×8192
- **THEN** the compiled kernels contain no Welford reduction, and every output is in Triton's band under every plan the policy admits

#### Scenario: Rows that keep Welford

- **WHEN** `var_mean` is compiled for split 16-bit rows (4×32769, 8×300000), a single 16-bit row, and a float32 row of 4096, and LayerNorm for a bfloat16 row of 4096
- **THEN** each compiles to a Welford reduction, as under Inductor with Triton

#### Scenario: Short persistent 16-bit rows

- **WHEN** `var_mean` is compiled for a 16-bit row shorter than the two-step minimum that is reduced in one pass (300×1023)
- **THEN** it compiles to two sums, as Inductor's persistent-reduction fallback does under Triton
