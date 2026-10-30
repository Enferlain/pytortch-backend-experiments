## ADDED Requirements

### Requirement: Tile candidates for row sums

The backend SHALL offer tile kernels as plan candidates for reduction kernels with exactly one
reduction, a sum, on rows of at least 2048 elements, with the kernel's fused per-element work
inside the tile kernel. Every other kernel SHALL see tile plans only among its rejected candidates,
each with a recorded reason. Tile candidates SHALL be added beside the existing plans: no existing
plan is removed, and first-use selection SHALL judge tile candidates like any other candidate.

#### Scenario: Row sums offered

- **WHEN** `x.sum(-1)`, `x.mean(-1)`, `x * x.sum(-1, keepdim=True)` or an RMSNorm with a weight is compiled for a 2048×8192 input under the `first_use` policy
- **THEN** the recorded candidate set includes a tile plan, or records the rule that rejected it

#### Scenario: Kernels outside stage 1

- **WHEN** a kernel with a max, min or Welford reduction, with several reductions, with a mask region, or forming part of a split reduction is compiled
- **THEN** no tile plan is a candidate, each appears among the rejected candidates with its reason, and the selected plan is one the kernel could select before this change

#### Scenario: Short rows

- **WHEN** a sum over rows shorter than 2048 elements is compiled under the `first_use` policy
- **THEN** tile plans are rejected with a recorded reason, while the plans offered before this change remain

#### Scenario: Pinned at any row length

- **WHEN** a tile plan is pinned for an admitted kernel at row lengths 31, 100, 4099 and 65539
- **THEN** the kernel runs and its outputs match the reference as the other plans for sum do

### Requirement: No read outside a tensor

A tile kernel SHALL read and write only elements inside its tensors' logical extents, at every row
length, including the last partial step of the last row. Before a tile kernel is admitted to the
catalog it SHALL be run with index validation at uneven shapes, with zero reports. Memory checkers
SHALL NOT be accepted as evidence, since pooled allocators hide these reads from them.

#### Scenario: Partial last step

- **WHEN** an admitted kernel runs under index validation at 2048×4099
- **THEN** no index past the tensor's logical length is reported

#### Scenario: Validation at uneven shapes

- **WHEN** each tile kernel family runs under index validation at row lengths 31, 100, 4099 and 65539
- **THEN** there are zero reports, and the validation detects a deliberately unguarded read in a negative control

#### Scenario: Zero-length rows

- **WHEN** an admitted kernel runs at a row length of 0 with at least one row
- **THEN** no element of any tensor is read for the reduction, and the outputs match the reference

### Requirement: Accuracy in Triton's band

Every tile plan first-use selection may choose SHALL be no less accurate than Inductor with Triton
for the same case, judged by the paired band test of `mean-variance-reductions`: the per-row sign
test and mean-difference test, both at p < 0.01, and non-finite rows, with no margin. A tile plan
outside the band for a class of kernels SHALL NOT be a candidate for that class, and the reason
SHALL be recorded.

#### Scenario: Float32 long rows

- **WHEN** `x.sum(-1)` and `x * x.sum(-1, keepdim=True)` run for float32 inputs at 2048×8192 and at least one longer row, with zero-mean and offset data, under a tile plan
- **THEN** either no output is less accurate than Triton's for the same inputs and the tile plan is a candidate, or the tile plan is rejected for that class with a recorded reason, and the measurements are recorded either way

#### Scenario: Result independent of the shape class

- **WHEN** the same admitted kernel runs under a tile plan at row lengths in different alignment classes
- **THEN** each class that is a candidate has its own band measurement recorded

### Requirement: Speed at or above Triton's

A tile kernel selected on a case SHALL NOT be measurably slower than the same graph compiled by
Inductor with Triton at matched settings. "Measurably" means outside the run-to-run spread of the
two measurements under the project's GPU timing method. A tile plan SHALL NOT raise the time of a
kernel it replaces in selection.

#### Scenario: Row timing

- **WHEN** the sum consumer and RMSNorm are timed with a tile plan pinned, for float32 inputs at 2048×8192 and 2048×4099, against Triton and the best existing plan, on the RTX 3080 and the RTX 3090
- **THEN** each tile time is within the run-to-run spread of the Triton time or faster, or the gap is recorded on a bead with its per-kernel breakdown and the plan is not a candidate for that case, and the measurements are recorded

### Requirement: Existing launch contract

A tile kernel SHALL launch through the same contract as other kernels: one device pointer per
buffer plus one metadata pointer, in PyTorch-owned storage on the current stream, under the same
launch validation. Its row and column extents SHALL be runtime values, so that one compiled kernel
serves every shape in an alignment class without a recompile.

#### Scenario: Launch without a new calling convention

- **WHEN** a tile artifact is built and loaded
- **THEN** it passes the same pointer-ABI check as every other artifact, and launches with no change to the launcher

#### Scenario: One kernel across shapes

- **WHEN** a tile plan runs at 2048×8192, then at 1024×8192 and 2048×4096
- **THEN** no further compilation occurs for the later shapes, and all outputs are correct

#### Scenario: Compilation without a device runtime

- **WHEN** the descriptor compiler builds a tile kernel
- **THEN** it builds in the compiler process without creating a CUDA context, and repeated compilations in fresh processes produce identical IR, CUDA and PTX bytes

### Requirement: Existing kernels unchanged

Adding tile plans SHALL NOT change the generated code of any existing plan. Differences in the
manifest SHALL be confined to candidate and rejected lists and to cases where a tile plan wins
selection, and each difference SHALL be reviewed before the manifest is re-recorded.

#### Scenario: Crate copied, no plan added

- **WHEN** the copied crate is compiled into the compiler and no tile plan exists yet
- **THEN** the manifest check reports no change to any case, region or file

#### Scenario: Plans added

- **WHEN** the manifest check runs after tile plans are added to the catalog
- **THEN** every reported change is a candidate list, a rejected list, or a case a tile plan won, and the IR, CUDA and PTX digests of every non-tile artifact are unchanged
