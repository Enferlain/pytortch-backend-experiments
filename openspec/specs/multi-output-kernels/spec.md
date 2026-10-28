# multi-output-kernels Specification

## Purpose

Lets one compiled kernel write several external outputs, so that regions Inductor fuses into one
kernel with several stores (optimizer steps, frexp, norm statistics with their normalized
output) compile into one kernel, as with Inductor+Triton, instead of one kernel per output.

## Requirements

### Requirement: Fused regions with several external outputs compile into one kernel

When Inductor's scheduler fuses nodes whose union writes several external outputs, the backend
SHALL compile them into one kernel that writes every output, for pointwise and reduction
kernels. It SHALL NOT split such a fusion into one kernel per output. Kernels that generate
random numbers or store atomically SHALL stay single-output.

#### Scenario: Adam-style step

- **WHEN** the functional Adam-style step of `float-arithmetic` (`exp_avg.lerp(grad, 1 - beta1)`, `exp_avg_sq.mul(beta2).addcmul(grad, grad, value=1 - beta2)`, `param.addcdiv(exp_avg, denom, value=-step_size)`, all three returned) is compiled over 4M float32 elements
- **THEN** one generated kernel writes all three outputs, as Inductor+Triton's does, and every output equals eager bit for bit

#### Scenario: Mantissa and exponent

- **WHEN** `m, e = torch.frexp(x); return m, e.to(torch.int64)` is compiled for float32 `x`
- **THEN** one generated kernel writes both outputs, and both equal eager

#### Scenario: Two reductions of one row

- **WHEN** `(x.sum(-1), x.amax(-1))` is compiled
- **THEN** one generated kernel writes both results

#### Scenario: Random-number kernels stay separate

- **WHEN** a region fuses a node that generates random numbers with a node writing another external output
- **THEN** the fusion is rejected with the existing effect-safety reason and each kernel writes one output

### Requirement: Norm forward matches Triton's kernel structure

`var_mean`, LayerNorm and GroupNorm forward SHALL compile with Inductor+Triton's kernel count,
and each kernel SHALL write the same outputs as Triton's corresponding kernel.

#### Scenario: LayerNorm statistics and output

- **WHEN** `native_layer_norm(x, (1280,), None, None, 1e-5)` is compiled for a 4096×1280 bfloat16 `x`
- **THEN** one kernel writes the per-row mean, the per-row rstd and the normalized output

#### Scenario: var_mean, unsplit and split

- **WHEN** `torch.var_mean(x, -1, correction=0)` is compiled for 300×4096 and 8×300000 float32 `x`
- **THEN** the 300×4096 case runs one kernel writing mean and variance, and the 8×300000 case runs a first stage writing the partial mean, m2 and weight and a second stage writing mean and variance

#### Scenario: GroupNorm

- **WHEN** `native_group_norm` is compiled for a 1×320×64×64 float32 input with 32 groups
- **THEN** it runs Triton's three kernels: a first stage writing three partial statistics, a second stage writing the group statistics, and the normalized output

### Requirement: Each reduction is computed once per kernel

A kernel whose fused nodes reduce the same value with the same reduction type SHALL compute
that reduction once and use its result for every output that depends on it. This SHALL hold
also when the reduced value is computed under a mask, as in a split reduction's first stage.

#### Scenario: Welford statistics of one row

- **WHEN** the LayerNorm kernel above is compiled
- **THEN** its descriptor contains one Welford reduction, which the mean, rstd and normalized-output stores all depend on

#### Scenario: Masked split first stage

- **WHEN** the 8×300000 `var_mean` first stage is compiled
- **THEN** its descriptor contains one `welford_combine` reduction, not one per output

### Requirement: Per-row and per-element outputs in one reduction kernel

A reduction kernel SHALL be able to store values that are constant along the reduction
(per row) and values that vary along it (per element) in the same kernel. Per-element stores
SHALL see the single per-row reduced values of their row.

#### Scenario: Normalized output beside its statistics

- **WHEN** the LayerNorm kernel is compiled under every eligible physical plan
- **THEN** each row's normalized output equals the value computed from the same row's stored mean and rstd

### Requirement: Buffer limit

A kernel SHALL be able to read and write up to 16 buffers in total, counting inputs and
outputs. Fusion that would exceed the limit SHALL be rejected with a reason naming the limit,
and the nodes SHALL compile as separate kernels.

#### Scenario: Fused Adam step

- **WHEN** the Adam-style step above (five inputs, three outputs) is compiled
- **THEN** it compiles into one kernel with eight buffers

#### Scenario: GroupNorm backward

- **WHEN** `F.group_norm` with weight and bias, followed by SiLU, is compiled forward and backward for a 2×320×32×32 bfloat16 input
- **THEN** no region is rejected for its input or buffer count

#### Scenario: Over the limit

- **WHEN** fusing two nodes would give a kernel 17 buffers
- **THEN** the fusion is rejected with a reason naming the 16-buffer limit, and the region still compiles

### Requirement: Descriptor buffer contract

A descriptor SHALL declare each buffer as read-only or write-only. The compiler SHALL reject,
with an error naming the problem and without producing an artifact, a descriptor that:

- loads from a written buffer or stores to a read buffer;
- never stores to a declared output;
- in a reduction kernel, stores to one output more than once;
- declares more than 16 buffers.

#### Scenario: Store into an input

- **WHEN** a hand-authored descriptor stores to a buffer declared read-only
- **THEN** compilation fails with an error naming the buffer and its access, and nothing is written to the cache

#### Scenario: Output never written

- **WHEN** a hand-authored descriptor declares two outputs and stores only to one
- **THEN** compilation fails with an error naming the unwritten output

#### Scenario: Two stores to one output

- **WHEN** a hand-authored reduction descriptor stores twice to the same output buffer
- **THEN** compilation fails with an error naming the buffer

#### Scenario: Fill, then a predicated store

- **WHEN** a hand-authored pointwise descriptor stores a fill value to its output, then stores to it again inside a masked region
- **THEN** it compiles, and each element holds the predicated value where the mask holds and the fill elsewhere

### Requirement: Selection compares every output

First-use plan selection SHALL check every output of each candidate plan against the
baseline plan's, each by the comparison its own value calls for, as it does today for a
single-output kernel of that kind. The user's outputs SHALL NOT be written until the final
launch.

#### Scenario: Comparison per output

- **WHEN** a candidate's outputs are checked
- **THEN** an output that depends only on exact reductions (max, min, the Welford weight) is compared bit for bit, one depending on a Welford mean or m2 under the Welford corruption guard, any other reduction output under the sum guard, and a pointwise output under the pointwise tolerance

#### Scenario: Weight beside mean and m2

- **WHEN** the split `var_mean` first stage is selected under first-use tuning
- **THEN** its weight output is compared bit for bit and its mean and m2 outputs under the Welford guard

#### Scenario: Corrupted second output

- **WHEN** a candidate plan writes a correct first output and a wrong second output in a test
- **THEN** the candidate is excluded, and the exclusion names the output that failed

### Requirement: Single-output kernels unchanged

Kernels that write one external output SHALL keep their generated code. The manifest check
SHALL report no change for any corpus region whose kernels all write one output. Every
changed region SHALL be one whose graph fuses several outputs, and SHALL be reviewed and
explained before the manifest is re-recorded.

#### Scenario: Manifest after the change

- **WHEN** the manifest check runs after this change
- **THEN** every reported case, region and file belongs to a graph with a multi-output fusion, and each is explained before re-recording

### Requirement: Multi-output kernels are not slower than Triton

The multi-output workloads SHALL NOT be measurably slower than Inductor+Triton under the
repository's timing method, apart from the recorded short-kernel exception. Each is timed on
the RTX 3080 and confirmed on the RTX 3090.

#### Scenario: Pointwise workloads

- **WHEN** the whole Adam-style step and `frexp` with both outputs, over 4M float32 elements, are timed under CubeCL and Inductor+Triton
- **THEN** each CubeCL time is within Triton's spread or faster, or the gap is fixed in this change, or recorded on a bead with its per-kernel breakdown if fixing it takes much more effort

#### Scenario: Norm workloads

- **WHEN** `var_mean` (37×129, 300×4096 and 8×300000 float32; 300×4096 bfloat16), LayerNorm (4096×1280 float32 and bfloat16) and GroupNorm (1×320×64×64 float32 and bfloat16) forward are timed
- **THEN** each CubeCL time is within Triton's spread or faster, or a short-kernel exception, or a gap recorded on a bead with its per-kernel breakdown
