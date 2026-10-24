# Spec Delta

## Purpose

States the numerical contract of mean/variance (Welford) reductions in compiled kernels: the
mean, m2 (sum of squared deviations) and weight (element count) that `var`, `std`, `var_mean`
and normalizations are built from. It is measured against an fp64 reference, with
Inductor+Triton as the floor.

## ADDED Requirements

### Requirement: Accuracy not worse than Triton

For finite inputs, the compiled mean and variance SHALL NOT be less accurate than
Inductor+Triton's on the same inputs. Accuracy is the error against an fp64 computation of the
same statistics, compared at matched settings (same split) row by row: each reduced row of every
input pairs CubeCL's absolute error with Triton's. A statistic is less accurate when CubeCL's
error is larger on significantly more rows than it is smaller (one-sided sign test over the rows
whose errors differ, p < 0.01), when its mean per-row error is significantly larger (paired,
p < 0.01), or when it is non-finite on a row where Triton's is finite. No tolerance margin
applies.

#### Scenario: Row statistics sweep

- **WHEN** `torch.var_mean(x, -1, correction=0)` is compiled for float32, float16 and bfloat16 inputs at short (37×129), medium (300×4096), long (8×300000) and split (2×1048579) shapes, with zero-mean and offset (mean 1e3, std 1) data
- **THEN** no case's mean or variance is less accurate than Triton's for the same case, and the measurements are recorded

#### Scenario: Normalization statistics

- **WHEN** GroupNorm (32 groups, SDXL shapes such as 1×320×64×64 and 1×640×32×32) and LayerNorm (4096×1280) are compiled in training mode
- **THEN** the saved mean and rstd and the normalized output are no less accurate than Triton's for the same inputs

### Requirement: Exact weight

The weight component SHALL equal the number of reduced elements exactly, for every plan,
including split plans and masked tails, up to 2^24 elements per reduction.

#### Scenario: Ragged split weight

- **WHEN** a split Welford reduction over a row of 1048579 elements is compiled and its partial weights are combined
- **THEN** the final weight is exactly 1048579.0

### Requirement: Special values

A Welford reduction SHALL propagate NaN from any element into mean and m2. An all-equal finite
row SHALL give m2 exactly 0. Combining a finite partial with an empty one (weight 0) SHALL leave
it unchanged. A row holding ±inf SHALL give a non-finite mean or variance, never a finite value.

#### Scenario: NaN element

- **WHEN** a row contains one NaN as its first, middle or last element
- **THEN** that row's mean and variance are NaN

#### Scenario: Constant row

- **WHEN** every element of a row has the same finite value v, including rows long enough to split
- **THEN** the mean is v and the variance is exactly 0.0

#### Scenario: Empty partials

- **WHEN** a split or cooperative plan combines partials where some have weight 0 (rows shorter than the number of cooperating units, ragged split tails)
- **THEN** no NaN appears in the result and the statistics equal those of the non-empty partials

#### Scenario: Infinite elements

- **WHEN** a row contains +inf or −inf, in an unsplit and in a split plan (as an overflowed float16 LayerNorm input can)
- **THEN** that row's mean or variance is non-finite (inf or NaN), as Triton's is, and never finite

### Requirement: One value per reduced row

Every unit that reads a Welford result in a kernel SHALL see the same mean, m2 and weight, so
that a fused epilogue normalizing each element of a row uses one value for the whole row.

#### Scenario: Fused normalization epilogue

- **WHEN** `(x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)`-style code compiles into one Welford kernel that stores normalized elements, under every eligible strategy
- **THEN** each row's normalized output equals the result computed from that row's single reduced mean and variance
