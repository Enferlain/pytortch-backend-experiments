# Spec Delta

## MODIFIED Requirements

### Requirement: Accuracy not worse than Triton

For finite inputs, the compiled mean and variance SHALL NOT be less accurate than
Inductor+Triton's on the same inputs. Accuracy is the error against an fp64 computation of the
same statistics, compared at matched settings (same split) row by row: each reduced row of every
input pairs CubeCL's absolute error with Triton's. A statistic is less accurate when CubeCL's
error is larger on significantly more rows than it is smaller (one-sided sign test over the rows
whose errors differ, p < 0.01), when its mean per-row error is significantly larger (paired,
p < 0.01), or when it is non-finite on a row where Triton's is finite. No tolerance margin
applies. This holds for every output of a multi-output kernel and for kernels that fuse a
Welford reduction with a sum.

#### Scenario: Row statistics sweep

- **WHEN** `torch.var_mean(x, -1, correction=0)` is compiled for float32, float16 and bfloat16 inputs at short (37×129), medium (300×4096), long (8×300000) and split (2×1048579) shapes, with zero-mean and offset (mean 1e3, std 1) data
- **THEN** no case's mean or variance is less accurate than Triton's for the same case, and the measurements are recorded

#### Scenario: Normalization statistics

- **WHEN** GroupNorm (32 groups, SDXL shapes such as 1×320×64×64 and 1×640×32×32) and LayerNorm (4096×1280) are compiled in training mode
- **THEN** the saved mean and rstd and the normalized output are no less accurate than Triton's for the same inputs

#### Scenario: Multi-output norm kernels

- **WHEN** the row statistics sweep and the normalization statistics are compiled with every statistic and the normalized output written by one kernel (`multi-output-kernels`)
- **THEN** no output is less accurate than Triton's for the same case, and the measurements are recorded

#### Scenario: Welford fused with a sum

- **WHEN** a graph whose kernel fuses a Welford reduction with a sum over the same rows (for example `(x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)` beside `x.sum(-1)`) is compiled at short, medium and long rows with zero-mean and offset data
- **THEN** no Welford-derived output is less accurate than Triton's for the same case, under the plan selection the backend makes for that kernel
