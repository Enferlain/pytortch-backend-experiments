# Spec Delta

## ADDED Requirements

### Requirement: Low-precision intermediates in fused kernels follow Triton

Inside a fused kernel, a float16 or bfloat16 intermediate SHALL stay in float32 unless the user
wrote the cast that produces it (`.to`, `.half`, `.type`, autocast) or Inductor's
`emulate_precision_casts` is on. This covers a cast a decomposition inserts and a low-precision
reduction result read by a fused consumer. A stored value SHALL round once, at its store, as in
Inductor+Triton.

#### Scenario: Fused normalization in bfloat16

- **WHEN** `(x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)` is compiled for bfloat16 `x`
- **THEN** the standard deviation and the mean stay float32 inside the kernel, and the normalized output is no less accurate against fp64 than Triton's on the same inputs

#### Scenario: A cast the user wrote

- **WHEN** `(x.float() * 2).to(torch.bfloat16) + 1` is compiled
- **THEN** the value is rounded to bfloat16 before the addition, as eager rounds it

## MODIFIED Requirements

### Requirement: Basic float kernels are not slower than Triton

Representative kernels using these ops SHALL NOT be measurably slower than Inductor with Triton on
the same GPU, judged per kernel at matched settings, under the repository's timing method and its
recorded short-kernel exception.

#### Scenario: Optimizer and pointwise kernels

- **WHEN** each Adam-style update (`lerp`, `addcmul`, `addcdiv`) alone, an `addcdiv` kernel and an `abs`/`relu` pointwise kernel over 4M float32 elements are timed under CubeCL and Inductor+Triton on the RTX 3080, and confirmed on the RTX 3090, with clock warm-up
- **THEN** each CubeCL time is within Triton's spread or faster, or the gap is recorded on a bead with its explanation (for example the same gap in an equivalent kernel without these ops)

#### Scenario: Whole Adam step

- **WHEN** the whole Adam-style step (the three updates in one graph) over 4M float32 elements is timed under CubeCL and Inductor+Triton on the RTX 3080, and confirmed on the RTX 3090, with clock warm-up
- **THEN** CubeCL runs one kernel with three outputs, as Triton does (`multi-output-kernels`), and its time is within Triton's spread or faster, or the gap is recorded on a bead with its per-kernel breakdown
