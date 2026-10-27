# Spec Delta

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
