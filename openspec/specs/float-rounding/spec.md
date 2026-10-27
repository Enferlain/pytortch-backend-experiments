# float-rounding Specification

## Purpose
States the numerical contract of float rounding to integral values in compiled kernels: `floor`,
`ceil`, `trunc`, `round` (half to even) and their `*_to_int` forms. Covers signed zeros, NaN,
infinities and constant folding that agrees with execution, measured against eager PyTorch and
Inductor with Triton.

## Requirements

### Requirement: Rounding to integral values

Float `floor`, `ceil`, `trunc` and `round` SHALL return the integral value toward −inf, toward
+inf, toward zero, and to nearest with ties to even. A zero result keeps the operand's sign, so
`ceil(-0.5)` and `round(-0.4)` are −0.0. ±inf and ±0.0 are returned unchanged and NaN stays NaN.
Every non-NaN result SHALL equal eager bit for bit, and Triton's except for float32 and bfloat16
subnormal operands: Triton flushes them to zero (`floor(-2^-149)` is −0.0 there, −1 in eager).

#### Scenario: Special values and ties

- **WHEN** `torch.floor`, `torch.ceil`, `torch.trunc` and `torch.round` are compiled for float32, float16 and bfloat16 `x` holding ±0.0, ±inf, NaN, ±0.4, ±0.5, ±1.5, ±2.5, values just below and above integers, the smallest subnormal, and values at and above 2^23 (float32) and 2^10 (float16)
- **THEN** every non-NaN result equals eager bit for bit, and every NaN result is NaN
- **AND** every result equals Triton's recorded output except for float32 and bfloat16 subnormal operands

#### Scenario: Round with decimals

- **WHEN** `torch.round(x, decimals=2)` is compiled for float32 `x`
- **THEN** it compiles through Inductor's decomposition and equals Triton's recorded output bit for bit

### Requirement: Folded rounding equals executed rounding

A rounding op whose operand is a constant in the kernel SHALL fold to the value the executed op
would return. In particular, `round` SHALL fold half to even, as it executes.

#### Scenario: Constant tie

- **WHEN** a kernel applies `round` to the constants 0.5, 1.5, 2.5 and −2.5
- **THEN** the results are 0.0, 2.0, 2.0 and −2.0, and the generated CUDA contains no rounding call for them, showing they were folded

### Requirement: Rounding to an integer dtype

`floor_to_int`, `ceil_to_int`, `trunc_to_int` and `round_to_int` SHALL equal the rounding op
followed by the existing float-to-integer cast to the requested dtype. A NaN, infinite or
out-of-range value converts as that cast defines.

#### Scenario: Ops-level rounding to int64 and int32

- **WHEN** a test op lowered to each `*_to_int` op with int64 and int32 targets is compiled for float32 values including ties, negative values and −0.0
- **THEN** every result equals rounding then casting in eager

### Requirement: Rounding ops are admitted in every kernel class

`floor`, `ceil`, `trunc`, `round` and the `*_to_int` ops SHALL be admitted for float32, float16 and
bfloat16 operands wherever other value ops are: pointwise, fused, reduction, RNG and indirect
kernels. A bool operand SHALL be rejected with a reason naming the op and the dtype.

#### Scenario: Fused into a reduction

- **WHEN** `torch.floor(x * 4).sum(-1)` and `torch.round(x).amax(-1)` are compiled
- **THEN** each runs as one generated reduction kernel and equals eager (the sum within the existing float32 reduction tolerance, the max bit for bit)

### Requirement: Existing kernels keep their generated code

Kernels that use none of this change's ops SHALL keep their generated IR, CUDA and PTX unchanged,
even though the CubeCL patch stack and compiler binary change.

#### Scenario: Manifest after the patch change

- **WHEN** the structural manifest check runs after the change
- **THEN** it passes with no changed case

### Requirement: Rounding kernels are not slower than Triton

Rounding kernels SHALL NOT be measurably slower than Inductor with Triton on the same GPU, judged
per kernel at matched settings under the repository's timing method.

#### Scenario: Pointwise rounding kernels

- **WHEN** `torch.floor(x)` and `torch.round(x)` over 4M float32 and bfloat16 elements are timed under CubeCL and Inductor+Triton on the RTX 3080, and confirmed on the RTX 3090, with clock warm-up
- **THEN** each CubeCL time is within Triton's spread or faster, or the gap is fixed before the change completes, or recorded on a bead with its explanation if fixing it takes much more effort
