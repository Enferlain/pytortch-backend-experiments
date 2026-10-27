# float-remainder Specification

## Purpose
States the numerical contract of float remainder and floor division in compiled kernels: `fmod`,
`remainder` and Inductor's float `mod`, plus floor division. Covers every sign combination, zero
and infinite operands, signed zeros and large quotients, measured against eager PyTorch and
Inductor with Triton.

## Requirements

### Requirement: Truncating remainder is exact

Float `fmod(a, b)` and Inductor's float `mod(a, b)` SHALL return the exact remainder of `a / b`
with the quotient truncated toward zero, as C `fmod`:
- the result has `a`'s sign, a zero result included;
- a zero divisor or an infinite dividend gives NaN;
- an infinite divisor with a finite dividend gives the dividend.

float16 and bfloat16 operands compute in float32; the exact remainder is representable in the
operand dtype, so the stored result equals eager's.

#### Scenario: Subnormal operands

- **WHEN** `torch.fmod(a, b)` is compiled for float32 `(0, 2^-149)` and `(2^-149, inf)`
- **THEN** the results are 0 and 2^-149, as eager, and the test records that Triton, which flushes float32 and bfloat16 subnormal operands to zero, returns NaN and 0

#### Scenario: Agreement with eager and Triton

- **WHEN** `torch.fmod(a, b)` is compiled for float32, float16 and bfloat16 operands covering every sign combination, ±0.0, ±inf, NaN, subnormals, fractional divisors, and quotients from below 1 up to 2^100
- **THEN** every non-NaN result equals eager bit for bit, and every NaN result is NaN in eager too
- **AND** every result equals Triton's recorded output except where an operand is a float32 or bfloat16 subnormal

#### Scenario: Large quotients

- **WHEN** `torch.fmod(a, b)` is compiled for float32 `a = 1e10, -1e10, 3e38` and `b = 3, 1.1, 0.1`
- **THEN** each result equals eager bit for bit (for example `fmod(1e10, 3) = 1`)

### Requirement: Floor remainder follows eager

Float `remainder(a, b)` SHALL equal eager CUDA's: `r = fmod(a, b)`, then `r + b` when `r` is
nonzero and its sign differs from `b`'s. The result therefore has `b`'s sign or is a zero with
`a`'s sign, and is exact apart from the one rounding of `r + b`. Triton's result differs for large
quotients, for signed zeros, and where its `frem` expansion lands outside `[0, b)`; there this
backend SHALL equal eager, not Triton.

#### Scenario: Agreement with eager

- **WHEN** `torch.remainder(a, b)` and `a % b` are compiled for float32, float16 and bfloat16 operands over the same value set as `fmod`
- **THEN** every non-NaN result equals eager bit for bit, and every NaN result is NaN in eager too

#### Scenario: Cases where Triton is wrong

- **WHEN** `torch.remainder(a, b)` is compiled for float32 `(1e10, 3)`, `(-1e10, 3)`, `(3e38, 1.1)`, `(1e7, 0.1)`, `(-0.0, 2)` and `(-4.0, 2)`
- **THEN** the results are `1`, `2`, eager's value for `(3e38, 1.1)`, eager's positive value for `(1e7, 0.1)`, `-0.0` and `-0.0`, and the test records that Triton's differ

#### Scenario: Agreement with Triton elsewhere

- **WHEN** the recorded inputs on which Triton equals eager are compiled
- **THEN** every non-NaN result equals Triton's recorded output bit for bit

#### Scenario: Infinite divisor

- **WHEN** `torch.remainder(a, b)` is compiled for `(1, inf)`, `(-1, inf)`, `(0, -inf)` and `(3, -inf)`
- **THEN** the results are `1`, `inf`, `0` and `-inf`, as eager

### Requirement: Remainders are exact under contraction

`fmod`, `mod` and `remainder` SHALL keep these results whatever the FMA contraction and division
policy settings, including the default (contraction on) and Inductor's emulation and
division-rounding switches.

#### Scenario: Contraction on

- **WHEN** the large-quotient and sign cases are compiled with contraction on, and through the FP audit route at contraction on
- **THEN** every result equals eager bit for bit

### Requirement: Float floor division matches Triton

Float floor division (`a // b`, `torch.div(a, b, rounding_mode="floor")` and `floor_divide`) SHALL
compile through Inductor's lowering and equal Triton bit for bit. That lowering floors the
correctly rounded quotient; eager floors the exact quotient. Where the rounded quotient crosses an
integer (for example float32 `1 // 0.1`: 10 here and in Triton, 9 in eager), the result follows
Triton.

#### Scenario: Agreement with Triton

- **WHEN** float floor division is compiled for float32, float16 and bfloat16 operands covering every sign combination, ±0.0, ±inf, NaN, zero divisors and large quotients
- **THEN** every non-NaN result equals Triton's recorded output bit for bit, and every NaN result is NaN in both
- **AND** where Triton flushes a float32 or bfloat16 subnormal operand or quotient to zero, the result instead equals the same lowering evaluated without flushing (eager's `floor(div_rn(a, b))` with the lowering's corrections), and the test lists those elements

#### Scenario: Divergence from eager is recorded

- **WHEN** float32 `1 // 0.1` and `7 // -0.7` are compiled
- **THEN** the results equal Triton's (10 and −10), and the test records eager's (9 and −11)

### Requirement: Remainder ops are admitted in every kernel class

`fmod`, `mod` and `remainder` SHALL be admitted for float32, float16 and bfloat16 operands, as for
integer operands, wherever other value ops are: pointwise, fused, reduction, RNG and indirect
kernels. A bool operand SHALL be rejected with a reason naming the op and the dtype.

#### Scenario: Fused into a reduction

- **WHEN** `torch.remainder(x, 1.0).sum(-1)` and `torch.fmod(x, w).sum(-1)` are compiled
- **THEN** each runs as one generated reduction kernel and equals eager within the existing float32 reduction tolerance

#### Scenario: Applied to a gather

- **WHEN** `torch.remainder(x[idx], 3.0)` is compiled with an index tensor
- **THEN** the kernel compiles and equals eager bit for bit

### Requirement: Remainder kernels are not slower than Triton

Remainder and floor-division kernels SHALL NOT be measurably slower than Inductor with Triton on
the same GPU, judged per kernel at matched settings under the repository's timing method.

#### Scenario: Pointwise remainder kernels

- **WHEN** `torch.fmod(x, y)`, `torch.remainder(x, y)` and `x // y` over 4M float32 elements, with quotients both small and large, are timed under CubeCL and Inductor+Triton on the RTX 3080, and confirmed on the RTX 3090, with clock warm-up
- **THEN** each CubeCL time is within Triton's spread or faster, or the gap is fixed before the change completes, or recorded on a bead with its explanation if fixing it takes much more effort
