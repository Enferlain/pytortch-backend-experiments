# Spec Delta

## Purpose

States the numerical contract of basic floating-point value ops in compiled kernels: absolute
value, square, relu, fused multiply-add, and explicitly rounded multiply and divide. It is
measured against Inductor with Triton (the floor) and against eager CUDA where Inductor's
lowering intends to match eager.

## ADDED Requirements

### Requirement: Float absolute value

Float `abs` SHALL return its operand with the sign bit cleared, so −0.0 becomes +0.0, ±inf becomes
+inf and NaN stays NaN with its sign cleared. The NaN payload is left open, as for `maximum`
(`float-extrema`): Triton keeps it when `abs` is the last operation (a bit mask), while eager and
this backend's `abs.f32` return the canonical NaN. Every non-NaN result SHALL equal Triton and
eager bit for bit. float16 and bfloat16 operands compute in float32 and round at their store, as
in Triton.

#### Scenario: Special values

- **WHEN** `torch.abs(x)` is compiled for float32, float16 and bfloat16 `x` holding −0.0, +0.0, ±inf, NaN of both signs, the smallest subnormal, the largest finite value and ordinary values
- **THEN** every non-NaN result equals Triton's recorded output and eager bit for bit, and every NaN result is NaN with its sign cleared

#### Scenario: Weight test in lerp

- **WHEN** `torch.lerp(a, b, w)` is compiled with a float32 tensor weight `w` holding values on both sides of ±0.5
- **THEN** the kernel compiles and the result equals eager bit for bit

### Requirement: Square

`square(x)` SHALL be computed as `x * x`, with one rounding, and is subject to the same
multiply-add contraction policy as any other multiply. Results SHALL equal Triton bit for bit.

#### Scenario: Square of floats and integers

- **WHEN** `torch.square(x)` is compiled for float32, float16, bfloat16, int32 and int64 `x`, including values whose square overflows (to inf for floats, wrapping for integers)
- **THEN** every result equals Triton's recorded output bit for bit, and equals eager

#### Scenario: Short-row variance

- **WHEN** `torch.var_mean(x, dim=-1)` is compiled for float32 rows of 8 or fewer elements
- **THEN** the kernel compiles and its results are within the Welford band test's acceptance against Triton

### Requirement: Relu

`relu(x)` SHALL equal `maximum(x, 0)` with the zero in `x`'s dtype. It therefore returns NaN for a
NaN operand, with the payload left open as for `maximum` (`float-extrema`), and +0.0 for a −0.0
operand, as Triton does. Eager returns −0.0 there and passes a NaN operand through unchanged, sign
included; PyTorch promises neither the sign of a zero result nor NaN bits.

#### Scenario: Relu of special values

- **WHEN** `torch.relu(x)` is compiled for float32, float16 and bfloat16 `x` holding −0.0, +0.0, ±inf, NaN and ordinary values, and for int32 `x` including the dtype minimum
- **THEN** every non-NaN result equals Triton's recorded output bit for bit, every NaN result is NaN, and non-NaN results equal eager except for the sign of the zero produced from −0.0

### Requirement: Fused multiply-add rounds once

`fma(x, y, z)` SHALL return `x * y + z` rounded once to float32, whatever the contraction setting
(`emulate_precision_casts` included) and whether or not its operands are compile-time constants.
Results SHALL equal Triton's `tl.fma` bit for bit.

#### Scenario: add with alpha

- **WHEN** `torch.add(a, b, alpha=s)` and `torch.sub(a, b, alpha=s)` are compiled for float32, float16 and bfloat16 tensors, with `s` not 1
- **THEN** the results equal eager and Triton's recorded output bit for bit

#### Scenario: Single rounding is observable

- **WHEN** `torch.add(c, a, alpha=s)` is compiled with float32 values where `a * s + c` rounded once differs from `a * s` rounded and then added to `c`
- **THEN** the result equals the singly rounded value (eager's), with contraction on and with `emulate_precision_casts` on

#### Scenario: Constant operands

- **WHEN** a kernel computes `fma` of three compile-time constants chosen so that single and double rounding differ
- **THEN** the folded result equals the executed single-rounding result

### Requirement: Rounded multiply is never contracted

`mul_rn(x, y)` SHALL return the float32 product rounded to nearest even, subnormal products
included. That product SHALL NOT be fused into a following add, even with contraction on. Results
SHALL equal Triton's `libdevice.mul_rn` bit for bit, except that a subnormal product is kept, as
eager keeps it, where Triton's flushes it to zero (`FMUL.FTZ`).

#### Scenario: Subnormal product

- **WHEN** `torch.addcmul(s, a, b, value=0.3)` is compiled with `s` = 0 and products `a * b` that are subnormal in float32
- **THEN** the results equal eager bit for bit, nonzero (Inductor+Triton returns 0 there)

#### Scenario: addcmul with a value

- **WHEN** `torch.addcmul(s, a, b, value=v)` is compiled for float32, float16 and bfloat16 tensors with `v` = 1 and `v` not 1, the float32 inputs including rows where an unrounded product or a doubly rounded fma would change the result (16-bit products are exact in float32, so no such rows exist there)
- **THEN** the results equal eager and Triton's recorded output bit for bit

### Requirement: Rounded divide is IEEE division

`div_rn(x, y)` SHALL return the correctly rounded float32 quotient, including for zero, infinite and
NaN operands and subnormal results. This holds under the default approximate division policy as
well as under `division_rounding`. Results SHALL equal Triton's `tl.div_rn` bit for bit.

#### Scenario: addcdiv with a value

- **WHEN** `torch.addcdiv(s, a, b, value=v)` is compiled for float32, float16 and bfloat16 tensors with `v` not 1, with divisors including 0, ±inf and NaN, a NaN dividend, and pairs whose quotient is subnormal in the tensors' dtype
- **THEN** the results equal eager and Triton's recorded output bit for bit

#### Scenario: addcdiv with value 1

- **WHEN** `torch.addcdiv(s, a, b, value=1)` is compiled for the same inputs
- **THEN** the kernel divides with `div_rn` and the results equal eager bit for bit: this backend disables Inductor's joint-graph constant folding (whose `remove_no_ops` also rewrites `x + 0` and loses a −0.0), so the `* 1` survives and Inductor's post-grad pass rebuilds `addcdiv`
- **AND** they differ from Triton's recorded output only where its approximate division rounds differently: with default settings Inductor deletes the `* 1` first and Triton divides with `div.full` (bead `4jf`)

#### Scenario: Plain division keeps its policy

- **WHEN** a kernel contains both an `addcdiv` and an ordinary `a / b`
- **THEN** only the `div_rn` is correctly rounded; the ordinary division follows the division policy (approximate by default) and equals Triton's result for it

### Requirement: Optimizer update kernels match eager

The fused optimizer update forms SHALL compile into generated kernels and equal eager bit for bit:
`lerp` with scalar and tensor weights, `addcmul`, `addcdiv`, and `add` with `alpha`, as composed in
an Adam step.

#### Scenario: Adam-style step

- **WHEN** one Adam update is compiled as a functional graph over float32 tensors: `exp_avg.lerp(grad, 1 - beta1)`, `exp_avg_sq.mul(beta2).addcmul(grad, grad, value=1 - beta2)` and `param.addcdiv(exp_avg, denom, value=-step_size)`, with `denom` an input (its own division follows the division policy)
- **THEN** every generated region compiles through CubeCL, and the updated tensors equal eager bit for bit

### Requirement: Basic float ops are admitted in every kernel class

`abs`, `square`, `relu`, `fma`, `mul_rn` and `div_rn` SHALL be admitted wherever other value ops are:
pointwise, fused, reduction, RNG and indirect kernels. Their operand domains: float and integer for
`abs`, `square` and `relu`, floating only for `fma`, `mul_rn` and `div_rn`. An operand outside the
domain SHALL be rejected with a reason naming the op and the dtype.

#### Scenario: Fused into a reduction

- **WHEN** `(torch.abs(x) * w).sum(-1)` and `torch.addcmul(x, x, w, value=0.5).sum(-1)` are compiled
- **THEN** each runs as one generated reduction kernel and equals eager within the existing float32 reduction tolerance

#### Scenario: Integer operand to fma

- **WHEN** a hand-written descriptor or graph applies `fma` to int32 values
- **THEN** compilation is rejected with a reason naming `fma` and the integer dtype

### Requirement: Existing kernels keep their generated code

Kernels that use none of the six ops SHALL keep their generated IR, CUDA and PTX unchanged, even
though the CubeCL patch stack and compiler binary change.

#### Scenario: Manifest after the patch change

- **WHEN** the structural manifest check runs after the change
- **THEN** it passes with no changed case

### Requirement: Basic float kernels are not slower than Triton

Representative kernels using these ops SHALL NOT be measurably slower than Inductor with Triton on
the same GPU, judged per kernel at matched settings, under the repository's timing method and its
recorded short-kernel exception.

#### Scenario: Optimizer and pointwise kernels

- **WHEN** each Adam-style update (`lerp`, `addcmul`, `addcdiv`) alone, an `addcdiv` kernel and an `abs`/`relu` pointwise kernel over 4M float32 elements are timed under CubeCL and Inductor+Triton on the RTX 3080, and confirmed on the RTX 3090, with clock warm-up
- **THEN** each CubeCL time is within Triton's spread or faster, or the gap is recorded on a bead with its explanation (for example the same gap in an equivalent kernel without these ops)
- **AND** the whole Adam step is timed and recorded: Triton fuses its three updates into one kernel with three outputs, which waits on multi-output fusion (bead `4z8.21`)
