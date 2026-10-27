# float-sign-exponent Specification

## Purpose
States the contract of ops on a float's sign and exponent in compiled kernels: `sign`, `signbit`,
`copysign`, `nextafter`, `ldexp` and `frexp`. They are exact by definition, so they are specified
bit for bit, subnormals and signed zeros included, against eager PyTorch and Inductor with Triton.

## Requirements

### Requirement: Sign of a float

Float `sign(x)` SHALL return 1.0 for positive `x`, −1.0 for negative `x`, and +0.0 for ±0.0 and
NaN, in `x`'s dtype, as eager and Triton.

#### Scenario: Special values

- **WHEN** `torch.sign(x)` is compiled for float32, float16 and bfloat16 `x` holding ±0.0, ±inf, NaN of both signs, subnormals and ordinary values
- **THEN** every result equals eager and Triton's recorded output bit for bit

### Requirement: Sign bit of a float

Float `signbit(x)` SHALL return the sign bit as bool, true for −0.0 and for a float32 or
bfloat16 NaN with its sign bit set. A float16 NaN reads as positive in eager and Triton too
(each widens it to float32 through a conversion that canonicalizes NaN), and here.

#### Scenario: Signed zeros and NaN

- **WHEN** `torch.signbit(x)` is compiled for float32, float16 and bfloat16 `x` holding ±0.0, ±inf, NaN of both signs and ordinary values
- **THEN** every result equals eager and Triton

### Requirement: Copy sign

`copysign(x, y)` SHALL return `x`'s magnitude with `y`'s sign bit, including −0.0 and NaN of
either sign as `y`, and NaN or ±0.0 as `x`. Integer operands promote to float32, as Inductor's
lowering does. Every non-NaN result SHALL equal eager and Triton bit for bit, apart from a
float16 negative-NaN `y` (below). A float32 NaN result SHALL carry `y`'s sign.

#### Scenario: Every sign source

- **WHEN** `torch.copysign(x, y)` is compiled for float32, float16 and bfloat16, and for int32 operands, with `x` and `y` drawn from ±0.0, ±inf, NaN of both signs, subnormals and ordinary values
- **THEN** every non-NaN result equals eager and Triton's recorded output bit for bit, and every NaN result is NaN, with `y`'s sign in float32

#### Scenario: Half-precision NaN signs are open

- **WHEN** `y` is a float16 negative NaN, or a NaN result is stored as float16 or bfloat16
- **THEN** the sign is lost: CUDA's half/float conversions make every NaN the positive canonical one. The test lists those elements, and bead `ddu` tracks keeping the sign, as for the NaN payloads of `abs` and `relu`

### Requirement: Next representable value in the operand's dtype

`nextafter(x, y)` SHALL return the next value after `x` toward `y` in the operands' dtype, as
eager:
- float16 and bfloat16 step in their own format, not in float32;
- float32 steps through subnormals and never flushes them;
- `x == y` returns `y`, so `nextafter(-0.0, 0.0)` is +0.0;
- either operand NaN gives NaN;
- the step from the largest finite value toward inf is inf.

Triton differs on subnormals and on float16/bfloat16; there this backend SHALL equal eager.

#### Scenario: Agreement with eager

- **WHEN** `torch.nextafter(x, y)` is compiled for float32, float16 and bfloat16 operand pairs covering ±0.0, the smallest subnormal, the smallest normal, 1.0, the largest finite value, ±inf, NaN and equal operands, in both directions
- **THEN** every non-NaN result equals eager bit for bit, and every NaN result is NaN

#### Scenario: Cases where Triton is wrong

- **WHEN** `torch.nextafter` is compiled for float32 `(0, 1)` and for float16 and bfloat16 `(1, 2)`
- **THEN** the results are the float32 smallest subnormal and the next float16 and bfloat16 value above 1.0, and the test records that Triton's differ

### Requirement: Scaling by a power of two

`ldexp(x, n)` for a float `x` and an integer `n` SHALL return `x · 2^n` rounded once to `x`'s
dtype. `n` is first converted to int32 by wrapping, as eager and Triton do. Results that overflow
are ±inf, results in the subnormal range are rounded once, `x = ±0.0` and `±inf` are returned
unchanged and NaN stays NaN. Subnormal `x` SHALL be scaled exactly, as eager, where Triton
is wrong.

#### Scenario: Agreement with eager

- **WHEN** `torch.ldexp(x, n)` is compiled for float32, float16 and bfloat16 `x` holding ±0.0, ±inf, NaN, subnormals and ordinary values, with int32 and int64 `n` from −300 to 300 and beyond the int32 range
- **THEN** every non-NaN result equals eager bit for bit, and every NaN result is NaN

#### Scenario: Non-integer exponent

- **WHEN** `torch.ldexp(x, n)` is compiled with a float `n`
- **THEN** Inductor's fallback `x * pow(2, n)` applies, and compilation is rejected naming `pow` until the transcendental family supports it

### Requirement: Mantissa and exponent

`frexp(x)` SHALL return a mantissa in `x`'s dtype, with magnitude in [0.5, 1) and `x`'s sign, and
an int32 exponent with `x = mantissa · 2^exponent`. Subnormals are included. ±0.0 gives
(±0.0, 0), and ±inf and NaN give (x, 0), as eager. Triton differs on subnormals and −0.0; there
this backend SHALL equal eager.

#### Scenario: Agreement with eager

- **WHEN** `torch.frexp(x)` is compiled for float32, float16 and bfloat16 `x` holding ±0.0, ±inf, NaN, subnormals, powers of two and ordinary values, and both outputs are used
- **THEN** every mantissa equals eager bit for bit (NaN as NaN), and every exponent equals eager

#### Scenario: One output used

- **WHEN** only the mantissa, or only the exponent, of `torch.frexp(x)` is used in a compiled graph
- **THEN** the kernel compiles and the used output equals eager

### Requirement: Sign and exponent ops are admitted in every kernel class

`sign`, `signbit`, `copysign`, `nextafter`, `ldexp` and `frexp` SHALL be admitted for float32,
float16 and bfloat16 operands (`ldexp` with an integer `n`) wherever other value ops are:
pointwise, fused, reduction, RNG and indirect kernels. An operand outside the domain SHALL be
rejected with a reason naming the op and the dtype.

#### Scenario: Fused into a reduction

- **WHEN** `torch.copysign(x, w).sum(-1)` and `torch.ldexp(x, n).amax(-1)` are compiled
- **THEN** each runs as one generated reduction kernel and equals eager (the sum within the existing float32 reduction tolerance, the max bit for bit)

### Requirement: Sign and exponent kernels are not slower than Triton

Kernels using these ops SHALL NOT be measurably slower than Inductor with Triton on the same GPU,
judged per kernel at matched settings under the repository's timing method.

#### Scenario: Pointwise kernels

- **WHEN** `torch.copysign(x, y)`, `torch.nextafter(x, y)`, `torch.ldexp(x, n)` and `torch.frexp(x)` over 4M float32 elements are timed under CubeCL and Inductor+Triton on the RTX 3080, and confirmed on the RTX 3090, with clock warm-up
- **THEN** each CubeCL time is within Triton's spread or faster, or the gap is fixed before the change completes, or recorded on a bead with its explanation if fixing it takes much more effort
