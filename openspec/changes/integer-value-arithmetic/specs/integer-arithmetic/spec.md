# Spec Delta

## Purpose

States the contract of integer and bool value arithmetic in compiled kernels: which integer value
dtypes exist, how each op behaves on them (including overflow, shifts and division edge cases),
and where they are admitted. It is measured against eager CUDA PyTorch, with Inductor+Triton as
the floor.

## ADDED Requirements

### Requirement: Integer value dtypes keep their width

A kernel value SHALL have one of the dtypes bool, int8, int16, int32, int64 or uint8. An integer
op's result SHALL have the dtype Inductor's dtype propagation gives it, computed at that width and
signedness, never widened silently. Values of these dtypes arise from constants, casts,
`index_expr`/`value_expr` and loads of admitted storage dtypes.

#### Scenario: Narrow arithmetic wraps at its own width

- **WHEN** `((x.to(torch.int8) + 100) > 0)` is compiled, with float32 `x` holding values where `x.to(torch.int8) + 100` exceeds 127
- **THEN** the result equals eager, the int8 sum having wrapped before the comparison

#### Scenario: int32 index values

- **WHEN** Inductor emits an `index_expr` or `value_expr` with dtype int32
- **THEN** the kernel compiles and the value equals the int64 index expression converted to int32

### Requirement: Wrapping integer arithmetic

Integer `add`, `sub`, `mul` and `neg` SHALL produce the two's-complement result modulo 2^width
of the operand dtype, as eager CUDA and Triton do.

#### Scenario: Overflow at every width

- **WHEN** add, sub, mul and neg are compiled over int8, int16, int32, int64 and uint8 values that include the dtype's minimum, maximum, 0, 1 and −1
- **THEN** every result equals eager bit for bit

### Requirement: Integer and bool comparisons, selection and extrema

`eq`, `ne`, `lt`, `le`, `gt` and `ge` SHALL accept two operands of the same bool or integer
dtype, comparing signed or unsigned as the dtype is, with a bool result. `where`, `maximum` and
`minimum` SHALL accept integer and bool operands. `masked` SHALL accept integer and bool results.

#### Scenario: Signed and unsigned order

- **WHEN** comparisons and `maximum`/`minimum` are compiled over int8 and uint8 values on both sides of 127/128
- **THEN** int8 orders them as signed and uint8 as unsigned, equal to eager

#### Scenario: Bool comparison

- **WHEN** `(a > 0) == (b > 0)` and `(a > 0) < (b > 0)` are compiled
- **THEN** the results equal eager, with False ordered below True

### Requirement: Logical operations

`logical_and`, `logical_or`, `logical_xor`, `logical_not`, `and_`, `or_` and `xor` SHALL compile
for bool operands with a bool result equal to eager. Inductor converts the operands of the aten
logical ops to bool before these ops.

#### Scenario: Padding mask

- **WHEN** `F.pad(x, (1, 2), mode="constant")`, `F.avg_pool2d` with padding, and `torch.logical_and(a > 0, b < 1) | torch.logical_not(c > 0)` are compiled
- **THEN** the kernels contain no rejected `and_`, `logical_and` or `logical_not` op, and the results equal eager

### Requirement: Bitwise operations

`bitwise_and`, `bitwise_or`, `bitwise_xor` and `bitwise_not` SHALL compile for bool and every
integer value dtype, equal to eager bit for bit. `bitwise_not` of a bool SHALL be logical negation.

#### Scenario: Every dtype

- **WHEN** `a & b`, `a | b`, `a ^ b` and `~a` are compiled over bool, int8, int16, int32, int64 and uint8 values including the dtype's minimum and maximum
- **THEN** every result equals eager

### Requirement: Shifts follow eager for every shift amount

`bitwise_left_shift`, `bitwise_right_shift`, `lshift` and `rshift` SHALL equal eager CUDA for
every shift amount. A left shift by a negative amount or by at least the width SHALL give 0. A
right shift by such an amount SHALL give the sign fill for signed dtypes and 0 for uint8. Signed
right shift SHALL be arithmetic.

#### Scenario: Out-of-range amounts

- **WHEN** `a << b` and `a >> b` are compiled with `b` in {−1, 0, 1, width−1, width, width+1, 100} for int8, int16, int32, int64 and uint8, and `a` covering negative, zero and positive values
- **THEN** every result equals eager

#### Scenario: Triton's behaviour is recorded

- **WHEN** the same cases run under Inductor with Triton
- **THEN** the reference data records Triton's results, and any case where Triton differs from eager is listed in the test as a documented improvement

### Requirement: Integer division and remainder

`truncdiv` SHALL round the quotient toward zero, `floordiv` SHALL round it toward negative
infinity, and integer `mod` SHALL return the remainder with the dividend's sign, all equal to
eager for every nonzero divisor. For a zero divisor, and for the dtype's minimum divided by −1,
the result SHALL be unspecified, but the kernel SHALL NOT trap, hang or fault.

#### Scenario: Signs and rounding

- **WHEN** `torch.div(a, b, rounding_mode="trunc")`, `torch.div(a, b, rounding_mode="floor")`, `a // b` and `torch.fmod(a, b)` are compiled over int32 and int64 operands in every sign combination, including divisors ±1 and values near the dtype's limits
- **THEN** every result equals eager

#### Scenario: Zero divisor does not fault

- **WHEN** the same graphs run with zero divisors in some elements
- **THEN** the kernel completes, and every element with a nonzero divisor equals eager

### Requirement: True division of integers

`int_truediv` SHALL return the float32 quotient Inductor with Triton computes: both operands
converted to float32 and divided under the backend's numerical policy for float division.

#### Scenario: Integer true division

- **WHEN** a graph whose Inductor lowering emits `int_truediv` on int64 operands is compiled
- **THEN** the result equals Inductor with Triton on the same inputs

### Requirement: Casts between bool, integer and float

`to_dtype` SHALL convert as eager does:
- float to integer truncates toward zero
- integer to narrower integer wraps modulo 2^width
- integer or float to bool is `!= 0` (NaN gives True)
- bool to a number is 0 or 1
- integer to float rounds to nearest

A float to integer conversion of NaN or of an out-of-range value SHALL follow the hardware
conversion, as eager and Triton do.

#### Scenario: Conversion table

- **WHEN** casts between float32, float16, bfloat16, bool and every integer value dtype are compiled over values including ±0, ±0.5, ±1.5, the integer limits and, for floats, NaN and ±inf
- **THEN** every in-range result equals eager, and every NaN or out-of-range float to integer result equals Inductor with Triton

### Requirement: Bit reinterpretation

`to_dtype_bitcast` SHALL reinterpret the bits of a value as a dtype of the same size, after
rounding a float16/bfloat16 value held in float32 compute back to its source dtype, as Triton does.
A size mismatch SHALL be rejected with a reason.

#### Scenario: Float bits as integers and back

- **WHEN** `x.view(torch.int32)` on float32, and `x.view(torch.int16)` on bfloat16 and float16, are each used in a fused computation and the results viewed back
- **THEN** the bits equal eager, including for NaN, ±inf and ±0

### Requirement: Integer ops are admitted in every kernel class

Integer and bool ops SHALL be admitted on equal terms in pointwise kernels, kernels with
indirect addressing or RNG, fused pointwise kernels and the pointwise body of reduction kernels.
Fusion SHALL NOT be refused because a node contains them.

#### Scenario: fmax fused into a reduction

- **WHEN** `torch.fmax(a, b).sum(-1)` and `torch.fmax(a, b) * 2` are compiled
- **THEN** each compiles to a single kernel, and the result equals eager within the sum's existing tolerance

#### Scenario: Mask in a reduction body

- **WHEN** `torch.where(torch.logical_and(x > 0, x < 1), x, 0).sum(-1)` is compiled
- **THEN** it compiles to one reduction kernel and matches eager within the sum's existing tolerance

### Requirement: Existing kernels keep their generated code

Admitting integer arithmetic SHALL NOT change the IR, CUDA or PTX of any kernel the backend
compiled before. Descriptor bytes MAY change only where an internal op spelling is replaced by the
shared one, and every such case SHALL be listed for review.

#### Scenario: Manifest after the change

- **WHEN** the manifest check runs after the change
- **THEN** every changed case differs only in descriptor and artifact-key digests, with identical IR, CUDA and PTX bodies, and the list of changed cases is recorded before re-recording

### Requirement: Integer kernels are not slower than Triton

On representative integer kernels, the backend's kernel time SHALL NOT be measurably slower
than Inductor with Triton, under the repository's timing method and its recorded short-kernel
exception. The representative kernels are a padding mask, an int64 floordiv, a bitwise/shift mix
and an int8 wrap.

#### Scenario: Timing against Triton

- **WHEN** the representative kernels are timed on the RTX 3080 and confirmed on the RTX 3090
- **THEN** each CubeCL time is within Triton's run-to-run spread or faster, or the gap is recorded on a bead with its SASS explanation

### Requirement: Out-of-scope integer work is rejected by name

Integer storage other than bool and int64, integer reductions, and bool `add`/`sub`/`mul` SHALL be
rejected with the unsupported-capability error, and a reason that names the dtype or op.

#### Scenario: Integer sum

- **WHEN** `x.to(torch.int64).sum(-1)` is compiled
- **THEN** the backend raises the unsupported-capability error naming the integer reduction, and no kernel is cached
