# Spec Delta

## Purpose

States the numerical contract of floating-point extrema in compiled kernels: `maximum`, `minimum`,
max/min reductions and `isnan`, including NaN, infinities and signed zero. It is measured against
eager PyTorch and Inductor with Triton, with Triton as the floor.

## ADDED Requirements

### Requirement: NaN propagation

Float `maximum`, `minimum`, and max and min reductions SHALL return NaN whenever any operand or
any reduced element is NaN, regardless of its position. This matches eager PyTorch and Triton.

#### Scenario: NaN in a row

- **WHEN** `x.amax(-1)` and `x.amin(-1)` are compiled and a row contains a NaN as its first element, its last element, or one in the middle (float32, float16, bfloat16)
- **THEN** that row's result is NaN

#### Scenario: Pointwise NaN operand

- **WHEN** `torch.maximum(a, b)` and `torch.minimum(a, b)` are compiled and either operand is NaN
- **THEN** the result is NaN

### Requirement: Ordered signed zero

Float `maximum` and the max reduction SHALL treat −0.0 as less than +0.0; `minimum` and the min
reduction SHALL treat +0.0 as greater than −0.0. The sign of a zero result therefore does not
depend on operand or reduction order. This matches Triton; eager PyTorch's reductions do not
promise a sign.

#### Scenario: Pointwise zeros

- **WHEN** `torch.maximum` and `torch.minimum` are compiled for operand pairs (−0, +0) and (+0, −0)
- **THEN** `maximum` returns +0.0 and `minimum` returns −0.0 for both orders

#### Scenario: Reduction over mixed zeros

- **WHEN** `x.amax(-1)` and `x.amin(-1)` are compiled for rows holding only −0.0 and +0.0 in any order and length, including split shapes
- **THEN** `amax` returns +0.0 whenever the row contains +0.0, and `amin` returns −0.0 whenever the row contains −0.0

### Requirement: Exact extrema results

Max and min results SHALL be exact: an element of the input, rounded only by the storage dtype
of the output. Results SHALL be bitwise identical to Inductor with Triton on the same inputs, apart from
the NaN payload, under every admitted plan.

#### Scenario: Agreement with Triton

- **WHEN** max/min reductions and `maximum`/`minimum` are compiled by both backends over seeded random inputs that include infinities, NaN and both zeros, at short, long and split shapes, static and dynamic, for float32, float16 and bfloat16
- **THEN** every non-NaN result is bitwise identical across the backends, and every NaN result is NaN in both

#### Scenario: Infinities

- **WHEN** a row contains +inf, or holds only −inf
- **THEN** `amax` returns +inf for the first and −inf for the second, and `amin` follows symmetrically

### Requirement: NaN-ignoring extrema through decomposition

`torch.fmax` and `torch.fmin` SHALL compile through Inductor's decomposition and return the
non-NaN operand when exactly one operand is NaN, and NaN only when both are. Float `isnan` SHALL be
supported as a pointwise op returning bool.

#### Scenario: fmax with one NaN

- **WHEN** `torch.fmax(a, b)` and `torch.fmin(a, b)` are compiled and exactly one operand is NaN
- **THEN** the result is the other operand

#### Scenario: isnan

- **WHEN** `torch.isnan(x)` is compiled for float inputs containing NaN, ±inf, zeros and finite values
- **THEN** the result equals eager's
