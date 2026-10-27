# Spec Delta

## ADDED Requirements

### Requirement: Infinity test

Float `isinf(x)` SHALL be supported as a pointwise op returning bool: true for +inf and −inf,
false for NaN, zeros, subnormals and finite values. float16 and bfloat16 operands are tested in
float32, where their infinities stay infinite. Integer operands never reach the op: Inductor's
lowering gives false.

#### Scenario: isinf

- **WHEN** `torch.isinf(x)` is compiled for float32, float16 and bfloat16 `x` holding ±inf, NaN of both signs, ±0.0, the largest finite value and ordinary values, and for int32 `x`
- **THEN** every result equals eager and Triton

#### Scenario: Finite mask in a reduction

- **WHEN** `torch.where(torch.isinf(x), 0.0, x).sum(-1)` is compiled
- **THEN** it runs as one generated reduction kernel and equals eager within the existing float32 reduction tolerance
