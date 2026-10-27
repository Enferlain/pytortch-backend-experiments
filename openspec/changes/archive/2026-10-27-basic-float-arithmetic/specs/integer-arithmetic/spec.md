# Spec Delta

## ADDED Requirements

### Requirement: Integer absolute value wraps at the minimum

Integer `abs` SHALL return the two's-complement absolute value at the operand's width. The dtype
minimum maps to itself, as eager CUDA returns. Unsigned operands are returned unchanged. The result
SHALL never depend on undefined behaviour in the generated code.

#### Scenario: Every width with its limits

- **WHEN** `torch.abs(x)` is compiled for int8, int16, int32, int64 and uint8 `x` holding the dtype minimum, maximum, 0, 1 and −1 (where representable)
- **THEN** every result equals eager bit for bit, the minimum of each signed dtype included
