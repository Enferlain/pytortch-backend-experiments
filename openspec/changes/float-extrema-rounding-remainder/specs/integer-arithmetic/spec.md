# Spec Delta

## ADDED Requirements

### Requirement: Integer sign and sign bit

Integer `sign(x)` SHALL return −1, 0 or 1 in `x`'s dtype, the dtype minimum included. `signbit(x)`
SHALL return `x < 0` for signed integers. For bool and unsigned operands Inductor's lowering
decides both (identity and false), so no op reaches the kernel.

#### Scenario: Every width with its limits

- **WHEN** `torch.sign(x)` and `torch.signbit(x)` are compiled for int8, int16, int32, int64, uint8 and bool `x` holding the dtype minimum, maximum, 0, 1 and −1 (where representable)
- **THEN** every result equals eager

### Requirement: Rounding integers is the identity

`floor`, `ceil`, `trunc` and `round` of an integer operand SHALL return the operand unchanged in
its dtype, as eager.

#### Scenario: Rounding int32 and int64

- **WHEN** `torch.floor`, `torch.ceil`, `torch.trunc` and `torch.round` are compiled for int32 and int64 `x` including the dtype limits
- **THEN** every result equals `x`
