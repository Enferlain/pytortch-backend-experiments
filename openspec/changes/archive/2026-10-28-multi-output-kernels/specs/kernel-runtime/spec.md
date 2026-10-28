# Spec Delta

## MODIFIED Requirements

### Requirement: Contract-driven launch validation

Before each launch, the runtime SHALL check tensor arguments against the artifact's declared
guards. A mismatch SHALL raise an error naming the mismatch instead of launching. Every output
tensor of a kernel with several outputs is guarded, not only the last argument.

#### Scenario: Wrong device

- **WHEN** a guarded tensor argument is on a different device than declared
- **THEN** the launch is refused with an error naming the device mismatch

#### Scenario: Wrong dtype

- **WHEN** a guarded tensor argument has a different dtype than declared
- **THEN** the launch is refused with an error naming the dtype mismatch

#### Scenario: Layout differs from the guard

- **WHEN** a guarded tensor argument's strides or offset differ from the declared layout
- **THEN** the launch is refused with an error naming the layout mismatch

#### Scenario: Extent outside the guard

- **WHEN** a tensor argument's extent violates the declared extent guard
- **THEN** the launch is refused with an error naming the extent mismatch

#### Scenario: Extent of a non-final output

- **WHEN** a kernel with a per-row output and a per-element output is launched with a per-row output tensor whose element count differs from the outer extent
- **THEN** the launch is refused with an error naming the extent mismatch, although the last tensor argument is correct
