## MODIFIED Requirements

### Requirement: Audited CubeCL source only

The compiler SHALL build only from the audited CubeCL source pin and floating-point patch state,
SHALL refuse when the source differs from that state, and SHALL record the state in each artifact's
provenance. Code copied from CubeCL into this repository SHALL record the upstream revision it was
copied from and every local change to it, and SHALL be covered by the compiler binary digest in
each artifact's provenance.

#### Scenario: Unexpected CubeCL modification

- **WHEN** the CubeCL source contains changes outside the audited patch stack (checked against a temporary copy or fake checkout, never the real one)
- **THEN** the build or compilation is refused with an error identifying the mismatch, and no artifact is produced

#### Scenario: Copied code changed

- **WHEN** a file of the copied crate changes
- **THEN** the compiler binary digest changes, so artifacts built before the change are not reused

#### Scenario: Copy traceable to upstream

- **WHEN** the copied crate is compared with the recorded upstream revision
- **THEN** every difference appears in its record of local changes
