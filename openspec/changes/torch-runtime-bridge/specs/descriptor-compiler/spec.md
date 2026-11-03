## MODIFIED Requirements

### Requirement: Audited CubeCL source only

The compiler and the runtime bridge SHALL build only from the audited CubeCL source pin and patch
stack (the floating-point patch and the bridge's runtime changes) and from the audited CubeK source
pin and its own patch stack, SHALL refuse when either source differs from its state, and SHALL record both states in each
artifact's provenance and in each bridged kernel's selection records. Code copied from CubeCL into
this repository SHALL record the upstream revision it was copied from and every local change to it,
and SHALL be covered by the compiler binary digest in each artifact's provenance.

#### Scenario: Unexpected CubeCL modification

- **WHEN** the CubeCL source contains changes outside the audited patch stack (checked against a temporary copy or fake checkout, never the real one)
- **THEN** the build or compilation is refused with an error identifying the mismatch, and no artifact is produced

#### Scenario: Unexpected CubeK state

- **WHEN** the CubeK checkout is at another revision than its pin, or has changes outside its audited patch stack (checked against a temporary copy or fake checkout, never the real one)
- **THEN** the bridge build is refused with an error identifying the mismatch

#### Scenario: Copied code changed

- **WHEN** a file of the copied crate changes
- **THEN** the compiler binary digest changes, so artifacts built before the change are not reused

#### Scenario: Copy traceable to upstream

- **WHEN** the copied crate is compared with the recorded upstream revision
- **THEN** every difference appears in its record of local changes
