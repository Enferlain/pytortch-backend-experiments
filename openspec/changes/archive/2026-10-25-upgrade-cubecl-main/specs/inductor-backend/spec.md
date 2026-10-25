## REMOVED Requirements

### Requirement: Structural artifact equivalence with the Stage 5 backend

**Reason**: The frozen Stage 5 backend runs only against CubeCL `6bba5fdf` and refuses the
upgraded checkout, so the two sides can no longer be compared. The last full comparison was
336/336 identical at `6bba5fdf`.
**Migration**: The artifact manifest, recorded from that identical state and unchanged after the
upgrade, is the structural gate ("Artifact manifest reproducibility").

### Requirement: Output equivalence under the deterministic policy

**Reason**: Needs the Stage 5 backend to run; see above.
**Migration**: Output correctness is covered by the package's eager and Triton comparisons in
the suite (contracts, band and special-value tests).

### Requirement: Selection equivalence

**Reason**: Needs the Stage 5 backend to run; see above.
**Migration**: The manifest records candidate sets and selections per corpus case, so a change
to either is reported by the manifest check.

## MODIFIED Requirements

### Requirement: Artifact manifest reproducibility

The backend SHALL be checkable against a recorded manifest of the structural corpus records, so
that later changes can tell whether they altered generated code. The manifest SHALL be re-recorded
only after every reported difference has been reviewed and explained.

#### Scenario: Unintended codegen change

- **WHEN** a code change alters any region record for any corpus case
- **THEN** the manifest check reports which cases, regions and files changed

#### Scenario: Compiler upgrade

- **WHEN** the CubeCL pin or its patch stack changes
- **THEN** the manifest check runs before anything is re-recorded, and each reported case, region and file is reviewed and explained first
