## MODIFIED Requirements

### Requirement: Deterministic compilation

Compiling the same descriptor and physical plan with the same compiler build and toolchain SHALL
produce byte-identical CubeCL IR, CUDA source and PTX, including when the compilations run in
separate compiler processes and the kernel declares several shared-memory arrays.

#### Scenario: Recompile from an empty cache

- **WHEN** the same descriptor is compiled twice with empty caches
- **THEN** both compilations produce identical IR, CUDA and PTX bytes

#### Scenario: Several shared arrays across compiler processes

- **WHEN** a Welford descriptor whose plan declares three shared arrays is compiled three times, each in a fresh compiler process with an empty cache
- **THEN** all three compilations produce identical IR, CUDA and PTX bytes
