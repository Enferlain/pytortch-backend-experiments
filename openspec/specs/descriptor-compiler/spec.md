# descriptor-compiler Specification

## Purpose
Compiles semantic kernel descriptors and physical plans into runtime kernel artifacts through
CubeCL, deterministically and only from the audited CubeCL source. It binds each descriptor's
symbolic sizes, strides and offsets into the artifact's launch contract.

## Requirements

### Requirement: Deterministic compilation

Compiling the same descriptor and physical plan with the same compiler build and toolchain SHALL
produce byte-identical CubeCL IR, CUDA source and PTX.

#### Scenario: Recompile from an empty cache

- **WHEN** the same descriptor is compiled twice with empty caches
- **THEN** both compilations produce identical IR, CUDA and PTX bytes

### Requirement: Descriptor binding into the launch contract

The compiler SHALL produce, for each artifact, launch parameters and tensor guards derived from the
descriptor's symbolic sizes, strides, offsets and element count. The runtime SHALL need no
descriptor knowledge to launch or validate it.

#### Scenario: Symbolic extents at launch

- **WHEN** a dynamic-shape descriptor's artifact is launched with concrete tensors of a newly seen extent within its guards
- **THEN** the bound launch parameters equal those Stage 5 computed for the same descriptor and tensors

### Requirement: Malformed descriptor rejection

The compiler SHALL reject a descriptor that doesn't conform to its schema with an error, and SHALL
NOT produce an artifact.

#### Scenario: Missing or inconsistent fields

- **WHEN** a descriptor lacks a required field or declares inconsistent buffers
- **THEN** compilation fails with an error naming the problem, and nothing is written to the cache

### Requirement: Audited CubeCL source only

The compiler SHALL build only from the audited CubeCL source pin and floating-point patch state,
SHALL refuse when the source differs from that state, and SHALL record the state in each artifact's
provenance.

#### Scenario: Unexpected CubeCL modification

- **WHEN** the CubeCL source contains changes outside the audited patch stack (checked against a temporary copy or fake checkout, never the real one)
- **THEN** the build or compilation is refused with an error identifying the mismatch, and no artifact is produced

### Requirement: Layer dependency

The descriptor compiler SHALL depend only on the runtime layer. It SHALL NOT import the Inductor or
custom-op front ends, Inductor, Dynamo or the overlay.

#### Scenario: Compile without Inductor

- **WHEN** a fresh process compiles and launches a hand-authored descriptor
- **THEN** no `torch._inductor`, `torch._dynamo` or front-end module is loaded
