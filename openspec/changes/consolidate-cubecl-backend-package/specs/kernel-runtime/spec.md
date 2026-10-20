# Spec Delta

## Purpose

Caches, validates and launches compiled CubeCL kernel artifacts on PyTorch-owned tensors, through
a contract that doesn't depend on which compiler produced the artifact. It serves every producer
(the descriptor compiler, later CubeK-based kernels) and every front end.

## ADDED Requirements

### Requirement: Layer independence

The runtime SHALL be importable and usable without importing the descriptor compiler, the Inductor
or custom-op front ends, Inductor, Dynamo or the PyTorch overlay.

#### Scenario: Import without upper layers

- **WHEN** a fresh process imports the runtime and launches an artifact
- **THEN** no `torch._inductor` or `torch._dynamo` module, no descriptor-compiler module, and no front-end module is loaded, and the overlay has not been installed

### Requirement: Producer-independent artifact contract

The runtime SHALL accept any kernel artifact that declares its compiled code, entry point, launch
shape, parameters and tensor guards. It SHALL NOT require a semantic descriptor or physical plan.

#### Scenario: Artifact from a non-descriptor producer

- **WHEN** an artifact built without the descriptor compiler (the Stage 0 programmatic `sin(x)+1` PTX) is described through the artifact contract and launched
- **THEN** it runs on PyTorch tensors and produces the same result as `torch.sin(x) + 1`

### Requirement: Free-threaded operation

On free-threaded CPython, importing the runtime and loading and launching artifacts SHALL leave the
GIL disabled, and SHALL NOT import Triton.

#### Scenario: GIL stays disabled through first launch

- **WHEN** a 3.14t process with the GIL disabled imports the runtime, loads an artifact and launches it
- **THEN** the GIL is still disabled afterwards and no `triton` module was imported

### Requirement: Integrity-checked artifact cache

Artifacts SHALL be cached under a key derived from the artifact's producer inputs and compiler
provenance. A cached artifact SHALL be reused only if its recorded integrity digests match.
Selection-policy sources SHALL NOT be part of the runtime key.

#### Scenario: Corrupted cache entry

- **WHEN** a cached artifact file has been modified after it was written
- **THEN** the runtime does not load it, and the producer recompiles or an integrity error is reported

#### Scenario: Warm reuse

- **WHEN** an artifact that is already cached and intact is requested again
- **THEN** it is loaded without invoking any compiler

#### Scenario: Policy edit does not invalidate artifacts

- **WHEN** only a selection-policy source changes
- **THEN** runtime artifact keys for unchanged producer inputs stay the same

### Requirement: Configurable cache location

Artifact cache locations SHALL be configurable by the caller or environment, and SHALL default to a
per-user cache directory rather than a path inside the source tree.

#### Scenario: Custom cache directory

- **WHEN** the caller configures a cache directory
- **THEN** all artifacts are written there and nowhere else

### Requirement: PyTorch storage and current-stream execution

Kernels SHALL read and write PyTorch-owned storage in place and SHALL be enqueued on the caller's
current CUDA stream. The runtime SHALL keep tensors alive until the work that uses them is ordered.

#### Scenario: Non-default stream ordering

- **WHEN** a producer on a non-default current stream writes a tensor, a kernel reads it, and a consumer then reads the kernel's output on the same stream
- **THEN** the consumer observes the kernel's result with no additional synchronization

### Requirement: Contract-driven launch validation

Before each launch, the runtime SHALL check tensor arguments against the artifact's declared
guards. A mismatch SHALL raise an error naming the mismatch instead of launching.

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

### Requirement: Compiler provenance and FP state

Artifacts SHALL record their compiler provenance, including the CubeCL source pin and
floating-point patch state. The runtime SHALL refuse to load artifacts whose provenance is not the
audited state.

#### Scenario: Artifact from a non-audited state

- **WHEN** an artifact records a CubeCL pin or patch state other than the audited one
- **THEN** loading is refused with an error identifying the provenance mismatch

### Requirement: Live historical runtime contracts are preserved

The runtime-facing assertions of the frozen Stage 0–5 gates that describe live behaviour SHALL hold
for the package runtime. Each such assertion SHALL be listed as ported, or as not ported with a
recorded reason.

#### Scenario: Contract inventory

- **WHEN** the gate inventory is reviewed
- **THEN** every runtime-facing assertion (bad layout, dtype, device or extents, corrupt cache, resource mismatch, stream lifetime) is either covered by a package test or listed with a reason it is historical only

### Requirement: Launch host overhead parity

The warm per-launch host time of package kernels SHALL be no worse than the Stage 5 backend's
beyond measurement noise, for the same artifacts.

#### Scenario: Warm launch microbenchmark

- **WHEN** the same prepared kernels are launched repeatedly through both implementations in alternating order
- **THEN** the package's median per-launch host time does not exceed Stage 5's by more than the measured noise
