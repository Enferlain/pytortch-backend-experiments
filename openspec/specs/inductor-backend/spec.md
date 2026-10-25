# inductor-backend Specification

## Purpose
Lets `torch.compile` run Inductor-generated kernels through CubeCL instead of Triton. It reuses
Inductor's graph capture, fusion and scheduling, and compiles kernels through the descriptor
compiler before running them on the runtime.

## Requirements

### Requirement: torch.compile backend entry point

The package SHALL expose a backend callable accepted by `torch.compile(backend=...)`, with the
existing physical-plan policies (`first_use`, `baseline` and the diagnostic policies) selectable as
before.

#### Scenario: Compile with the default policy

- **WHEN** a supported function is compiled with the package backend and called
- **THEN** its generated kernels execute through the runtime, and the results match eager PyTorch under the established numerical policy

### Requirement: Observable rejection

When the backend can't compile part of a graph, it SHALL raise a distinct unsupported-capability
error with a human-readable reason, and SHALL never silently produce a different computation.

#### Scenario: Unsupported operation

- **WHEN** a graph contains an operation the backend does not support
- **THEN** compilation fails with the unsupported-capability error, and its message names the operation or construct

#### Scenario: Inventory lists every reason

- **WHEN** a graph is compiled with `rejections="all"` (the scoreboard's inventory mode) and has several unsupported features, in one or more nodes or in its forward and backward graphs
- **THEN** it is rejected once, and the error's `reasons` list every distinct reason found before each graph's pre-fusion boundary; with the default `rejections="first"` it is rejected at the first reason

#### Scenario: Scoreboard coverage parity

- **WHEN** the workload scoreboard runs its `cubecl` path in fail-fast mode with the package backend
- **THEN** the region counts and first rejection reasons equal those produced with the Stage 5 backend (inventory mode lists more reasons per region since 2026-10-25)

### Requirement: Live historical front-end contracts are preserved

The rejection and admission assertions of the frozen Stage 0–5 gates that describe live behaviour
SHALL hold for the package backend. Each such assertion SHALL be listed as ported, or as not
ported with a recorded reason.

#### Scenario: Contract inventory

- **WHEN** the gate inventory is reviewed
- **THEN** every front-end assertion (rejection reasons, admission guards, descriptor/plan validation, stream and offset controls) is either covered by a package test or listed with a reason it is historical only

### Requirement: Free-threaded compilation path

Compiling and running graphs through the backend on free-threaded CPython SHALL leave the GIL
disabled, and SHALL NOT import Triton.

#### Scenario: Scoreboard GIL check

- **WHEN** a scoreboard workload compiles and runs on the `cubecl` path
- **THEN** the recorded GIL state is disabled at every phase

### Requirement: Overlay confinement

PyTorch overlay installation SHALL happen only when the Inductor backend is requested, SHALL refuse
a PyTorch build other than the recorded wheel commit, and SHALL NOT modify the installed PyTorch
files.

#### Scenario: Unrecorded PyTorch build

- **WHEN** the backend is requested in an environment whose PyTorch git version differs from the recorded commit
- **THEN** installation is refused with an error, before any Inductor module is imported

#### Scenario: Installed files untouched

- **WHEN** the overlay is installed and a graph is compiled
- **THEN** the digests of every file in the installed `torch/_inductor` directory are identical before and after

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
