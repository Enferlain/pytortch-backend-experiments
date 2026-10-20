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

### Requirement: Structural artifact equivalence with the Stage 5 backend

For every case in the Stage 5 fixed corpus, the backend SHALL produce the same ordered list of
regions as the Stage 5 backend at its recorded revision. Each region SHALL match in identity, role,
descriptor digest, physical plan and IR, CUDA and PTX digests, with multiplicity preserved.

#### Scenario: Corpus artifact comparison

- **WHEN** the 168-case corpus is compiled once by the Stage 5 backend and once by the package backend, in fresh processes with separate empty caches
- **THEN** each case yields identical ordered region records from both

#### Scenario: Duplicate artifacts are not collapsed

- **WHEN** two regions in a case map to byte-identical artifacts in one backend but not the other
- **THEN** the comparison reports a mismatch

### Requirement: Output equivalence under the deterministic policy

Under the `baseline` policy, the backend's outputs SHALL be bitwise identical to the Stage 5
backend's for every corpus case on identical inputs, except outputs that are not bitwise
reproducible within one implementation (float atomic accumulation with colliding indices).
Those SHALL match within the Stage 4 tolerance (2e-5).

#### Scenario: Corpus output comparison

- **WHEN** both backends run the corpus with the `baseline` policy on identical inputs
- **THEN** every output tensor that repeats bitwise within each implementation is bitwise identical across them

#### Scenario: Nondeterministic atomic output

- **WHEN** an output differs between two runs of the same implementation
- **THEN** the two implementations' outputs match within the Stage 4 tolerance

### Requirement: Selection equivalence

The set of eligible physical-plan candidates per corpus case SHALL equal the Stage 5 backend's.
Given identical injected deterministic candidate timings, the backend SHALL select the same
candidate and produce the same selection-cache key data, excluding compiler provenance (which
differs by construction). Real `first_use` winners are not required to match.

#### Scenario: Candidate sets match

- **WHEN** both backends enumerate candidates for the same corpus case
- **THEN** the candidate sets are identical

#### Scenario: Injected timings

- **WHEN** both backends' benchmarkers are replaced with the same deterministic timing table and the corpus is compiled under `first_use`
- **THEN** the selected candidate and the selection-cache key data (without compiler provenance) are identical for every case

### Requirement: Observable rejection

When the backend can't compile part of a graph, it SHALL raise a distinct unsupported-capability
error with a human-readable reason, and SHALL never silently produce a different computation.

#### Scenario: Unsupported operation

- **WHEN** a graph contains an operation the backend does not support
- **THEN** compilation fails with the unsupported-capability error, and its message names the operation or construct

#### Scenario: Scoreboard coverage parity

- **WHEN** the workload scoreboard runs its `cubecl` path in fail-fast and inventory modes with the package backend
- **THEN** the region counts and rejection reasons equal those produced with the Stage 5 backend

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
that later changes can tell whether they altered generated code.

#### Scenario: Unintended codegen change

- **WHEN** a code change alters any region record for any corpus case
- **THEN** the manifest check reports which cases, regions and files changed
