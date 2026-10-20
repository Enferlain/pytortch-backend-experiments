# Proposal

## Why

The live CubeCL backend is spread across six experiment directories. Its code is 33 Python modules
(about 5,600 lines) in a `stage5 → stage4 → stage3 → stage2 → stage1` subclass chain, plus the
`stage0` native launcher and Rust compiler and the `fp_hardening` patch stack. Around it sit
roughly 160 experiment harnesses in `stage5/` alone. The next coverage work, shown necessary by the
workload scoreboard (beads `4z8.6`, `4z8.7`, `4z8.15`), adds new kernel arguments and reduction
kinds. If it lands in the subclass chain, consolidation only gets more expensive later.

The project direction also requires the CubeCL kernel machinery to serve two callers: the
`torch.compile` backend, and hand-written or CubeK-based custom ops used next to `torch.compile`.
Today nothing below Inductor can be used without going through Inductor. The user chose to
consolidate before adding capabilities (beads `sil.5`).

## What Changes

- Add one importable package, working name `cubecl_torch`, with four layers:
  - **Runtime.** A generic kernel-artifact contract: PTX and entry point, plus declared
    parameters, tensor guards and launch shape. Also the integrity-checked artifact cache, native
    launch, validation driven by the artifact's declared contract, PyTorch storage, current-stream
    and lifetime rules, and compiler provenance/FP-state checks. It knows nothing about descriptors,
    Inductor or selection policy, so artifacts from other producers (CubeK kernels later) can use it.
  - **Descriptor compiler.** The semantic descriptor schema, the indexing/expression model, the
    physical-plan representation, the adapter to the Rust `cubecl-stage0` compiler and NVRTC, and
    the binding that turns a descriptor's symbolic sizes, strides and offsets into an artifact's
    launch contract.
  - **Inductor front end.** It holds the scheduling, kernel and ops classes from the stage
    subclass chain. Inheritance links are kept but regrouped by responsibility, not flattened. It
    also owns capability checks, candidate generation, plan legality and admission, reduction
    policy and qualification, the final, column and reference512 plans and oracle, first-use
    tuning, and the PyTorch overlay installer.
  - **Custom-op front end.** It registers kernels as `torch.library` ops, usable from eager code
    and as opaque extern calls inside compiled graphs.
- Copy the native sources (Rust crate, C launcher, CubeCL patch files) into the package's
  `native/` directory as the live build. `stage0/` stays as a frozen, still-buildable copy for
  the historical verifiers.
- Make the artifact and tuning cache locations configurable instead of fixed under `stage5/cache`.
- Add an equivalence check against the current `stage5` backend:
  - the 168-case corpus must produce structurally identical artifacts: for each case, the same
    ordered regions with the same roles, descriptors, plans and IR, CUDA and PTX bytes;
  - outputs must be bitwise equal under the deterministic policy;
  - selection must be identical under injected deterministic timings;
  - warm per-launch host time must be no worse beyond noise;
  - the GIL must stay off;
  - the scoreboard's coverage results must be identical.
- Inventory the old gate scripts' assertions and port the ones that are live runtime or rejection
  contracts (bad layout, dtype, device or extents, corrupt cache, resource mismatch, malformed
  descriptors, stream lifetime, rejection errors). Record why each remaining assertion isn't
  ported.
- Switch the scoreboard's `cubecl` path to the new package.
- Add one proof custom op, built from a hand-authored descriptor, to show the runtime and
  descriptor compiler work without Inductor. Add a runtime test that launches an artifact not
  produced by the descriptor compiler, to keep the seam for CubeK-based kernels real. The artifact
  contract stays minimal and is expected to change when the first CubeK op arrives.
- No kernel behaviour, numerical policy, eligibility or selection policy changes. Stage directories
  aren't modified or deleted.

## Capabilities

### New Capabilities

- `kernel-runtime`: caches, validates and launches compiled kernel artifacts on PyTorch-owned
  storage and the current stream, through a producer-independent artifact contract. It checks
  compiler provenance and FP state, and keeps the GIL disabled.
- `descriptor-compiler`: compiles semantic descriptors and physical plans into kernel artifacts
  deterministically from the audited CubeCL source. It binds descriptor symbols into the
  artifact's launch contract and rejects malformed descriptors.
- `inductor-backend`: the `torch.compile` backend built on the descriptor compiler and runtime.
  It covers capability rejection with observable reasons, structural equivalence with Stage 5,
  selection under injected timings, overlay confinement, and reproducibility against a recorded
  manifest.
- `custom-op-frontend`: registers kernels as `torch.library` custom ops that work from eager code
  and as opaque extern calls inside compiled graphs, with no dependency on the overlay or Inductor
  internals.

### Modified Capabilities

None. No specs exist yet in `openspec/specs/`.

## Impact

- **New code:** `cubecl_torch/` (Python package plus `native/`). New project metadata for the
  package (`pyproject.toml`) and one test entry point.
- **Scoreboard:** `scoreboard/run_one.py` imports the new backend instead of `stage5.backend`.
- **Unchanged:** every `stage*/` and `fp_hardening/` module and their artifacts, which remain the
  historical record and the equivalence reference. The existing regression runners keep using
  them.
- **Dependencies:** the same pinned CubeCL checkout (`.stage0-sources/cubecl` at `6bba5fdf`) with
  the same patch stack, CUDA 13.0 NVRTC and the same PyTorch wheel commit for the overlay.
- **Follow-up work that builds on this:** `4z8.6`, `4z8.7`, `4z8.15`, `4z8.3` (fallback) and `s2m`
  (native Windows). The package must not add new Linux-only assumptions, though porting isn't part
  of this change.
