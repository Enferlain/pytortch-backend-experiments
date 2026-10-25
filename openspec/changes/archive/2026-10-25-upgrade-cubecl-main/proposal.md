## Why

The backend compiled against CubeCL `6bba5fdf` (2026-10-11), while upstream moved 82 commits in
two weeks, including #1816, which removes most of the float folds our patch stack deletes. Two
problems also came up:
- CubeCL's shared-memory pass created slices in `HashMap` order, so `kernel.cu` was not
  byte-reproducible for kernels with more than one shared array (all Welford kernels; bead `g57`,
  upstream tracel-ai/cubecl#1878). That broke the "Deterministic compilation" requirement, which
  only tested kernels with zero or one array.
- The frozen Stage 5 reference refuses any CubeCL other than `6bba5fdf`, so the Stage 5
  equivalence requirements cannot be checked after an upgrade.

## What Changes

- CubeCL moves to `main` `907676dc` (2026-10-24, the revision CubeK pins). `native/cubecl-fp.patch`
  is rebased onto it. Hunks now upstream (#1816) are dropped. The integer `x % 1` simplifiers
  (still on `main`), LLVM `frem` redirection, implicit-contraction removal, the C++ float literal
  fix and FMA-formation permission are kept. A declaration-order fix for shared-memory slices is
  added.
- Reference checkouts match what runs: `pytorch` at the installed wheel `v2.14.1`, `triton` at
  `v3.8.0`; `cubek`, `tilelang` and `KohakuMossTrainer` at `main`.
- **BREAKING (checks):** structural, output and selection equivalence with the Stage 5 backend are
  removed. Their last result was 336/336 identical at `6bba5fdf`. The manifest is the structural
  gate from here: it passed unchanged on the new CubeCL, and an upgrade re-records it only after
  every difference is reviewed. `manifest_check record` compiles the corpus itself.
- Deterministic compilation gains a cross-process scenario for kernels with several shared arrays.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `inductor-backend`: Stage 5 equivalence requirements removed; manifest reproducibility gains the
  compiler-upgrade scenario.
- `descriptor-compiler`: deterministic compilation covers separate compiler processes and
  multi-array kernels.

## Impact

- `.stage0-sources/cubecl` (patched checkout), `cubecl_torch/native/{cubecl-fp.patch,build.sh,Cargo.lock,src/main.rs}`,
  `cubecl_torch/runtime/native.py`.
- `cubecl_torch/testing`: `test_equivalence.py` removed, `manifest_check.py` records without an
  equivalence file, `test_compiler_determinism.py` and `data/welford.json` added.
- Frozen `stage0/`-`stage5/` keep their binary built at `6bba5fdf` as history.
