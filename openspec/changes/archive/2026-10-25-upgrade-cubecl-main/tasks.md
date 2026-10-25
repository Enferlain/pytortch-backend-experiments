## 1. Reference checkouts

- [x] 1.1 Pull `cubek`, `tilelang` and `KohakuMossTrainer` to `main`. Check out `pytorch` at the installed wheel commit `5c48869` (`v2.14.1`) and `triton` at `v3.8.0`. Update the `.stage0-sources` README table.

## 2. CubeCL upgrade

- [x] 2.1 Move `.stage0-sources/cubecl` to `907676dc`. 3-way apply `native/cubecl-fp.patch`, take upstream's text where #1816 made the same change, and save the rebased patch.
- [x] 2.2 Fix shared-memory slice order: `SharedLiveness` keeps declaration order, `alloc_shared_memory` iterates it (bead `g57`).
- [x] 2.3 Update the pin in `native/build.sh`, `runtime/native.py` and the compiler manifest. Update `Cargo.lock` minimally. Remove the unused historical patch copies.
- [x] 2.4 Verify: clippy `-D warnings` (default, `hip-probe`, `llvm`); FP rewrite suite 7/7; a 244-case descriptor matrix byte-identical across compiler processes (165/165 compiled); full suite green apart from the retired equivalence tests; manifest PASS.

## 3. Checks

- [x] 3.1 Add a cross-process determinism test for a three-array Welford kernel.
- [x] 3.2 Retire Stage 5 equivalence: remove `test_equivalence.py`; `manifest_check record` compiles the corpus itself; update `CLAUDE.md`, the package README and the contract inventory.

## 4. Upstream

- [x] 4.1 Comment on tracel-ai/cubecl#1812 (float folds fixed by #1816; integer `x % 1` still live) and #1878 (cause, CUDA repro, fix).
