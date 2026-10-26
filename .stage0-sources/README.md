# .stage0-sources — reference repositories

Upstream checkouts kept as reference material for the experiments in this repo.
The directory is git-ignored from the parent repository (see [`.gitignore`](.gitignore));
only this README is meant to be tracked. Note that `stage0/` builds directly against
the local `cubecl` checkout via path dependencies, so that one is a live dependency,
not just reading material.

Checkouts are not auto-updated — verify the pinned commit before citing exact line
numbers or behavior.

| Repo | Upstream | Checkout | Upstream commit date | Cloned |
| ------ | ---------- | ---------- | ---------------------- | -------- |
| [cubecl](cubecl/) | [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl) | detached `907676dc` (`main`, the revision cubek pins) | 2026-10-24 | 2026-10-12, updated 2026-10-25 |
| [cubek](cubek/) | [tracel-ai/cubek](https://github.com/tracel-ai/cubek) | `main` @ `08659297` (`v0.3.0-151-g08659297`) | 2026-10-24 | 2026-10-15, updated 2026-10-25 |
| [pytorch](pytorch/) | [pytorch/pytorch](https://github.com/pytorch/pytorch) | detached `5c48869085` (`v2.14.1`, the installed wheel's commit) | 2026-09-29 | 2026-10-12, updated 2026-10-25 |
| [tilelang](tilelang/) | [tile-ai/tilelang](https://github.com/tile-ai/tilelang) | `main` @ `c58a86f0` (`v0.1.15-111-gc58a86f0`) | 2026-10-23 | 2026-10-18, updated 2026-10-25 |
| [triton](triton/) | [triton-lang/triton](https://github.com/triton-lang/triton) | detached `c01b6774b1` (`v3.8.0`, the Triton reference venv's version) | 2026-08-28 | 2026-10-15, updated 2026-10-25 |
| [KohakuMossTrainer](KohakuMossTrainer/) | [KohakuBlueleaf/KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer) | `main` @ `da5dd97` (`v0.1.0-73-gda5dd97`) | 2026-10-21 | 2026-10-18, updated 2026-10-25 |

`pytorch` and `triton` match the versions that actually run (the `stage0/.venv` wheel and the
Triton reference venv), so their source answers "how does the pinned Inductor/Triton do X";
moving either is part of upgrading that wheel, not a pull.

## Summaries

- **cubecl** — Rust GPU compute framework from the tracel-ai (Burn) stack. Defines the
  `cube` kernel language and IR, compiling to CUDA (NVRTC / LLVM PTX), wgpu and CPU.
  The live backend `cubecl_torch` builds on it; the frozen `stage0/`–`stage5/` history was built on `6bba5fdf`.
- **cubek** — Kernel library on top of CubeCL from the same organization: matmul/GEMM
  families, random generation, and related ops. Reference for kernel strategies and
  idiomatic CubeCL API usage.
- **pytorch** — PyTorch itself. Integration target for the experimental backend:
  dispatcher and tensor semantics, plus reference numerics and kernels.
- **tilelang** — "A tile level programming language to generate high performance code":
  a Python-embedded DSL over TVM for writing tiled GPU kernels with autotuning.
  Reference for tiling and scheduling approaches.
- **triton** — The Triton language and compiler: a Python block-level GPU kernel DSL
  over an MLIR/LLVM pipeline. Reference for kernel DSL design, layouts and codegen.
- **KohakuMossTrainer** — "Finetuning framework for MOSS-Image: LyCORIS algorithms and
  full finetune, per-arch kernels, fp16/bf16, offload, multi-GPU, Windows + Linux."
  Real-world training workload reference.

## Local modifications

`cubecl` carries the audited patch stack `cubecl_torch/native/cubecl-fp.patch` as uncommitted
changes; `cubecl_torch/native/build.sh` refuses to build unless the working tree diff equals it.
Do not reset or commit them from the parent repo. To move the pin, save the diff, check out the
new commit, `git apply --3way` the patch, resolve, and update the patch file, `build.sh` and
`cubecl_torch/runtime/native.py` together; then run the checks in `cubecl_torch/README.md`, including
`native_matrix` against a build of the previous pin. The patched files:

- `crates/cubecl-cpp/src/shared/binary.rs`
- `crates/cubecl-cpp/src/shared/value.rs`
- `crates/cubecl-ir/src/dialect/math.rs`
- `crates/cubecl-llvm/src/nvptx/codegen.rs`
- `crates/cubecl-llvm/src/shared/math_library.rs`
- `crates/cubecl-llvm/src/shared/to_llvm/math.rs`
- `crates/cubecl-opt/src/analyses/liveness.rs`
- `crates/cubecl-opt/src/passes/alloc_shared_memory.rs`
- `crates/cubecl-opt/src/passes/inst_combine.rs`

The frozen `stage0/` binary (`stage0/target/release/cubecl-stage0`) was built at the previous
pin `6bba5fdf` and is kept as history; rebuilding it against the current checkout is not
supported.

All other checkouts are clean at the commits listed in the table above.
