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
| [cubecl](cubecl/) | [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl) | detached `6bba5fdf` (`v0.11.0-25-g6bba5fdf`) | 2026-10-11 | 2026-10-12 |
| [cubek](cubek/) | [tracel-ai/cubek](https://github.com/tracel-ai/cubek) | `main` @ `4ca0235e` (`v0.3.0-46-g4ca0235e`) | 2026-10-14 | 2026-10-15 |
| [pytorch](pytorch/) | [pytorch/pytorch](https://github.com/pytorch/pytorch) | detached `e342e9a9d7` (from `main`) | 2026-10-11 | 2026-10-12 |
| [tilelang](tilelang/) | [tile-ai/tilelang](https://github.com/tile-ai/tilelang) | `main` @ `fe6d319c` (`v0.1.15-66-gfe6d319c`) | 2026-10-16 | 2026-10-18 |
| [triton](triton/) | [triton-lang/triton](https://github.com/triton-lang/triton) | `main` @ `4eda6cc91e` | 2026-10-14 | 2026-10-15 |
| [KohakuMossTrainer](KohakuMossTrainer/) | [KohakuBlueleaf/KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer) | `main` @ `001b3aa` (`v0.1.0-13-g001b3aa`) | 2026-10-18 | 2026-10-18 |

## Summaries

- **cubecl** — Rust GPU compute framework from the tracel-ai (Burn) stack. Defines the
  `cube` kernel language and IR, compiling to CUDA (NVRTC / LLVM PTX), wgpu and CPU.
  The backend under study in `stage0/` and `stage5/` is built on it.
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

`cubecl` carries uncommitted local changes (math dialect, LLVM/NVPTX codegen and opt
pass areas — likely from earlier experiment passes). Do not reset or commit them from
the parent repo without checking the stage experiment notes:

- `crates/cubecl-cpp/src/shared/value.rs`
- `crates/cubecl-ir/src/dialect/math.rs`
- `crates/cubecl-llvm/src/nvptx/codegen.rs`
- `crates/cubecl-llvm/src/shared/math_library.rs`
- `crates/cubecl-llvm/src/shared/to_llvm/math.rs`
- `crates/cubecl-opt/src/passes/inst_combine.rs`

All other checkouts are clean at the commits listed in the table above.
