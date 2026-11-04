# .stage0-sources — reference repositories

Upstream checkouts kept as reference material for the experiments in this repo.
The directory is git-ignored from the parent repository (see [`.gitignore`](.gitignore));
only this README is meant to be tracked. `cubecl_torch` builds directly against the local
`cubecl` and `cubek` checkouts, so those two are live dependencies, not just reading material.

Checkouts are not auto-updated — verify the pinned commit before citing exact line
numbers or behavior.

| Repo | Upstream | Checkout | Upstream commit date | Cloned |
| ------ | ---------- | ---------- | ---------------------- | -------- |
| ~~cubecl~~ | superseded: the code we build and develop is `sources/cubecl` in this repository (2026-11-04, bead `zah.12`; `sources/LOCK.toml`, `sources/NOTES.md`) | | | |
| ~~cubek~~ | superseded: the code we build and develop is `sources/cubek` in this repository (2026-11-04, bead `zah.12`; `sources/LOCK.toml`, `sources/NOTES.md`) | | | |
| [pytorch](pytorch/) | [pytorch/pytorch](https://github.com/pytorch/pytorch) | detached `5c48869085` (`v2.14.1`, the installed wheel's commit) | 2026-09-29 | 2026-10-12, updated 2026-10-25 |
| [tilelang](tilelang/) | [tile-ai/tilelang](https://github.com/tile-ai/tilelang) | `main` @ `c58a86f0` (`v0.1.15-111-gc58a86f0`) | 2026-10-23 | 2026-10-18, updated 2026-10-25 |
| [triton](triton/) | [triton-lang/triton](https://github.com/triton-lang/triton) | detached `c01b6774b1` (`v3.8.0`, the Triton reference venv's version) | 2026-08-28 | 2026-10-15, updated 2026-10-25 |
| [burn](burn/) | [tracel-ai/burn](https://github.com/tracel-ai/burn) | `main` @ `35589bf` (shallow; pins cubecl `fdba0a84`, cubek `06e1c8ec`) | 2026-11-02 | 2026-11-02 |
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
- **burn** — Deep-learning framework from the same team, built on CubeCL's runtime and CubeK.
  Reference for how a framework drives CubeK: `burn-cubecl-fusion` fuses element-wise work into
  CubeK reduce and matmul kernels (`FusedReduceArgs` implements CubeK's `ReduceArgs`) and tunes
  CubeK routines with CubeCL's `LocalTuner`.
- **KohakuMossTrainer** — "Finetuning framework for MOSS-Image: LyCORIS algorithms and
  full finetune, per-arch kernels, fp16/bf16, offload, multi-GPU, Windows + Linux."
  Real-world training workload reference.

## Local modifications

None here any more. CubeCL and CubeK, with our changes, are `sources/cubecl` and `sources/cubek`
in this repository (2026-11-04, bead `zah.12`): see `sources/NOTES.md` for what we changed and how
upstream updates come in. The `cubecl` and `cubek` checkouts in this folder are superseded; the
others are read-only references.

