# Artifact capture and verified interfaces
Verified 2026-10-13. Check relevant source and local capabilities before running examples. Replace filenames/interpreters with actual reproducer locations; filenames below are user-created examples. No GPU execution is claimed by this reference.

## PyTorch
Example shell capture for an existing `repro.py`:
```sh
TORCH_LOGS="graph_code,aot_graphs,output_code,schedule,fusion" TORCH_COMPILE_DEBUG=1 python repro.py
```
Verified keys: [logging registrations](https://github.com/pytorch/pytorch/blob/60de3b7f3316d28ab80fb8029742e201d5c6ab7b/torch/_logging/_registrations.py); trace enable/defaults: [trace config](https://github.com/pytorch/pytorch/blob/60de3b7f3316d28ab80fb8029742e201d5c6ab7b/torch/_inductor/config.py); writers: [DebugFormatter](https://github.com/pytorch/pytorch/blob/60de3b7f3316d28ab80fb8029742e201d5c6ab7b/torch/_inductor/debug.py). Select fewer keys if possible. `fx_graph_transformed.py`, `ir_pre_fusion.txt`, `ir_post_fusion.txt` and `output_code.<extension>` are trace artifacts when the relevant path/writer runs; do not promise every file on cache hits or every backend. Generated wrapper code can contain kernels and external calls, not only Triton.
Context7 /pytorch/pytorch supplied observability docs and old troubleshooting examples; these keys/writers were cross-checked in pinned source.

## CubeCL
- Enable the applicable `pliron-dump` Cargo feature and set `CUBECL_DEBUG_PLIRON` to a fresh directory. `PMConfig` and the CPP/LLVM compiler implementations control per-pass dumps. Feature propagation: [CUDA manifest](https://github.com/tracel-ai/cubecl/blob/3b47011522dbe79544388267c2b6cf8278913a8b/crates/cubecl-cuda/Cargo.toml), [CPP manifest](https://github.com/tracel-ai/cubecl/blob/3b47011522dbe79544388267c2b6cf8278913a8b/crates/cubecl-cpp/Cargo.toml), [LLVM manifest](https://github.com/tracel-ai/cubecl/blob/3b47011522dbe79544388267c2b6cf8278913a8b/crates/cubecl-llvm/Cargo.toml). Setting the variable alone is not a guarantee of per-pass dumps.
- [LLVM debugging](https://github.com/tracel-ai/cubecl/blob/3b47011522dbe79544388267c2b6cf8278913a8b/crates/cubecl-llvm/README.md) verifies `CUBECL_DEBUG_PLIRON=./debug cargo test -p cubecl-cpu`, feature example `cargo test -p cubecl-cpu --features cubecl-llvm/pliron-dump`, and offline GPU-lowering tests `cargo test -p cubecl-llvm --features nvptx,amdgpu`. These are upstream examples; select an actual test/reproducer and target. Offline lowering tests do not validate execution/ABI correctness on hardware.
- LLVM target artifacts documented there include numbered after-pass `.plir`, CPU `llvm.ll` / `llvm.opt.ll`, AMDGPU `amdgpu.ll` / `amdgpu.s`, NVPTX `nvptx.ll` / `nvptx.ptx`. Read code to establish whether each dump is before/after native optimization and which feature/environment gates apply.
- `CUBECL_CUDA_DUMP_PTX` is verified in [dump_ptx](https://github.com/tracel-ai/cubecl/blob/3b47011522dbe79544388267c2b6cf8278913a8b/crates/cubecl-cuda/src/compute/modules.rs); it writes the PTX handed to the driver for both CUDA routes, including load-time cache behavior. Archive unique run directories; generated names may collide across captures. Do not equate PTX with loaded machine code.
- For SPIR-V, [debugging docs](https://github.com/tracel-ai/cubecl/blob/3b47011522dbe79544388267c2b6cf8278913a8b/crates/cubecl-spirv/README.md) describes `pliron-dump` plus `CUBECL_DEBUG_PLIRON`. Do not invent an analogous HIP binary-dump environment variable.

## Triton
`CompiledKernel.asm` stores available stage artifacts; `AsmDict` lazily handles NVIDIA `sass` via `cubin`: [CompiledKernel / AsmDict](https://github.com/triton-lang/triton/blob/0bea346bfbb357072cdd16fa6ebd7afaa8a0399f/python/triton/compiler/compiler.py). Discover actual keys from the selected backend `add_stages` and retained artifacts. Cache policy can suppress intermediates. Save compiler options, metadata and binaries, not only pretty IR.

## No convenient dump
Read current source for pass-manager/configuration and emission points. In read-only tasks deliver conditional source findings and an instrumentation design. If edits are authorized, insert a minimal boundary observer, preserve its diff and verify it does not alter optimization flags/IR or timing. Do not invent an “all stages” command.
