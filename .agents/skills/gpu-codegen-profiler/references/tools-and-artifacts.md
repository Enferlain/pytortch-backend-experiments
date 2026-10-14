# Verified tooling and artifact boundaries
Checked 2026-10-13 against official documentation and pinned source. These are small collection examples, not exhaustive profiler instructions. Reverify version/device support, executable availability and local paths; substitute actual reproducer and kernel. No tools are asserted installed in the user's environment.

## NVIDIA collection
| Interface/example | Verified source | Purpose / limit |
| --- | --- | --- |
| `nsys profile --trace=cuda,nvtx --output=timeline python repro.py` | [Nsight Systems User Guide: profile, trace, output](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | Host/GPU timeline; scope collection using existing Nsight skill |
| `ncu --set basic --kernel-name 'ACTUAL_KERNEL_NAME' --launch-count 1 --export kernel-report python repro.py` | [Nsight Compute CLI: set, kernel-name, launch-count, export](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) | Replace exact kernel name; first matched launch may be warmup, so choose intended invocation with verified filtering |
| `ptxas --verbose --gpu-name=SM_TARGET -o kernel.cubin kernel.ptx` | [CUDA Compiler Driver: PTXAS options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) and [make_cubin](https://github.com/triton-lang/triton/blob/0bea346bfbb357072cdd16fa6ebd7afaa8a0399f/third_party/nvidia/backend/compiler.py) | Replace SM_TARGET with verified architecture; offline experimental assembly, not proof of driver-JIT output |
| `cuobjdump --dump-sass --dump-resource-usage kernel.cubin` | [CUDA Binary Utilities: cuobjdump](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) | Machine code/resources for captured compatible binary |
| `nvdisasm kernel.cubin` | [CUDA Binary Utilities: nvdisasm](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) | Cubin disassembly; richer control-flow analysis available via verified options |
| `torch.profiler.profile` | [official PyTorch profiler API](https://docs.pytorch.org/docs/2.14/profiler.html) | Delegate activity/schedule selection to existing PyTorch skill; inspect matching installed version |

SM_TARGET, ACTUAL_KERNEL_NAME, report/reproducer/binary filenames are placeholders, not actual commands ready for every environment. Capture options and outputs. Do not pass CUDA compiler flags to a Rust backend or NVRTC unless its real invocation supports them. Nsight counter availability and replay support depend on device/tool/platform permissions; report unavailable evidence rather than manufacture metrics.

## Runtime versus offline compilation
CubeCL CUDA Cpp uses NVRTC to obtain PTX; CUDA Llvm emits PTX directly; both load PTX through the driver. Inspect [CudaArtifactCompiler](https://github.com/tracel-ai/cubecl/blob/3b47011522dbe79544388267c2b6cf8278913a8b/crates/cubecl-cuda/src/compute/artifact.rs) and [CudaModules::load](https://github.com/tracel-ai/cubecl/blob/3b47011522dbe79544388267c2b6cf8278913a8b/crates/cubecl-cuda/src/compute/modules.rs). Do not label the loaded code “ptxas output” without evidence of that actual path.

Use `nvcc` only if the real build uses it, or as an explicitly separate controlled source-compilation experiment; locate current flags from the [compiler manual](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html). NVRTC has its own [documented API/options](https://docs.nvidia.com/cuda/nvrtc/index.html). Inspecting an offline cubin can explain an assembler hypothesis but cannot stand in for an uncaptured runtime binary.

## AMD and portable artifacts
`rocprofv3 --hip-trace --kernel-trace -- ./repro` combines host HIP and kernel-dispatch tracing; `repro` is an actual executable chosen by the investigator. Verified in [current ROCprofiler-SDK tracing guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html). Preserve the actual output paths/format reported by the installed version; current docs use rocpd by default and distinguish HIP API from kernel tracing. Hand counter/PC-sampling/thread-trace configuration to target-specific skills/docs. Do not reuse Nsight or PTX/SASS commands for AMD.

CubeCL LLVM debug paths can provide AMDGPU LLVM IR/assembly and linked code objects; Cpp HIP uses HIPRTC. Save code object, ABI/HSA metadata, compile/link diagnostics and actual launch resource data where accessible. Verify an installed compatible disassembler before giving its command.

## Artifact request by stage
| Stage | Preserve | Distinguish |
| --- | --- | --- |
| Framework/compiler scheduling | Captured/transformed graph, pre/post fusion, wrapper | Captured operator versus generated launch |
| CubeCL/Pliron | Modes/IR and after-pass dumps | Dialect operations versus infrastructure |
| Backend | Source, native LLVM/target IR, emitted PTX/assembly | Source compiler versus LLVM selection |
| Final compilation/load | Tool identity/options, diagnostics, binary/cache key | Offline compiler versus driver-JIT |
| Hardware | Correctness outputs, unprofiled samples, trace/counter reports | Observations versus contract/causality |

Use lowering-trace's capture reference through that skill for exact project dump interfaces. If only this skill is installed, locate current compiler debug hooks from source instead of assuming the companion exists.
