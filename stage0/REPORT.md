# Stage 0: CubeCL compiler on PyTorch CUDA storage

Experiment date: 12 October 2026, Europe/Budapest.

## 1. Result

**Stage 0 works through both compiler routes.** A kernel built programmatically
with CubeCL IR computes `y = sin(x) + 1` on ordinary PyTorch-owned CUDA buffers,
using the stream PyTorch considers current. There are no intermediate x/y copies,
CubeCL-managed device allocations, Triton compilation, or per-kernel extension
builds. The C launcher enqueues work without any synchronization call.

Both routes passed sizes 0, 1, 255, 256, 257 and 1,048,583, output canaries,
NaN/±inf/signed-zero/ordinary/extreme inputs, delayed producer→kernel→consumer
checks on two changing non-default streams, five allocator-pressure trials,
CUDA Graph capture and three replays with changed input, and twenty launches
from two barrier-started threads on independent streams. Float32 tolerance:
`rtol=2e-6, atol=2e-6, equal_nan=True`.

Both passed on CPython **3.14.5t with the GIL disabled throughout**, with Triton
unavailable to import. This is evidence for this bounded bridge/test workload;
it does not establish general concurrent compilation or training safety.

Evidence: [C++ results](artifacts/cpp/results.json),
[LLVM results](artifacts/llvm/results.json), [test code](verify.py),
[route timing](artifacts/comparison.json).

## 2. Source versions and assumption validation

| Component | Exact version used |
|---|---|
| CubeCL source and compiled crates | `6bba5fdf04206d9c3710dfbc46aa9850450ab405`, workspace 0.11.0 |
| PyTorch investigation source | `e342e9a9d7123c2b78940da0ea3fdb0dfa9beeb0` |
| Executed PyTorch wheel | `2.14.1+cu130`, commit `5c4886908584029761b579af026dcfb627c84070` |
| Python / Rust | CPython 3.14.5t / rustc and cargo 1.97.1 |
| NVIDIA target | CUDA device 0 in PyTorch: GeForce RTX 3090, SM 8.6, WSL driver 610.43.02 |
| NVRTC | CUDA 13.0, compiler V13.0.88; compute_86, C++17, fmad disabled |
| LLVM | CubeCL bundled LLVM `23.1.0-3`; pliron/pliron-llvm 0.19.0 |
| LLVM math library | NVIDIA `nvidia-nvvm==13.0.88`, standalone libdevice, supplied with CUDA_PATH |

The installed wheel is deliberately used for execution: rebuilding the report's
full PyTorch head is unnecessary for this direct pointer/stream experiment.
The report source and runtime wheel commits are distinct, recorded above, and
must not be conflated. Dependencies are fixed by [Cargo.lock](Cargo.lock).

| Investigation assumption | Classification | Evidence / qualification |
|---|---|---|
| KernelBuilder usable without `#[cube]` | Confirmed by running code | [src/main.rs](src/main.rs) registers two buffers directly |
| Scope/dialect body construction | Confirmed by running code | ReadBuiltin, BufferLen, unsigned comparison, branch, index/load/sin/add/store |
| KernelDefinition/public compiler API | Confirmed by running code | builder.build() → Compiler::compile() on both compilers |
| CUDA C++→NVRTC→PTX | Confirmed by running code | Unmodified generated .cu and final .ptx retained |
| LLVM→NVPTX | Confirmed by running code | PlironCompiler produces NvptxModule; final PTX loads and runs |
| Three-pointer compiler ABI in chosen mode | Confirmed by running code | Signatures inspected and rejected if not three 64-bit parameters; internal metadata word order is source-derived |
| Public runtime artifact finalizer | Confirmed by source only: private | CudaArtifactCompiler is pub(crate); independent NVRTC adapter is necessary |
| BufferArg raw parts imports a Tensor pointer | Confirmed by source only: it does not | Argument is a managed Handle; GPU storage resolves an allocation-map identity |
| Direct PyTorch pointer interop | Confirmed by running code | data_ptr() values supplied as host u64 argument storage; output views/canaries pass |
| Current-stream launch | Confirmed by running code | Both delayed side-stream pipelines pass with one loaded module |
| Allocator lifetime | Confirmed by running code for tested contract | Explicit record_stream; release references, enqueue clone consumer and pressure allocations |
| CUDA Graph compatibility | Confirmed by running code after wrapper correction | Warm metadata on capture stream; native launch captures/replays |
| Triton independence / 3.14t | Confirmed by running code | No importable Triton, no attempted imports or loaded native Triton, all GIL checks false |
| Other metadata modes, devices and generalized IR | Still untested | No grid constants, scalars, tensors/strides, reductions or non-SM86 validation |

Graph discovery did not return the cloned Rust symbols. Verification therefore
used exact checked-out source. Coverage reported partial parsing in CUDAStream.h
and CUDACachingAllocator.h; the reported declaration ranges were read directly.
Coverage's clean signal on other paths was not treated as proof of completeness.

## 3. Exact compiler paths

The out-of-tree compiler executable is built once. Its `build()` constructs:

```text
KernelBuilder::new(KernelSettings{256×1×1, Unchecked, U64})
→ buffer(f32), buffer(f32)
→ ReadBuiltinOp(AbsolutePos), BufferLenOp(buffer 0)
→ ULessThanOp(i,N), branch::IfOp
→ then Scope: IndexOp → LoadOp → SinOp → ConstantOp(1f32)
               → FAddOp → IndexOp → StoreOp → YieldOp
→ else Scope: YieldOp
→ ReturnOp
→ KernelBuilder::build → create_info → KernelIntegrator::integrate
→ KernelDefinition
```

`Unchecked` disables redundant compiler bounds instrumentation. The explicit
`i<N` branch contains **both memory accesses**. Scalar vector width is one;
AddressType::U64 governs index/metadata width. N is dynamic metadata, not baked
into the compiled body, so the same PTX serves every tested size.

Route A: public `CppCompiler<Cuda>::compile` → `compile_ir` → verification and
Pliron passes, including LowerInfo/CheckedIo/builtin/target lowering → source
emission → [bridge.py](bridge.py) `nvrtcCompileProgram` → `nvrtcGetPTX`.

Route B: public `PlironCompiler::compile` → `compile_ir` → `compile_nvptx`
→ common lower() and PtxKernelParams ABI lowering → `nvptx::codegen::emit_ptx`
→ LLVM NVPTX target emission, redirecting unsupported math intrinsics to
NVIDIA libdevice. No CubeCL CUDA runtime is initialized.

The saved `kernel.ll` is CubeCL's pre-final-target LLVM logging representation:
it still contains `llvm.sin.f32`. The final PTX includes the linked math
implementation. It is not claimed to be a dump after every LLVM/native pass.

Each route directory retains `cubecl.ir`, `info.txt`, `kernel.cu` or `kernel.ll`,
`kernel.ptx`, `manifest.json`, `metadata-257.bin`, `ptx-signature.txt`, and
`results.json`. The C++ directory additionally retains NVRTC options/log/timing.

## 4. Measured kernel ABI

The emitted C++ signature, excluding local SSA variable names, is:

```cpp
struct info_st { uint64_t static_meta[2]; };
extern "C" __global__ void __launch_bounds__(256) cubecl_sin_add(
    float const* const __restrict__ input,
    float* const __restrict__ output,
    info_st const* const __restrict__ info);
```

The exact emitted signature is in [kernel.cu](artifacts/cpp/kernel.cu).

| Parameter | Host argument storage passed to driver | Device meaning |
|---|---|---|
| 0 | Address of u64 containing x.data_ptr() | Read-only float32 input |
| 1 | Address of u64 containing y.data_ptr() | Write-only float32 output |
| 2 | Address of u64 containing metadata.data_ptr() | Two native-endian u64 lengths in elements |

InfoBuilder packs `[N,N]` into 16 bytes with static metadata at offset zero,
dynamic-metadata offset 16, no scalar fields and no dynamic metadata.
Because both lengths equal N, runs cannot distinguish a swap of metadata words.
Word order and field-zero selection are established from source and emitted IR,
not independently distinguished by numerical output.
Metadata is generated by the Rust library on setup, allocated/uploaded by
PyTorch once per N. The compiler uses field zero for this kernel's bound.

Block: `(256,1,1)`; grid: `(ceil(N/256),1,1)`. N=0 skips cuLaunchKernel.
Static shared memory: absent; dynamic shared memory: **0 bytes** on both routes.
No cluster, scalar argument, grid constant, tensor map or stride buffer.

Both PTX entries have exactly three `.param .u64` arguments in the same order.
LLVM adds `.ptr .global .align 1` annotations and `.reqntid 256,1,1`;
C++/NVRTC emits PTX 9.0, LLVM emits PTX 7.1. Both target SM86. These differences
do not change the driver argument packing. C++ ComputeKernel carries source,
shared-memory size, buffers and read/write attributes, but no full standalone
manifest/entrypoint field. LLVM NvptxModule additionally carries its entrypoint,
PTX and LLVM logging IR. The fixed manifest records settings/Info and is checked
against the actual PTX; it is not a proposed permanent compiler ABI.

## 5. Runtime, ordering and lifetime contract

```text
torch.Tensor → public data_ptr()/device
→ Python keeps x/y references during enqueue and calls record_stream
→ current_stream(device).cuda_stream, retrieved anew for every invocation
→ ctypes → prebuilt launcher.c
→ current CUcontext checked against the module's load context
→ host argument values → cuLaunchKernel on supplied PyTorch stream
```

The public stream getter reaches native `getCurrentCUDAStream`; source also
exposes `Stream.__cuda_stream__()` as a supported stream protocol. The launcher
uses Driver cuCtxGetCurrent, so it neither creates another context nor borrows
CubeCL's streams. `cuModuleLoadData`/`cuModuleGetFunction` execute only in
preparation. A loaded object is tied to its context; the launcher rejects a
different current context. No context cache shared across devices is assumed.

Producer, CubeCL kernel and consumer order follows the same stream. Cross-stream
data readiness is the caller's normal wait/event responsibility. The bridge's
record_stream calls protect input/output/metadata storage against allocator reuse
after Python references disappear; they do **not** establish data dependencies.
Same-allocation-stream use already has allocator ordering, while foreign-stream
use needs recording or an equivalent explicit lifetime arrangement.

The pressure test allocates on default, delays a side stream by 30 million GPU
cycles, enqueues kernel and clone consumer, deletes input/output references,
then writes sixteen same-sized allocations on default: approximately 134 MB per
trial. Five trials per route remain correct. A completion event distinguishes
addresses recycled after side-stream work from reuse observed before completion;
the test requires the latter count to be zero. Address recycling alone is not
proof of premature reuse, and querying after allocation cannot exclude a race
that has already completed. Numerical output and explicit allocator recording
are therefore also part of the evidence.
This is a focused stress test, not an exhaustive allocator/stream audit.

Immutable metadata readiness is checked with a nonblocking event query; if it
has not completed, the current stream waits on the upload event outside capture.
Capture requires completed setup first. No raw stream handles are memoized.
`prepare_capture(N)` waits only on the metadata-upload event outside capture;
the completion flag then applies to all streams. CUDA event queries are also
prohibited during capture, so the capturing path reads only that completed flag.
Native modules deliberately remain loaded until
process exit; graph users must retain k (metadata owner) and captured tensors
through replay. Compilation, module loading and metadata allocation are outside
capture. Tests synchronize only for observing results or timing endpoints;
there is no device-wide synchronization in the launch path.

## 6. Comparison and discrepancies

| One recorded sample | C++/NVRTC | LLVM/NVPTX |
|---|---:|---:|
| CubeCL compilation | 16.90 ms | 60.58 ms |
| Additional NVRTC compilation | 19.83 ms | None |
| Module load/JIT | 1.18 ms | 2.00 ms |
| Event-timed median per kernel, 1,048,583 elements | 11.70 µs | 11.58 µs |

Kernel timing uses seven warmed graph replays containing 128 sequential kernels;
it includes graph scheduling. The LLVM samples varied up to 15.83 µs. Host-call
measurements in results.json include Python validation, locking, recording and
ctypes overhead and are not pure kernel timings. Driver cache state/GPU clocks
were uncontrolled; these values support feasibility, not a performance ranking.
The first observed LLVM module load took 203.73 ms before the driver's JIT cache
was warm, illustrating why the later load sample is not a cold-compilation result.

The original architecture is confirmed for this kernel. Its implementation
guidance omitted several details that matter:

1. CUDA Graph capture initially failed with CUDA_ERROR_STREAM_CAPTURE_ISOLATION
   because the wrapper waited on an event recorded outside capture. Warming that
   metadata dependency per stream fixes capture without adding synchronization.
   The exact original failure is retained in the C++ artifacts. Review identified
   raw stream-handle memoization as a potential reuse hazard, so the final wrapper
   checks event completion outside capture and never caches stream identity.
   An intermediate correction attempted an event query during capture and failed
   with cudaErrorStreamCaptureUnsupported. The final `prepare_capture(N)` setup
   step records completed metadata once; captured launches make no event query.
2. LLVM's `sin` path requires NVIDIA libdevice. It is not standalone merely
   because it emits PTX without NVRTC. The local machine lacked a toolkit;
   standalone NVIDIA libdevice was installed, independently of Triton's copy.
3. NvptxModule.ptx is a NUL-terminated `Vec<c_char>`, not a CString; SmArch takes
   packed capability `86` and a tensor-core boolean. Report pseudocode was not
   an exact executable API specification.
4. Preventing *every* find_spec probe breaks PyTorch import. In the isolated
   environment, probes correctly return None; actual Triton imports remain
   prohibited. Both final runs recorded three availability probes, zero imports
   and zero loaded Triton modules.
5. There is no automatic module/tensor lifetime management across graph replay.
   This prototype makes those ownership requirements explicit.

No report assumption concerning Stage 0 required an upstream patch. Inductor
claims and generalized compiler-mode/operation support remain outside this task.

## 7. Source changes and validation

Experiment files: `stage0/Cargo.toml`, `Cargo.lock`, `src/main.rs`, `src/lib.rs`,
`launcher.c`, `bridge.py`, `verify.py`, `measure.py`, `setup.sh`,
`isolate_python.py`, `.gitignore`, `README.md`, this report, `llvm-build.log`,
and the generated artifact files enumerated above plus `verification.log` for
each route, `comparison.json`, `measurement.log`, `validation.json`, raw review
reports and the intermediate event-query failure. The complete file list is
recorded in [file-inventory.json](artifacts/file-inventory.json).
Local source copies live in
`.stage0-sources/{cubecl,pytorch}`; `.stage0-sources/.gitignore` excludes them.
Local environments, build products and toolkit files are ignored setup outputs.
Beads task: `pytortch-backend-experiments-0pq`; its passive export changed.

**PyTorch patch: none. CubeCL patch: none.** Both source checkouts have clean
status. Pre-existing AGENTS.md/CLAUDE.md and other workspace changes were preserved.
No commit, push or Dolt remote sync was performed.

Validation commands and reproduction instructions are in README.md. The native
C build treats warnings as errors; both real GPU test suites and the bounded
route comparison passed. Required review/check results are recorded separately
in [validation.json](artifacts/validation.json).

## 8. Recommendation for Stage 1

It now makes sense to attempt one actual Inductor contiguous float32 pointwise
schedule that replays load/sin/add/store into this same compiler/launcher bridge.
Use C++/NVRTC first for the smaller required toolchain and inspectable source;
retain LLVM as the verified alternative. Neither route is a clear performance
winner from this small measurement.

Minimum next experiment: emit the exact current two-buffer kernel description
from one SIMDScheduling node; demonstrate one fused CubeCL kernel, matching
forward output and cosine gradient, with Triton unavailable. Validate the
reached Inductor import/availability gates explicitly. **Stage 1 was not
implemented here.**

## Source references

Pinned source used for verification:

- [KernelBuilder](https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-core/src/compute/builder.rs),
  [Scope](https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-ir/src/scope.rs),
  [CppCompiler](https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-cpp/src/shared/base.rs).
- [C++ metadata lowering](https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-cpp/src/shared/metadata.rs),
  [LLVM compiler](https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-llvm/src/shared/base.rs),
  [NVPTX ABI](https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-llvm/src/nvptx/abi.rs),
  [libdevice lookup](https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-llvm/src/nvptx/libdevice.rs).
- [Private artifact finalization](https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-cuda/src/compute/artifact.rs),
  [managed BufferArg](https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-core/src/frontend/container/slice/launch.rs).
- [PyTorch CUDA stream APIs](https://github.com/pytorch/pytorch/blob/e342e9a9d7123c2b78940da0ea3fdb0dfa9beeb0/c10/cuda/CUDAStream.h),
  [allocator recording](https://github.com/pytorch/pytorch/blob/e342e9a9d7123c2b78940da0ea3fdb0dfa9beeb0/c10/cuda/CUDACachingAllocator.h),
  [Python stream protocol](https://github.com/pytorch/pytorch/blob/e342e9a9d7123c2b78940da0ea3fdb0dfa9beeb0/torch/cuda/streams.py).
- NVIDIA documents [libdevice's role](https://docs.nvidia.com/cuda/archive/13.0.1/libdevice-users-guide/introduction.html);
  the independently installed math library came from [NVIDIA's package](https://pypi.org/project/nvidia-nvvm/13.0.88/).
