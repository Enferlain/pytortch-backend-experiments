# Stage 1: Inductor semantic replay to CubeCL

12 October 2026, Europe/Budapest. Beads task: `pytortch-backend-experiments-f41`.

## Result

**Stage 1 works.** Normal Dynamo/AOTAutograd/Inductor compilation of
`torch.sin(x) + 1` emits one generated CubeCL pointwise kernel per nonempty
forward invocation. AOTAutograd's compiled backward emits one CubeCL kernel
computing `upstream_gradient * cos(x)`. A unit upstream gradient gives `cos(x)`.
No pointwise operation falls back to eager ATen or Triton.

The tested surface is deliberately narrow: static, 1D contiguous float32 CUDA
tensors, one or two equal-sized inputs, one disjoint output, scalar execution.
Sizes 0, 1, 255, 256, 257 and 1,048,583 pass forward and backward; special values,
repeat sizes, two non-default streams and allocator pressure also pass.
For size zero, Inductor removes the empty computation and launches no kernel.
This is normal compiler elimination, not eager execution.

The warmed forward profiler records exactly one GPU **kernel** event,
`cubecl_66627c35b250b257`. Its separate GPU user annotation is not another kernel.
The required forward/backward descriptions invoke the actual Rust/CubeCL compiler
twice and are reused across sizes. A supplementary two-input forward adds one
description, for three compiler invocations and 44 native launches in the full suite.
Native atomic counters independently agree with Python instrumentation.

Sources of evidence: [results](stage1/artifacts/results.json),
[profiler](stage1/artifacts/forward-profiler.json),
[operation replay and actual stacks](stage1/artifacts/schedule-replay.json),
[native/compiler events](stage1/artifacts/runtime-events.json).

## Exact versions and source tracing

The full [Stage 0 report](stage0/REPORT.md) was read before changes. Its working
compiler/launcher is the baseline. Stage 1 executes the same installed wheel:
PyTorch **2.14.1+cu130**, embedded commit
`5c4886908584029761b579af026dcfb627c84070`, CPython **3.14.5t**, CUDA 13.0,
GeForce RTX 3090 (CUDA device 0), SM86. CubeCL remains
`6bba5fdf04206d9c3710dfbc46aa9850450ab405` with the same locked Rust dependencies.
NVRTC remains CUDA 13.0 with compute_86, C++17 and `--fmad=false`.

The earlier investigation's PyTorch source commit
`e342e9a9d7123c2b78940da0ea3fdb0dfa9beeb0` is **not** used to infer Stage 1 internals.
[source_trace.py](stage1/source_trace.py) extracts exact installed function bodies,
line spans, source paths and SHA256 hashes into
[source-trace.json](stage1/artifacts/source-trace.json), before overlay installation.
Runtime evidence comes from Python call profiling plus stacks collected at the
first actual replayed load of each kernel, including backward's lazy compilation.
The latter also covers paths not observed by the Python profile hook during
backward. Source inspection alone is not treated as execution evidence.

Graph queries did not expose the installed Inductor source because the dependency
symlinks are outside the graph project and excluded. Coverage checks recorded
that limitation; exact files and pertinent definitions were read directly.
The experimental code is indexed without recorded gaps, a best-effort signal.
Graph edges into source-text artifacts are not evidence for live compiler calls;
the exact source and runtime stacks take precedence.

## Actual source-traced path

Forward reaches the following concrete path. Functions between the listed
boundaries are preserved in the actual replay stacks, not inferred from a diagram.

```text
torch.compile(f, backend=stage1.backend.compile_backend,
              fullgraph=True, dynamic=False)
→ Dynamo OutputGraph._call_user_compiler
→ stage1.backend.compile_backend
→ torch._inductor.compile_fx.compile_fx / _compile_fx_main
→ torch._dynamo.backends.common.AotAutograd.__call__
→ torch._functorch.aot_autograd.aot_module_simplified
→ graph_compile.aot_stage2_autograd / _aot_stage2b_fw_compile
→ _aot_stage2b_compile_forward_or_inference
→ compile_fx._compile_fx_main.<locals>.fw_compiler_base
→ compile_fx_forward → compile_fx_inner → _compile_fx_inner
→ fx_codegen_and_compile → codegen_and_compile
→ GraphLowering.run [lowering], compile_to_module / _compile_to_module
→ GraphLowering.codegen → _update_scheduler → Scheduler
→ Scheduler.create_backend [conditional requirement; registered CUDA factory]
→ CubeCLScheduling
→ Scheduler.codegen / _codegen_partitions / _codegen
→ SIMDScheduling.codegen_node → _codegen_nodes
→ generate_node_schedule → codegen_node_schedule
→ CubeCLScheduling.get_tiling_and_scores: {x: logical_numel}
→ create_kernel_choices → CubeCLKernel(SIMDKernel)
→ SIMDScheduling.codegen_node_schedule_with_kernel
→ split_and_set_ranges, indexing passes, SchedulerNode.codegen(index_vars)
→ SimplifyIndexing → LoopBody.__call__ → LoopBodyBlock.__call__
→ InterpreterShim / FX interpreter, replay through V.ops
→ common.CSEProxy → CubeCLOps / CubeCLKernel.load and store
→ typed CubeCLValue / experimental semantic descriptor
→ CubeCLKernel.codegen_kernel → CubeCLScheduling.define_kernel
→ ordinary PythonWrapperCodegen definition and tensor allocation
→ stage1.runtime.compile_kernel
→ existing stage0 Rust executable with descriptor argument
→ KernelBuilder / Scope / dialect IR → KernelDefinition
→ existing CppCompiler<Cuda> → CUDA C++
→ existing stage0.bridge.nvrtc_compile → PTX
→ existing stage0_load → module / function in PyTorch's context
→ emitted wrapper calls CompiledKernel with PyTorch tensors
→ existing stage0.bridge.Kernel: data_ptr, current_stream, record_stream
→ stage0_launch / bounded stage1_launch argument packing
→ shared original enqueue logic → cuLaunchKernel
```

The forward pointwise expression is already one `SchedulerNode` named `op0`
with both sin and add inside its replayed body. No separate sin launch precedes
add. This graph does not require a `FusedSchedulerNode` to achieve fusion.

Backward is compiled lazily by AOTAutograd:

```text
autograd.Function.apply_boxed
→ runtime_wrappers.backward / _backward_impl / get_or_compile
→ common._wrapped_bw_compiler
→ compile_fx._compile_fx_main.<locals>.bw_compiler
→ compile_fx_backward → compile_fx_inner
→ same GraphLowering / Scheduler / SIMD semantic replay / CubeCL path
```

The replay stack for backward explicitly contains `bw_compiler`,
`codegen_node_schedule_with_kernel`, `SchedulerNode.codegen`, `LoopBody` and
`CSEProxy.load`. [compiler-path.json](stage1/artifacts/compiler-path.json) and
[schedule-replay.json](stage1/artifacts/schedule-replay.json) retain the evidence.

## PyTorch changes and encountered Triton assumptions

Only one executed PyTorch source change is needed, in an isolated copy of the
wheel's `_inductor` package:

| File / function | Change and reason | Classification |
|---|---|---|
| `torch/_inductor/scheduler.py`, `Scheduler.create_backend` | Replace unconditional `if not has_triton()` with `if get_current_backend(device.type) == "triton" and not has_triton()` | Small generic upstream-refactor candidate for existing named backend selection; a future registration capability such as `requires_triton` would be more extensible |

[Exact patch](stage1/artifacts/pytorch.patch) and
[original/patched hashes](stage1/artifacts/overlay.json) are retained.
The [unpatched live probe](stage1/artifacts/unpatched-gate.json) reaches
`TritonMissing`. `has_triton()` is never modified, made to return true, or replaced.
No fake module, native-import suppression, PyTorch binary rebuild or CubeCL
upstream patch is used. [overlay.py](stage1/overlay.py) copies the actual wheel's
Python package locally, checks its embedded commit and verifies the patch matches
exactly once. Shared dependency source files remain untouched.

Other assumptions encountered:

- `init_backend_registration` normally selects `CUDACombinedScheduling`, which
  sends ordinary nodes to `TritonScheduling.codegen_node`. We use the existing
  device registration API with a selector that routes only experimental
  `cuda_backend="cubecl"` to `CubeCLScheduling`; the previous factory remains
  available for other backend settings. This is local registration, not another
  source patch. The experimental string is not a supported PyTorch release option.
- Registration imports PyTorch's Python Triton codegen modules even when the
  actual `triton` package is absent. Their definitions are importable in this
  wheel. They are not instantiated for CubeCL or used for source generation.
- `CSEProxy._default` runs dtype/shape propagation for a named list of backends
  excluding CubeCL. No patch is required here for float32: our values carry dtype
  and scalar shape before CSE, and `CSE.generate` passes those values through.
  `utils.upcast_compute_type` promotes half/bfloat16 only for named Triton;
  those dtypes are rejected by this experiment rather than borrowing that rule.
- Generic SIMD code consults `config.triton` host settings for tiling, reductions,
  coalescing and wrapper-related behavior. The local compiler selects one static
  flattened range and disables coalescing, combo kernels, in-place reuse,
  benchmarking, CUDA Graph automation and compile-thread fanout. Advanced
  Triton autotuning/config object paths are not reached. The report does not claim
  SIMD is generally backend independent beyond this executed path.
- The default SIMD launch is abstract/backend-specific. Our kernel emits an
  ordinary Python callable invocation through the existing wrapper instead of
  a Triton `.run`, `XBLOCK` launch or copying Triton wrapper code.

The absent package triggers eight already-guarded top-level import attempts in
the accepted test. Each stays absent and is handled by existing PyTorch code:

| Reached PyTorch location | Existing handling |
|---|---|
| `utils/_triton.py:39`, `has_triton_package` | ImportError → false |
| `_inductor/runtime/triton_compat.py:10` | ImportError → compatibility definitions without package |
| `utils/flop_counter.py:110` | Optional JIT import → warning |
| `_inductor/ir.py:158` | ImportError → unavailable flag |
| `_dynamo/repro/after_aot.py:45` | Optional diagnostic classes |
| `_inductor/kernel/mm.py:72` | ImportError → unavailable flag |
| `_inductor/heuristics/template/tlx.py:8` | Optional registry import → pass |
| `_inductor/autows_utils.py:12`, `has_meta_ws` | ImportError → false |

There was no need to patch these guarded attempts. Import blocking distinguishes
top-level `triton` from relative imports such as `.triton` within PyTorch.

## Backend implementation surface

| Category | Actual surface |
|---|---|
| Reused unchanged from SIMDScheduling | Grouping, fusion framework, `codegen_node`, `_codegen_nodes`, `generate_node_schedule`, `codegen_node_schedule`, kernel creation, two-pass schedule replay, node marking, launch cleanup and buffer freeing |
| Reused unchanged from SIMDKernel/common | Range-tree construction, static range splitting/mapping, index simplification, index dtype choice, CSEProxy, CSEVariable pass-through, store-cache and removed-buffer handling |
| Small generic hook required | Only Scheduler's conditional dependency gate above |
| CubeCL-specific | Typed scalar values; minimal operation overrides; index/layout validation; semantic CSE keys; descriptor construction; fixed physical plan in Rust; kernel definition via Stage 0 compiler and Python wrapper invocation |
| Triton-specific assumption avoided | Combined CUDA generated-kernel routing; backend-specific launch; tile/autotune/advanced SIMD paths; named dtype propagation handled with already typed backend values |

[backend.py](stage1/backend.py) defines `CubeCLScheduling(SIMDScheduling)` and
`CubeCLKernel(SIMDKernel)`. It never subclasses `TritonKernel`. `CubeCLOps`
implements constant, sin, cos, add, mul and identity float32 casts; unknown
operations raise `UnsupportedCubeCL`. Casts were not replayed in the accepted
forward/backward graphs. No general vector, reduction, atomic, random, template,
GEMM, indirect-index or arbitrary-stride implementation was added.

`CubeCLValue` subclasses `CSEVariable`; its name is the backend SSA identity,
with float32 dtype, local scalar shape `()` and `i<N` mask provenance. Operation
emission returns that value directly. `CSE.generate` tightens bounds and tracks
uses without producing assignment strings. Backend semantic keys avoid duplicate
IR emission. These are semantic names and records, not Triton-like source text.

The physical mapping is isolated in [descriptor.rs](stage0/src/descriptor.rs):
one scalar per absolute CUDA unit/thread, 256 threads, explicit unsigned `i<N`
branch enclosing every load and store. Logical Inductor range symbols resolve
to `i`; for N=1, a constant zero index is equivalent under the bound and accepted.
No logical `XBLOCK` array is mapped into physical CubeCL vector units.

The experiment rejects unsupported schedules at tiling/layout/operation checks.
These are deliberately fatal compilation errors, not a generalized fallback
policy. Future capability selection must happen before committing fused groups
and layouts, in lowering/backend selection and scheduler fusion legality.
Existing naturally selected ATen/vendor extern scheduling is retained; its broad
coverage is not established by this test.

## Actual operation replay and IR mapping

| Forward replay | Descriptor → CubeCL IR |
|---|---|
| `load(primals_1, x0)` | buffer 0, index i → IndexOp + LoadOp |
| `sin(tmp0)` | SinOp |
| `constant(1.0, torch.float32)` | f32 ConstantOp |
| `add(tmp1,tmp2)` | FAddOp |
| `store(buf0,x0,tmp3,None)` | buffer 1, index i → IndexOp + StoreOp |

| Backward replay | Descriptor → CubeCL IR |
|---|---|
| `load(tangents_1,x0)` | buffer 0 → IndexOp + LoadOp |
| `load(primals_1,x0)` | buffer 1 → IndexOp + LoadOp |
| `cos(tmp1)` | CosOp |
| `mul(tmp0,tmp2)` | FMulOp; required for arbitrary incoming gradient |
| `store(buf0,x0,tmp3,None)` | buffer 2 → IndexOp + StoreOp |

The physical builder adds ReadBuiltinOp(AbsolutePos), BufferLenOp(buffer 0),
ULessThanOp, IfOp, YieldOp and ReturnOp. These are physical mapping/control
operations, not extra high-level semantic replay. Inductor's common indexing
passes run before replay; no additional bookkeeping value operations were needed.

Stage 0 reuse required only a descriptor argument to the existing compiler CLI,
a new IR builder module, bounded three-buffer metadata, and host pointer-array
packing for backward. Existing manual Stage 0 building remains the no-descriptor
path. CppCompiler/NVRTC/module loading and enqueue code are shared, not recreated
inside Inductor. LLVM support remains built and independently regression-tested;
Stage 1 does not advertise LLVM backend integration. The descriptor checker also
validates buffer access roles and operation dtype before building IR.

| ABI | Forward | Backward |
|---|---|---|
| Pointer order | input, output, info | tangent, primal, output, info |
| Metadata | 2 lengths / 16 bytes | 3 lengths / 24 bytes |
| Input/output transfer | None; original data_ptr values | None; original data_ptr values |
| Grid / block | ceil(N/256), block 256 | Same |
| Shared memory | 0 | 0 |
| Stream | PyTorch current stream, fetched per call | Same |

All equal-size buffers have their own InfoBuilder length in descriptor order;
word ordering remains source-derived because equal lengths do not distinguish
swaps numerically. Runtime validation enforces equal shapes, dtype, contiguity,
device and disjoint output. Every tensor and metadata allocation records the
current stream. The module/metadata process-lifetime ownership contract remains
as in Stage 0. Setup may upload small metadata once per size; x/y are never copied
to CubeCL storage, and CubeCL CUDA runtime allocation/stream management is unused.

## Correctness and independence

[verify.py](stage1/verify.py) runs a real fresh-process acceptance test. Each
nonempty required size has one forward launch and one backward launch; an
additional random upstream gradient verifies multiplication. N=0 has zero launches
and correct empty output/gradient. Tolerance: rtol/atol 2e-6, equal_nan=True.

- NaN, ±inf, signed zero, tiny and ordinary/extreme values pass forward/backward.
- Different supported sizes repeat without another CubeCL compiler invocation.
  Inductor specializes static wrapper sizes, while the two PTX bodies use dynamic
  buffer-length metadata. No advanced dynamic-shape backend support is implied.
- Two delayed producer→compiled forward/backward→consumer stream pipelines pass.
- Three side-stream allocator-pressure trials pass; no address reuse is observed
  before completion. This reuses Stage 0's contract and event-based observation
  method, including its limitation that a completed race cannot be excluded only
  by a later event query. Numerical checking and record_stream are complementary.
- Plain inference forward also passes every required size. A supplementary
  `sin(x)+other` forward verifies the existing two-input ABI directly. Backward
  pointer order is checked with both unit and arbitrary upstream gradients.
- Unsupported tan, reduction, float64, strided-input, CPU-input, indirect-index
  and non-finite-constant cases fail with CubeCL
  diagnostics. `fullgraph=True` and default exception propagation are retained.
- Warmed profiler observation excludes reference ATen work and backward. It has
  exactly one CubeCL kernel and no Triton kernel. Pointer and stream assertions
  compare the real wrapper arguments with caller tensors/current stream.

Triton availability probes: **3**, returning no package. Actual top-level import
attempts: **8**, all blocked, all existing optional PyTorch imports. Loaded Triton
Python modules: **0**. Native mappings containing Triton: **0**. No `PYTHON_GIL`
setting is present; all five GIL checks remain **false**, including after Inductor
import, AOT backward, profiling and unsupported-case checks. This proves the
bounded workload on 3.14t, not unrestricted concurrent Inductor compilation.

Both full independent Stage 0 GPU suites pass again after the extension: C++
and LLVM, including their CUDA Graph capture, lifetime and prepared-thread tests.
Their logs are retained in Stage 1 artifacts. Stage 1's own Inductor-managed CUDA
Graph support is outside scope; its compiler disables that automation.

## Generated artifacts

Representative forward artifacts:
[descriptor](stage1/artifacts/kernels/21d834bc279b6d28/descriptor.json),
[CubeCL IR](stage1/artifacts/kernels/21d834bc279b6d28/cubecl.ir),
[CUDA C++](stage1/artifacts/kernels/21d834bc279b6d28/kernel.cu),
[PTX](stage1/artifacts/kernels/21d834bc279b6d28/kernel.ptx),
[manifest](stage1/artifacts/kernels/21d834bc279b6d28/manifest.json).

Representative backward artifacts:
[descriptor](stage1/artifacts/kernels/4c0f8eabbd050d4f/descriptor.json),
[CubeCL IR](stage1/artifacts/kernels/4c0f8eabbd050d4f/cubecl.ir),
[CUDA C++](stage1/artifacts/kernels/4c0f8eabbd050d4f/kernel.cu),
[PTX](stage1/artifacts/kernels/4c0f8eabbd050d4f/kernel.ptx),
[manifest](stage1/artifacts/kernels/4c0f8eabbd050d4f/manifest.json).

Each directory also contains compiler/NVRTC logs, metadata and inspected PTX
signature. [runtime-events.json](stage1/artifacts/runtime-events.json) records
PTX/descriptor hashes. Emitted Inductor Python modules are preserved under
`stage1/artifacts/wrappers/`; their call bodies allocate ordinary PyTorch outputs
and invoke the CubeCL callable. Final quality/review results and an artifact
inventory are recorded separately in `stage1/artifacts/validation.json`.

## Discrepancies and limitations

1. The earlier conceptual AOT/Inductor path is valid, but current AOT compilation
   uses `_functorch/_aot_autograd/graph_compile.py` and separate lazy backward
   wrappers. Live bodies/stacks, not older function names, were necessary.
2. CSEVariable can carry typed backend values directly as proposed. For this
   narrow float32 path, the named dtype-propagation cases do not require a patch.
   Broadening dtypes will require a principled backend-independent type policy.
3. The initial graph contains both pointwise operations in one node. Fusion proof
   is replay and one GPU kernel, not the existence of a FusedSchedulerNode class.
4. Stage 0's exact three-pointer ABI only fits one input. Real backward needs
   tangent and primal, hence a bounded fourth pointer and third metadata word.
   This is an extension of the same compiler/launcher, not a changed execution model.
5. Loading Inductor makes optional package import attempts even without Triton.
   Zero attempts from Stage 0 is not a general Inductor property. Zero successful
   package/native loads and correct blocking are the relevant independence evidence.
6. Forward launch count is zero for an empty tensor because Inductor removes its
   body. A blanket requirement of one launch at N=0 would conflict with normal
   Inductor optimization.
7. The profiler includes a CUDA user annotation in addition to its one GPU kernel.
   Counting every CUDA activity as a kernel would incorrectly report two kernels.

The descriptor is experimental schema 1, not a promised permanent serialization
ABI. The native compiler launches a subprocess once per unique description;
wrapper specialization and process-lifetime metadata caching are intentionally
simple. No performance tuning or Triton comparison was performed. Concurrent
source-overlay installation, general free-threaded compiler safety, alternate
hardware, arbitrary gradients/strides and production fallbacks are unproven.

## Recommendation

**Next choose 2: refactor the Inductor integration boundary first.** There is no
fundamental problem exposed by pointwise replay: both forward and backward execute
through CubeCL on PyTorch storage. The useful next improvement is a small generic
backend dependency declaration, type/shape policy hook and explicit physical-plan
boundary, with capability checks before fusion. The one named dependency patch
and typed-value pass-through experiment give concrete requirements for that work.

After those boundaries are stable, broaden pointwise/dtype/index coverage.
Reductions require a new physical mapping and should be a separate investigation.
The compiler/launcher design needs no replacement on this evidence. No next-stage
implementation is started automatically.

## Reproduction and validation

[Build/run instructions](stage1/README.md) include the local experimental option.
Significant code changes are subject to the repository's required review-mcp
review and affected checks before handoff. The required review found no wrong-result issue on the tested path. Its scope,
diagnostic and evidence findings were addressed: CUDA-only entry gating, explicit
advanced-operation/non-finite-constant rejection, CSE use counts, backward ABI
assertions, a supplementary two-input forward, collision-checked wrapper capture,
loud missing-source-target errors and single-pass metadata serialization.
Exact final results/dispositions are
retained in `stage1/artifacts/validation.json` and review artifacts.

Changed implementation files: Stage 1 backend/runtime/overlay/evidence/verification
and source probe scripts, plus Stage 0 `src/descriptor.rs`, `src/main.rs`,
`src/lib.rs`, `launcher.c` and `bridge.py`. Existing Stage 0 LLVM code and tests
remain. No commit, push or remote issue synchronization is performed.
