# Stage 1.5: explicit Inductor backend boundaries

12 October 2026. Beads task `pytortch-backend-experiments-n7v`.

## Result

**boundary refactor works**. The supported execution surface remains static,
contiguous, zero-offset 1D float32 CUDA pointwise computation, with the existing
constant/sin/cos/add/mul/load/store operations and identity float32 casts.
`torch.compile(sin(x)+1)` and AOTAutograd's `upstream*cos(x)` still run through real
SIMD scheduler replay, CubeCL IR, the Stage 0 C++/NVRTC compiler and native current
stream launcher. Nonempty forward executes one CubeCL kernel. No operation
coverage, reduction algorithm, fallback system or compiler/launcher replacement
was added. Empty computations still launch nothing.

The Stage 0 and Stage 1 reports were read in full before changes. Executed versions
remain PyTorch 2.14.1+cu130, wheel commit
`5c4886908584029761b579af026dcfb627c84070`, CPython 3.14.5t, CUDA 13.0, RTX 3090/SM86,
and CubeCL `6bba5fdf04206d9c3710dfbc46aa9850450ab405`. The older investigation
checkout is not evidence for the executed wheel.

Source discovery used Tier 2 graph verification for the experimental files.
Coverage identifies the symlinked installed Inductor dependency as excluded;
its registration, scheduling and type-policy bodies were read directly and
preserved in exact original/overlay snapshots. Graph coverage is best effort,
not a completeness claim for those dependency sources.

## Before / after architecture

| Stage 1 shortcut | Stage 1.5 boundary |
|---|---|
| Scheduler checks the selected backend's name before requiring Triton | Scheduler asks the selected scheduling object to validate its requirements |
| Float32 assumptions scattered through values and buffer checks | A bounded, explicit `ScalarFloat32Policy` records semantic/storage/compute dtype and local shape; typed values remain the existing extension point |
| Replay descriptor contains `physical=scalar_256` and physical index `i` | Replay emits logical `linear` accesses; a separate checked `PhysicalKernelPlan` is passed to the wrapper/compiler adapter |
| Reduction/layout/operation failures occur in tiling or emission | Input contract checks at backend selection; generated-node checks before fusion; backend fusion vetoes; emission checks remain defensive |
| Generic SIMD reads historical `config.triton` settings | Actual reads are recorded and classified; no namespace sweep or new setting |

```text
torch.compile → Dynamo → AOTAutograd → Inductor GraphLowering
  → registered scheduling factory selects CubeCLScheduling
  → Scheduler.create_backend → selected.check_backend_available(device)
  → SchedulerNode logical ranges / LoopBody
  → existing _pre_fusion_custom_pass: capability validation
  → existing backend can_fuse_vertical/horizontal: fusion legality
  → unchanged SIMDScheduling.codegen_node / operation replay through V.ops
  → CubeCLKernel typed scalar semantic descriptor (linear logical accesses)
       + PhysicalKernelPlan from the logical schedule
  → ordinary Python wrapper → stage1.runtime.compile_kernel
  → plan.bridge_description: legacy Stage 1 wire descriptor
  → unchanged Rust KernelBuilder/Scope → CppCompiler<Cuda> → NVRTC → PTX
  → unchanged native launcher → PyTorch-owned tensors / current stream
```

No CubeCL IR object appears in Inductor logical replay. Inductor range-tree
symbols are logical coordinates; no XBLOCK is interpreted as a CubeCL vector.

## PyTorch changes

These four Python files change only in the pinned, isolated overlay. The installed
wheel and upstream checkouts are untouched. The exact [patch](stage1_5/artifacts/pytorch.patch)
is preserved. Registration structures and type/shape/SIMD sources are unchanged.

| File / function | Old responsibility → new responsibility | Reason / classification |
|---|---|---|
| `_inductor/scheduler.py`, `Scheduler.create_backend` and `BaseScheduling.check_backend_available` | Scheduler owned a global GPU Triton gate → it constructs the selected backend and invokes its requirement check; default hook is a no-op | Generic candidate: compiler dependencies follow selected implementation, not device category or backend-name list |
| `_inductor/codegen/triton.py`, `TritonScheduling.check_backend_available` | No owned check → owns the original `has_triton`, `GPUTooOldForTriton`, `TritonMissing` logic | Triton-specific implementation of the generic hook; no probe is made to lie |
| `_inductor/codegen/cuda_combined_scheduling.py`, `CUDACombinedScheduling.check_backend_available` | Relied on Scheduler's global gate → delegates to its existing Triton scheduling member | Preserves current ordinary generated-kernel routing and requirements; mixed CUDA scheduling still requires Triton |
| `_inductor/codegen/xpu/xpu_combined_scheduling.py`, `XPUCombinedScheduling.check_backend_available` | Same global gate → same delegation | Preserves current XPU Triton requirement; no XPU kernel execution was tested |

Unused Scheduler imports for the moved gate are removed. Error frames now identify
the owning requirement method. Default CUDA still fails with the real
`TritonMissing` class and diagnostic after CubeCL registration. The legacy GPU
major-version branch is retained verbatim in meaning; no old GPU was available
to hardware-test that branch. A Triton-present run is also untested.

The hook is a plausible small upstream refactor independently of CubeCL. It needs
upstream device/extension regression tests before upstream acceptance. An extension
which depended on the old accidental global gate must now declare its own
requirements, e.g. by inheriting TritonScheduling or overriding the hook.
The isolated overlay installer and experimental CubeCL selector remain out of tree.

## Backend requirement mechanism

`DeviceCodegen` currently contains scheduling and Python/C++/FX wrapper
constructors. `register_backend_for_device` additionally records custom passes and
config. CUDA and CPU scheduling constructors are config-dependent factories,
not always fixed backend classes. This makes a static per-device dependency flag
an awkward owner: it could describe CUDACombinedScheduling when the factory has
actually selected Halide, Pallas or CubeCL.

The smallest mechanism is therefore a raising availability method on the selected
`BaseScheduling` object, invoked once per device creation before grouping. No
registry signature, capability framework, dependency declaration DSL, callback
map or backend-name conditional is needed. Construction already has to tolerate
`scheduler=None` for `get_backend_features`; checking availability is separate
from those feature queries. In the executed constructors, constructing the
selected object does not compile a kernel or require loading the Triton package.

| Existing selection | Observed source behavior / effect of refactor |
|---|---|
| CPU C++ | Default no-op; compiler/toolchain checking remains in existing C++ compilation. The old global gate never rejected CPU |
| CPU Triton | Retains the old GPU-only failure semantics in the moved gate; CPU support/package errors remain where existing Triton code handles them |
| CUDA default | CUDACombinedScheduling → TritonScheduling check, preserving failure for absent Triton |
| XPU default | XPUCombinedScheduling → same check; direct availability-method probe passes the expected missing-package diagnostic |
| MPS Metal | Default no-op; previously explicitly excluded from the global GPU Triton gate |
| Halide CPU/CUDA | Default no-op; existing Halide import/type/code generation and async compilation still validate Halide when needed. CUDA no longer has an unrelated global Triton requirement |
| Pallas CPU/CUDA/TPU | Default no-op; emitted JAX/Pallas imports and `AsyncCompile.pallas` remain its compiler/loading boundary |
| MTIA Triton | Inherits TritonScheduling's gate |
| Private/extension devices | Existing registration/custom-mod discovery unchanged; selected scheduling object owns its requirements |

Only CUDA CubeCL and default CUDA are end-to-end tested here. Other entries are
source findings, supplemented by CPU no-op and XPU requirement method probes.

CubeCL's check validates PyTorch CUDA availability and presence of the existing
Rust executable, metadata library and native launcher. It does not compile as
an availability probe. Missing readiness requirements raise
`CubeCLBackendUnavailable(RuntimeError)`, distinct from unsupported semantics;
a missing-profile probe exercises this without deleting any build products.
The setup hint uses the C++-sufficient `bash stage0/setup.sh`; LLVM remains the
verified optional build feature and is not a mandatory CubeCL dependency.
Native library loading, NVRTC toolchain/linking errors
and compilation errors retain their existing actionable compiler/runtime failures;
the hook is a readiness check, not a guarantee that arbitrary compilation succeeds.

## Type/shape policy

No PyTorch type-policy patch is justified for this scope. The existing
`Kernel.create_cse_var` hook and `CSE.generate`'s typed-value pass-through already
provide an extension point. CubeCL operations produce `CubeCLValue(CSEVariable)`
with SSA identity, explicit dtype, scalar shape and bounds provenance before CSE.
The trace-only mask provenance label is now `logical_in_bounds`, replacing `i<N`;
the final saved replay evidence includes this label. Physical bounds remain
the unchanged explicit `i<N` branch in Rust-generated CubeCL IR.
The local fixed `ScalarFloat32Policy` describes the supported contract; it is
not a new generic promotion API.

| Concern | Current owner / decision |
|---|---|
| Semantic result dtype | Inductor lowering and `DtypePropagationOpsHandler` generic promotion rules describe operator semantics; this backend accepts only float32 semantic operations |
| Storage dtype | Graph buffer dtype, input capability checks, load/store checks and Rust descriptor validation; float32 only |
| Compute dtype | CubeCL operation emission / typed SSA values; explicitly float32, coinciding with storage only within this scope |
| Backend value shape | `CubeCLValue`, `Kernel.create_cse_var`; shape `()` represents a local scalar, independent of whole tensor size/logical tile |
| Backend promotion | Explicitly none within this float32-only contract. Future load/cast/store conversions belong to CubeCL overrides and typed result construction |

Named or concrete-backend decisions found in the pertinent live sources:

* `common.CSEProxy._default`: masked results handled specially for Triton and C++;
  automatic handler propagation enabled for Triton/C++/MPS; non-null dtype required
  for Triton/C++; missing C++ value dtype repaired. Reached by CubeCL, whose values
  are already typed and pass through. CubeCL is not added to any named list.
* `utils.upcast_compute_type`: half/BF16 → float32 only for named Triton with
  `triton.codegen_upcast_to_fp32`. Called by dtype handler constant/load/seed/cast,
  index and masked rules and Triton overrides. This mixes storage lookup with
  backend compute policy; CubeCL does not invoke those rules for value emission.
* `shape_propagation.ShapePropagationOpsHandler.constant`: concrete TritonKernel
  uses singleton block dimensions, others return scalar shape. `dot` requires
  TritonKernel and returns YBLOCK/XBLOCK. Neither is used by CubeCL's typed
  pointwise emission; dot remains unsupported.
* `common.check_dtype`, `check_shape`, `_is_runtime_triton_assert_target`,
  `check_nan`, and the CSE assignment assertion path generate backend-specific
  debug source. Their assertions are disabled on this accepted path; these are
  debug code-generation owners, not generic tensor semantics.

For Stage 2, retain generic semantic promotion (`promote_types` and applicable
operator rules); explicitly apply backend compute conversions in CubeCLOps
load/to_dtype/store and return typed values through `create_cse_var`. Avoid
calling the named Triton compute upcast helper as a semantic dtype oracle.
If an untyped/string-producing backend needs common CSE propagation, a kernel-owned
operation-result policy at the handler invocation inside `CSEProxy._default`
is the precise candidate boundary. That refactor is postponed until a real
non-float32 replay demonstrates its inputs/outputs; no speculative generic API
or half/BF16 implementation is introduced now.

## Physical plan

The actual immutable [representation](stage1/plan.py) is:

```text
PhysicalKernelPlan
  iteration_mapping = absolute_linear_unit
  workgroup = (256, 1, 1)
  elements_per_unit = 1
  local_vector_width = 1
  bounds_strategy = explicit_less_than_numel
  reduction_mapping = none
```

`CubeCLKernel` obtains the plan from its static logical pointwise numel and
reduction_numel using `plan_pointwise`. Semantic replay only produces logical
`linear` loads/stores and typed operation SSA. The wrapper passes semantic
description and plan separately to runtime. `bridge_description` validates the
supported plan, maps logical `linear` to the existing physical `i`, adds the
legacy `physical=scalar_256`, and derives the unchanged entrypoint hash.
The existing Rust descriptor builder then constructs absolute-position reads,
buffer length, explicit `i<N` branch and semantic ops in CubeCL IR.

The Rust schema stays version 1. This is a compatibility adapter, not a new
permanent serialization ABI. The launcher still fixes 256 units; plan validation
rejects every different physical mapping before Rust IR/compiler invocation.
Future reductions can select a different plan and physical builder while keeping
semantic replay distinct; no reduction fields/algorithm beyond `none` are designed.
Representative [semantic](stage1_5/artifacts/kernels/21d834bc279b6d28/semantic.json),
[plan](stage1_5/artifacts/kernels/21d834bc279b6d28/physical-plan.json) and
[bridge descriptor](stage1_5/artifacts/kernels/21d834bc279b6d28/descriptor.json)
make the boundaries inspectable. Both accepted forward/backward IR, CUDA C++,
PTX and legacy descriptors are byte-identical to the committed Stage 1 baseline.

## Capability path

The exact live Scheduler `_init` runs `_pre_fusion_custom_pass` after dependency
analysis/dead-node elimination and before `fuse_nodes`. The experimental compiler
composes the existing configured pass, then validates generated SchedulerNodes.
It inspects their already-traced LoopBody operation graph, reduction/range shape,
indirect/nested bodies, buffer dtype/layout/size and linear indexing. `get_index`
is accepted as LoopBody's symbolic lookup bookkeeping, not a new semantic op.
Extern/nop nodes retain their natural scheduling; extern fusion is vetoed.

`CubeCLScheduling.can_fuse_vertical/horizontal` combines a bounded ABI/range
veto with unchanged SIMD legality checks. A real two-node branching fixture
observes a veto for two externally observable outputs and deliberately stops in
the post-fusion pass before codegen. It does not add a multi-output execution
acceptance case. Pallas already overrides these same fusion methods to reject
incompatible reduction read-index patterns; CUDA combined scheduling and C++
also use backend-specific vetoes. A new generic fusion framework is unnecessary.

| Concrete case | Actual decision trace |
|---|---|
| Supported sin+add | backend_selection accept → selected availability check → pre_fusion accept (`load,sin,constant,add,store`) → SIMD schedule/replay → semantic+plan → CubeCL compile/launch |
| Unsupported reduction, N=257 | backend_selection accepts f32 contiguous input → pre_fusion rejects `reductions unsupported`; no tiling, replay or compiler invocation |
| Unsupported float64 | backend_selection rejects `float32 required` before `compile_fx`/AOT/Inductor lowering |
| Unsupported strided input | backend_selection rejects `1D contiguous zero-offset input required` before `compile_fx` |
| Unsupported tan / nonfinite constant | pre_fusion operation/constant rejection; no schedule/kernel event |
| Indirect-index fixture | its integer index input violates the input dtype contract and rejects at selection; body indirect-index checks remain defensive for later IR transformations |
| Unsupported fusion | backend returns False; scheduler can retain separate supported groups |
| Unsupported physical schedule | plan validation raises before Rust IR/compiler; explicit negative plan probe |
| Backend compiler failure | Existing subprocess/NVRTC/native exceptions propagate; no fallback or Triton retry |

Acceptance tests assert the expected reject boundary, no subsequent schedule or
kernel event, and unchanged native compile count. Concrete traces are in
[results](stage1_5/artifacts/results.json), [boundary probes](stage1_5/artifacts/boundary-probes.json)
and local `schedule-replay.json`.

This rejects before scheduler fusion, not before Inductor's earlier lowering has
inlined pointwise expressions into a LoopBody. Future operator-directed fallback
must participate in lowering/realization or backend choice before that inlining.
The current input contract is deliberately global to the tiny CubeCL experiment;
it is not a general per-extern dtype capability system. It can reject a
differently-shaped unused tensor argument which Stage 1 could eliminate before
late buffer validation; this is an intentional tightening of the stated input
contract, not a claim that every previously untested graph is preserved.
The custom pre-fusion
pass is an existing experimental hook rather than a new upstream capability API.
The multi-node fusion accept/delegate/codegen path is untested: the accepted
workload has one node, and the separate fixture exercises only the veto. A first
genuinely fused multi-node acceptance case should accompany any Stage 2 change
that affects fusion.

## config.triton audit

The acceptance harness observes real `SubConfigProxy.__getattr__` reads and caller
locations, then restores the method. Config hashing/serialization inventories
are excluded. [Read evidence](stage1_5/artifacts/config-triton-reads.json) records
eleven settings on the accepted path.
This is a best-effort caller-based audit; generated wrapper reads outside the
Inductor source directory are not instrumented. Classifications also use the
exact source snapshots.

| Setting(s) actually read | Classification / owner / action |
|---|---|
| coalesce_tiling_analysis | Generic SIMD analysis under historical namespace; existing experimental scope disables it; no independent policy needed yet |
| native_matmul | Triton-specific native dot recognition read by SIMDKernel constructor; false, no dot behavior reached |
| cudagraphs | Generic capture/partition policy under historical namespace; disabled in experimental config; no move |
| cudagraph_min_partition_size, cudagraph_trees_generation_cloning | Generic graph/partition infrastructure; reads occur even with capture disabled; no CubeCL capture claim |
| autotune_at_compile_time | Wrapper/compiler integration policy for existing autotuned kernels; None; CubeCL callable avoids Triton tuning objects |
| enable_host_side_tma | Triton/TMA-specific compile_fx gate; no TMA path reached |
| debug_sync_graph, debug_sync_kernel | Generic debug synchronization policy under historical namespace; false; no generated sync |
| proton_profiling | Triton-specific profiling integration gate; false; ordinary torch.profiler used |
| store_cubin | Triton-specific binary-export wrapper gate; false; existing PTX evidence adapter used |

Not reached by accepted CubeCL emission: max_tiles, prefer_nd_tiling,
tile_reductions, mix-order/nested/cooperative/persistent reduction policies,
codegen_upcast_to_fp32 and Triton autotuning configurations. Generic SIMD fusion
tiling settings can be consulted when fusing multiple generated nodes; these are
a future tuning-policy concern. The two-output probe vetoes before those reads.
No config setting was renamed, moved or given a CubeCL-specific duplicate.

## Regression evidence

[Stage 1.5 results](stage1_5/artifacts/results.json) preserve the full existing
suite: sizes 0,1,255,256,257,1,048,583; inference and training forward; unit and
arbitrary upstream backward; NaN/infinities/signed zero/extremes; repeated sizes;
two delayed non-default stream pipelines; three allocator-pressure trials.
The profiler records exactly one CubeCL forward kernel, with no pointwise ATen
or Triton kernel. Native counters agree: three descriptions compiled (including
the existing supplementary two-input ABI case), 44 actual native launches.
No x/y copy into CubeCL storage; original pointer and stream assertions remain.

Triton remains unavailable: three package probes, eight already-optional blocked
top-level imports, zero Python/native loaded Triton modules/maps. Five GIL
checkpoints remain false under CPython 3.14.5t; `PYTHON_GIL` is absent. Separate
boundary probes also retain GIL-disabled/no-Triton evidence. This is bounded
workload evidence, not general concurrent compilation safety.

[Stage 0 regressions](stage1_5/artifacts/regression.json): both C++ and LLVM full
suites pass, including canaries, special values, two changing streams, five
allocator-pressure trials, CUDA Graph capture/replay, and twenty barrier-started
prepared launches. Six GIL checks per route stay false. The compiler, Rust builder,
native launcher, metadata ABI and stream/lifetime code have no source changes.

Preserved locally under `stage1_5/artifacts`: original/patched exact source
snapshots and hashes, scheduler/replay stacks, config reads, profiler trace,
runtime pointer/stream/compiler events, patch, availability/fusion probes,
Stage 0 logs, and representative semantic/plan/IR/CUDA/PTX/manifest artifacts.
Only representative generated evidence is eligible for Git; full unchanged
upstream source snapshots, caches, local logs and raw traces stay local.
Reproduction commands are in [stage1/README.md](stage1/README.md). Run overlay
installing scripts sequentially in fresh processes; concurrent installation is
outside the established experiment contract.

Required review and final validation disposition are recorded in
`stage1_5/artifacts/review-disposition.md` and `validation.json` before handoff.
No commit, push or Dolt remote sync is performed.

## Why these abstractions exist

| Boundary | Concrete requirement / source owner | Other beneficiary / suitability |
|---|---|---|
| check_backend_available | Stage 1 was blocked by Scheduler's global Triton gate; selected scheduling object owns compiler requirements | Halide/Pallas CUDA and extension backends; small generic upstream candidate |
| Existing typed CSE value + explicit f32 contract | Replay must carry backend identity/type/shape without source strings; Kernel.create_cse_var/CSE.generate already own it | Any typed IR backend; no speculative PyTorch patch |
| PhysicalKernelPlan + adapter | Logical range `x` must not silently imply CUDA units or vectors; kernel planning/runtime descriptor boundary owns execution mapping | Later distinct mappings; experimental backend-local representation |
| pre_fusion checks + fusion veto | Unsupported nodes and multi-output groups must be rejected before codegen; existing Scheduler custom pass and BaseScheduling vetoes own it | Pattern already used by alternative backends; CubeCL checks out of tree, no general fallback framework |
| Config-read evidence | Distinguish historical namespace from actual Triton dependence | No code refactor justified by current reads |

## Recommendation

**Exactly one next direction: 1. Stage 2 pointwise/dtype/index coverage.** The
requirements hook removes the name-based availability gate; existing typed-value
hooks suffice for this IR backend; semantic and physical descriptions are separate;
unsupported cases now fail before scheduler fusion/codegen. Broaden in small,
source-replayed increments, choosing an explicit compute/storage conversion
policy when the first new dtype is actually tested. General fallback and reduction
mapping remain separate investigations. Stage 2 is not started automatically.
