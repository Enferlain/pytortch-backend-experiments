# Stage 3: cooperative reductions and reduction→pointwise fusion

12 October 2026. Beads task `pytortch-backend-experiments-ble`.

## Result

**passed** for the bounded sum/mean/RMSNorm surface below. Real Dynamo,
AOTAutograd, Inductor scheduling and semantic replay produce cooperative CubeCL
GPU kernels, including column addressing, naturally split transposed reductions,
dynamic extents, lazy masks, reduction epilogues and normalization backward.
Inference RMSNorm and mean-with-epilogue each execute one generated kernel.
Training RMSNorm executes two forward kernels and one backward reduction kernel;
the forward split preserves the inverse norm that AOTAutograd saves.

The original investigation/roadmap and all Stage 0–2 and FP-hardening reports were
read in full before code changes. Their compiler/runtime and conservative numeric
boundaries remain. Executed versions: PyTorch 2.14.1+cu130, wheel commit
`5c4886908584029761b579af026dcfb627c84070`; CPython 3.14.5t; CUDA/NVRTC 13.0;
RTX 3090/SM86; Rust 1.97.1; CubeCL pin
`6bba5fdf04206d9c3710dfbc46aa9850450ab405` with the exact existing FP patch.
The investigation's older PyTorch checkout is not evidence for executed code.

[Acceptance results](stage3/artifacts/results.json): **21 category checks,
50 actual CubeCL compiler invocations, 149 launches matching the native counter**.
[Components](stage3/artifacts/components.json): eight separately compiled artifacts,
35 GPU checks, canaries and descriptor/physical rejection checks. Counts belong to
separate fresh processes; the HIP source probe is additional compiler evidence.
**Independent native GPT-6.1/high review completed under explicit user
authorization** after review-mcp failed with HTTP 429 / provider error 1308.
It reproduced two bugs: misplaced shared-memory launch dispatch and omitted
masked-result SSA dependencies. Both are fixed and rechecked by the reviewer.
[Review disposition](stage3/artifacts/review-disposition.md) records the findings,
corrections and limits. This is a reviewed experimental gate; changes are uncommitted.

## Supported reduction surface

| Concern | Executed surface | Boundary |
|---|---|---|
| Operator | Sum; mean through sum/scaling; sum-of-squares | No extrema, product, Welford, arg/tuple/scan or online-softmax reduction |
| Storage | FP32, FP16, BF16 | No FP64/integer tensor reduction |
| Accumulator | FP32 local partials and shared tree | No target-implied relaxed math; lowp result rounding remains explicit |
| Domains | One flattened outer range and one logical reduction range | Multiple unsimplified reduction dimensions reject before fusion |
| Layout | Row, column, transpose, sliced offset/non-unit stride, stride-zero | Stage 2 direct signed-i64 index grammar; no contiguous input copy |
| Extents | Dynamic B/N; 0/1 specializations; 31/32/33, 255/256/257, 4099, 65539 | Per-kernel reduction extent ≤1,048,576; normal Dynamo/common SIMD guards remain |
| Physical strategy | 256 scalar units, one workgroup per outer output, shared binary tree | No planes/vectorization/persistent topology/autotuning |
| Fusion | Reduction→scalar epilogue, reduction→full-size pointwise; two sums+consumer | One externally observable output; one to four external reads; different reduction domains veto |
| Multi-stage | Inductor's natural transposed partial→final sum | Existing scheduler/wrapper allocation and sequencing; no independent general split planner |
| Backward | Row-sum gradients and real RMSNorm reduction backward, arbitrary gradients | No column-gradient, full LayerNorm/Welford or learned-scale/bias acceptance claim |
| Masks | Real cat→sum lazy load regions; split-axis semantic mask; physical loop tails | No reduction/barrier or store effect inside a semantic masked subblock |

The validator bounds a schedule to four scalar sums. Runtime acceptance exercises
one and two; arbitrary three/four-reduction DAGs are not a coverage claim.
Naturally selected extern/nop handling remains inherited. Accepted profiler
regions contain only CubeCL generated kernels, without ATen pointwise/reduction
or Triton kernels.

## Live call path

```text
torch.compile(..., backend=stage3.backend.compile_backend, fullgraph=True)
 → Dynamo OutputGraph._call_user_compiler
 → compile_fx / AOTAutograd forward or lazy backward compiler
 → GraphLowering.run
 → lowering.sum_ / mean → Reduction.create and possible multilayer decomposition
 → GraphLowering.codegen / _update_scheduler → Scheduler._init
 → selected CubeCLScheduling.check_backend_available
 → _pre_fusion_custom_pass: operation/dtype/index/domain checks
 → Scheduler fusion → CubeCL can_fuse → unchanged SIMDScheduling.can_fuse
 → SIMDScheduling.codegen_node / _codegen_nodes
 → generate_node_schedule → SIMDKernelFeatures
 → get_tiling_and_scores: logical {x: outer, r0_: reduction_extent}
 → create_kernel_choices → CubeCLKernel(SIMDKernel)
 → codegen_node_schedule_with_kernel, split_and_set_ranges, indexing passes
 → SchedulerNode.codegen → LoopBody / LoopBodyBlock / V.ops
 → CSEProxy.reduction and store_reduction → typed semantic SSA
 → semantic descriptor + ReductionPlan → ordinary Python wrapper
 → stage3.runtime → existing cubecl-stage0 CLI → descriptor3::build
 → KernelBuilder / Scope → KernelDefinition → CppCompiler<Cuda>
 → existing NVRTC (--fmad=false) → PTX → existing module loader
 → validated tensor/scalar arguments → shared Kernel._enqueue
 → stage3 pointer/grid/shared packing → original launch_args → cuLaunchKernel
```

The [source trace](stage3/artifacts/source-trace.json) records actual call locations
and exact function/file hashes. Full replay stacks/source snapshots remain local
and ignored. [Lowering](stage3/artifacts/lowering.json) preserves real bodies and
hints. Graph Tier 2 coverage reports no recorded gaps in local implementation;
installed PyTorch and the CubeCL clone are deliberately excluded, so their actual
source was read directly. Graph coverage is not proof of completeness.

## Semantic reduction model

Schema 3 extends the Stage 2 typed DAG with `outer`, `reduction_extent` and a
`reduction` operation containing source/result/accumulator dtypes and `sum`.
Logical `linear` now denotes the outer coordinate in this schema; a separate
`reduction` index leaf denotes the reduction coordinate. Stores declare the
logical `outer` or full `reduction` iteration domain. These describe tensor
semantics, not block/thread ownership. Scalar runtime arguments remain separate.

Representative replay:

```text
row:    load(x, outer*N + r) → reduction(dst,src,sum,value)
        → store_reduction(y, outer, result)
column: load(x, outer + r*N) → same semantic sum/store
mean:   sum node → DisableReduction → load forwarded sum
        → index_expr(N,float32) → divide → add → store → EnableReduction
```

`index_expr(N,float32)` is integer symbolic metadata converted explicitly to a
floating value. It is not a floating runtime scalar ABI. The shared signed-i64
grammar, floor/mod semantics and final pointer conversion remain unchanged.

Common CSE/store forwarding and removal of kernel-local buffers are retained.
Lowp sum results round to their semantic type before a fused consumer reuses
them. No CubeCL IR object, CUDA builtin or arena identity appears in replay.
Dependency closure follows both operation arguments and masked-region result
edges, including nested results referencing enclosing CSE values. Five descriptor
boundary GPU tests cover empty masked bodies returning enclosing constants, with
true/false and nested predicates, exact output checks and storage canaries.
The small `finalize_descriptor` hook reuses Stage 2 serialization without copying
it or mutating another backend's trace buffer.

## Physical reduction model

[ReductionPlan](stage3/plan.py) independently specifies:

```text
iteration_mapping = one_workgroup_per_outer
workgroup = (256,1,1), scalar local vector width = 1
elements_per_unit = strided_loop_step_workgroup
reduction_mapping = shared_binary_tree
accumulator_dtype = float32, shared_elements = 256
barriers = uniform_after_publish_and_each_tree_level
result_owner = unit_zero_store_all_units_read
bounds_strategy = loop_reduction_coordinate_less_than_extent
phases = 1 per generated kernel
```

Rust binds the logical outer coordinate to CubePosX and assigns each unit
`r=unit; r<extent; r+=256`. Every unit initializes a private FP32 accumulator to
+0. Units with no elements do no loads and contribute that neutral value.
Each publishes one partial into a 256-element shared array. A barrier follows
publication, and tree strides 128→64→…→1 each have a predicated half-workgroup
add followed by an unconditional workgroup barrier. All units then load shared
element zero; another uniform barrier protects scratch reuse for a later sum.

An outer-only store runs on unit zero. A full-size epilogue runs another parallel
strided loop, reusing the workgroup-uniform reduction result. Pointwise input
expressions may be recomputed rather than retained across loops. Work is O(N)
per reduction/output pass, without single-thread serial or O(N²) computation.

No barrier is placed inside a variable-trip-count reduction loop, semantic mask,
or lane predicate. The enclosing outer-bound branch depends only on CubePosX
and runtime metadata and is workgroup-uniform. Component negatives reject a
divergent barrier policy, unsupported topology/accumulator, masked reduction,
missing compute conversion and an outer store depending on a varying coordinate
**before constructing IR**. The public C++ path needs `branch::RangeLoopOp`;
direct `scf::ForOp` was not emitted by that compiler and was replaced before
acceptance. This was an API selection correction, not a compiler redesign.

Plane/subgroup operations were inspected: FSumOp/shuffles/broadcast synchronize
at plane scope; Scope exposes shared allocation, SyncScope::Cube and mutable
locals. Stage 3 chooses the explicit shared tree, avoiding an untested active-lane
or target-plane-width contract. No plane execution or LLVM reduction claim is made.

## Numerical and accumulator policy

| Case | Input storage/semantic | Accumulator/partial | Result semantic/store | Barriers/conversions |
|---|---|---|---|---|
| FP32 sum | FP32 | FP32 | FP32 | Normal FP32 additions |
| FP16/BF16 sum | Respective lowp type | FP32 | Requested lowp type | Widen load; round reduced result; fused consumers widen again |
| Lowp mean | Lowp input, generic lowering promotes to FP32 | FP32 sum | Lowp mean/store | Divide in FP32 before generic result rounding |
| Natural lowp split | Lowp original input | FP32 intermediate tensor | Lowp final result | Retain actual Inductor intermediate dtype and masks |
| Accepted RMSNorm | Lowp or FP32 original; authored `x.float()` | FP32 statistics and saved inverse norm | Original output/gradient dtype | Explicit final conversion; arbitrary upstream gradients |

PyTorch documents that floating reductions need not be bitwise identical across
evaluation orders/platforms ([numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)).
Our selected tree is an explicit sum evaluation order; it does not enable a
compiler reassociation permission or waive special-value handling. The live
`_strict_sum_layout_eligible` machinery is specifically tied to Triton eager-order
emulation. Stage 3 rejects `config.numerics='strict'` deliberately rather than
claiming that ordering. No new PyTorch backend-name condition is added.

FP32 acceptance uses `rtol=2e-5, atol=2e-5`, allowing ordinary sum/tree and rsqrt
rounding differences; pointwise Stage 2 retains its tighter tolerance. FP16 uses
`0.002/0.001`, BF16 `0.016/0.008`, reflecting storage rounding. Tolerances were
chosen before the matrix; failures were not repaired by widening them. NaN and
both infinity classifications and dtypes are exact checks. Tested all-±0
cooperative sums produce +0; cancellation and empty/single-element cases pass.
No NaN payload, universal signed-zero reduction or all-overflow-order contract
is advertised. A lowp accumulation fixture confirms FP16 sum can overflow on
`257*1000`, while its promoted mean remains finite and matches eager.

The full FP patch hash is
`9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2`.
Availability checks verify the pinned checkout's complete diff against that exact
stack. No CubeCL source modification is added in Stage 3. Unsafe partial folds
stay removed; default InstCombine contraction remains disabled, LLVM ordinary
arithmetic has no contract flags, and NVRTC retains `--fmad=false` without fast
math. The FP-hardening suites rerun unchanged.

## Fusion and normalization evidence

Normal scheduling accepts actual reduction and consumer nodes, then supplies
DisableReduction/EnableReduction markers to common replay. The scalar mean
epilogue executes with its sum in one kernel. Inference RMSNorm executes the
sum-of-squares/statistic/full-output computation in one kernel. A separate
three-node fixture, `x.sum(-1)+(x*x).sum(-1)`, naturally fuses two reductions and
a consumer and exercises shared scratch reuse in one kernel.

Training saves the reduced inverse norm: normal external-output legality keeps
that statistic kernel separate from the final multiply, giving two forward
kernels. Backward fuses a genuine dot-like sum with the input-gradient epilogue,
giving one CubeCL reduction kernel for all three storage dtypes. Fresh saved
tensors are used for profiling, preserving AOTAutograd's donated-buffer contract.
Reference work is outside acceptance profiler regions. A neighboring two-output
sum/mean graph triggers the existing external-output veto and executes two
supported CubeCL kernels. Scheduler thresholds/realization policy are unchanged.

## Dynamic shapes and natural split sequencing

For each dtype, `(B,N)` changes through `(3,31),(4,32),(2,33),(5,255),(3,256),
(4,257),(2,4099),(3,65539)` with no new artifact after the first. Returning to
257 reuses it. N=1 specializes to pointwise, N=0 to a generated zero-fill output
kernel; B=0 has an empty output and no launch. Components additionally reuse an
actual cooperative kernel at reduction lengths 0 and 1.

Workgroup topology is compile-time fixed; logical extents/strides are runtime
signed-i64 scalar fields. Grid is the runtime outer extent. The existing
InfoBuilder ABI carries a reserved first scalar `launch_units=outer*256`, then
logical symbols and independent accessible buffer lengths. Runtime symbol values
are absent from the descriptor/artifact key and present in metadata-cache keys.
Representative guards include `2 <= N <= 1048576`, B's 0/1 specialization and
common SIMD `B*N <= 2147483647`. The physical/index arithmetic remains i64;
these conservative common guards are retained, not erased.

The transposed fixture naturally lowers to partial and final reduction nodes,
with a PyTorch intermediate shaped `[B,3]`. Its first phase contains a semantic
mask for a nondivisible partition; FP16/BF16 partial storage is FP32. Both
phases use the same validated scalar cooperative plan and compiler/launcher.
Inductor's normal wrapper owns allocation, dependency ordering and freeing;
record_stream protects intermediate lifetime and current-stream enqueue orders
the two phases. There is no CubeCL-owned partial buffer or device synchronization.

This is not a universal split-reduction API. A static N=1,048,577 fixture is
rejected before fusion because its natural split emits an unsupported int32
index_expr; it is recorded as an index capability failure, not a compiler failure
or proof that every larger reduction is impossible. The direct physical-plan
component rejects reduction extents beyond 1,048,576 before enqueue. Future
split/layout coverage belongs at these existing capability/descriptor boundaries.

## Compiler/runtime evidence and portability probe

[Representatives](stage3/artifacts/representatives) preserve semantic descriptor,
physical plan, schedule/replay, CubeCL IR, CUDA C++, PTX/signature, NVRTC options,
launch arguments and compiler/patch hashes for row/lowp/dynamic/tail, column/split,
mean fusion, normalization/backward and masks. Full raw profiler traces, caches,
logs and unchanged upstream source snapshots remain local and ignored.

Every cooperative launch has `(256,1,1)` units and 1024 dynamic shared bytes.
Independent C counters, incremented only after a successful nonempty
`stage3_launch`, verify 131 reduction launches and 134,144 requested shared bytes
in the matrix, plus 35/35,840 in components. Python planned-launch traces are
checked against these actual native-entry counters.

The initial pre-review run was insufficient evidence for shared allocation:
dispatch had mistakenly been placed in `_prepare`, which Stage 3 overrides,
while `_enqueue` reached old entries requesting zero shared bytes. Its numerical
passes and planned 1024-byte logs did not establish valid native shared-memory
allocation. Those launch claims are superseded by the corrected dispatch and
fresh native assertions; the complete matrix and baseline suites were rerun.
Driver function attributes record physical registers and local bytes; measured
representatives have no local-memory allocation. CUPTI's occupancy details also
record 1024 allocated shared bytes even when its top-level shared-memory field
is zero; occupancy accounting alone did not prove the old native request.
[One gross timing probe](stage3/artifacts/timing.json) compares the same
row-sum shape with eager PyTorch; CUDA-event intervals include Python enqueue
gaps and are not a pure-kernel benchmark or performance ranking. The simple tree,
few-output underutilization, uncoalesced column loads, repeated expression work,
metadata validation/caching and host overhead are left for optimization stages.

The same executed row descriptor/plan constructs CubeCL IR and compiles through
`CppCompiler<Hip>` to [HIP source](stage3/artifacts/noncuda/kernel.hip), including
shared memory and workgroup synchronization. No HIPRTC/device/ROCm launcher is
tested. The semantic descriptor remains target-neutral; a 256-unit topology and
its feature/size requirements still need validation for any other target.

Original view pointers and current stream are asserted. PyTorch owns all tensor
and metadata storage; no tensor copy into CubeCL storage, CubeCL stream, or
device-wide synchronization is introduced. Capture remains unsupported for the
Stage 3 wrapper, as in Stage 2. Process-lifetime modules and unbounded metadata
caching retain the established limitations.

## Source modifications

**Additional PyTorch source patches: zero. Additional CubeCL source patches: zero.**
The four Stage 1.5 availability-overlay changes and existing FP stack remain.

| Owner/files | Reason and new responsibility | Classification |
|---|---|---|
| `stage3/backend.py`, `capabilities.py`, `plan.py` | Reduction replay, sum/result policy, logical coordinates, capability/fusion boundaries and physical plan | Experimental out-of-tree backend, existing Inductor extension points |
| `stage3/runtime.py` | Bind plan to existing pointer/scalar ABI; validate outer/reduction domains and record launch/compiler evidence | Backend-specific adapter; compiler/stream/lifetime route retained |
| `stage2/backend.py` | Parameterize input count and add final-descriptor/trace hooks | Small reusable local hooks; Stage 2 defaults unchanged |
| `stage2/capabilities.py` | Parameterize the common checks for reduction replay and integer-shape→float values | Reuse typed/index validation without admitting those features in Stage 2 |
| `stage2/runtime.py` | Separate launch-extent validation from common layout/argument validation | Existing pointwise behavior unchanged; reduction adapter supplies its plan |
| `stage0/src/descriptor3.rs` | Validate schema 3, select dependency closures, build cooperative IR | Backend physical builder using public CubeCL primitives |
| `stage0/src/descriptor2.rs` | Expose scalar/index/mask emission helpers; support actual rsqrt replay and output-only zero fill | Shared bridge semantics, not copied Triton generation |
| `stage0/src/main.rs`, `src/lib.rs` | Dispatch schema 3; accurate manifest; allow one-buffer metadata for empty-reduction fill | Bounded experimental descriptor/ABI extension |
| `stage0/launcher.c`, `bridge.py` | Extend shared original enqueue with checked grid/shared packing; output-only fill; read-only driver attributes | Host ABI extension required by actual shared IR; original entries still pass shared=0 |

No `config.triton` namespace sweep is made. Default persistent/cooperative
heuristics remain false; advanced mixed/nested/template/combo paths remain gated.
The typed DAG overrides `disable_reduction` instead of flushing Triton-style
source buffers: logical stage markers remain generic, and physical loops are
created by Rust. Physical mapping never interprets XBLOCK/RBLOCK as local vectors.

## Regression evidence and remaining limitations

[All nine baseline gates](stage3/artifacts/regressions.json) pass: Stage 0 C++ and
LLVM, Stage 1, Stage 1.5 boundaries, complete Stage 2 and components, FP IR,
FP runtime and FP Inductor. They retain streams, allocator pressure, Stage 0
capture/prepared-thread tests, 34 Stage 2 category checks and its 94 compiles/165
launches, seven FP IR groups/272 unknown-operand combinations, 100 FP runtime
cases and real contraction backward tests. Stage 2 also reruns after its final
serialization hook is added; its matrix remains unchanged.

Stage 3 records three absent package probes, eight optional blocked imports,
zero loaded Python/native Triton modules, four false GIL checkpoints and no
`PYTHON_GIL` setting. Components, baseline suites and the timing probe independently
retain free-threaded/no-Triton evidence. This is not a claim of concurrent Inductor
compilation safety. Clippy with all targets/warnings-as-errors, rustfmt, Ruff,
Python syntax and diff checks pass. Review-mcp failed on provider quota; the
user-authorized native GPT-6.1/high review then completed, verified both fixes and
returned no further actionable findings. It did not independently rerun every
baseline suite or run CUDA racecheck. Parent verification covers the complete
matrix and all nine baseline gates. Changes remain uncommitted.

Limits remain: one logical reduction dimension; bounded sum only; limited natural
split patterns; unknown/static int32 mask replay; no general fallback, reduction
algorithm selection, extrema/Welford/arg semantics, arbitrary mutation/aliasing,
portable launcher, concurrent/disk cache or capture support. Numerical equality
is the stated default reduction policy, not eager bit ordering or all extreme
overflow equivalence. No Stage 4 feature or performance tuning is started.

## Completion decision

**Exactly one next direction: 1. Stage 4 — RNG, atomics/indirect operations and
explicit fallback boundaries.** The bounded Stage 3 gate now proves nonlocal
values, uniform barriers, reduced-value reuse, dynamic domains, legal fusion and
compiled normalization backward without breaking the semantic/physical split.
The remaining reduction classes are coverage/algorithm work rather than a
demonstrated architectural blocker. Stage 4 is not started automatically.
