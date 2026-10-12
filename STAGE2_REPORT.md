# Stage 2: pointwise semantics, layouts and runtime extents

12 October 2026, Europe/Budapest. Beads task `pytortch-backend-experiments-1rr`.

Subsequent local hardening is documented in [FP_HARDENING_REPORT.md](FP_HARDENING_REPORT.md).
It supersedes this report's negative-zero add identity and implicit-FMA limitations:
the full add-zero simplifier is removed, contraction is disabled by default, and
the remaining audit defects are corrected. The acceptance surface stays unchanged;
the Stage 2 suite and representative artifacts have been regenerated with that patch stack.

## Result

**Stage 2 gate passed** for the bounded surface below. Real Dynamo/AOTAutograd/
Inductor replay now drives CubeCL float32/float16/bfloat16 pointwise kernels with
explicit storage/compute conversions, comparisons/selects, broadcasting, direct
strides, runtime extents and lazy semantic masks. A genuine two-SchedulerNode
fusion is accepted by normal SIMD fusion and executes one CubeCL kernel.
No reduction or optimization stage was started.

All four baseline reports were read in full before changes. The Stage 0 manual
path and Stage 1/1.5 entry point remain independent; Stage 2 uses the local,
experimental `stage2.backend.compile_backend` spelling. It delegates to normal
Inductor capture and the existing registration/availability boundary.

Executed versions remain PyTorch **2.14.1+cu130**, wheel commit
`5c4886908584029761b579af026dcfb627c84070`, CPython **3.14.5t**, CUDA 13.0/NVRTC
13.0.88, RTX 3090/SM86, Rust 1.97.1, and CubeCL
`6bba5fdf04206d9c3710dfbc46aa9850450ab405`. CubeCL now has one precisely retained
numeric correctness patch, explained below; its compiler/runtime architecture
has not changed. The older investigation's PyTorch checkout is not the live wheel.

The final CUDA matrix passes 34 semantic-category checks: **94 actual compiler
invocations and 165 actual native kernel launches**. Native atomic counters agree
with Python instrumentation. The separate component process tests index widths
and nested masking through the same compiler/launcher. These counts include many
intentionally distinct fixture/dtype/static graph descriptions, not recompilation
for every dynamic size. Evidence: [results](stage2/artifacts/results.json),
[coverage](stage2/artifacts/coverage.json), [components](stage2/artifacts/components.json).

## Supported semantic surface

| Concern | Implemented and executed surface | Explicit limit |
|---|---|---|
| Unary | sin, cos, neg, exp, sqrt, reciprocal | Other ops such as tan remain capability errors |
| Binary | add, sub, mul, true division | Floating arithmetic; no integer/bool tensor arithmetic |
| Comparison | eq/ne/lt/le/gt/ge | Floating inputs, or internal signed index values; bool result |
| Selection | where | Scalar value selection; protected loads use separate lazy regions |
| Constants/casts | f32/f16/BF16/bool/internal i64; explicit numeric casts | No float64, bit reinterpretation, broad integer tensor storage |
| Storage | float32, float16, bfloat16, bool | Integer values are for index/mask computation, not integer tensors |
| Compute | float32 for the three floating types; bool predicates; signed i64 indices | No vectors or mixed-precision reduction accumulator |
| Layout | Direct strided rank 0..4 representation; executed 1D/2D, transpose, sliced offset views, stride zero | Positive/nonnegative direct strides; no indirect indexing, arbitrary alias/mutation or negative strides |
| Broadcasting | `[B,N]+[N]`, `[B,N]+[B,1]`, zero-stride input | No expanded input allocation; broadcast gradients needing reductions are outside scope |
| Indices | constants, logical linear coordinate, runtime scalars, Add/Mul (including subtraction by negative coefficient), FloorDiv, Mod/ModularIndexing, Identity; small positive integer powers normalized to multiplication | Provably positive divisors, signed i64 arithmetic within range; unsupported expression nodes reject before fusion |
| Dynamic | Backed symbolic dimensions and emitted symbolic strides through `dynamic=True` | Normal Dynamo specialization/guards remain; no unbacked-shape or arbitrary scalar-value ABI claim |
| Masks | Real concatenation's two lazy masked loads/fills; component-tested nested lazy regions and predicated stores | Scalar literal fill, bounded structured regions; no generalized mask/control-flow surface |
| Fusion | Single-node inlining and real normal two-node fusion, one external output | One to four external input buffers; externally observable multi-output fusion is vetoed |
| Physical plan | One scalar per unit, 256 units, explicit tail guard | No reduction mapping, vectorization, tuning or persistent kernels |

No reductions, atomics, RNG, indirect/gather indexing, GEMM/templates, tensor cores,
general fallback, combo kernels or portable PyTorch launcher is implemented.
Naturally selected extern scheduling remains inherited; the accepted fixtures
need no pointwise ATen fallback and the warmed profilers show none.

## Architecture and dtype policy

```text
Dynamo → AOTAutograd → GraphLowering → Scheduler / pre-fusion capability
 → CubeCLScheduling(SIMDScheduling), normal fusion delegate
 → SchedulerNode.codegen → LoopBody / V.ops / typed CSE values
 → schema-2 semantic descriptor + separate PhysicalKernelPlan
 → stage2.runtime bridge adapter
 → validated Rust descriptor2 → KernelBuilder / Scope / CubeCL IR
 → existing CppCompiler<Cuda> → existing NVRTC adapter → PTX
 → existing module loader / stream readiness / record_stream
 → native pointer packing → shared Stage 0 enqueue → current PyTorch stream
```

Semantic descriptors contain no CUDA builtin, CUDA stream, CubeCL arena handle or
IR object. Accesses are expression trees over logical `linear` and scalar slots.
The adapter attaches the checked physical plan and entrypoint. Runtime tensors
and scalar values are separate arguments, never embedded into the dynamic kernel
cache key. The physical builder owns coordinate builtins and the tail branch.

| PyTorch storage | Semantic/result dtype | CubeCL compute | Conversion points | Store |
|---|---|---|---|---|
| float32 | Generic lowering/promotion yields f32 | f32 | Identity where appropriate | f32 |
| float16 | f16 unless generic promotion/cast changes it | f32 | load f16→f32; explicit precision barrier f32→f16→f32; constants round to f16 then widen | f16 |
| bfloat16 | BF16 unless generic promotion/cast changes it | f32 | load BF16→f32; explicit precision barrier f32→BF16→f32; constants round to BF16 then widen | BF16 |
| bool | comparison/predicate bool | bool | bool numeric casts where requested | bool |
| internal index | signed i64 | signed i64 | logical expression→i64; final address→CubeCL IndexType/U64 | No integer tensor store |

This uses generic PyTorch `_prims_common.get_computation_dtype` and lowered
semantic casts, plus `torch.promote_types` for typed operands. It never invokes
`utils.upcast_compute_type` or adds a backend name to dtype rules. `CubeCLValue`
tracks actual compute dtype and a distinct semantic dtype; descriptors record
both. Rust validates storage types, cast results, operands, masks and SSA before
constructing IR. Existing typed `CSEVariable` pass-through suffices; **no new
PyTorch type framework or CSE patch** was required.

Live `lowering.make_pointwise` and `config.emulate_precision_casts` explicitly
specify f32 opmath for f16/BF16 and optional eager rounding barriers. Stage 2
chooses that existing eager-precision mode. For example half sine/add replays
`load → f16/f32 conversions → sin → round/widen → constant(f16) → add →
round/widen → store(f16)`. Explicit casts and mixed f16/f32 or f16/BF16 promotion
are executed. Floating `1/2` lowers to multiplication by 0.5; the two-input
fixture forces actual `truediv`. `0/x` additionally exposed `reciprocal` replay;
it emits an ordinary f32 `1/x`, not an approximate reciprocal instruction.
NaN comparisons use float comparison IR, yielding false for ordered comparisons
and true for `ne`; where uses `SelectOp` only after its value operands are safe.

Tolerances: f32 rtol/atol `2e-6`; f16 `0.002/0.001`; BF16 `0.016/0.008`, with
`equal_nan=True`. Bool comparisons and the signed-index component results are
exact. Separate IEEE tests check sign bits, not just closeness, for finite
results of neg/add-zero/sub-self/mul-zero/zero-divide. Arbitrary upstream gradients
cover sine, exp/neg, sqrt, casts, where, binary division and supported layouts
for all three floating types. Broadcast-to-original-base gradients would require
reductions and remain deliberately unsupported.

## Index lowering and layouts

The descriptor's index policy is **signed i64**; CubeCL memory indexing uses U64
only at the final pointer conversion. Generic SIMD still installs conservative
32-bit selection guards before this backend sets its actual index type to i64.
Those guards restrict graph reuse; they do not narrow descriptor arithmetic.
This is recorded rather than hidden or patched into a new generic policy API.
The experimental scalar/expression validator accepts `[-2^63+1, 2^63-1]` and
rejects overflowing metadata arithmetic; it does not implement wrapping integer
tensor arithmetic or general runtime floating scalar parameters.

Representative live expressions:

| Fixture | Inductor indexing → descriptor → CubeCL |
|---|---|
| `[B,N]+bias[N]` | `p0*N+p1`, `p1` → flattened `linear`, `ModularIndexing(linear,1,N)` → signed modulo → IndexOp |
| `[B,N]+column[B,1]` | `p0*N+p1`, `p0` → `linear`, `FloorDiv(linear,N)` → signed quotient → IndexOp |
| Sliced input `[1::step]` | `p0*symbolic_stride` → `Mul(linear,scalar_slot)` → IMulOp → IndexOp |
| Stride-zero expand | lowered input coordinates omit the expanded dimension → modulo for input, linear output | No broadcast-vector instruction |
| Concatenation | `Identity(p0)`, extent `s+1`, predicates `p0<s` and `p0>=s` → logical trees + typed comparisons + structured lazy load regions |

Transpose may be traversed in its physical contiguous order by Inductor's normal
loop reordering. The executed descriptor and output layout reflect that fact;
we do not falsely require a transpose to retain the source tensor's original
row-major logical traversal. The input pointer is unchanged. A distinct sliced
view forces non-unit symbolic multiplication, and broadcast fixtures force
floor/mod address transforms. No `.contiguous()` or expanded-tensor copy exists.

For a positive divisor `b`, floor division emits truncating `SDivOp(a,b)` and
`SRemOp(a,b)`, then subtracts one from the quotient only when the remainder is
negative. This avoids both truncation/floor confusion and overflow from computing
`a-floor_mod(a,b)` near i64's lower boundary. `SModFloorOp` uses CubeCL's signed
floor-mod polyfill: adjust a nonzero remainder whose sign differs from the divisor.
Scalar declarations preserve proven positive domains; Rust requires positive
constant/scalar divisor provenance, and the launch adapter checks declared-positive
runtime scalars before enqueue. Zero/negative constants, undeclared-positive scalar
divisors and violated runtime domains have neighboring rejection tests.
Subtraction is an Add/Mul tree with a signed negative constant, not unsigned wrap.

Unit/reference checks cross 2^31/2^32 with negative and positive operands. Real
GPU component kernels use runtime scalar values beyond 32-bit and near the i64
boundary, storing small modulo results so no huge allocation or lossy float
comparison is needed. Physical coordinate construction widens block position
before `block*256 + unit`; it never relies on a potentially overflowing u32
absolute-position multiply. Bounds remain `linear<numel`.

Runtime validation matches guarded sizes/strides, validates available storage
spans, device/dtype and disjoint output. `Tensor.data_ptr()` already refers to the
view's storage offset; that offset is not added twice. Inductor's own layout offset
is retained in its expressions. Input slice offset 3 and symbolic strides 2/3/5
are executed. This is direct-stride support, not a claim about every PyTorch view.

## Dynamic extents and ABI

`dynamic=True` captures real SymInt inputs. Scalar slots carry symbolic numel,
dimensions and emitted strides in signed-i64 InfoBuilder scalar fields. Each
buffer independently carries its accessible element span. Numel comes from the
logical schedule, not the first input length (essential for broadcast and cat).
Grid is `ceil(numel/256)`; zero skips launch. Device metadata is allocated by
PyTorch and uploaded once per process-local metadata key; x/y are never copied.

For each dtype's sine forward/backward, sizes 255→256→257→1,048,583 reuse the same
graph and forward/backward artifacts without another compiler invocation. Size
1 gets a specialized artifact; size 0 gets its normal empty graph, no launch.
Returning to 257 reuses the dynamic graph. Repeated equal dimensions canonicalized as `B**2` also reuse one graph/artifact. Both row and column broadcast graphs
reuse artifacts across `(3,255),(5,256),(2,257),(4,513)`; emitted symbolic stride
changes 2→3→5 reuse a guarded graph. Guard/log evidence is retained locally in
`verification.log` and `guards.json`, including dtype/stride and lower/upper bounds.

Dynamo specializes 0/1, tensor dtype, rank, layout class and requires shape
relationships for broadcasting. A representative common SIMD guard is
`2 <= x.size()[0] <= 2147483647`. The backend does not erase those guards or claim
fully unrestricted dynamic shapes. [PyTorch's dynamic-shape documentation](https://github.com/pytorch/pytorch/blob/v2.14.0/docs/source/user_guide/torch_compiler/compile/dynamic_shapes_core_concepts.md)
provided a lead; live guards and actual graph/artifact counts are the evidence.

Cache keys retain static semantic/index structure, dtype/conversion operations,
physical plan and device. Runtime symbol *values* live only in scalar arguments
and the metadata key. Process-local compilation dedup is unchanged in principle;
there is no cross-process binary-cache or concurrent compiler safety claim. The
metadata cache retains one small device tensor/event for every distinct runtime
shape/stride tuple and has no eviction; a long-lived workload with unbounded shape
variation would grow that cache.
Toolchain is fixed during a process. Representative manifests retain the CubeCL
patch hash; compiler outputs are rebuilt by fresh test processes.

Binary division backward required up to four external reads: tangent, two saved
values and a previous gradient buffer. A bounded six-pointer native entry packs
those four reads, one output and info. Stage 0/1 pointer entries remain unchanged.
Their shared `launch_args` function still owns context checking, grid limits,
launch counters and cuLaunchKernel. New metadata uses **InfoBuilder**, not manually
invented scalar offsets. CUDA half/BF16 compilation adds existing bundled CUDA
headers to NVRTC's include search; no toolchain installation was needed.

## Masking and fusion evidence

Real `torch.cat([x,x[:1]])+1` lowers to two masked subblocks. Their loads execute
only inside respective true branches; else branches supply zero. A subsequent
where selects the already safely computed values. Physical `linear<numel` is an
outer execution guard and is represented separately from these semantic masks.
The backend scopes semantic CSE and common store caches for each lazy region.
No where operation speculates a protected memory access.

No clean accepted fixture was used to claim a real nested masked-store replay.
The narrow descriptor/IR boundary is instead tested on GPU: nested predicates
`linear<128` / `linear<64` protect `input[2*linear]`; an initial fill store is
conditionally overwritten by stores inside the region. False lanes retain the
fill, and out-of-range loads in inactive lanes never execute. This establishes
nested lazy body/predicated load and store components without inventing a Dynamo
graph. Fill must be a scalar literal; more general masked results remain gated.

The genuine fusion fixture performs 20 ordinary `sin(a)+0.01` steps, then returns
`a*a+a`. Existing lowering's normal operation-count/reuse heuristic realizes an
intermediate, giving `op0` and `op1`. No scheduler or realization threshold is
modified to force this. Pre-fusion accepts both → CubeCL capability accepts →
unchanged `SIMDScheduling.can_fuse` accepts → normal fused replay includes both
nodes → one CubeCL GPU kernel. Its warmed profiler records one kernel.

Stage 1.5's tiling override was only tested as an instance call. Live fusion calls
it through a class method; Stage 2 makes the override a class method to satisfy
both actual call paths. Fusion output checks now include graph outputs and users
outside the candidate group: a consumed intermediate can still be externally
observable, as binary backward exposed. A two-output branch is vetoed and executes
two supported CubeCL kernels. This preserves normal scheduler legality and avoids
late ABI rejection. Externs, incompatible ranges and over-limit ABIs remain vetoes.

## Capability boundaries and compiler corrections

Input selection rejects unsupported tensor dtype/device/rank. Pre-fusion checks
operation graphs (including masked subblocks), storage/value types, direct index
forms, reductions, indirect variables and bounded ABI. Fusion separately checks
external consumers/output count and normal SIMD legality. The physical adapter
rejects any non-scalar-256 plan before Rust compilation. Rust rejects malformed
SSA/operand/type/region descriptions before constructing IR. Compiler/NVRTC/native
failures retain their distinct errors; no retry or Triton route exists.

Negative tests include float64/int64 tensors/rank5 at selection; reductions, tan
(including a real masked cat body), non-int64 numeric runtime value expressions,
indirect indexing, five input buffers, bool arithmetic, float64 casts and minimum
before fusion. Assertions prove no later schedule/kernel emission and no compiler
invocation. Existing Stage 1.5 non-scalar physical-plan rejection remains tested.
This is not operator fallback before lowering inlining; that limitation remains
precisely as documented in Stage 1.5.

Special-value execution exposed actual CubeCL correctness problems:
`x-x→0`, `0*x→0`, `0/x→0`, and adding positive zero as an identity violate
NaN/infinity or signed-zero semantics. [cubecl-ieee.patch](stage2/cubecl-ieee.patch)
corrects only those rules in `cubecl-ir/src/dialect/math.rs`: retain actual
constant/constant folding, drop the unsafe runtime identities, and recognize
only negative zero as the IEEE addition identity. The patch is generic numeric
correctness, not an Inductor backend-name case. It is applied to the pinned local
checkout and retained as authored source; unmodified upstream copies stay ignored.

Inductor's `fx_passes/joint_graph.remove_no_ops` also drops `x+0`, changing `-0`.
Stage 2 disables the existing `joint_graph_constant_folding` setting locally.
This avoids a new source patch and trades an optimization for tested semantics.
CubeCL's explicit FMA combine can still change ordinary last-bit rounding; numeric
tolerances cover the tested f32 cases and low-precision barriers remain explicit.
This is a bounded numeric gate, not bitwise equality for every floating expression.

## PyTorch modifications and baseline regressions

**No additional PyTorch source modification.** The Stage 2 availability override reuses the Stage 1 dependency check and also validates the existing libraries export the schema-2 metadata and pointer-packing ABI, before grouping.
The four Stage 1.5 overlay changes
and scheduler-owned availability hook are reused unchanged. Registration selects
the experimental scheduler; type/index/fusion/launch hooks remain out of tree.
No backend-name check was added for dtype or requirements.

Bridge changes are bounded: schema-2 Rust builder/validation; CLI dispatch and
optional HIP source compiler probe; independent length/scalar InfoBuilder C ABI;
a six-pointer packing entry sharing original enqueue; optional NVRTC include paths;
and extraction of the existing stream/lifetime block into `Kernel._enqueue` so
both validated ABIs reuse it. Manual Stage 0 builder, schema-1 builder and LLVM
route remain. Stage 1 implementation files are unchanged.

[Regression results](stage2/artifacts/regression.json) retain both complete Stage
0 routes (canaries/special values/streams/five pressure trials/capture replay/
prepared barrier threads), the full Stage 1 suite and Stage 1.5 requirement/fusion/
physical-plan probes. Stage 1 still has three compiles/44 native launches and one
CubeCL forward kernel. Stage 2 also passes three delayed side-stream/allocator
pressure trials; pointer/stride/storage-offset/current-stream assertions are saved.

Stage 2 has three absent Triton availability probes, eight already optional blocked
imports, zero loaded Python/native Triton modules/maps. Four GIL checkpoints remain
false; no `PYTHON_GIL` setting. Separate component and baseline processes also keep
the GIL disabled. This proves tested CPython 3.14t paths, not unrestricted
concurrent compilation or general free-threaded training safety.

Warmed profilers show exactly one CubeCL kernel for sine, broadcast, sliced view,
semantic-mask cat and genuine fusion. No extra ATen pointwise or Triton kernel
appears. Reference operations are outside profiler regions. The descriptor component also validates an empty runtime extent and observes zero native kernel launches. Readiness events and
record_stream retain Stage 0's distinction between data readiness and allocator
lifetime; native launch contains no device-wide synchronization or CubeCL storage
allocation/stream ownership.

## Generated evidence and non-CUDA probe

[Representative artifacts](stage2/artifacts/representatives) retain typed semantic
and physical descriptions, exact replay, CubeCL IR, CUDA C++, PTX and manifests
for half, BF16, half backward, division, broadcast, strided input, real masks,
genuine fusion and the narrow index/nested-mask components. PTX/source/IR hashes
are in coverage.json. Full schedule stacks, runtime pointer/scalar/grid/stream
trace, profiler traces, compiler/NVRTC logs, exact source snapshots and guard logs
are preserved locally and ignored. Only authored files and representative evidence
are eligible for Git.

The same representative masked Stage 2 semantic descriptor successfully builds
CubeCL IR and compiles through existing `CppCompiler<Hip>` to
[HIP source](stage2/artifacts/noncuda/kernel.hip). This is an executed compiler/
frontend probe, not HIPRTC binary compilation, GPU execution or a PyTorch ROCm
launcher. No HIP stack was installed. The semantic descriptor has no CUDA-specific
assumption; the attached physical plan still requires target-specific validation
before any future launcher integration. The probe does not establish portability.

Graph verification uses Tier 2 for local implementation; installed PyTorch and
cloned CubeCL sources are excluded, so their exact files/bodies were read directly
and preserved locally. Coverage reports are best effort, not proof of completeness.
The required review found no wrong-result bug on the executed surface. Its
recursive-mask whitelist and divisor-domain findings were fixed and tested; the
unsupported inherited capture API now rejects explicitly. Example metadata uses
the same registration order as the runtime serializer. The pinned CUDA-header
layout and unbounded metadata cache remain deliberate experimental limitations.
[Review disposition](stage2/artifacts/review-disposition.md) records validation.

## Remaining gap before Stage 3 and recommendation

Numerical compute/storage conversion, direct affine/floor/mod indexing, dynamic
scalar metadata, physical tails, lazy semantic loads/stores and accepted/vetoed
normal fusion now have executed invariants sufficient to investigate reductions.
Unresolved: unbacked/dynamic-rank shapes; nonpositive divisors/overflowing address
expressions; broader integer semantics; arbitrary views/mutations; general masked
fills; reduction gradients for broadcast; independent precision/FMA modes; removal
of conservative SIMD 32-bit guards; concurrent/disk cache and portability contracts.
These are explicit limits, not advertised coverage.

**Exactly one next direction: 1. Stage 3 reduction architecture.** The Stage 2
pointwise gate is now broad enough to study inter-unit values and uniform reduction
control flow separately. Keep the tested scalar semantic frontend and require a
new, explicit reduction physical plan. Do not automatically implement it.
