# Stage 4 — stateful/effectful generated training semantics

## Result

**Passed, for the bounded surface below.** Real Dynamo/AOTAutograd/Inductor replay now drives generated uniform RNG, checked indirect memory reads and FP32 atomic scatter updates through CubeCL. A representative embedding → linear → RMS normalization → dropout → loss graph executes compiled forward/backward, followed by explicit ordinary SGD momentum updates. Four controlled training steps match reference losses, gradients, parameters and momentum state.

The final main acceptance process passes 15 categories: **61 compiler invocations, 375 generated GPU launches, independently confirmed by the native launcher**. Of these, 360 use the Stage4 pointwise ABI and 15 use the unchanged Stage3 reduction ABI (15,360 requested shared-memory bytes). Five launches exercise the new five-read-input ABI. Separate runtime checks pass 11 cases with 27 native launches. Descriptor components pass 30 seed/shape cases, four counter-bit cases, two executed lazy-mask cases, nine malformed descriptors and two stale-ABI tests. Five invalid-index subprocesses execute and report a device launch failure. All 11 earlier-stage regression gates pass.

No new PyTorch or upstream CubeCL source patch is required. The existing Stage1.5 availability overlay and exact conservative floating-point patch remain active. Stage4 does not implement Stage5 optimization, a universal fallback, arbitrary user mutation or general concurrency safety.

Executed on 2026-10-13: CPython 3.14.5t, PyTorch 2.14.1+cu130 (`5c4886908584029761b579af026dcfb627c84070`), CUDA/NVRTC 13, RTX 3090 / SM86, CubeCL `6bba5fdf04206d9c3710dfbc46aa9850450ab405`. The unchanged FP patch SHA256 is `9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2`.

## Supported Stage 4 surface

| Concern | Executed support | Deliberate boundary |
|---|---|---|
| RNG | Seeded uniform `rand` replay, `rand_like`, two independent random operations, dropout at p=.25, p=0/1 and evaluation mode | No generated normal/integer RNG, eager-aligned RNG or strict bitstream mode |
| Dtype | FP32/FP16/BF16 output storage; FP32 uniform generation; explicit low-precision conversion; bool predicates | i32 only for random counters; i64 for seeds, indices and required internal bookkeeping; no general integer tensor arithmetic surface |
| Indirect reads | `gather`, ordinary advanced index, embedding; i64 tensor-derived coordinates; checked direct affine addressing | No indirect reductions or quotient/modulo of tensor-derived values; no arbitrary nested indirect Inductor subblocks |
| Layout | Existing direct rank≤4 nonnegative-stride policy; executed 1D source slice with stride 2 / offset 3; multidimensional embedding coordinates | No contiguous copy to satisfy CubeCL; unsupported expressions/layouts still reject |
| Atomics | Actual `atomic_add` replay into an FP32 destination; FP32 or explicitly converted FP16/BF16 sources; scatter and embedding backward | No low-precision atomic destination, CAS emulation, other atomics or general mutation |
| Dynamic extents | Genuine `dynamic=True` graphs; symbolic source, index/update counts, strides and destinations | Normal 0/1 specializations remain; random logical extent guarded to ≤2^31−1 |
| Fusion | Existing pure pointwise/reduction fusion; effectful generated nodes remain separated | Fusion involving RNG/indirect nodes is deliberately vetoed pending effect-safe replay proof |
| Reduction | Stage3 sum/RMS/loss and backward reduction strategy reused unchanged | No expanded reduction surface |
| Runtime | Existing current-stream, PyTorch-storage and lifetime path; two prepared calls on independent threads/streams | CUDA graph capture and concurrent compilation are unsupported |

Unknown operations remain capability failures. The experimental selector is `cubecl_stage4`; it is not a production or upstream backend name. The callable entry is `stage4.backend.compile_backend`, with experimental `rng_policy="generated"` by default or `"aten_extern"` for an explicit existing random fallback.

## Live call paths

[source-trace.json](stage4/artifacts/source-trace.json) retains 65 observed Python function identities, source spans and hashes. Exact dependency source snapshots stay ignored. [lowering.json](stage4/artifacts/lowering.json) records the initial live pre-fusion probes; representative `evidence.json` files retain the final operation replay and scheduling decisions.

```text
torch.compile(fullgraph=True, dynamic=True)
→ OutputGraph._call_user_compiler
→ stage4.compile_backend → compile_fx → aot_module_simplified
→ forward/inference compiler or AOT backward compiler
→ replace_random_passes / replace_random / fuse_seed_creation_pass
→ GraphLowering.run → live lowering → GraphLowering.codegen
→ GraphLowering._update_scheduler → Scheduler
→ registered backend availability → Stage4 pre-fusion capability pass
→ normal scheduler fusion, with explicit CubeCL vetoes
→ SIMDScheduling.codegen_node / _codegen_nodes
→ codegen_node_schedule / codegen_node_schedule_with_kernel
→ SchedulerNode.codegen → LoopBody / LoopBodyBlock / InterpreterShim
→ CSEProxy / V.ops → CubeCLKernel typed operation replay
→ semantic descriptor + unchanged scalar/reduction PhysicalKernelPlan
→ Stage4 wrapper definition → stage4.runtime.compile_kernel
→ reused stage3.CompiledKernel and Stage0 compiler executable
→ validated descriptor4 (or unchanged descriptor3 for reductions)
→ KernelBuilder / Scope / CubeCL IR → CppCompiler<Cuda>
→ NVRTC (--fmad=false) → PTX / module / function
→ existing tensor-layout/metadata preparation and current-stream enqueue
→ stage4_launch (or stage3_launch) → shared native launch_args
```

Concrete live lowering includes `inductor_seeds`, `inductor_lookup_seed`, `inductor_random`, `gather`, `scatter_add`, `scatter_reduce_` and `index_put_impl_`. AOT backward reaches the same scheduler/replay/compiler/launcher stack. `FallbackKernel.create` is observed for the explicit random policy; `ExternKernelOut.codegen` is observed for seed creation and vendor matrix multiplication.

## RNG architecture and contract

The default live random replacement inserts `RandomSeeds`, an Inductor-owned `aten.randint.low_out` extern. Its actual IR generates full signed-i64 seeds, horizontally combining seed creation for two random operations. The eager implementation of the custom seed prim has different bounds; **the executed `RandomSeeds` IR is the contract tested here**.

The generated body replays `load_seed` → `index_expr(..., int32)` → `rand` → explicit store/casts. The descriptor stores seed/counter identities and `algorithm="philox4x32_10_top24"`; `effects.rng="explicit_seed_counter"` describes the supplied seed dependency. It contains no generator object, host RNG state, CUDA offset reservation or CubeCL arena handle.

Rust builds Philox4x32-10 using CubeCL unsigned integer arithmetic, shifts, XORs and casts. The 64 seed bits become two key words; the counter's 32 bits become the first counter word, with the remaining words zero. Products use u64; low words and key increments explicitly wrap modulo 2^32. The top 24 bits of word zero convert exactly to FP32 and scale by 2^-24. The independent Python reference checks the published zero-key/zero-counter known-answer vector and executed GPU values. The algorithm comes from [Random123](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf), rather than copied Triton source or a Rust host RNG.

FP32 uniforms lie in [0,1); rounding to FP16/BF16 can produce 1. Low-precision dropout replay rounds the random value to the semantic storage dtype **before** comparing to p. Tests use that real precision barrier when constructing the reference mask. FP32 computation/storage policy otherwise remains Stage2's explicit policy. Backward consumes the corresponding saved mask and arbitrary upstream gradient; it does not regenerate randomness.

State ownership remains PyTorch's CUDA generator. Tests restore its actual state and independently predict the same signed-i64 seed draw before comparing generated samples. Cold and warm invocation results/state match exactly after restoring state. Repeated calls advance state; two operations draw distinct keys in the executed fixture. The allocation of keys follows normal Inductor seed creation, not an invented counter allocator or a mathematical promise that finite random keys can never collide.

For a nonempty random body, generator advancement is that of the normal seed extern (one seed per operation), not one generator update per output element. Empty random output is naturally eliminated without advancement. Compatible sizes 255, 256, 257 and 65,539 reuse the dynamic graph/kernel artifact; 0 and 1 retain normal specializations. Shape values and seed contents are runtime arguments, not cache-key seed values.

Compilation, capability probes and unsupported compilation do not consume user-visible RNG. The probe records state before/after failed pre-fusion compilation; cold acceptance compares state against precisely the runtime seed extern, excluding hidden compile/warm consumption. Calling a stochastic graph for warmup **is execution** and advances state normally; no backend autotuner or implicit warmup was introduced.

A source reread corrected the review's initial fits-in-i32 assumption: `inductor_random` explicitly asks for i32 `index_expr`. Stage4 therefore adds a normal `SizeVarAllocator.guard_or_false(extent <= 2^31-1)` before fusion, plus defensive runtime and constant-descriptor checks. [counter-guard.json](stage4/artifacts/counter-guard.json) proves live CUDA FakeTensor lowering of length 2^31 rejects before compiler invocation or RNG consumption, without allocating the data. Components separately test counter bit patterns 0, −1, i32 minimum and maximum, with an unsigned-before-widen conversion. Memory/index arithmetic remains signed i64.

Default compiled uniform samples intentionally differ from eager CUDA's distribution mapping. PyTorch's ordinary reproducibility guidance does not promise identical sequences across implementations; the local Inductor prim explicitly documents backend-specific RNG. See [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html). `align_random_eager` and strict numerical mode are rejected. The explicit ATen random policy instead preserves eager random output/state exactly in its accepted fixture.

## Indirect addressing and masks

Live `CSEProxy.indirect_indexing` handles advanced-index negative wrapping with `lt(value,0)` / `add(value,size)` / `where`, then delegates to the backend. Gather and embedding use `wrap_neg=False`. Stage4 emits a semantic `indirect(value,size,bounds="require_in_range")` operation and a distinct checked SSA identity. Direct symbolic address trees can use that identity through `{"op":"value","id":...}` leaves; ordinary logical coordinate FloorDiv/Mod and strides continue to use Stage2's direct index machinery.

An executed sliced-source gather lowers to:

```text
idx = load_i64(index_buffer, logical_coordinate)
checked = indirect(idx, source_length, require_in_range)
value = load_float(source_buffer, checked * runtime_source_stride)
store(output, logical_coordinate, value)
```

The source tensor's original pointer already includes storage offset 3. The descriptor's relative offset is zero; applying `Tensor.storage_offset()` again would be wrong. Pointer/stride/offset evidence and the non-default-stream test confirm original storage is passed directly.

The bridge's local `cubecl_bridge.require` is a side-effecting, no-memory-effect CubeCL IR primitive. It checks `0 <= coordinate < size` before dependent memory access. Its CUDA target lowering emits a device trap; HIP source lowering uses the target's builtin trap. This is an IR/compiler extension through public operation interfaces, not handwritten kernel CUDA.

Rust validates type, checked provenance and region scope before IR construction. A raw i64 value cannot be substituted for a checked address; checked identities cannot escape a lazy region. Physical tail bounds are still separate from semantic masks. Two descriptor-boundary GPU tests use varying/nested runtime predicates with invalid 2^32 indices in inactive regions, prove those loads never occur, and retain fill-value/canary checks. These are explicitly component tests, not fabricated PyTorch masked graphs. The natural embedding backward's actual `masked` replay is also retained.

Out-of-range upper, negative gather, excessively negative advanced index, 2^32 and i64-minimum indices all execute in separate disposable processes and fail with asynchronous CUDA error 719 (`unspecified launch failure`), while the generated launch is recorded. This is a clear device failure, but it is not eager's exact assertion diagnostic and poisons that context. There is no clamp/fill/retry for an invalid protected access. `check=False` replay callers still receive the bounds precondition; valid-domain unsafe backward indices remain protected.

## Mutation and atomic model

The accepted contract is a functional PyTorch scatter/embedding result, whose fresh PyTorch-owned destination is initialized before a generated atomic update. Arbitrary user-visible mutation is rejected at backend entry: mutable schemas, `_` methods, `out=` and `inplace=True` patterns.

Live lowering supplies separate initialization and atomic scheduling nodes. Existing Inductor dependencies and wrapper buffer lifetime/reuse establish producer → update → consumer ordering. Canonicalization through `scheduler.mutation_real_name` matches `KernelArgs`' existing physical argument naming; it fixes the logical buf1/physical buf0 distinction without discarding mutation dependencies or changing common CSE invalidation.

Atomic stores append effectful operations directly, bypassing pure expression CSE. The descriptor states `atomic_add`, FP32 storage and `effect="atomic_rmw"`; `effects.mutation=true` is recursively validated against its operations. CubeCL's existing `AtomicFAddOp` lowers to `atomicAdd` and the executed PTX contains `atom.relaxed.add.f32`. There is no non-atomic emulation. The destination's prior initialization is represented by Inductor's graph dependencies, not a backend-owned initialization buffer. The update grid uses update count, rather than destination element count.

The accepted target/op combination is FP32 add on the tested SM86 CUDA compiler path. FP16/BF16 sources explicitly convert to FP32; low-precision destinations reject before fusion. Existing i64 `minimum` / `maximum` and bool conjunction were added only for observed embedding-backward validity bookkeeping. Floating extrema remain unsupported.

Collision-free, two-way, many-to-one and sparse updates pass for 0, 1, 255, 256, 257 and 4,099 updates, including positive/negative low-precision source values. Dyadic test inputs permit exact result comparisons at those bounded counts. Arbitrary-gradient embedding and training comparisons use FP32 rtol/atol 2e-5; floating atomic order is not claimed bitwise stable. Deterministic testing is separate: normal PyTorch lowering selects `ScatterFallback` / `aten.scatter_reduce_.two`. Toggling deterministic mode on an already compiled graph causes normal reselection, and no generated atomic survives. The backend additionally rejects an atomic schedule presented under deterministic mode.

## Fallback policy and census

[fallback-census.json](stage4/artifacts/fallback-census.json) describes all 15 main categories and both deterministic cases. It separates `cubecl_generated`, `aten_extern`, `vendor_library`, `explicit_fallback`, unsupported graphs, explicit eager optimizer work and autograd runtime work. Entries retain node/phase identity, semantic IR owner, operations, selection reason, fused group/entrypoint and category-wide launch counts. Per-member counts must not be summed across a fused group. Warmed profiler CPU/GPU counts are separately scoped.

Classification uses scheduler IR/target and exact native/runtime entrypoints. Non-CubeCL GPU events are correlated through profiler **External id** to their actual CPU operator owner; they are not categorized by kernel naming alone. There are no unclassified GPU kernels in the preserved warmed regions.

| Warmed fixture | CubeCL kernels | Explicit other GPU work |
|---|---:|---|
| FP32 uniform | 1 | 1 seed-creation extern |
| FP16/BF16 uniform | 2 | 1 seed extern; separate precision conversion stays generated |
| Two random operations | 3 | 1 combined seed extern |
| Dropout forward + arbitrary-gradient backward, each dtype | 4 | 1 seed extern |
| Sliced-source gather | 1 | 0 |
| FP32-destination scatter, each source dtype | 2 | 0; initialization + atomic update are generated |
| Embedding forward/backward | 3 | 0; gather + initialization + atomic update |
| Complete training step | 9 | 1 seed extern, 4 vendor GPU kernels from 3 mm calls, 1 autograd upstream fill, 6 explicit SGD kernels |
| Explicit normal-RNG delegation + add | 1 | 1 ATen normal-RNG fallback |

The normal RNG policy uses existing `config.fallback_random=True` **before** random replacement/lowering. The fixture `randn_like(x)+1` executes ATen random generation and one generated CubeCL add, with exact eager output/state. Default generated `randn` rejects. This is deliberate early delegation, not reconstruction of arbitrary fused kernel IR.

Matrix multiplication remains naturally selected `extern_kernels.mm` / `aten.mm.out`. Deterministic scatter remains naturally selected external scatter. Neither is a failed CubeCL compiler retry. Fullgraph compilation and `suppress_errors=False` exclude hidden eager graph-break fallback. No generated work silently routes to Triton.

Compiler metadata transfers and ordinary PyTorch seed, mask, output and intermediate allocation remain. No input layout materialization or CubeCL tensor copy is introduced. Vendor-library internal workspace/materialization is not claimed absent; its GPU work is accounted for separately. Test-harness reference/producers/allocator-pressure operations are outside the compiled acceptance regions.

## Training-step evidence

The authored model uses a learnable [17,7] embedding table through ordinary `F.embedding`, a bias-free [7,9] `nn.Linear`, RMS normalization, p=.25 dropout and a squared flattened mean loss. Repeated indices `[0,2,2,16,1]` force actual backward atomic accumulation. Ordinary AOTAutograd saves/reuses the tensors and mask; backward also produces a real generated normalization reduction.

Eleven generated pre-fusion nodes become **nine** CubeCL kernels under normal fusion: five forward and four backward, with three reduction kernels total. Three natural mm externs produce four vendor GPU kernels (one library operation uses a split-K helper). Seed creation is one additional extern. These are measured in the same training-step profiler region.

The eager reference starts from identical model/optimizer state. It derives the same seed from a saved CUDA generator state and uses an independently computed Philox mask, then restores state before invoking the compiled stochastic graph. Thus optimizer comparison uses the same stochastic samples without inserting a custom random implementation into the captured graph. Four steps compare loss, every parameter gradient, every updated parameter and every momentum buffer at rtol/atol 2e-5. No tolerance was widened to hide a systematic error.

The optimizer is explicitly **outside** fullgraph forward/backward: `torch.optim.SGD(lr=.01,momentum=.9,foreach=False,fused=False)`. The warmed region records its two multiply and four add kernels. Scalar-loss `backward()` also creates its normal upstream with `ones_like`/fill. These are visible runtime work, not an Inductor graph break or generated CubeCL optimizer claim. No foreach/fused optimizer support is advertised.

A plain `nn.Embedding` dynamic capture in development lifted its float `norm_type` attribute into a CPU scalar input, beyond the established input ABI. The acceptance model uses normal `F.embedding` on its Parameter instead. CPU floating scalar hyperparameter capture remains unsupported; it was not generalized to make the fixture pass.

## Runtime, determinism and compiler evidence

The original scalar PhysicalKernelPlan remains one logical element per unit, workgroup [256,1,1], vector width 1 and explicit physical tail predicate. Reduction kernels retain Stage3's existing cooperative plan. Semantic descriptors contain no CUDA builtins, target pointer objects, stream ownership or allocator handles.

Existing buffer layout/span validation, signed-i64 scalar metadata, InfoBuilder serialization, module loading and native enqueue are reused. Stage4 adds one bounded extra read argument because the observed normalization/dropout backward genuinely requires five inputs. Older stage argument limits remain unchanged.

Three delayed-producer trials on non-default streams exercise both data and index readiness, initialization/update/consumer ordering and release under eight large pressure allocations. They assert original pointers and launch stream identities. Readiness uses the user's stream/event contract; `record_stream` protects lifetime separately. No device-wide synchronization was added to launch. Test synchronization only establishes correctness observation.

Two prepared wrappers start behind a thread barrier on independent streams. Their distinct outputs and exact two-seed state advancement pass, with per-call launch packing. Compilation concurrency is deliberately untested; no new global runtime RNG lock or mutable host generator cache was introduced.

All fresh processes verify Triton package absence before installing the import guard. The main process records eight optional import attempts at PyTorch `_triton`, `ir`, `triton_compat`, `flop_counter`, AOT repro, mm, TLX registry and autows locations. **Zero Triton modules or native mappings load**. These optional probes are distinct from executing a generated Triton kernel. `has_triton()` is not falsified; no fake package is installed.

GIL checkpoints before Torch, after Torch/Inductor and after final execution are false under CPython 3.14.5t. `PYTHON_GIL` is absent. This proves the tested paths remain GIL-disabled; it does not establish universal concurrent compiler/library safety.

[coverage.json](stage4/artifacts/coverage.json) indexes 79 curated executed representatives/components and retains compiler/native/source/FP patch hashes. Each includes semantic descriptor, physical plan, validated CubeCL IR, CUDA C++, PTX, signature, NVRTC options, launch manifest, replay/schedule and scoped launches/profile evidence. Actual driver attributes show 10–36 registers per thread, zero static shared memory and zero local bytes per thread in these representatives; reduction launches request their existing 1,024 dynamic shared bytes. RNG and scatter work is linear in output/update count, with no serial-host or quadratic algorithm. No tuning or Triton performance claim is made.

The narrow HIP probe compiles an executed indirect semantic descriptor plus the same scalar physical plan to CubeCL IR/HIP source, including the target's `__builtin_trap`. It passes source generation only. No HIP runtime, ROCm launcher or general portability is claimed; target trap syntax remains in target lowering rather than the common descriptor.

## Source modifications

There are **zero additional upstream PyTorch/CubeCL source modifications**. All changes below are experimental local frontend/bridge code, using existing extension points.

| File / responsibility | Change and reason |
|---|---|
| `stage4/backend.py` | Stage4 typed operations, early input/semantic/extent checks, effect-aware fusion veto, mutation argument canonicalization and wrapper integration; directly consumes common SIMD replay |
| `stage4/runtime.py` | Existing runtime adapter with int64 storage, schema4 metadata export and update-count extent; capture/extent contracts enforced |
| `stage0/src/descriptor4.rs` | Typed schema4 validation, region-scoped checked addresses, explicit Philox IR and actual atomic IR; semantic/physical separation retained |
| `stage0/src/require.rs` | Small out-of-tree side-effecting require operation, CUDA/HIP target compiler lowering; needed because checked indirect loads cannot rely on eager assertion or speculative invalid loads |
| `stage0/src/descriptor2.rs` | Extract reusable validated scalar builder and structured-region emitter callback; keep old schema/type/buffer limits; avoid duplicating compiler/physical construction |
| `stage0/src/main.rs` | Route schema4 to its validator/builder and record its signed scalar ABI; existing C++/LLVM/HIP routes remain |
| `stage0/src/lib.rs` | Reuse unchanged InfoBuilder serialization behind a bounded Stage4 six-buffer metadata export; old export limit preserved |
| `stage0/launcher.c`, `stage0/bridge.py` | Bounded seven-pointer launch (five reads, output, metadata) via existing `launch_args`; independent schema4/five-read counters; schema dispatch/manifest validation |
| `stage0/Cargo.toml`, lock | Direct Pliron/linkme dependencies for public out-of-tree IR operation macros; versions already in the dependency graph |
| `stage2/backend.py`, `capabilities.py`, `runtime.py` | Small typed-value/storage/capability/metadata extension hooks with unchanged default support; no backend-name dtype lists |
| `stage3/backend.py`, `runtime.py` | Backend-owned trace, runtime root/kernel/events injection; existing plan/compiler/stream/launcher logic reused unchanged |
| `stage4` test/capture/probe files | Real execution, independent reference, failure controls, state/counter/profiler census and curated evidence |

The generic hooks are plausible reusable local seams, not a proposed upstream plugin framework. Backend-specific schema4/require and launch-count extension remain experimental. The preexisting Stage1.5 four-file PyTorch availability overlay is unchanged.

## Regression and review evidence

[regressions.json](stage4/artifacts/regressions.json) records all eleven commands with exit zero. [regression-details.json](stage4/artifacts/regression-details.json) preserves fresh lower-stage result evidence; old tracked baseline artifacts were restored after retaining this evidence to avoid unrelated machine-specific churn.

| Gate | Result |
|---|---|
| Stage0 C++ / LLVM independent compiler-launch paths | Both pass |
| Stage1 capture/forward/backward | Pass, 3 compiles / 44 launches |
| Stage1.5 availability/capability boundaries | Pass |
| Complete Stage2 + components | Pass, 34 acceptance categories / 94 compiles / 165 launches |
| Stage3 + components | Pass, 21 acceptance categories / 50 compiles / 149 launches, including 131 reduction launches |
| FP IR rewrite tests | 7 pass |
| FP runtime | 100 cases pass, 100 compiles / 114 launches |
| FP Inductor | 3 accepted / 2 unsupported, 6 compiles / 12 launches |
| Stage4 main / runtime / components / bounds / oversized-counter control | Pass; separate machine evidence retained |
| Rust clippy and fmt, Python Ruff checks, C11 Wall/Wextra/Werror, diff whitespace | Pass |

The exact CubeCL source diff matches the conservative patch byte-for-byte. Unsafe floating identities, implicit FMA contraction, ordinary LLVM contract flags and target-implied relaxed math remain disabled. Stage4 uses the working CUDA C++ route; earlier LLVM remains verified, without claiming schema4 LLVM support.

Required review-mcp completed against HEAD and the new untracked core. It found no supported-surface wrong-result/memory-safety bug, and its small availability/diagnostic concerns were addressed. The final counter-domain/source correction and checked-address validation controls are documented in [review-disposition.md](stage4/artifacts/review-disposition.md). The reviewer saw an earlier in-progress acceptance snapshot; final numerical/profile/regression results are independent runtime evidence. No native review fallback was used.

## Remaining limits

There is no generalized operation catalog, arbitrary alias mutation, indirect reduction, floating extrema, low-precision atomic destination, generated normal/integer random algorithm, eager-aligned RNG, CUDA graph capture or concurrent compilation guarantee. Effectful fusion remains conservatively vetoed. Stochastic graph inputs requiring CPU floating scalar capture are outside the current contract.

Caches, traces and loaded modules remain process-lifetime experimental structures. Invalid indices produce a context-poisoning device failure with a less specific diagnostic than eager. These limitations are explicit boundaries rather than silent fallback. The normal ATen seed extern, vendor matmul, deterministic scatter extern, explicit random delegation, autograd upstream creation and outside-capture optimizer are all accounted for.

## Completion decision

**1. Stage 5 — vectorization, launch/reduction autotuning, register/occupancy feedback.** The bounded state/effect gate is now executed and inspectable: RNG ownership, checked data-dependent addresses, atomic ordering, gradients, optimizer state and mixed execution are established without changing the semantic/physical architecture. Performance work can build on these measured contracts while preserving conservative math and the current effect vetoes. Stage5 has not been started.

## Reproduction

```bash
bash stage4/setup.sh
stage0/.venv/bin/python stage4/probe.py
stage0/.venv/bin/python stage4/verify.py
stage0/.venv/bin/python stage4/components.py
stage0/.venv/bin/python stage4/runtime_checks.py
stage0/.venv/bin/python stage4/counter_guard_probe.py
stage0/.venv/bin/python stage4/verify_bounds.py
stage0/.venv/bin/python stage4/regressions.py
stage0/.venv/bin/python stage4/capture.py
```

Run overlay-installing suites sequentially. `verify_bounds.py` isolates each device-failure context. Tests regenerate local evidence and lower-stage result files; nothing here stages, commits or pushes. Generated representative artifacts are eligible for commit; downloaded unmodified dependency sources, raw snapshots/profiler traces, local build products, caches and logs remain ignored.
