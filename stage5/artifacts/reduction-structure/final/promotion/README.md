# Guarded ordinary final-plan admission (cwk.9)

Current decisions live in [STAGE5_REPORT.md](../../../../../STAGE5_REPORT.md).
This pass promotes the already-qualified cwk.7 mechanism into an ordinary Stage 5
physical candidate. Stage 5 remains active; Stage 6 is deferred. No new reducer,
partial topology, planner, synthetic GPU activity, warmup policy or memory primer.

## Physical choice and eligibility

`independent_planes_final9` has a distinct schema-2 physical identity: 32×4×1,
scalar reads, four outputs/block, plane width32, one private FP32 accumulator.
The ordinary schema-5 `cpp` adapter calls the existing qualified final emitter:
+0 initialization, protected scalar reads, descending XOR16,8,4,2,1 with FP32 adds,
lane0 result ownership and the original cast/store sequence. No shared memory or
block barrier. The algorithm still naturally supports small extents1..32 through
the retained diagnostic; this initial **automatic admission** is intentionally9.
The generic `independent_planes` plan retains its different ascending FSum order.

Python admission (`final_plan.semantic_reason`) and the native descriptor adapter
both require the qualified semantic forms: constant9; one positive i64 outer
scalar; contiguous FP32 input[B,9]; output[B] in FP32/FP16/BF16; direct row address;
one sum with FP32 accumulation; outer owner store; no effects or extra consumers.
The lower-precision descriptors retain all seven operations, including their
existing casts and semantic/source type annotations. FP32 has three operations.
SSA names may differ. The exact address/cast/layout forms are a conservative
qualification boundary, not a claim that all equivalent expressions are unsafe.

A queried 32-lane target and sufficient block capacity are required. Ordinary
storage capacity, dtype/device/layout, offset, aliasing, metadata, readiness,
allocator lifetime and current-stream checks still run. The Kernel constructor
checks the final contract before loading it; enqueue revalidates its physical
contract after ordinary ABI validation. Valid offset views and nondefault streams
remain supported. Zero outer violates this particular positive-scalar contract;
it is rejected, not silently converted into a different domain.

Unsupported descriptors exclude this candidate and retain existing choices.
Dynamic extents, nearby counts8/10, strided/noncanonical loads, fused consumers,
modified epilogues, unqualified cast sequences and other plane widths do not
inherit eligibility. Existing long-row compensation and reference512 diagnostic
ineligibility are unchanged. The frozen partial and its split9 remain diagnostic.

**Eligibility is separate from selection.** `first_use` compiles and validates the
new candidate against the existing frozen final, using the existing strict
comparison. It is excluded from `qualified_for_runtime`'s coarse compensated-plan
corruption guard. Existing rotated empirical selection and its baseline/MAD
criterion decide the winner; deterministic/baseline/effect bypasses remain intact.
There is no forced winner or blanket default promotion.

The distinct plan enters artifact entrypoint/cache keys, tuning catalogs and
persisted selections. The admission source is hashed in both compiler-adapter
and tuning identities. All new admission facts are descriptor-static and already
part of the semantic key; no new unencoded runtime size fact enters proof caching.
The live ABI facts are rechecked every call. Reloads cannot silently reinterpret
the old independent-plane identity as this descending mechanism.

## Reuse and code evidence

No new arithmetic topology was implemented. This is the same CubeK independent
output/plane mapping and CubeCL ShuffleXor/FAdd lowering selected in cwk.7 after
inspection of the exact retained Triton final, CubeK/cubek-reduce, CubeCL plane and
shared reductions, CUB/CCCL and ATen Reduce.cuh. See the [prior source inventory](../inventory.json)
and [qualified mechanism report](../graphs/README.md) for revisions and comparisons.
Runtime/API differences are handled through the existing IR boundary.

The old final maps256 threads to each row with nine inputs and neutral lanes,
shared publication plus eight halving stages and nine block barriers. The reused
final maps one row to one32-lane plane, four rows/block. The abbreviated nonzero
root is `(((p0+p8)+p4)+(p2+p6))+((p1+p5)+(p3+p7))`; CPU models include every zero
addition. The fixed9 mechanism is byte-equivalent to the prior qualified probe
in IR/CUDA/PTX for all three storage types with the same symbol. The ordinary plan
uses a new symbol/identity; its executed SASS instructions match the qualified
probe after normalizing relocated branch addresses.

The64-output counter control records old/new34752/2816 executed warp instructions,
1600/0 shared loads,1280/0 shared stores and16/16 allocated registers. SourceCounters
SASS records4608/0 executed BAR operations and0/320 SHFL operations. The raw barrier
metric itself is unavailable; the BAR count comes from the SASS instruction table.
Driver queries record10/15 registers and zero local memory. These are distinct
resource measures, not an attribution of latency by register or instruction count.

`ncu-command.json` pins the PATH-resolved2026.3.1 runner, SHA and section assets.
`final-code.ncu-repz`, `ncu-raw.csv`, `executed-sass.txt`, `sass-analysis.json` and
`counter-kernels.json` retain the actual production driver-JIT code and queries.
Kernel replay is code/counter evidence, not performance timing. Compilation stays
CUDA13.0 NVRTC with conservative FP options; no offline13.1 binary is substituted.
`retained/` holds actual generated artifacts and `selections/` raw tuning records.

## Numerical, runtime and cache evidence

The CPU rerun precedes GPU validation: all148 existing corpus cases pass the
unchanged reference-or-better policy;1332 neighboring comparisons at1,4,8,9,10,16,
17,31,32 preserve the old tree. NaN/Inf/signed-zero/subnormal/cancellation controls
are included. No tolerance was loosened and no arithmetic was changed in this pass.

Ordinary production kernels pass148 GPU corpus comparisons bitwise against the
old final with the exact frozen uniform-loop producer. Thirty direct controls
cover all storage types, output counts1,3,4,5,8,31,32,33,64,257, canaries, exceptional
values and CPU-model finite bits/classes. CPU/CUDA NaN encodings can differ; the
GPU-to-frozen comparison remains byte-exact even for NaNs. Eighteen live ABI
rejections and21 descriptor rejections are checked; the latter are rejected in
both Python admission and the native compiler. Unsupported target topology is
also rejected, and generic dynamic reductions keep ordinary fallbacks. Three
additional top-level malformed JSON values return clean admission reasons. The
post-review alias controls use genuine shared storage for all three dtypes and
require the disjoint-output error before any native launch.

Cold first_use selection, warm calls and a fresh-process persistent reload pass
for all three storage forms. All three64-output selections chose the new final;
reload has zero compiler invocations. The9→8→9→10→9 control proves separate keys,
correct exclusion outside9 and reuse on return. Existing generic choices may win
other sizes or future captures; these observed winners are not a selection promise.

Ordinary `compile_backend` emits and selects the final on a static9 reduction axis
with dynamic outer dimension. Its normal large-workload producer/selector also
passes all three storage types. The existing guarded planner chooses32 partials
for8×1M and4 for64×65539 in this capture; these finals correctly exclude the new
candidate. No planner change manufactures eligibility. An additional ordinary `first_use`
wrapper uses the existing inherited split policy to retain9 and pairs the normal
compensated producer with the new final at both workload shapes. Its strict
ATen-relative checks pass; the frozen producer remains a separate control.

Separate real generated-wrapper controls retain the existing diagnostic frozen
producer and route its final through ordinary first_use. A test-only scheduling
subclass chooses those existing per-kernel policies at wrapper emission; it does
not patch compiler/runtime methods or substitute artifacts. All6 dtype/shape
pairs preserve frozen-final output bits and partial artifacts. This verifies
integration while keeping the producer automatically ineligible.

## Measurements and limits

RTX3090/SM86; same process, six rotated rounds,16 calls/cell; all60 admitted raw
CUPTI traces retained and independently checked. These are kernel-duration sums
from real wrappers, not host latency, queue-inclusive intervals or CUDA Graph replay.
Ordinary execution for compilation/readiness occurs before capture. No synthetic
GPU sleeps, memory primers, clock forcing or production warmup is introduced.

| Scope | Original final | New final |
| --- | ---: | ---: |
| 8 outputs, isolated final | 1.888µs | 1.535µs |
| 8×1M frozen producer, final only | 2.270µs | 1.857µs |
| 8×1M frozen-producer graph GPU sum | 43.790µs | 43.459µs |
| 64 outputs, isolated final | 5.983µs | 4.547µs |
| 64×65539 frozen-producer graph GPU sum | 311.910µs | 312.769µs |

The secondary capture has substantial variation, with round medians around280–648µs
for the old graph and280–643µs for the promoted graph. It does **not** establish a
complete-graph gain. Raw samples/MAD/extrema are in graphs/analysis JSON; final-body
medians there are9.985→8.962µs. No cross-capture/device-clock-normalized claim.
The established memory/history benchmarking confound stays parked in cwk.5;
these samples are reported without another hardware-state investigation.

The normal guarded graphs have different split counts and a compensated producer;
their timings in `graphs.json` are separate workload evidence, not a comparison
attributing changes to this final. This pass adds no new Triton implementation or
retuning; the prior [same-process retained Triton comparison](../graphs/README.md)
remains the qualified reference scope. The new ordinary final's code is identical
to that already compared mechanism.

The first graph attempt completed its primary timing before an incorrect harness
expectation about the normal planner's split count stopped the secondary case.
Its raw traces and primary medians567.637→572.812µs remain in `failures/first-graph`.
The accepted rerun fixes that expectation; it was not selected for faster timing.
Earlier harness errors (cast-form guard, positive-zero domain, singleton model
stride and CPU/GPU NaN encoding) and the missing-LLVM-feature build attempt are
retained. None was a GPU parity failure. The full-feature build reruns the affected
production controls and cache/generated-wrapper checks; its evidence is separate.

## Validation disposition

Required review-mcp returned **no blocking defects**. All actionable minor
findings are addressed: clean malformed-descriptor reasons; true storage-alias
checks in all dtypes; explicit ordinary-winner assertions; guarded SASS parsing;
semantic role lookup and per-call choice records across outer-size buckets; one
admission evaluation per components fixture. Static enqueue revalidation is kept
intentionally; this pass makes no host-latency claim. See [review disposition](review/disposition.md).

All18 established regression groups have passing evidence in
`regressions/gates.json` / `gate-status.json`. Three unchanged groups are reused
from the preserved full run;15 run after the review fixes, including repeats of
all8 affected groups. The corpus retry,78 forward/9 backward precision matrix,
precision harness, frozen-order qualification, accumulation checks/harness and
25 long-row checks pass. Historical-preservation checks report matching bytes
and no added historical files. `analysis.json` independently verifies all60
admitted timing traces, selected artifacts and the201 retained files.

The first suite attempt lacked the optional LLVM build feature; that is corrected
with `--features llvm,hip-probe`. The second stopped in the BF16 bias-row case on
`invalid GPU kernel timestamp`, outside this final's contract. Both failures and
the unchanged profiler gate's passing retry are retained. No timestamp validity
check, tolerance or timing method was loosened. Full-suite and continuation phases
are explicit in the accepted manifest; this is not claimed as one uninterrupted
18-group run. Fresh corpus/precision JSON outputs are captured before restoration.

The full-feature pre-review storage control recorded warm-creation choices;
BF16's small-output capture retained baseline. Post-review records now capture
actual choices after each call. All6 current dtype/shape pairs select the new
final and preserve outputs, while all three64-output cache probes and the ordinary
static9 graph select it as well. This is an observed capture, not a forced-winner
rule or a guarantee for other devices. Compiler-feature changes and the defensive
JSON guard rotate cache identities; matching-plan IR/CUDA/PTX stays byte-identical.

The graph service returned `Transport closed` for discovery/coverage attempts.
Affected source and artifacts were read directly; no graph coverage/completeness
claim is made. Production changes are bounded to plan registration, admission,
cache identity and ordinary routing to the existing emitter.

The guard is intentionally conservative: these exact constant9 forms and CUDA32
targets, no unqualified fusion/epilogue, broader constant/dynamic reduction or
cross-device promise. Stage5 stays active; Stage6 and broader reuse trial8x0 are
separate work. No commit, push, synchronization or broad default promotion.
