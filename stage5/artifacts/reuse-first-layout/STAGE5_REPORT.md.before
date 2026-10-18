# Stage 5 — current findings

**Start here.** This is the consolidated Stage 5 summary, updated 2026-10-18.
The other Stage 5 reports preserve individual experiments; their recommendations
and status describe those captures, not necessarily the current decision.

## Current state

The original bounded Stage 5 gate passed. Physical candidates, guarded shape
regimes, empirical selection and the compiler/current-stream bridge work on the
tested surface. **A broadly qualified, competitive long-row reduction default
has not been established.** `cwk` remains open; later-stage work is deferred.

| Area | Established finding | Current decision |
| --- | --- | --- |
| Compiler/runtime boundary | Real Inductor replay, typed semantics + separate physical plan, CubeCL IR, CUDA13.0 NVRTC/PTX and native PyTorch-stream execution remain viable. | Preserve this boundary; no replacement is justified by the measured issues. |
| Pointwise physical plans | Bounded vector/scalar alternatives and cached selection work with explicit layout/alignment proofs and precision barriers. | Keep their capability limits; no blanket speedup or general vector-layout claim. |
| Constructor precision | Generic overlay corrections distinguish FP16/BF16 tensor-storage rounding from scalar opmath precision; forward/backward neighbors pass. | Installed in the normal experimental overlay. Earlier candidate-only wording is historical. |
| Shape history | An initial short shape could freeze an unsuitable long-row split/topology. Guarded regimes now reuse Inductor's existing split machinery. | Bounded fix, not a universal planner; measured regime boundaries are experimental. |
| Compensated reductions | Compensation corrects important cancellation cases but has measurable cost. | Retained for experimental continuity, not accepted as the final default. |
| Four-chain experiment | Nested protected-load/accumulator regions generated expensive reconvergence. One load-batch plan sharply reduces that cost. | Diagnostic-only: all 116 tested partial buffers and final outputs match the old plan bitwise, including its heldout_10017 output4 qualification failure. Never automatically eligible. |
| Uncompensated private merge | An exact CPU model reproduces executed partial/final bytes. Adjacent pairing resolves the held-out failure but fails three existing adversarial elements under the 256-unit assignment. | `adjacent_merge_diagnostic` remains rejected. Its failure is preserved; the subsequent complete-topology experiment below resolves the numerical question. [Evidence](stage5/artifacts/reduction-structure/rounding/README.md). |
| Complete reference-derived topology | The qualified 512-thread/four-chain assignment, adjacent merge, descending XOR orders, split9/final and numerical policy are frozen. All 148 original-model cases qualify. | `reference512_diagnostic` stays automatically ineligible. Continue Stage 5 code-generation work; the earlier stop recommendation was rejected by the user. [Arithmetic evidence](stage5/artifacts/reduction-structure/rounding/topology/README.md). |
| Frozen-topology code generation | Compiled actual CubeK readers/Plane/Cube components on the local toolchain. Translating its uniform chunk-loop expression removes executed BSSY/BSYNC while preserving every chain/update. FP32 prepared partials improve about 72.65→41.60 µs at 8×1048576 and 26.67→21.51 µs at 64×65539; registers 48→40. | Retain only the uniform loop in the existing diagnostic. Owner-only publication and bounded physical-counter probes qualify but are not installed; their combination does not improve the retained expression. No default/policy/Stage 6 change. [Evidence](stage5/artifacts/reduction-structure/codegen/README.md). |
| Small owner-only final | The cwk.7 descending independent-plane mechanism now has a distinct ordinary `independent_planes_final9` physical plan, guarded in Python and native code. CPU/GPU corpus parity, live ABI guards, cache transitions/reload, ordinary generated final emission and frozen-producer wrapper controls pass. | cwk.9 is qualified: ordinary `first_use` may select only the qualified constant9 forms; no forced winner, partial promotion or planner change. Required review has no blockers; all18 regression groups pass, including affected post-review repeats. [Promotion evidence](stage5/artifacts/reduction-structure/final/promotion/README.md). |
| Reuse beyond the long-row path | `5nb` implements actual pinned CubeK PlaneReader + Sum through unchanged descriptor replay, CppCompiler/NVRTC and the existing launcher. CPU-before-GPU and GPU qualification pass66 cases/945,534 output elements; live ABI/stream/offset controls pass. | Qualified negative result: the selected32×8 Lazy-plane configuration loses all six alternating rounds in all nine RMS graphs. Primary isolated2.333→4.132µs; generated graph2.340→4.162µs. Stop condition reached; retain explicit diagnostic, no fallback mechanism, tuning sweep or automatic eligibility. [Trial evidence](stage5/artifacts/reuse-first-rms/README.md); [8x0 selection](stage5/artifacts/reuse-first-selection/README.md). |
| RMS reuse attribution | `hs6` verifies that CubeK requires an explicit routine choice; conditional Plane packing matches the trial, while default 128-bit I/O policy admits FP32 input width4 at1024 and scalar at255/4099. The surviving row-coordinate remainder accounts for1,536/47,680 executed warp instructions; the rank loop and adapter position round trip are optimized away. | The selected vector4 follow-up is now completed as wqc below. hs6 itself introduced no GPU candidate/timing;5nb and its attribution packet remain closed and unchanged. [Attribution](stage5/artifacts/reuse-first-rms-attribution/README.md). |
| Vector4 RMS reuse | `wqc` composes actual vector4 PlaneReader/Sum plus stock component fold. CPU-before-GPU and GPU qualification pass36 cases/868,800 outputs; ABI/alignment/stream and12static controls pass. Actual primary warp instructions fall47,680→43,648 while shuffles rise320→1,280; scalar descriptor loads/output mapping remain. | Stop: vector4 loses all six paired graph rounds against ordinary in all nine cases on the display-disabled RTX3080. It improves FP32 graphs versus scalar but generally regresses FP16/BF16. Clock states vary210→1710MHz; isolated primary cost is unqualified across that transition. Required review has no blockers and affected checks pass. Explicit diagnostic only; no second mechanism or promotion. [Evidence](stage5/artifacts/reuse-first-rms-vector4/README.md). |
| Packed-load reuse capability | `9e8` locates the scalar loads in our virtual read adapter and identifies the existing descriptor5 storage-vector load → scalar extraction → unchanged descriptor2 replay pattern. Four storage elements require16-byte FP32 or8-byte FP16/BF16 alignment; the P4 FP32 operand/reduction order can remain unchanged. Host enumeration checks all260 admitted B/N pairs and11,600,160 scalar address identities. | Bounded source investigation closed after user-authorized native GPT-6.1 medium review; original review-mcp HTTP429 preserved. The separate implementation2b9 is reported below. [Capability evidence](stage5/artifacts/reuse-first-rms-packed-load/README.md). |
| Packed-storage RMS trial | `2b9` changes only the input storage access using that pattern, retaining the byte-identical CubeK P4 component/mapping body and original casts/epilogue/runtime. CPU and GPU36-case policy passes; all final bytes and qualification-observer FP32 operands/per-lane sums match frozen wqc. Actual SASS confirms128-bit FP32 and64-bit FP16/BF16 input loads; metadata loads excluded. | Stop after this single qualified diagnostic: FP32 graph medians improve about3–4% versus wqc, but ordinary wins all54 paired graph comparisons across nine cases; low-precision deltas are small/mixed.216 raw traces retained; primary isolated FP32 cost remains unqualified across execution/clock transitions. Required review approves with no blockers; affected checks pass,2b9 closed. No second mechanism, tuning or production eligibility. [Trial evidence](stage5/artifacts/reuse-first-rms-packed-trial/README.md). |
| Transpose/layout reuse | `d39` distinguishes original transpose-view graphs (linear physical reads/stores, transposed output retained) from explicit contiguous materialization. One actual CubeCL perpendicular gather/register-transpose/store body composes through existing virtual tensors and unchanged descriptor replay. CPU24 mapping cases/1,215,843 addresses; GPU54 bitwise cases,42 live guards,6 stream controls and9 view controls pass. | Qualified negative: all108 paired rounds across18 materializing graphs lose to ordinary. Actual FP32 wide loads/stores and instruction savings coexist with one/three-block grids and long worker loops. Clocks vary; no stable intrinsic-cost or stock-launcher claim. Stop this mechanism without tuning/promotion; required review-mcp found no blockers, with seven minor diagnostic observations documented. [Evidence](stage5/artifacts/reuse-first-transpose/README.md). |
| Secondary execution state | Disjoint-address controls retain read-last fast/write-last slow selection. A1.5MiB writer lowers both target-window writes and target latency in every paired trial; larger footprints and idle delays do not define a clean one-variable mechanism. | Final bounded pass completed:192 raw captures,176 admitted coupled comparisons;24MiB contrast unqualified due a preserved clock-window failure. Exact internal source remains unresolved. Keep cwk.5 parked as an environmental benchmarking confound; the qualified fixed-final graph validation(cwk.7) is reported above. No production warmup. [Causal evidence](stage5/artifacts/reduction-structure/state-memory-causal/README.md); [prior counters](stage5/artifacts/reduction-structure/state-counters/README.md). |
| Host dispatch | Redundant warm-path work was reduced; the measured small training graph still has substantial host/framework costs. | No new runtime/interface redesign is established by those captures. |
| LLVM vectors | Scalar LLVM remains an independent verified route; a vector CompositeConstruct verification issue is isolated. | `q6l` remains open; no LLVM vector tuning candidate is admitted. |

The current cwk.9 integration reuses that qualified final through the normal
schema-5 `cpp` route. It admits only the exact pure constant9 scalar FP32-input
sum forms, supported FP32/FP16/BF16 output/cast sequences, owner-only stores and
queried CUDA32 topology. The new plan/cache identity cannot collide with generic
ascending `independent_planes`. Ordinary ABI validation remains active and the
new plan does not use the compensated reducer's coarse corruption guard.
Eligibility and empirical selection stay separate. Other split counts and
unqualified descriptors retain existing candidates. The full-feature compiler
controls, required review and all18 established regression groups pass. The minor
review findings are fixed and affected checks pass again. The primary isolated
final is1.888→1.535µs and its frozen-producer graph sum43.790→43.459µs; the secondary
capture is unstable and does not establish a graph gain. Full raw evidence and
preserved setup/profiler failures are in the promotion packet.

The normal guarded planner selects final counts32/4 for8×1M/64×65539 in this
capture, so these graphs correctly exclude the new candidate. Ordinary inherited
split9 graphs pair the existing compensated producer with the new final and pass
strict checks at both shapes. The qualified reference512 producer remains an
explicit diagnostic; its artifacts and numerical policy are unchanged.

The following cwk.7 paragraphs describe the earlier diagnostic-only capture.
The cwk.7 generated-graph pass retained the mature CubeK independent-plane
mapping, CubeCL ShuffleXor/FAdd mechanism and existing descending FP32 order.
The explicit probe now admits constant final extents1..32 after CPU qualification;
the normal backend final policy and physical catalog remained unchanged in that pass. Fresh
same-process rotated measurements include real CubeCL generated wrappers and
the retained Triton generated two-kernel device sequence loaded without its
runtime. These are CUPTI kernel-duration sums, not host latency, queue-inclusive
time, or full Triton-wrapper comparisons. No synthetic activity or memory primer
is used. The 64-row case remains slower than that reference despite a repeatable
final-body gain; automatic eligibility was unaccepted at that point. cwk.5 stays parked.

Required review-mcp found the cwk.7 change sound. Its cache-environment ordering
finding is fixed; the affected33 extra graph cases and current-stream/offset
control pass again with identical recorded rows and kernel artifacts. The
independent analyzer checks all108 raw traces and the three storage-type
substitutions; all18 established gates pass. [Review disposition](stage5/artifacts/reduction-structure/final/graphs/review/disposition.md).

## Lessons to carry into the next pass

- **Reuse first across the backend.** The user confirmed this broader rule on
  2026-10-16: start with actual CubeK/CubeCL components before designing custom
  physical expressions or launching elaborate tuning investigations. The lack
  of trials outside reductions is a coverage gap, not a limit on this rule.
  Runtime/API differences warrant a bounded adapter trial. Custom machinery
  needs a concrete missing capability, unsafe-to-adapt semantic mismatch or
  measured deficiency. Make a quick component/workload match and try one
  available mechanism; preserve the existing semantic/runtime contracts.
- **Control history and both reduction phases.** Match input bytes and efficient
  reference regime; hold logical split count and final physical artifacts fixed
  before attributing partial-kernel differences to an algorithm.
- **Follow executed code.** More registers, barriers, accumulators or verbose CUDA
  source alone do not establish cost. The latest causal experiment changed our
  physical loop expression after retaining grouped loads. Uniform control, safe
  read predication and actual allocation matter together; no CubeCL correctness
  bug was established.
- **Keep timing scopes separate.** Prepared partial batches, complete generated
  graph CUPTI sums, cold Nsight replay and host/event intervals answer different
  questions. Historical timing numbers are not current universal targets.
- **Judge behavior and cost together.** Mature Triton/ATen/CUB/CubeK structures are
  references, not exact-imitation requirements or accuracy ceilings. Preserve
  PyTorch contracts; extra FP64 accuracy alone does not justify a materially
  slower default. The current experiment qualifies outputs per case/element;
  it does not hide failures by averaging or waive them because a plan is faster.
- **Retain the runtime and FP invariants.** PyTorch owns storage/current streams;
  lazy access protection, precision barriers, readiness and allocator lifetime
  remain hard constraints. Conservative FP hardening remains active.

The latest comparison is on RTX3090/SM86. CubeCL acceptance runs exclude Triton
and keep CPython3.14t GIL-disabled; the separate installed Triton reference enables
the GIL, qualifying host comparisons. Production CUDA13.0 NVRTC/driver JIT,
Windows CUDA13.1 offline utilities and Linux Nsight measurements remain distinct.

## What remains unresolved

cwk.9 establishes guarded automatic eligibility for the qualified constant9
final. It does not establish a broadly competitive long-row default, admission
of other final forms/counts or general Triton parity. The earlier cwk.7 secondary
workload still trails the retained reference; the new secondary capture is too
variable to establish a whole-graph gain.
The following paragraphs preserve the earlier passes and their evidence scopes.

The original large FP32 gap was substantially caused by how our physical IR
expressed the already-qualified work. A thread-dependent induction variable
made NVRTC peel/unroll divergent loop-control paths. The isolated uniform-loop
translation from CubeK removes both executed reconvergence regions, preserves
all active scalar loads/shuffles and keeps both existing publication barriers.
All 152 bitwise corpus/stream/special-value checks pass before timing. Prepared
partial medians are about 41.60 versus efficient Triton 44.78 µs at 8×1048576,
and 21.51 versus 21.32 µs at 64×65539, with fixed split9/final and identical inputs.
These component measurements do not by themselves establish complete-graph or
broad default competitiveness. Fresh graph checks preserve all 178 outputs and
input hashes, including both existing eager-relative failures. Three replacement
fixed-final graph rounds pin the retained Triton configurations through PyTorch's
existing lookup table: primary graph sums are 47.4–47.8 versus 47.8–48.7 µs;
secondary graph sums are 23.7/27.6/27.7 versus 25.0/25.1/24.9 µs. The secondary
still loses in two rounds and is variable. Its unchanged CubeCL final costs
about 2.3–2.4 versus reference 1.28 µs. Earlier unpinned captures with final-config
drift are preserved and excluded from these timing conclusions.


The frozen partial's causal primer matrix establishes that read/write history
selects roughly22 versus26µs execution for the identical target. Earlier register
compute, reused compact reads and sparse address touches do not explain the
transition by generic active duration or target-address coverage alone. The
subsequent selector-preserving CUPTI PM pass observes substantially more
device-wide DRAM writes in the slow target window with nearly equal measured
SM/memory clocks. The __stwb/__stwt intervention changes neither split. These
historical facts and failed/perturbing counter methods remain retained in the
[primer packet](stage5/artifacts/reduction-structure/state-primers/README.md) and
[counter packet](stage5/artifacts/reduction-structure/state-counters/README.md).

The final bounded A/B/C pass queries6MiB L2 through CUDA device properties and
predeclares all addresses, footprints, delays and repetition counts. All192 raw
activity captures preserve exact artifacts/input/complete partial/final bits.
176 captures form fully admitted coupled comparisons. A delayed timestamp anchor
makes one24MiB window's unchanged conservative clock guard exceed target duration;
that entire16-capture contrast stays unqualified, without replacement samples,
narrowed guards or favorable-prefix inference.

Same-address revisiting is unnecessary: writeA→readB remains fast(~22.14µs), and
readA→writeB remains slow(~26.21µs). Reducing writer footprint to1.5MiB lowers target
latency24.24 versus matched26.23µs and write bounds1.643–1.658 versus2.753–3.280MB;
all eight pairs both improve latency and conclusively reduce writes. This supports
a causal contribution from write-footprint-sensitive memory state. It does not
prove DRAM writes are the sole mediator, a per-byte cost, or a particular cache.
3/6/12MiB results do not form a monotonic capacity curve. CUDA host-function
scheduling makes short requested idle intervals imprecise; actual gaps are
retained. Even2.7–3.7ms without diagnostic GPU work leaves~2.5MB interior writes
and~25.68µs target cost. Callback0 already changes latency; no clean joint traffic/
latency drain curve or precise25µs threshold is established.

The exact internal hardware/driver source remains unresolved. Under the user's
final-pass stop condition, **park cwk.5 as an environmental benchmarking confound**.
No more hardware-state candidates, production synthetic reads/warmup, kernel,
precision, eligibility or Stage6 change. Stage5 returns to the already-qualified
fixed-final component's generated-graph validation(cwk.7), with history/producer
identity recorded rather than hidden. The new diagnostics pass local admission,
identity and static checks. Postcommit review-mcp of4fd9337 accepts the change
as-is with no material correctness, admission-waiver or provenance defect and
corroborates the reported numbers. Optional diagnostic robustness notes are
retained; setup-warmup/schedule documentation is clarified. Executable source
and every retained capture stay unchanged; native review remains disabled.
[Review disposition](stage5/artifacts/reduction-structure/state-memory-causal/review/4fd9337/disposition.md).
[Full causal evidence, preserved failure and limitations](stage5/artifacts/reduction-structure/state-memory-causal/README.md).


The separate final investigation also identifies a producer-history confound:
after retained Triton partials, the original CubeCL final costs about 1.73µs;
after frozen CubeCL partials it costs about 2.30µs. The explicit plane final
reduces those to about 1.41 and 1.92µs. The same retained Triton final shifts from
about 1.25 to1.66µs across those producers. These are crossed, fixed-artifact,
bitwise-identical controls, so the earlier2.3-versus1.2 comparison was not solely
a final-body cost. The candidate's remaining extra shuffle, reconvergence and
i64/metadata work are traced through IR/CUDA/PTX/executed SASS; no fraction of
the latency is assigned to a proxy alone. Actual ATen nine-value dispatch is
modeled and GPU-checked over148 cases, but its stock FP32 sum_out costs about
2.43µs here and is not adopted. [Final evidence](stage5/artifacts/reduction-structure/final/README.md).

The ordinary uniform-loop partial and ordinary final regenerate byte-identical
IR/CUDA/PTX for all three storage types (18 file comparisons). The new final is
an explicit probe only; default selection, split9, numerical/precision policy,
PyTorch storage/current-stream ownership and Stage 6 status are unchanged.
Seventeen established regression gates pass in the full run; the last initially
has an empty CUPTI census and passes unchanged in a fresh affected repeat. The
initial missing-LLVM build failure and empty-census failure are retained; no
assertion is waived. Historical artifacts are restored after every gate run.
The fixed-final/primer pass had no completed independent review; native
review was disabled by the user. The separate counter pass completed two review-mcp reviews. Its evidence
validator and failure/FFI guards were strengthened; all96 counter captures
revalidate with15 admission controls. The last small comparison/guard fixes
were checked locally; review disposition is retained in the counter packet. The earlier [repeatability controls](stage5/artifacts/reduction-structure/repeatability/README.md)
remain useful evidence, including failed NCU state preservation and the different
RTX3080 behavior, rather than a universal display/clock explanation.

The original 48 versus 29 register difference is now 40 versus 29 for this
expression. The isolated bounded-counter probe reaches 36 registers but is
slower than the retained uniform loop; combining them allocates 46 and regresses
the secondary case. Removing owner-only root publication is numerically safe
under the tested proof, but gives negligible isolated benefit. Those probes
remain offline controls. The normal fused result-visibility and scratch-reuse
barriers remain. Generic signed-i64 split/address bookkeeping is still visible;
its remaining practical costs are not a reason to weaken semantics.

CubeK's actual virtual-tensor/reduction bodies compile and execute without its
runtime/allocator ownership, after small documented compatibility edits in an
isolated copy. Stock reducers have different arithmetic and output ownership,
so this does not establish an arbitrary Inductor-to-CubeK fusion adapter. The
retained change translates an existing mechanism behind our current boundary.
CubeK Mask still reads safe redirected addresses for inactive lanes and cannot
replace our lazy protected reads. **Continue Stage 5; Stage 6 remains deferred.**

All 18 established regression gates passed in the earlier codegen pass, with historical packets
restored byte-for-byte. Current bitwise integration, counter-boundary and code
identity evidence is in the codegen packet. With review-mcp unavailable, the user
authorized a native adversarial review. Its harness-provenance findings are
resolved: the gate runner restores every protected historical file and symlink
and relocates new evidence; CubeK setup extracts into a fresh generated tree.
The follow-up review reports no unresolved findings. CPU fixtures, a 14,195-file
preservation check and the clean CubeK rebuild pass; kernel and measurement
artifacts are unchanged. [Review disposition](stage5/artifacts/reduction-structure/codegen/native-review.json).
Source-hash invalidation still incurs one-time tuning revalidation after rebuild.

The preceding load-batch investigation reran all 18 regression gates successfully after preserving
an initial unseeded Stage3 eager-relative failure. Rebuilt HEAD/current Stage3
component CUDA and PTX match. Required review findings were resolved and affected
checks rerun. These are bounded results, not a production-readiness declaration.

## Supporting records — consult for proof, not to reconstruct current state

| Record | What it preserves |
| --- | --- |
| [Warm dispatch](STAGE5_LAUNCH_REPORT.md) | Launch/validation changes and qualified host measurements. |
| [Attribution](STAGE5_ATTRIBUTION_REPORT.md) | History/host analysis and the precursor constructor-precision candidate. |
| [Precision correction](STAGE5_PRECISION_REPORT.md) | Installed overlay correction and neighboring semantics/backward evidence. |
| [Accumulation](STAGE5_ACCUMULATION_REPORT.md) | Compensation experiment and the current practical behavior/performance requirement clarification. |
| [Matched references](STAGE5_REDUCTION_REFERENCE_REPORT.md) | Exact input corpus, efficient versus pathological histories and reference physical structure. |
| [Guarded long rows](STAGE5_LONG_ROW_REPORT.md) | Shape-regime fix, fixed-split candidate rejection and numerical qualification policy. |
| [First cost isolation](STAGE5_LONG_ROW_DIAGNOSIS_REPORT.md) | Partial/final confounds, redundant scratch-barrier fix and code/counter evidence. |
| [Latest structure diagnosis](STAGE5_REDUCTION_STRUCTURE_REPORT.md) | Fixed-split/final refresh, load-organization cause, sole candidate and retained numerical failure. |

Reusable findings also live in keyed Beads project memory, loaded by `bd prime`.
Update this summary and those findings after each completed pass. Keep supporting
reports and artifact packets as the historical evidence; inspect them when exact
source, measurements or reproduction details are needed.

## Original Stage 5 gate record (historical)

The captured report below is preserved verbatim. Its candidate counts, tool
versions, open issues, defaults and next-step recommendations describe that
original capture. The current summary above takes precedence for project status.

<details>
<summary>Expand the original Stage 5 gate report</summary>

# Stage 5 — physical candidates, cached tuning and performance diagnosis

## Result

**Passed for the bounded C++/NVRTC Stage 5 gate.** All five implementations pass the fixed 168-case corpus; all 16 established Stage 0–4/FP gates and the Stage 5 controls pass. Physical vector/reduction alternatives, validated empirical selection, persisted cache reuse and hardware counter diagnosis are real. No Stage 6 implementation is included. LLVM vector verification and generic low-precision tensor-constant provenance remain explicitly isolated limitations.

Experimental entrypoint: `torch.compile(f, backend=stage5.backend.compile_backend, fullgraph=True, dynamic=True)`. First-use empirical tuning is the default. `functools.partial(compile_backend, tuning="baseline")` selects the named frozen physical strategy and carries that choice through AOT backward compilation.

## Architecture and frozen baselines

```text
Dynamo / AOTAutograd / GraphLowering / Scheduler
  → existing Stage 4 capabilities, SIMDScheduling fusion and LoopBody replay
  → typed semantic schema 3 reduction or schema 4 pointwise/effect DAG
  + PhysicalCandidate(version 2)
  → schema 5 envelope adapter
  → validated Rust descriptor / CubeCL IR
  → CppCompiler<Cuda> → NVRTC 13.0 (--fmad=false) → PTX
  → existing PyTorch pointer/InfoBuilder/current-stream/lifetime boundary
  → explicit native launch geometry
```

Semantic operations stay scalar, including explicit FP16/BF16 cast barriers. Physical vector loads are extracted into scalar values for replay; scalar results are assembled for vector stores. No CubeCL arena handles, CUDA coordinates or vector-shaped semantic values enter the descriptor. Stage 4 remains an independent entrypoint; Stage 0 manual C++/LLVM tests remain independent.

`scalar_pointwise_baseline` retains one element per unit, 256 units and the explicit tail predicate. `shared_tree_reduction_baseline` retains one workgroup per outer output, 256 strided scalar accumulators, FP32 partials, 256 shared elements, eight binary-tree levels and uniform barriers. Semantic correctness fixes apply to both physical baselines and optimized paths; they are not credited as physical speedups.

## Corpus and methodology

The fixed [corpus](stage5/artifacts/corpus.json) contains 168 cases: FP32/FP16/BF16 launch-bound and bandwidth pointwise, fused arithmetic, row/column broadcasts, transpose, slices, stride-zero views, lazy concat masks, row and column reductions, natural Inductor split reductions, mean epilogues, RMSNorm inference/backward, gather, atomic scatter, embedding backward, dropout backward and the Stage 4 training graph. Sizes include 1/255/256/257/4099, 1,048,583 and 16,777,219; reduction lengths include 31/32/33/255/256/257/1024/4099/65539. Component tests independently cover zero extents and boundaries that Dynamo specializes away.

The original frozen capture was performed before physical optimization and is retained as `baseline-original-cubecl_baseline.json`. Final captures compare the corrected semantic frontend consistently and use the same release compiler bridge. Correctness uses the established dtype-specific tolerances, arbitrary upstream gradients, restored RNG state and reset model/optimizer state. Atomic benchmark updates use exact dyadic values to isolate reset/order mechanics; random and adversarial semantics remain covered by the existing regression suites.

Twenty unprofiled event/host samples follow warmup. Whole invocation CUDA-event intervals include Python enqueue gaps, so they are not intrinsic kernel execution times. Separate profiler calls record every generated/vendor/extern GPU kernel. Profiler sums are compared only with similarly instrumented sums and are indicative process/regime measurements, not timing confidence intervals. Early separate-process captures showed large differences for normalized-byte-identical frozen CUDA/PTX. Those cannot be attributed to physical optimization. Controlled activity warmup reduced that anomaly; physical improvement claims use the same-process rotated candidate measurements. The CUDA sleep warmup consumes no RNG or model state and is excluded from measured kernels and host/event timing. Final family comparisons apply the same CUDA activity warmup outside every profiler region, then use seven independently profiled sums of actual GPU kernel durations, with state reset and synchronization outside every region. Tuning uses generic Inductor event warmup followed by 384 exact named CUPTI kernel durations per rotated round. GPU user annotations and queue gaps are excluded from selection. Event enqueue intervals remain a separate diagnostic; the generic `_CALLABLE` GPU annotation was experimentally found to include driver gaps on WSL. Compilation finishes before timing. CUDA synchronization occurs at measurement boundaries, never inside the launch implementation.

Triton references run in a separate CPython 3.14t environment with the same Torch cu130 version. Importing its native extension enables the GIL. Host comparisons are therefore environment-qualified; this is not a concurrency benchmark. Device intervals, qualified by their measurement scope and numerical policy, remain useful. The CUDA device is RTX3090/SM86; exact UUID, clocks/power/temperature snapshots and compiler identities are preserved in the data.

## Research findings and CubeK assessment

See [source research](stage5/artifacts/research.json). CubeK was inspected at `a680b7012088537f3ed8573693746493d273c78a`; its CubeCL dependency `92f9afec75a03fd0806c34332039ae87d6cdd5da` differs from this experiment's pin. Its public reduction launch APIs use ComputeClient/TensorBinding/runtime ownership, and are fixed algorithm entrypoints rather than our arbitrary Inductor prologue/epilogue DAG. Direct API/Cargo reuse would require an unrelated dependency/runtime integration. We reused algorithm designs—plane partials plus shared final reduction, independent packed planes and finite shape buckets—through the existing compiler-only bridge. No CubeK runtime allocation or stream is introduced. Matrix/GEMM/attention/tensor-core capabilities were inventoried and remain Stage 6 work. [Pinned reduction launch source](https://github.com/tracel-ai/cubek/blob/a680b7012088537f3ed8573693746493d273c78a/crates/cubek-reduce/src/launch/base.rs).

CubeCL provides vector pointer types, vector memory operations, scalar extraction/assembly, shared arrays and plane sum polyfills. C++ packing/unrolling/optimization is reused; the backend explicitly chooses work assignment/coalescing/topology. It does not assume compiler passes invent them. The plane sum polyfill uses XOR shuffles and CUDA active masks. The executed physical paths keep whole participating planes active and use neutral private accumulators for reduction tails.

Inductor's generic `PersistentCache.lookup` and `InductorBenchmarker.benchmark_gpu` host the adapter without Triton Config/launcher objects. The former's actual source invokes its callback on cache miss independently of `max_autotune_gemm`, despite an outdated docstring. We supply exact backend compiler/device provenance instead of CacheBase's Triton system key. Template-oriented AlgorithmSelectorCache was inspected and not adopted. Generic stride/coalescing analysis and Triton launch/autotune/resource heuristics were inspected; exact direct-index descriptors already provide the narrow proofs needed here. Historical `config.triton.coalesce_tiling_analysis` remains disabled; no sweeping configuration rename was necessary.

ATen's ReduceConfig separates input/output cooperation and vectorized/coalesced memory work. Its FP16/BF16 sums use FP32 opmath accumulation. CUDA scalar opmath functors retain scalar literals in higher precision than tensor storage. [Pinned CUDA scalar helper source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cuda/Loops.cuh).

## Physical candidates and legality

The catalog is finite: at most twelve pure pointwise choices and fourteen reduction choices before layout/dtype/precision rejection (ordinary rows admit at most twelve; degenerate unit-stride outer/reduction patterns can admit the two column choices too). Executed selection sets and rejections are preserved. All are explicit compile-time identities; runtime pointers, seeds and tensor contents are absent.

| Family | Candidates | Boundaries |
|---|---|---|
| Pointwise | scalar64/128/frozen256; scalar128 with four elements/unit; vector2/4 at64/128/256; vector8 at128/256 for16-bit storage | At most16-byte memory access. Exact direct-linear addressing or scalar uniform load; actual pointer alignment. One observable vector store. |
| Row reduction | shared64/128/frozen256; plane+shared64/128/256; packed independent planes at128/256 units; plane+shared128/256 with vector2/4 inputs | FP32 accumulation, full planes, lazy scalar tails, aligned row bases/unit stride for vector inputs. |
| Column | shared128/frozen256; coalesced columns with block(32,4)/(32,8) | Neighboring outer memory addresses proven affine; invalid output lanes still participate in all barriers. |
| RNG/indirect/atomic | frozen scalar256 automatically | Controlled scalar64/128 alternatives test equivalent reset state; automatic effect tuning is deliberately bypassed. |

Packed planes assign four/eight outputs per workgroup and allocate zero shared memory and require no shared workgroup barriers. Plane+shared reduction publishes two/four/eight FP32 partials, synchronizes uniformly, has the first complete plane reduce neutral-filled partials, publishes the result and synchronizes before reuse. Coalesced columns assign neighboring outputs to UnitPosX and cooperation to UnitPosY, publishing shared scratch for all lanes including invalid outputs. Neither the logical replay nor the DAG contains these identities.

A focused FP64 diagnostic exposed excessive private accumulation error in smaller-cooperation long-row candidates. The existing acceptance tolerances were retained. Reduced-cooperation candidates are excluded once private accumulation exceeds 256 elements/unit; the frozen/256-unit assignment remains available. This is a bounded candidate surface, not a claim of universal numerical stability for every parallel ordering. [Diagnostic](stage5/artifacts/reduction-accuracy.json).

The vector reduction proof is mirrored in Python and Rust: literal signed-i64 reduction coefficients 0/1 only; runtime scalar multipliers are excluded even when their current value is 0/1. A literal `r*0` load stays scalar/uniform. A misaligned minimal backing-allocation test proves that distinction. Vector tails use separate guarded scalar accesses; they never issue invalid vector loads then mask the result. Nonlinear broadcast/masked/indirect addresses are rejected from vector plans. Scalar candidates still consume the established broadcast/stride/mask semantics. Genuine Inductor split reductions remain intact; there is no new split planner, template path, autotuned RNG or generalized fallback.

## Autotuning, cache and reset

First use compiles all legal physical candidates before timing, evaluates each on isolated PyTorch-owned scratch outputs, checks against the frozen strategy, and stops on mismatch. Inputs and user-visible outputs remain untouched during tuning. Pure selection does not consume RNG; the state equality is explicitly asserted. Three rotated rounds avoid a fixed candidate order. Each round requires 384 actual kernel events. An incomplete CUPTI buffer is discarded and retried at most three times, preserving all attempt counts; persistent incompleteness fails rather than timing partial data. A stable best candidate must win at least two rounds and have its median±3 MAD range separated from the baseline (a conservative stability heuristic, not a statistical confidence interval); otherwise selection retains the baseline.

The cache key combines semantic DAG/layout, device UUID/architecture/driver, compiler and metadata/launcher hashes, exact FP patch, CUDA headers/NVRTC/builtins hashes, conservative math policy, adapter/tuner version, legal catalog and finite shape/proof classes. Pointwise/reduction/outer buckets are finite. Alignment and affine stride predicates are finite classes; exact runtime scalar values remain metadata. Warm calls reuse proof classes and local selections. There is no one-kernel-per-runtime-size compilation strategy.

Artifact files are hash-checked under a file lock. A complete expected artifact set is required. PTX can reload in a fresh process without repeating CubeCL lowering or NVRTC; CUDA modules stay process-local. Selection persistence uses Inductor's atomic cache writer plus per-key locking. Selection identity is compared using canonical JSON to survive tuple/list serialization. A fresh process reused an empirical scalar64 selection with zero compiler invocations. Deterministic mode can reuse a valid selection and otherwise keeps the frozen baseline without fresh empirical timing; an explicit cold-cache control confirms the latter and unchanged RNG.

Both semantic validators and an early Python assertion require exactly one final write buffer. Prepared native batches live in an explicit timing scope, retain buffers only for that scope, and release references afterward. Ordinary launches do not add user-tensor lifetime caches. RNG and mutation batches are rejected. Controlled tests restore explicit seeds/counters or zero destinations before each candidate/trial; the generated dropout graph's user RNG advancement is identical under baseline and first-use tuning of its pure consumers.

## Numerical policy

The conservative FP patch remains exactly `9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2`: no unsafe algebraic identities, implicit contraction, LLVM contract flags or target-implied fast math. NVRTC retains `--fmad=false`. FP32 compute and FP16/BF16 explicit storage barriers are preserved by vector and reduction strategies; accumulation stays FP32.

Configuration controls are not treated as proof of equivalence. The per-operation probe tests add/multiply, contraction-sensitive arithmetic, division, sin/cos/exp/sqrt/reciprocal, precision chains, select, casts and constants with special values. CubeCL and configured Triton match bit-for-bit on all 42 operation/dtype probes after correcting a Stage 2 literal conversion error. Nonrepresentable FP16/BF16 scalar coefficients now stay in FP32 compute precision until a materialization/cast/store boundary.

A separate tensor-constant fusion fixture produces an eager discrepancy shared bit-for-bit by configured Triton and CubeCL; it is tracked separately as `pytortch-backend-experiments-iqw`. Scalar opmath precision is not treated as proof of tensor-constant materialization semantics; it is explicitly qualified rather than used to infer eager-exact behavior. Default Triton differs from CubeCL in 10/42 byte comparisons, including contraction-sensitive expressions, division, reciprocal, sin and precision chains; one FP16 precision-chain case changes signed-zero behavior. It is measured independently; its relaxed/default math is not labelled equivalent from settings alone. RNG implementation streams are not assumed bit-identical across backends. Signed-zero/NaN/Inf classification and low-precision barriers are checked separately from reduction ordering tolerances.

## Performance decomposition and comparisons

The [full 168-case comparison](stage5/artifacts/results.json), [representative table](stage5/artifacts/performance-table.md), [figure](stage5/artifacts/plots/qualified-family-comparison.svg) and [candidate distributions](stage5/artifacts/tuning.json) retain different measurement scopes. No universal percentage or arbitrary parity threshold is claimed. Seven-sample medians and MAD are available for every family. A default-Triton profiling process stalled after 108 completed records; the remaining 60 were captured in a fresh process with identical warmup, timing, math policy and strict counts. One incomplete baseline profiler record was recaptured independently. Neither event lowered the count/correctness gate or changed the Nsight path; capture qualifications and exact commands remain in final-gates.json.

* Fusion/scheduling: the existing Stage 4 capability/fusion machinery is unchanged; actual generated/vendor/extern kernel census accompanies every family. Optimizer work remains outside the compiled training graph.
* Physical mapping: vector memory transactions, changed workgroup sizes, plane/shared reductions and packed/coalesced outputs are measured against the frozen strategies. Prologue/epilogue fusion is retained.
* Compiler/codegen: CUDA 13.1 offline disassembly confirms 128-bit vector memory operations. Actual production register/local/shared-memory/occupancy queries are separate. Precise transcendental math is a separate policy cost.
* Runtime/launch: tiny whole-invocation event intervals are dominated by Python validation/tracing/metadata/driver enqueue gaps. Prepared batches supply actual named CUPTI kernel durations for selection; driver queue intervals remain separately labelled. There is no Stage 5 CUDA Graph capture implementation.

## Pointwise results by regime

The bandwidth FP32 case 16,777,219 elements is already memory-bound: baseline 163.718us, selected vector 161.370us, configured Triton 160.655us, ATen 159.967us. The small difference is not a broad parity claim. FP16 improves 103.710→81.238us and BF16 improves 97.493→81.240us; the configured references are 80.760/80.924us. At 1,048,583 elements fused FP16/BF16 bodies improve 10.640→9.122us and 10.588→9.108us; scalar arithmetic/barriers remain explicit beneath vector memory transactions. Same-process candidate trials include approximately 2.12x/2.07x low-precision vector wins for specific semantic/bucket keys; this is distinct from whole-graph ratios.

Tiny FP32 add remains launch-bound (1.665/1.878us GPU for baseline/Stage 5); physical tuning adds no compelling graph benefit there. Broadcast and transpose cases mostly use scalar strategies because the proof does not admit arbitrary vector layout transformations. For example transpose 256x513 is 2.690→2.596us, while configured Triton is 3.745us and ATen 4.285us. Preserved logical scheduling/fusion and the kernel census, rather than a new semantic operation, explain the fused graph advantage over eager multi-kernel bodies.

## Reduction and normalization results

Short row 64x31 improves 2.241→1.567us with packed independent planes, and FP32 RMSNorm 64x1024 improves 3.016→2.113us. RMSNorm backward is 7.808→6.941us across three generated kernels; configured Triton is 5.999us and ATen 29.422us. Low-precision RMS forward improvements are smaller (FP16 2.689→2.634us, BF16 2.819→2.718us), with per-operation/store barriers retained.

Long row 8x65539 retains the frozen plan: 10.465/10.335us versus configured Triton 5.961us. The CUDA/PTX bodies are identical after entrypoint normalization, so this variation is not credited as optimization. Eight outer outputs yield only eight workgroups, vector rows fail alignment proofs, and the bounded candidate space has no greater-than 256 cooperation or new split planner. This is a concrete topology/utilization gap. Column 4099x64 also retains frozen cooperation (5.847/5.508us); coalesced lower-cooperation plans fail the private-accumulation bound. For compatible shorter/coalesced families, paired trials select column plans, while the diagnostic 1024x256 column shows why that strategy can lose on a small grid.

There are 182 measured semantic/bucket/proof selections: 64 frozen pointwise, 40 frozen tree, 23 scalar alternatives, 22 independent-plane, 16 smaller shared-tree, 11 vector and 6 coalesced-column winners. Plane+shared/input-vector paths pass controls but were not forced to win. At most 14 reduction choices are available in the degenerate overlap of row/column address classes; ordinary rows admit at most 12. No new split-reduction machinery or operation coverage was introduced.

## Stateful and memory results

Gather/scatter/dropout remain deliberate frozen generated implementations, with pure consumers eligible for tuning. Representative FP32 GPU sums are gather 2.785/2.848us, collision scatter 13.012/11.905us and dropout backward 9.744/8.678us (baseline/Stage 5). Because stateful kernels retain identical physical strategy, these process differences are not advertised as stateful optimization. Configured Triton is 2.806/11.443/5.885us, ATen 4.741/11.487/4.586us. Per-backend restored-state replay and explicit Philox/atomic controls remain the correctness proof; cross-backend stochastic masks are not assumed identical. Indirect checks and atomic initialization/order were not weakened.

## End-to-end training

The exact Stage 4 graph uses embedding 17x7, linear 7x9 without bias, RMS normalization, dropout 0.25 and squared flattened mean loss. SGD(lr=0.01,momentum=0.9,foreach=False,fused=False) remains outside the compiled graph.

| Implementation | Unprofiled step event ms | Host enqueue ms | Warm GPU kernel sum us | GPU kernels |
|---|---:|---:|---:|---:|
| CubeCL frozen |4.868|4.450|36.722|20|
| CubeCL Stage 5 |3.944|3.918|45.733|20|
| Configured Triton |2.225|2.144|27.710|18|
| Default Triton |1.941|1.856|29.910|18|
| ATen |2.528|2.452|61.478|37|

Both CubeCL paths execute eight generated kernels: five forward and three backward; one forward RNG seed extern, one forward GEMM, one backward gradient fill, three backward vendor GEMM/reduction kernels and six external optimizer kernels account for the rest. Triton has two fewer forward generated kernels; this is scheduling/fusion quality, not a CubeCL arithmetic speed issue. Optimizer and seed externs are recorded explicitly. [Live trace](stage5/artifacts/schedule-trace.json) retains actual schedule/replay, three-kernel RMS backward and one-kernel fused pointwise execution.

The separately annotated Stage 5 census measures forward 41.604us, backward 45.576us and optimizer 20.898us, with 7/7/6 kernels. These single-call annotated durations differ from the seven unannotated whole-step samples and are retained only as instrumentation-qualified contributor evidence; they are not combined or divided with unprofiled step timing. Host overhead dominates this small model. The optimized GPU step does not beat the frozen one in this capture, so local wins are not claimed to survive end-to-end composition.

## Production resources and hardware counters

The 76 component configurations use 10–44 actual production registers/thread and report zero local bytes/thread. Offline disassembly finds no local load/store spills in the retained candidates and confirms FP32 LDG.E.128/STG.E.128 preservation. Register counts are CUDA Driver API JIT attributes, not C++ variable counts or offline 13.1 estimates.

| Nsight representative | Block | Registers | Kernel dynamic shared bytes | Achieved occupancy | Diagnostic duration us |
|---|---|---:|---:|---:|---:|
| scalar add 1,048,583 |256x1|16|0|74.98%|13.248|
| vector4 add 1,048,583 |128x1|18|0|65.13%|13.056|
| row tree 256x1024 |256x1|39|1024|46.14%|5.984|
| row plane/shared 256x1024 |128x1|39|16|23.52%|5.600|
| row packed 256x1024 |32x4|39|0|7.77%|5.312|
| column tree 1024x256 |256x1|40|1024|47.78%|9.600|
| column coalesced 1024x256 |32x8|40|1024|16.63%|12.544|

All static kernel shared allocations are zero; the driver separately reserves its own block overhead. The row strategies trade barriers/shared traffic for fewer active blocks; higher achieved occupancy is not automatically a faster kernel. The profiled coalesced column loses because a small outer grid and more private work offset neighboring accesses. Memory throughput counters identify the large add as primarily bandwidth-bound. Nsight replay/cache treatment is distinct from warm CUPTI distributions, so these durations are not mixed into the benchmark table.

## Compilation, persistence and dynamic buckets

[Initial first-invocation data](stage5/artifacts/compile-performance.json) spans about 0.7ms–4.99s for CubeCL Stage 5, depending on graph specialization, existing artifacts and first-use tuning. It is not labelled an all-artifact-cold compile measurement. Manifest fields separately retain descriptor parse, IR construction, CubeCL emission, NVRTC and module-load costs; immutable artifact hashes distinguish those from warm selection/launch work.

After population, a fresh process reports zero compiler invocations and a validated empirical cache hit. Dynamic 257/259/1001/4099/4101 tests reuse compatible artifacts/proof classes rather than recompiling for every size; moving buckets may reselect a physical plan while preserving runtime extent metadata. Normal Dynamo specialization of 0/1 and guarded layouts remains. Failed selection retries reconstruct canonical key data; cached tuple/list identity survives process serialization. Deterministic cold selection stays frozen and does not benchmark. The exact compiler/device/math/source identities remain in curated manifests and tuning keys.

## Compiler/tooling boundaries

Production uses PyTorch cu130/CUDA 13.0 NVRTC and the CUDA driver JIT. Windows CUDA 13.1.115 `ptxas`, `nvdisasm` and `cuobjdump` are verified offline utilities; paths are converted with `wslpath -w`. Their cubins/SASS/register estimates are labelled offline diagnostics, not production artifacts. Actual function attributes and theoretical occupancy come from Driver API queries for the production PTX. Theoretical occupancy is not achieved occupancy; small grids can leave SMs idle.

The existing Linux Nsight Compute 2025.4.1.0/build37053803 payload was staged into an ignored whitespace-free path. Initial `ERR_NVGPUCTRPERM` was a Windows-host counter permission denial, not an installation/WSL failure. After the user enabled unrestricted host counter access, all seven representative production kernels were successfully profiled with LaunchStats, Occupancy, SpeedOfLight and MemoryWorkloadAnalysis. [Hardware evidence](stage5/artifacts/profiling.json) includes exact executable/hash/version/commands. Windows `ncu.exe` was never substituted. Nsight replay/default cache treatment and profiler instrumentation differ from warm CUPTI timing, so these diagnostic durations are not mixed into the comparison table.

The scalar sin+add descriptor is executed through both C++/NVRTC and LLVM under conservative policy. At 1,048,583 elements the paired actual kernel medians are 11.896us(C++) and 11.521us(LLVM), with 22 versus 19 production registers and zero local/shared memory. Queue-inclusive native batch medians are 18.183us and 19.012us and are not substituted for GPU time. The representative Rust build/C++ emission/NVRTC costs are 1.39/1.43/18.16ms; LLVM build/emission is 2.29/83.45ms. Module load measurements are cache-qualified; these are a single descriptor probe, not a compiler-wide ranking. A separate vector LLVM probe fails verification with `CompositeConstructOp: expected cube.f32, got builtin.fp32`; it is isolated as `pytortch-backend-experiments-q6l` and is not admitted into the tuner. This does not replace the verified Stage 0 LLVM route. A narrow HIP vector source probe is retained as source-only evidence, with no HIP launcher/runtime claim.

## Source modifications

| Files | Reason / classification |
|---|---|
| `stage5/backend.py` | Existing semantic scheduling injection and descriptor-construction timing. |
| `stage5/plan.py`, `legality.py` | Finite physical identities, exact affine/alignment/target/precision proofs and bounded keys. |
| `stage5/artifact.py`, `runtime.py`, `timing.py` | Validated schema 5/cache/native adapter, generic Inductor persistence, isolated empirical selection and named CUPTI timing. |
| `stage5/corpus.py`, `bench.py`, `analyze.py`, `curate.py` | Fixed semantic corpus, qualified measurements/census, distribution summaries and inspectable artifacts/figures. |
| `stage5/fixtures.py`, `components.py`, `controls.py`, `proof_controls.py`, `descriptor_controls.py`, `cache_probe.py` | Existing-graph fixture extraction and independent physical/stream/reset/cache/negative controls. |
| `stage5/math_policy.py`, `reduction_accuracy.py`, `timer_probe.py`, `compiler_compare.py` | Executed numerical, accumulation, timing-scope and compiler-route investigations. |
| `stage5/offline.py`, `profile.py`, `ncu_probe.py`, `trace_probe.py` | Verified offline binaries, Linux hardware profiling and actual replay/native kernel census. |
| `stage5/regressions.py`, `final_gates.py`, `.gitignore`, `__init__.py` | Sequential gate orchestration, package entry and curated evidence/raw-cache separation. |
| `stage0/src/descriptor2.rs`, `descriptor4.rs` | Separate existing semantic validation from fixed-plan construction; existing compiler/emitter paths reused. |
| `stage0/src/descriptor3.rs` | Parameterize physical work assignment/shared and plane strategies while retaining existing semantic reduction closure replay. |
| `stage0/src/descriptor5.rs`, `main.rs` | Validated envelope/physical adapter and inspectable IR/compiler timing. No handwritten CUDA. |
| `stage0/launcher.c` | Explicit geometry/prepared batch/actual Driver API resource queries; original entries retained. No allocation, stream ownership or launch synchronization. |
| `stage0/bridge.py`, `stage2/runtime.py` | Small native-geometry and cached-PTX-loading hooks; common readiness/record_stream/storage checks retained. |
| `stage2/backend.py` | Executed correctness fix: scalar literals use opmath compute precision, not low-precision storage rounding. |
| `stage4/backend.py` | Out-of-tree scheduling-constructor injection/experimental selector registry, preserving capability and fusion behavior. No backend-name dtype policy. |

No new upstream PyTorch or CubeCL source patch was needed. The local conservative FP source stack remains exact. These backend/descriptor interfaces remain experimental; the literal fix is a semantic frontend correction, not a new generic dtype framework.

## Regression evidence and remaining limitations

Stage 5 component coverage is 76 physical/dtype/family combinations, 754 executed extent checks and 6 deliberate long-row vector rejection checks. Stream/lifetime/allocator-pressure, canary/alignment/special-value, explicit RNG and atomic reset, deterministic and descriptor rejection controls pass. The complete Stage 0 C++/LLVM, Stage 1, Stage 1.5, Stage 2/components, FP IR/runtime/Inductor, Stage 3/components and Stage 4 main/runtime/components/bounds/counter-control suites pass. [Regression evidence](stage5/artifacts/regressions.json), [details](stage5/artifacts/regression-details.json). CubeCL execution retains absent Triton/native modules and unforced GIL-disabled CPython 3.14t, PyTorch allocations/current streams and allocator lifetime protection.

Remaining limits include narrow vector address proofs, capped smaller-cooperation reduction candidates, first-use compilation/tuning cost, Python wrapper overhead, process-lifetime module/immutable metadata retention inherited from prior stages, no graph capture, no new effectful fusion, and no LLVM vector candidate. The data does not establish broad Triton parity or production-grade autotuning.

## Independent review and completion decision

Two independent review-mcp reviews returned successfully. The first found a static/runtime reduction-axis proof mismatch; it was fixed and tested. The second confirmed the fixes and reported no blocking correctness or benchmark-validity defect. Two low-severity robustness notes were also fixed: exact signed-int literal matching and full selection identity on exception/retry. [Disposition](stage5/artifacts/review-disposition.md), [post-review gates](stage5/artifacts/post-review-gates.json). Bounded profiler retries and common activity warmup preserve the reviewed strict-count/reset design; persistent failures still stop collection.

**Completion decision: 2 — more Stage 5 physical/tuning work is required first.** Prioritize the warm Python launch/validation/tracing path (`pytortch-backend-experiments-xld`) and then resource-justified long-row cooperation. The small training step is dominated by host/launch work, and local GPU wins do not consistently survive composition; adding GEMM/attention kernels first would miss the dominant measured cost. The LLVM vector bug (`q6l`) and tensor-constant provenance discrepancy (`iqw`) are isolated follow-ups. None has been started.

</details>
