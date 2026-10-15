# Stage 5: matched reduction structure and load-organization diagnosis

> **Supporting experiment record.** Start with [the current Stage 5 findings](STAGE5_REPORT.md). Results and recommendations below retain this pass's historical context.

## Result

The matched post-fix investigation isolates a **backend physical-IR construction
problem**, rather than a demonstrated CubeCL correctness bug. The four-chain
experiment serialized protected loads and accumulator updates through nested
control flow. One reference-derived load-batch candidate greatly reduces that
control-flow cost. It remains **diagnostic only**: its arithmetic is bitwise
identical to the rejected four-chain plan on the executed corpus, including the
same failing held-out output. No production default or split policy changes.

The earliest material cost difference in the currently selected compensated
partial is our explicit TwoSum arithmetic and its dependency chain. Removing
compensation in an existing-plan ablation removes measurable cost, but does not
itself establish numerical/default eligibility. Merely replacing the shared tree
with the existing generic plane collective does not materially help this case.
The new experiment explains why adding four accumulators previously made things
much worse; it does not explain every remaining difference from Triton.

## Evidence and controls

Packet: `stage5/artifacts/reduction-structure/`. Baseline revision is
`e96bc185cf1c585191ea46be4eb7a8c3325303b3`, after the final scratch-barrier fix.
`protocol.json`, `analysis.json`, command files, raw-data hashes, selected artifacts
and production driver-JIT SASS bind the comparisons. Historical captures are
preserved. Raw CUPTI traces, NCU reports/CSV, arrays and build logs stay in the
ignored cache. The selected text/JSON artifacts are the reviewable evidence.

Real `torch.compile(dynamic=True, fullgraph=True)` captures start with 8×65539.
The existing `compensated_diagnostic` policy and inherited logical split policy
freeze nine partials and the existing shared-tree final. Triton uses the efficient
configured reduction, not the pathological short-first graph. Across retained
rounds and B=8/64, the analyzer asserts identical partial **and final** descriptor/
PTX identities for CubeCL and launcher/configuration/cache identities for Triton,
per dtype. NCU verifies partial grids 72 and 576. The final algorithms differ
between backends; “fixed final” means no final-plan change confounds the partial
comparison, not that their arithmetic is identical.

Three rounds rotate backend order C/T, T/C, C/T. Each case has seven independent
CUPTI captures, with the established activity warmup outside the region, two-kernel
census and input-byte hashes. Partial/final timings are extracted separately.
Prepared components use the established 384-event native-batch method in three
rotated rounds. These are different measurement scopes. NCU uses kernel replay,
cache-control all and clock-control boost: its replayed durations are diagnostic,
not substituted for warm timing samples. Nsight Compute is Linux 2026.3.1.0 build
38829034; CubeCL production remains CUDA 13.0 NVRTC/driver JIT. No CUDA 13.1 offline
assembly substitutes for executed SASS. Triton's environment enables the GIL;
no host-time parity claim is made.

Two collection corrections are explicit. Initial r2 Triton/component captures
overlapped and were excluded, then rerun sequentially. An FP16 final autotune
choice varied in initial r1; the actually selected r0 result was replayed through
Inductor's existing best-config cache, then the complete r1 pair was rerun.
`final-cache-seed.json` records it; no autotuner/global backend monkeypatch or
unmeasured configuration was installed. The identity assertions then pass.

## Refreshed partial baselines

CUPTI partial-kernel medians in microseconds, rounds 1/2/3:

| Case | CubeCL compensated | Efficient configured Triton |
| --- | --- | --- |
| float32-8-65539 | 6.817/6.506/6.664 | 5.026/4.672/4.962 |
| float32-64-65539 | 26.572/23.058/26.467 | 23.839/23.486/23.611 |
| float16-8-65539 | 5.696/3.888/5.747 | 3.996/4.023/3.967 |
| float16-64-65539 | 17.276/17.159/17.023 | 14.108/14.017/14.099 |
| bfloat16-8-65539 | 5.526/5.862/5.630 | 4.160/3.964/4.036 |
| bfloat16-64-65539 | 17.435/17.313/17.558 | 14.109/14.107/14.229 |

All samples and MADs remain in `analysis.json`. Some rounds vary substantially
(e.g. the 64-row FP32 and 8-row FP16 CubeCL samples); do not turn a single round
or the aggregate median into a universal speed ratio. The partial remains the
main median graph cost. Final-kernel medians are independently retained; no final
optimization is folded into this diagnosis. All three 178-case compensated
captures pass the existing per-element corresponding-regime qualification.
Fresh 65K Triton references are used there; retained efficient regime references
remain explicit for other extents. Special classifications/dtypes remain hard
checks in the runner, separately from finite-error qualification.

## Complete executed structure

For 65539 elements, a logical partial consumes ceil(N/9)=7283 coordinates.
Its offset is `r + N*floor(outer/9) + ceil(N/9)*(outer%9)`, with the logical split
mask protecting the final part. The semantic descriptor remains unchanged.

| Concern | Selected CubeCL compensated partial | Efficient Triton partial |
| --- | --- | --- |
| Cooperation | 256 units, 8 planes; one partial/workgroup | 512 threads, 16 warps; one partial/block |
| Private work | coordinate unit+k*256, one FP32 sum/correction pair | four scalar FP32 chains, unit+j*512+k*2048 |
| Arithmetic | explicit finite-aware TwoSum residual/correction | ordinary uncontracted FP32 additions |
| Protection | loop extent plus lazy semantic load region | combined extent/split/output predicates on each scalar load |
| Loop | plain CubeCL RangeLoop; NVRTC peeling/unroll-by-four variants | four scalar loads per tiled LLVM/PTX loop; no vector memory load |
| Private merge | high+correction, then compensated shared merges | (a0+a1)+(a2+a3), then decreasing-distance warp butterflies |
| Collective | 256-entry high+low shared tree | five intra-warp shuffles, 16-value shared exchange, four inter-warp shuffles |
| Synchronization | 9 uniform workgroup barriers after the fix | 1 publication barrier |
| Publication | shared root readable by cooperating units; unit 0 stores output | layout-replicated reduced value; one output owner |
| Addressing | exact signed i64 semantic arithmetic | narrowed loop/predicate arithmetic plus 64-bit addresses |
| Memory policy | NVRTC `ld.global.nc`, executed `LDG.E.CONSTANT` | scalar predicated `ld.global...evict_first`, executed `LDG.E.EF` |
| Resources | 30 registers/thread, no spills, 2048 shared bytes | 29 registers/thread, no spills, 64 shared bytes |

FP16/BF16 have their existing explicit storage/compute conversion barriers and
FP32 accumulation; low-precision references and input bytes are matched. Do not
infer operation-policy equivalence solely from similarly named config knobs.
This experiment changes no arithmetic policy or conversion primitive.

Actual LLVM, PTX and SASS correct an easy mistaken assumption: the old CubeCL
private merge is `(a0+a2)+(a1+a3)`, whereas this executed Triton merge is
`(a0+a1)+(a2+a3)`. Their warp butterfly orders differ too. Four accumulators are
not an identical numerical topology. Signed divide/modulo expressions that look
expensive in CUDA source are hoisted outside the hot loops; repeated division
is not an established explanation for this gap.

## Existing-plan ablations and earliest expensive divergence

Holding the real partial semantic packet, nine-way split, input and final packet
fixed, prepared partial component round medians are:

| Plan | 8 rows, µs | 64 rows, µs |
| --- | --- | --- |
| Compensated tree | 4.133 / 4.129 / 4.131 | 23.461 / 23.025 / 23.102 |
| Plain tree | 3.339 / 3.346 / 3.336 | 22.335 / 22.289 / 22.043 |
| Single-chain plane/shared | 3.430 / 3.421 / 3.421 | 22.166 / 22.109 / 21.926 |
| Old four-chain plane/shared | 9.738 / 9.722 / 9.733 | 38.043 / 37.839 / 36.785 |

The first demonstrated expensive arithmetic divergence is already in our physical
plan's private compensation. At 8 rows, removing it saves about 0.8 µs in this
component scope. That is causal plan-ablation evidence, not a license to select
the numerically ineligible plain tree. The shared-tree barrier count alone is
an inadequate explanation: the old single-chain plane plan has two barriers yet
roughly ties the uncompensated tree with nine.

The four-chain version is worse **before its collective**. The actual SASS
repeatedly issues a protected scalar load, reconverges (`BSYNC`), and only then
updates an accumulator. Predicated-off branches can still carry reconvergence
structure even when most data accesses are valid. NVRTC unrolls the four-load
body again, giving a bulky peeled/main control-flow structure. The nested source
regions originate in `descriptor3::build_physical`; CubeCL C++ faithfully emits
them as ordinary `if`/`for` constructs. The additional peeling/unrolling and
machine reconvergence appear downstream of that C++ boundary.

Attribution is therefore **our IR/control-flow organization interacting with
ordinary downstream compilation**, not evidence of a semantic CubeCL miscompile.
The descriptor, protected loads and numerical outputs remain valid. We do not
claim that all CUDA compilers would generate this exact code or that branch count
alone predicts time.

## One mechanism candidate

`plane_shared_load_batch` is exactly one additional catalog identity: scalar
256-unit/four-chain/plane-shared, no vector loads or tuning search. Each protected
semantic body writes a zero-initialized private pending value. All four values
are collected before four independent accumulator updates. Bounds and semantic
lazy bodies still guard the loads themselves. Inactive positions add the neutral
zero; the original active chain/order and `(a0+a2)+(a1+a3)` merge stay fixed.
Workgroup, collective expansion, shared root broadcast, barriers, final packet,
logical split and conservative FP remain fixed.

This adapts ATen's proven load-before-accumulate organization without adopting
its tensor ownership, vector-memory path or dispatcher. It is a small change in
our physical builder using CubeCL IR, not handwritten CUDA or a replacement
compiler/launcher. `load_batch_diagnostic` explicitly selects it for long partials;
its numerical-policy hook rejects automatic selection at every extent.

NCU dynamic **warp instruction** counts at 8 rows / 72 partials:

| Metric | Old four-chain | Load batch |
| --- | ---: | ---: |
| BSSY | 27504 | 1152 |
| BSYNC | 27648 | 1296 |
| BRA | 33624 | 7848 |
| FADD | 22248 | 23112 |
| BAR | 1152 | 1152 |
| SHFL | 3240 | 3240 |
| Registers/thread | 35 | 48 |
| Spill instructions | 0 | 0 |
| Eligible warps/cycle active | 0.051162 | 0.143278 |

The new plan improves despite more registers and a few more inactive-neutral
adds. Register-limited residency drops from six to five blocks/SM; it does not
spill. Thus “lower register count” was not the missing mechanism. The SASS change
and preserved arithmetic isolate the protected-load/accumulation organization
much more strongly than a new unexplained topology would.

Prepared component round medians, same rotated five-plan run:

| Rows | Old four-chain µs | Load batch µs |
| --- | --- | --- |
| 8 | 9.688 / 9.707 / 9.711 | 3.874 / 3.902 / 3.878 |
| 64 | 37.948 / 37.478 / 37.454 | 24.060 / 23.909 / 23.879 |

Both satisfy the existing >=2/3-round and disjoint median±3 MAD criterion against
the old four-chain component. The candidate also beats compensation at 8 rows in
this prepared scope, but not at 64. These do not establish acceleration against
Triton or qualify a production default. A separate real generated-graph capture
has partial medians 5.539/27.106 µs at 8/64 rows, and two generated kernels per
call; it is not rotated against the earlier baseline rounds. Its final identities
are asserted identical to the frozen baseline finals.

## Cache, collective lowering and remaining uncertainty

Cold NCU L2 lookup misses are 65801 versus 65774 for compensated CubeCL and Triton
at 8 rows; the gross external traffic is similar. L1 hit rates differ (22.28%
versus 19.59%), and eviction instructions differ. These data do not identify a
cache-policy bug or establish the benefit of changing cache controls. Warm
prepared batches, isolated graph calls and cold counter replay remain separate.

The generic CubeCL `FSumOp` expands through `polyfills::plane_reduce` to an
increasing-distance XOR loop. CUDA emission calls `__shfl_xor_sync(__activemask(),
...)`; PTX/SASS retain a five-iteration shuffle/control loop. Triton's fixed
layout yields static decreasing-distance shuffles. CubeCL's plan records queried
plane width 32, but that fact is not currently used to specialize the generic
polyfill expansion. This is a real lowering opportunity, not a proven explanation
for the entire residual cost. No second candidate is added to test it here.

CUB's warp-reduction specialization demonstrates a small shared publication
followed by a leader final merge; its result is only promised to the output owner.
CubeK's cube reduction distinguishes plane partials, shared publication and
shared-scratch reuse synchronization. Our all-unit epilogue broadcast requires
additional publication work, so “Triton has one barrier” does not justify deleting
the second CubeCL barrier. These mature implementations inform the ownership
and topology analysis; none was benchmarked here as an interchangeable backend.
Pinned source URLs/revisions and observations are in `references.json`.

## Numerical qualification and runtime

The 109 exact original RNG-history/held-out inputs are reused with input hashes.
The direct same-packet comparison adds FP16/BF16 storage neighbors and a
non-default stream: **116 partial buffers and final outputs are bitwise identical to the old
four-chain plan**. Partial-buffer hashes were added in the review follow-up;
their diagnostic host reads are outside every performance measurement. Existing tolerance-or-no-greater-FP64-error qualification still fails
heldout_10017, output 4: candidate 0.11673736572265625, efficient Triton
0.1167755126953125, FP64 0.1167629364572349. The real 178-case generated-graph
capture preserves that one failing output. No aggregate error budget hides it.
Special NaN/Inf/zero checks remain separate in the real graph runner.

All 18 RMS phase comparisons (FP32/FP16/BF16, long/short/long, forward/arbitrary
upstream backward) pass matched retained efficient-Triton references at unchanged
operation-specific tolerances and identical input/upstream bytes. Profiler regions
contain generated CubeCL kernels only. This is bounded RMS evidence, not universal
normalization or contraction equivalence. User storage, current streams,
protected accesses, allocator/lifetime and the compiler route remain inherited.

CubeCL captures retain CPython 3.14.5t with GIL disabled, Triton unavailable,
no loaded Triton Python/native modules and no `PYTHON_GIL` forcing. FP patch
`9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2`
remains identified. No fast math, implicit FMA, LLVM contract flags or unsafe
algebraic identities are enabled.

## Source modifications

- `stage0/src/descriptor3.rs`: private pending-value group in the physical builder;
  unchanged semantic replay, existing collective and launch contract.
- `stage0/src/descriptor5.rs`, `stage5/plan.py`: exactly one validated topology;
  old plan identities preserved. Python/Rust neighboring negatives reject wrong
  workgroup, width, accumulator count/type, outputs and plane width.
- `stage5/legality.py`: new strategy always diagnostic, never automatic eligibility.
- `stage5/runtime.py`, `backend.py`: explicit experimental diagnostic selection.
- Diagnostic probes/analyzer/checks and normalization output/policy options: matched
  controls, preserved captures, actual graph and backward evidence.
- Stage5 ignore rule: retain the new curated packet; raw captures/caches and bulk
  probe representatives stay ignored.

No PyTorch source patch, CubeCL checkout patch, compiler/launcher redesign,
CUDA graph support or split-planning change is introduced.

## Regression and review disposition

Validation is recorded in `gates.json`, `checks.json`, `stage3-first-failure.txt`,
`stage3-code-identity.json` and `review-disposition.md`. The first full gate attempt
stopped on the old **unseeded** Stage3 eager-relative random reduction assertion
(absolute difference 6.1035e-5 versus its 2e-5 bound). It was not relaxed or
relabelled as a candidate failure. The original failed input was not saved by
that older harness, limiting retrospective reproduction. Baseline source/code
identity and an unchanged-suite rerun are recorded separately below.

All **18 regression gates pass** on the unchanged-suite rerun: Stage0 C++/LLVM,
Stages1–4 and FP hardening, Stage5 structural/cache/compiler/math checks,
168-case corpus, precision matrix, frozen-order and accumulation gates, and the
long-row checks. The first failed attempt remains separate. The Stage3 component
kernels were also rebuilt with the exact HEAD Rust sources and current sources:
all three dtypes emit **byte-identical CUDA and PTX** through the existing NVRTC
route. This supports an existing random-input/tolerance issue, rather than a
code change from the new diagnostic strategy; it does not recover the lost
failing input or erase the assertion failure. Historical tracked artifacts were
restored by the established runner.

Required `review-mcp` review returned with no kernel correctness or automatic
eligibility bug. Its two medium harness findings were fixed: normalization
captures now record policy provenance, and diagnostic runs require explicit
directories distinct from baseline defaults. A CubeCL-only policy on a Triton
run is rejected. The low note about final-only bitwise checks was addressed by
also hashing/comparing partial buffers. A neutral-arithmetic assumptions comment
was added; no physical algorithm changed after review. Duplicated experimental
policy allowlists and the intentionally clearer redundant validator remain.

After these small review corrections, build/rustfmt/Clippy, 4 new CLI negatives,
all topology/integrity controls, the 116 partial/final comparisons and 18 matched
RMS phases were rerun successfully. The CPU analyzer verifies the refreshed
policy/capture/array/trace hashes. `review.md`, `review-fix-gates.json` and
`review-disposition.md` preserve the review and resolution. Full 18-gate results
precede these small harness/comment corrections; their affected checks above
were rerun without weakening any assertion.

## Decision

Retain exactly this load-organization experiment as diagnostic evidence. It
isolates an avoidable cost in our old four-chain physical IR without changing
its arithmetic; its numerical failure blocks default acceptance. `cwk` stays
open. Continue focused topology/numerical qualification with the now-understood
load organization and explicit partial/final controls; do not add another
candidate, planner or later-stage feature automatically.

## Reproduction

Use `STAGE0_CARGO_PROFILE=release` and
`CUDA_PATH=$PWD/stage0/cuda13-toolkit/nvidia/cu13`. Triton reference also uses
`TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib` and its recorded libdevice path.

```bash
cargo build --release --locked --manifest-path stage0/Cargo.toml --features llvm,hip-probe
stage0/.venv/bin/python -m stage5.reduction_structure_probe --candidate
stage0/.venv/bin/python -m stage5.reduction_structure_qualification
stage0/.venv/bin/python -m stage5.reduction_structure_checks
stage0/.venv/bin/python -m stage5.reduction_structure_analyze
```

`baseline-commands.json`/`protocol.json` describe rotated full capture and retries;
`ncu-commands.json` retains every prewarm/capture/export command. Warm the exact
Python probe before Nsight; collect one process at a time. The analyzer needs the
ignored raw files to verify original capture hashes. Selected IR/source/PTX/SASS
and numerical/timing results remain inspectable without those profiler binaries.
