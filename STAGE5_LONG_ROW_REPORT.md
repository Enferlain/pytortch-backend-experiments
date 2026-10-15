# Stage 5 follow-up: long-row candidate and guarded shape reuse

> **Supporting experiment record.** Start with [the current Stage 5 findings](STAGE5_REPORT.md). Results and recommendations below retain this pass's historical context.

## Result and decision

**The boundary/shape-history implementation works; the new physical candidate is
not accepted as a default.** The combined scalar256/four-chain/plane/shared plan
is implemented and measured through the existing compiler/launcher. It fails a
matched held-out output element and has material long-row slowdowns. No numerical
failure is hidden by averaging across inputs or outputs.

Compensation remains the experimental implementation used for continuity. This
does **not** accept compensation as the final physical policy; `cwk` remains open.
More accurate FP64 agreement alone does not justify a slower default. A simpler
qualified candidate at comparable cost would not need a statistically significant
speed win for adoption. This candidate fails numerical eligibility before that
tie rule could apply.

The latest user clarification in `STAGE5_ACCUMULATION_REPORT.md` governs this
experiment. References guide behavior and performance; exact Triton matching and
maximal FP64 accuracy are not objectives. PyTorch contracts, storage precision,
safe masks, current-stream ownership and conservative floating arithmetic remain
hard constraints.

## Physical candidate and fixed-split evidence

One catalog addition uses the existing `plane_shared` strategy with
`private_accumulators=4`, `workgroup=[256,1,1]`, `vector_width=1`, one output per
workgroup and queried plane width 32. The old serialized identities are unchanged.
Rust validates this exact combination and rejects neighboring counts, vectors and
workgroup dimensions.

Accumulator j visits `unit + j*256`, advancing by 1024. Every load is guarded before
access. Private values merge as `(a0+a2)+(a1+a3)`, then pass through CubeCL plane
sum and the existing eight-float shared merge. Uniform barriers and result
broadcast for epilogues remain inherited. This is scalar loading, not vector memory
transactions. No handwritten CUDA or new CubeCL compiler/runtime route is added.

The private-chain IR builder already existed in `descriptor3.rs`; only its
catalog/validation combination needed extending. Input storage conversion,
semantic lazy masks, FP32 accumulation and store precision barriers are reused.

`physical-fixed.json` executes all 109 exact original FP32 inputs on the five
physical plans before changing logical split selection. The combined one-stage
plan fails two output elements under reference-or-better, including the million
row. The later fixed nine-part generated graph still fails heldout_10017, element
4: CubeCL 0.11673736572265625, efficient Triton 0.1167755126953125, FP64
0.1167629364572349. Its FP64 error is larger and it exceeds the unchanged
Triton-relative tolerance. Both guarded histories preserve that failure.

Explicit `combined_diagnostic` and `compensated_diagnostic` policies use a fixed
long-row candidate with the same frozen final short reduction. They validate
structural topology and are excluded from automatic selection. `baseline` remains
an explicit historical diagnostic. These options are experimental, not fallback.

## Logical split and shape-history boundary

Actual live path:

```
torch.compile / Dynamo / AOTAutograd
  -> compile_fx forward/backward callback
  -> scoped V.set_choices_handler(CubeCLChoices())
  -> GraphLowering / reduction lowering / Reduction.create
  -> Reduction.num_splits
  -> prepare_reduction_split (including hint-only early returns)
  -> reduction_split_factor_with_extents
  -> inherited reduction_split_factor / create_multilayer
  -> Scheduler / normal SIMD replay / semantic descriptor
  -> unchanged physical compiler and native current-stream launch
```

Two small generic overlay hooks preserve default behavior. The symbolic companion
delegates to existing hint-only overrides. A preparation hook is also necessary:
`num_splits` returns early for initial rows of <=32 elements, before reaching the
split-factor method. Guarding only the companion leaves the original short-first
problem intact. There are no CubeCL-name conditions in the PyTorch changes.

`inductor_choices_class` identifies the versioned policy in config/cache state.
The inner compiler callback additionally binds its handler because choices are
thread-local and lazily cached; changing the config factory alone does not reliably
replace a handler already initialized by feature queries. This binding is carried
into lazy AOT backward compilation and restored afterward, without global method
monkeypatches for policy selection. The diagnostic split observer is restored
after capture and only records inherited method calls.

Execution exposed a second history effect: million-first chooses 32 partials,
whereas 65K-first chooses nine. A short/long guard alone therefore leaves topology
dependent on the initial long shape. The bounded policy uses representative
planning hints through the existing heuristic; runtime expressions, extents,
addresses and metadata stay symbolic and are never replaced by these hints.

On the tested SM86 target, candidate regimes are:

| Original reduction extent | Representative extent | Small-outer split from existing heuristic |
| --- | ---: | ---: |
| <= inherited no-split threshold 8192 | 8192 | 1 |
| 8193..73728 | 73728 | 9 |
| 73729..131072 | 131072 | 16 |
| 131073..1048576 | 1048576 | 32 |
| >1048576 | inherited actual hints | no new topology claim |

The first cooperative bound is the measured nine-part representative times the
existing heuristic's private-work quantum; the next is its enclosing power of
two. These and the million-element bound are **experimental candidate boundaries**,
not architectural constants or universal optima. The initially proposed 65536
boundary was redundant with respect to the tested split regimes and was omitted.
Outer candidate buckets reuse the existing Stage5 <=31 bucket and representative
8/64 outputs, plus the inherited sufficient-output condition at 2*SM count.
The SM-count threshold itself is used only when the inherited major>=10 no-split
heuristic distinguishes it; it is redundant on this tested target.

Inductor still creates and owns partial buffers and the final reduction node.
Derived partial extents use the representative hints from their originating
regime rather than independently specializing on the first partial length.
Large-output, unbacked and non-single-symbol cases retain inherited choices.
This is a bounded scheduling experiment, not a general planner or claim of
universal history-independent scheduling.

Three tested histories start at short, 65K and million-element rows. Sizes inside
a regime reuse their graph; crossings recompile and returning to a prior regime
reuses it. The ordinary short->65K->131K->million sequence uses four variants,
not exact-size specialization. Singleton and empty sizes retain normal Dynamo
specialization. Checks also cross outer-output and immediate extent boundaries.
The exhaustive combined sweep uses nine graph variants per dtype, including
empty/singleton specialization and outer-size crossings. Its test-only cache
limit is 16 (restored afterward), with an explicit variant-count assertion and
fullgraph failures preserved. The reference matrix uses its existing diagnostic
limit of 256. The backend does not globally increase Dynamo's default limit;
applications combining enough regimes can still hit that normal compiler limit.

## Proven mask-index widening

Bounded shape guards cause Inductor's `dtype_from_size` to emit int32 split-tail
comparison operands, including normalization backward. The established descriptor
and address ABI use signed i64. A Stage5 pre-fusion adapter widens only
`index_expr(..., int32)` nodes whose users are comparisons and whose **raw
get_index expression**, before casting, provably fits int32 under LoopBody bounds.
The values and comparisons are therefore unchanged; arithmetic is not silently
widened and no int32 tensor storage support is added.

Unbounded expressions, values exceeding int32 and arithmetic users remain
unsupported. Component controls cover each neighbor. Runtime traces retain the
bound intervals and show the real normalized replay reaching the usual
pre-fusion capability checks. i64 descriptor/Rust validation is preserved.

## Per-case numerical qualification

`reference-or-better-v1` qualifies each finite output element separately. It uses
the existing dtype tolerance against the efficient configured Triton reference
for the corresponding regime, or requires CubeCL absolute FP64 error no greater
than that corresponding reference element. There is no aggregate failure budget.
Reference captures start at 31, 8192, 65539, 131071 and 1048576 respectively.
Per-case references follow the corresponding nonsplit/split regime; the original
65K-first topology is not presumed best for every larger row. The pathological
short-first long graph is never the qualifying or performance oracle.

FP32 uses rtol=atol=2e-5; FP16 uses rtol=.002/atol=.001; BF16 uses
rtol=.016/atol=.008. ATen-relative results and FP64 errors remain visible. NaN,
infinity and required all-negative-zero classifications are checked independently;
dtype, protected loads, precision barriers and synchronization are not negotiable
timing criteria. This does not promise arbitrary reduction zero signs or bitwise
identity across reassociation orders.

The matrix retains all 109 exact FP32 core inputs and six storage cases, then adds
guard/tail/low-precision neighbors for 178 cases per capture. Input byte hashes,
dtype, shape, eager and FP64 vectors must match across captures. Every failing
element and which qualification clause passed each element is retained in
`analysis.json`.

<!-- NUMERICAL_START -->
| Capture | Per-case failures | Failing output elements | Eager-relative diagnostic failures |
| --- | ---: | ---: | ---: |
| baseline-fixed | 1 | 1 | 1 |
| compensated-fixed | 0 | 0 | 1 |
| combined-fixed | 1 | 1 | 2 |
| combined-guarded-short | 1 | 1 | 2 |
| combined-guarded-large | 1 | 1 | 2 |
| production-guarded-short | 0 | 0 | 1 |
| production-guarded-large | 0 | 0 | 1 |
| triton-short | 24 | 29 | 21 |
| triton-large | 0 | 0 | 2 |
| triton-mid | 4 | 4 | 3 |
| triton-medium | 0 | 0 | 1 |
| triton-million | 0 | 0 | 2 |
<!-- NUMERICAL_END -->

Structural legality and offline numerical eligibility are separate functions.
The rejected combined candidate remains diagnostic-only. Runtime first-use
validation retains strict pointwise comparisons; for the already executed
compensated reduction scope, finite differences from the old uncompensated tree
are recorded rather than making that tree the sole oracle. Dtype/shape and special
classification checks remain hard. The numerical-policy source/version enters
tuning identities. No runtime tensor values determine the qualification policy.
Qualified finite-difference diagnostics additionally reject gross finite
corruption using a versioned coarse guard; this is not an accuracy oracle or a
universal reassociation bound. The current sole-compensated-candidate path does
not invoke mixed-candidate tuning. Any future overlapping qualification domain
still needs an input-aware validation policy. Controls preserve the executed
accurate-1 versus old-tree-0 example and reject a grossly wrong finite scale.

## Device measurements and selection

Matched RMS captures additionally test FP32/FP16/BF16 forward and AOT backward
with identical input/upstream hashes, through long->short->long reuse. All 18
phase cases pass every-element configured-Triton comparisons at the existing
dtype tolerances. `normalization-comparison.json` records errors and both kernel
censuses. Long CubeCL forward and backward each execute three generated kernels;
the profiler regions contain no pointwise/reduction ATen or Triton kernel.
Raw output arrays stay in the ignored cache, bound to captures by hashes. This
executes the relevant square/mean/rsqrt/multiply/cast composition; matched controls
alone are not used to assert universal operation equivalence.

Three rounds of seven independent CUPTI kernel sums use the established 50M-cycle
activity warmup outside each region. Compilation, first-use tuning, host enqueue
gaps and annotations are excluded. Every sample/MAD and kernel census is retained.
Fixed logical-split comparisons separate the physical candidate from the new
guard policy. Prepared component timings are separate and are not substituted for
the generated graph's kernel sums. Measurements are profiler-qualified;
small-percent/end-to-end performance claims are not made.

<!-- PERFORMANCE_START -->
| Input | Baseline fixed µs | Compensated fixed µs | Combined fixed µs | Guarded production short-first µs | Guarded production long-first µs | Efficient Triton µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float32-8-65539 | 7.482 | 8.463 | 18.191 | 8.026 | 8.084 | 6.276 |
| float32-64-65539 | 27.877 | 28.704 | 42.559 | 27.673 | 27.809 | 24.960 |
| float16-8-65539 | 6.951 | 7.773 | 17.816 | 7.458 | 7.499 | 5.428 |
| float16-64-65539 | 18.567 | 19.182 | 32.847 | 17.771 | 17.852 | 15.533 |
| bfloat16-8-65539 | 6.994 | 7.916 | 17.585 | 7.386 | 7.345 | 5.500 |
| bfloat16-64-65539 | 18.791 | 19.407 | 32.792 | 17.623 | 17.527 | 15.540 |
| float32-64-31 | 6.353 | 6.264 | 6.254 | 2.110 | 2.175 | 1.757 |
| float32-64-257 | 6.426 | 6.428 | 6.332 | 2.493 | 1.889 | 2.261 |
| float32-8-131071 | 10.900 | 12.932 | 25.534 | 10.394 | 10.545 | 8.244 |
| float32-64-131071 | 47.149 | 49.175 | 63.080 | 46.619 | 46.955 | 44.801 |
| float32-8-1048576 | 48.163 | 65.569 | 159.536 | 46.303 | 46.590 | 49.623 |
| float16-8-131071 | 9.344 | 10.772 | 24.427 | 8.919 | 9.019 | 7.363 |
| float16-8-1048576 | 42.462 | 54.175 | 155.648 | 28.689 | 28.979 | 33.130 |
| bfloat16-8-131071 | 9.451 | 11.146 | 24.187 | 8.832 | 8.924 | 7.550 |
| bfloat16-8-1048576 | 43.053 | 54.436 | 155.395 | 28.825 | 28.981 | 33.101 |
<!-- PERFORMANCE_END -->

The analyzer records the existing two-of-three-round/disjoint median +/-3 MAD
criterion for material wins/slowdowns. Simplicity at comparable cost is a distinct
possible adoption reason, not an acceleration claim. Numerical ineligibility
rejects the combined candidate regardless of timing. The continuation policy is
not relabelled as accepted because a replacement failed.

## Nsight and generated-code diagnosis

WSL-native Nsight Compute 2026.3.1.0 build 38829034 collected the actual combined
nine-part graph through CUDA 13.0 NVRTC/driver JIT. `tools.json` verifies the binary
and matching section hashes. `ncu-metrics.json` retains the exact command, raw CSV
path/hash and unnormalized metric rows. No CUDA 13.1 offline diagnostics substitute
for production measurements.

| Metric | Combined partial | Frozen final |
| --- | ---: | ---: |
| Grid | 72 | 8 |
| Block size | 256 | 256 |
| Registers per thread | 35 | 16 |
| Dynamic shared bytes | 32 | 1024 |
| Waves per SM | .15 | .02 |
| Nsight duration ns | 18560 | 3360 |
| Compute throughput % | 3.72 | .90 |
| Memory throughput % | 15.67 | 1.85 |

Driver attributes report zero local bytes for the combined partial; this is not
evidence of spills or resource-limited theoretical occupancy. The CUDA contains
four separately guarded scalar/lazy-load regions with repeated signed index
arithmetic, followed by the balanced merge and plane/shared reduction. This
supports investigating instruction/control-flow lowering rather than assuming
four independent chains necessarily improve execution. Source and counters do
not isolate a single causal instruction; no unproven optimization is applied.
Nsight replay/cache/clock conditions perturb durations and remain separate from
the primary timing table.

## Source changes and interfaces

| Owner | Change / reason | Suitability |
| --- | --- | --- |
| Stage1 overlay, PyTorch choices.py / ir.py | Symbolic split companion and preparation before early returns; defaults preserve existing overrides | Generic extension points; private API prototype needs upstream review |
| Stage4 compile backend | Optional choices factory/config and scoped inner-compiler binding, including lazy backward | Experimental adapter; no shared PyTorch wheel edits |
| Stage5 choices | Versioned guarded representative hints using existing split machinery | Bounded out-of-tree experiment |
| Stage5 pre-fusion adapter | Proven comparison-only int32->i64 widening after real replay exposed narrower mask indices | Experimental backend adaptation; no dtype/backend-name patch |
| Stage0 descriptor5 validator / Stage5 catalog | Permit exact four-chain plane/shared scalar256 combination | Experimental physical adapter; existing CubeCL builder/compiler unchanged |
| Stage5 legality/runtime | Separate structural/numerical decisions, explicit diagnostic policies, versioned tuning identity and qualified finite-difference diagnostics | Experimental selection boundary |
| Stage2 capability error | Include the rejected dtype in diagnostics | Diagnostic only |
| Diagnostic runners/analyzer/checks | Fixed-split comparison, matched histories, per-element rule, integrity controls and generated artifacts | Evidence/testing |

No CubeCL checkout source or compiler/launcher design is changed. The conservative
FP patch remains `9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2`.
Implicit contraction, LLVM contract flags and unsafe algebraic identities remain
disabled. The Stage0 manual C++ and LLVM routes remain independent gates.

## Regression evidence and limitations

The final gate runner writes current results under this packet and restores
historical tracked artifacts afterward. The original accumulation/frozen-order suites explicitly
requests `split_policy="inherited"` to preserve its one-stage ordering/kernel-count
and pointer assertions unchanged; the guarded default has separate full execution,
kernel census and two-launch pointer/stream/lifetime tests. This is not a weakened
numerical test or acceptance of a fallback.

Final gate/review disposition is recorded in `gates.json` and
`review-disposition.md`. CPython 3.14.5t stays GIL-disabled on CubeCL, Triton remains
unavailable/import-blocked and no native Triton module loads. No PYTHON_GIL forcing
is used. The Triton reference enables the GIL; its device measurements are useful
but host timing is not parity evidence. Naturally selected allocation/view/extern
bookkeeping is not claimed as CubeCL-generated computation.

All **18 gates pass**: the 17 historical gates pass in the final full run, and
the final guarded integration gate passes a fresh retry after its test-only
cache-budget correction. `gates.json` explicitly records that retry and retains
the failed attempt separately. The integration packet has 25 focused checks;
five additional harness controls reject mismatched/incomplete evidence. The
168-case Stage5 corpus, 78/9 precision matrix and 75 accumulation checks retain
their original assertions. Rustfmt, release Clippy with all targets and warnings
as errors, Ruff and diff checks pass. Required review completed with no blocker;
all actionable notes and affected checks are recorded in its disposition.

This does not qualify arbitrary reductions, targets, dynamic layouts or topology
ranges. Candidate guard bounds and representative hints require more measurement
before broad deployment. Above the bounded regime, inherited scheduling remains
and no general history-independence claim is made. The combined physical candidate
is implemented but rejected; compensation's final suitability remains unresolved.

**Next direction:** further focused Stage5 physical/codegen diagnosis and candidate
qualification. Do not broaden operations, introduce another planner or proceed to
the next stage automatically. Nothing is committed or pushed by this task.

## Reproduction

```
export STAGE0_CARGO_PROFILE=release
export CUDA_PATH="$PWD/stage0/cuda13-toolkit/nvidia/cu13"
cargo build --release --locked --manifest-path stage0/Cargo.toml --features llvm,hip-probe
stage0/.venv/bin/python -m stage5.reference_reduction_physical --combined --output stage5/artifacts/long-row/physical-fixed.json
stage0/.venv/bin/python -m stage5.long_row_matrix
stage0/.venv/bin/python -m stage5.long_row_checks
stage0/.venv/bin/python -m stage5.long_row_gates
stage0/.venv/bin/python -m stage5.long_row_analyze --update-report --curate
stage0/.venv/bin/python -m stage5.long_row_analyze
```

The exact Nsight command is retained in `ncu-metrics.json`. Source hashes and
artifact integrity manifests distinguish the captured implementation from caches;
failed exploratory attempts are recorded separately from complete qualification.
