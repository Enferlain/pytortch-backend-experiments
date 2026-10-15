# Stage 5: production long-row cost isolation

## Result

The remaining loss is primarily in the partial reduction. This investigation
establishes a small redundant synchronization in our Rust bridge, fixes it, and
preserves bitwise output on the executed component corpus. It does **not**
establish a CubeCL compiler correctness bug or solve the remaining performance
gap. Compensation remains provisional; `cwk` stays open. No algorithm, catalog,
split policy, numerical qualification rule or default eligibility is changed.

**Independent review accepted:** the initial `review-mcp` attempt returned GLM
API 429 before a verdict. The user then explicitly authorized native review.
The native adversarial reviewer verified the synchronization reasoning, harness
and artifact/report consistency and returned no actionable findings. Review
scope and execution limits are recorded in `review-disposition.md` and
`native-review.md`; the remaining performance/default question stays open.

Evidence is under `stage5/artifacts/long-row-diagnosis/`; raw profiler reports,
traces, compiler binaries and logs stay in the ignored corresponding cache.
Historical Stage 0–5 captures remain separate.

## Scope and method

Lead: GPU codegen profiling, supported by source/IR provenance. Source discovery
used the project graph at Tier 2 with relevant coverage checks and direct source
reads. Baseline revision is `49a903b`. Production compilation remains CUDA 13.0
NVRTC then the CUDA driver JIT, PyTorch `2.14.1+cu130`, CPython 3.14.5t, conservative
FP patch `9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2`.

RTX 3090 SM86, 82 SMs, UUID `4cec666b-991c-fe31-e63f-6455bc828119`, driver 610.62.
Nsight Compute is the Linux 2026.3.1.0 build 38829034. The saved SASS is from
the actual profiled driver-loaded kernels. No offline CUDA 13.1 assembly is used.
The CLI source export is the documented `--page source --print-source sass`
interface ([NVIDIA CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)).

There are three deliberately separate measurement scopes:

1. Original matched single-call CUPTI traces: 3 rounds × 7 captures. The analyzer
   verifies their existing SHA256s and extracts partial/final durations. Graph
   kernel sums exclude host enqueue gaps. Individual medians need not sum to the
   median graph sum. These remain profiler-qualified timing, not end-to-end latency.
2. New Nsight replay counters: exact generated kernels, kernel replay, default
   cache flushing and boost clock policy, two launches inside profiler markers.
   Their durations are not substituted for CUPTI samples or used for stable-win
   claims. Stall ratios are per issue-active, not percentages of wall time.
3. Before/after physical components: actual split semantic descriptors and plans,
   prepared native batches, rotated 3 rounds with the established benchmarker.
   These timings omit graph dispatch and have a different cache/queue regime.

Triton's reference environment enables the GIL. Only device work is compared.
Division/FTZ/libdevice/emulated-cast configuration remains the captured reference
policy; this plain sum does not prove equivalence for unrelated operations.

## Where the time was going

Verified original CUPTI medians, microseconds:

| Path | Shape | Partial | Final | Complete graph kernel sum |
| --- | --- | ---: | ---: | ---: |
| CubeCL production | 8×65539 | 6.455 | 1.634 | 8.084 |
| Efficient configured Triton | 8×65539 | 4.963 | 1.313 | 6.276 |
| CubeCL production | 64×65539 | 25.867 | 1.920 | 27.809 |
| Configured Triton, 8-row-first | 64×65539 | 23.649 | 1.312 | 24.960 |

The 8-row case has the same nine partials per row. The 64-row case does **not**:
CubeCL's existing guarded outer regime chooses four, while Triton's reused
8-row-first graph retains nine. That was an attribution confound in treating the
64-row timing as a pure physical-kernel comparison. Neither the guard nor the
default is changed here. A new CubeCL `inherited` control retains nine partials
on the exact same 64-row input, without forcing a scheduler split count.

Final-kernel first-use selection also differs between captures: the historical
64-row timing selected independent planes, while the fresh guarded counter
capture used the frozen shared-tree final kernel. Counter conclusions below
therefore concern the mapped partials; their final durations are not used to
explain the older final-kernel timing. The fixed-nine counter control selected
independent planes. Each capture retains its actual selected-plan metadata.

## Counter and executed-instruction evidence

Actual collection command for the baseline 8-row production graph:

```bash
ncu --target-processes all --profile-from-start off --launch-count 2 \
  --section-folder /home/imi/tools/nvidia/nsight-compute-2026.3.1/sections \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats --section InstructionStats \
  --force-overwrite --export stage5/cache/long-row-diagnosis/production-8 \
  --csv --log-file stage5/cache/long-row-diagnosis/production-8.csv \
  stage0/.venv/bin/python -m stage5.reference_reduction_probe \
  --impl cubecl --history large_first --cubecl-policy first_use \
  --cubecl-split-policy guarded --ncu-case float32-8 \
  --output stage5/artifacts/long-row-diagnosis/production-8.json
```

The command JSONs preserve every corresponding invocation. `collection.json`
records compiler profile and warmup differences. Raw metric names/units/values,
CSV hashes, captured environment, and per-opcode dynamic counts are retained in
`diagnosis.json`. The analyzer reconciles SASS execution counts to Nsight's
`smsp__inst_executed.sum`, rejects unmapped kernels, checks byte-identical inputs
and validates GIL/Triton/FP-patch identities.

Selected raw values for the matched nine-part partials, before this fix:

| Metric | CubeCL 8 rows | Triton 8 rows | CubeCL 64 rows, fixed 9 | Triton 64 rows |
| --- | ---: | ---: | ---: | ---: |
| launch__grid_size | 72 | 72 | 576 | 576 |
| launch__block_size | 256 | 512 | 256 | 512 |
| smsp__inst_executed.sum | 366264 | 371160 | 2930112 | 2969280 |
| Dynamic warp FADD count | 121896 | 32256 | 975168 | 258048 |
| Dynamic warp BAR count | 5760 | 1152 | 46080 | 9216 |
| gpu__time_duration.sum, µs, replay only | 8.384 | 6.016 | 29.056 | 24.768 |

For the 8-row partial, `smsp__warps_eligible.avg.per_cycle_active` is
0.154186 versus 0.303219. CubeCL has 30 registers/thread reported by
the launch metric, Triton 29; the spilling metric is zero for both. The CubeCL
barrier stall ratio is 1.711771 and long-scoreboard ratio 8.423753; the corresponding
Triton ratios are 1.236618 and 11.526277. These ratios do not assign additive
causal percentages. The small grid and different cooperating-thread counts limit
what can be inferred from occupancy/eligible-warp comparisons.

**Established:** compensation and its shared merge execute much more FP addition;
the full shared tree executes ten workgroup barriers versus Triton's one. Total
warp-instruction counts are nevertheless similar, so total instruction count
alone does not explain runtime. Zero measured spills argues against a spill
explanation for these captures. This is a difference in physical strategies, not
evidence of incorrect compiler lowering.

**Not established:** which precise dependency/cache/issue sequence accounts for
the remaining microseconds. Eligible-warps and scoreboard observations suggest
latency-hiding/load-scheduling investigation, but this is not a causal diagnosis
of one instruction or a mandate to increase the block size.

## Boundary attribution

| Construct | First established owner | Evidence / conclusion |
| --- | --- | --- |
| Partial number and split-tail semantic mask | Inductor lowering and existing choices regimes | Four versus nine at 64 rows; separated with inherited-policy control |
| Signed i64 floor/mod addressing | Inductor expressions → Stage 2 descriptor/emitter | Required general semantics preserved; no narrowing or floor/truncation substitution |
| Repeated quotient/remainder text in generated CUDA | Generic symbolic index emission | NVRTC/PTX/SASS hoist the observed invariant work out of the hot loop; CUDA text alone was an insufficient bottleneck argument |
| TwoSum correction and compensated shared merges | `descriptor3::compensated_add` / `build_physical` | Explicit additions/subtractions and finite correction handling survive to SASS as intended |
| Full shared tree | Our selected physical plan | Existing scratch publication + each tree level; ten barriers included an unnecessary final reuse barrier |
| Extra final scratch-reuse synchronization | `descriptor3::build_physical` | Our emitter explicitly requested it; CubeCL correctly retained synchronization |
| Remaining partial-kernel loss | Multiple physical/codegen differences | No isolated CubeCL compiler bug established; no claim that all remaining cost is compensation |

This is not a blame assignment by profiler label. The semantic descriptor, IR,
CUDA, PTX and actual loaded SASS are adjacent evidence. No incorrect arithmetic,
address, conversion or compiler transformation was isolated in this work.

## One bounded fix and independent component proof

`stage0/src/descriptor3.rs::build_physical` now retains the post-read barrier only
when a subsequent reduction will overwrite the shared scratch. Publication and
all tree-level barriers stay. Between reductions all units still finish reading
before reuse. After the last reduction, epilogues consume reduced SSA values and
write global tensors; there is no later scratch writer. Kernel completion still
waits for the workgroup's execution. No relaxed floating-point rule is introduced.

This is an experimental bridge correction, not a CubeCL/PyTorch source patch.
The compiler/launcher ABI, logical replay, physical plan identities, launch
topology and reduction arithmetic order are unchanged. Compiler content hashes
already invalidate affected artifacts/tuning entries.

`long_row_scratch_probe.py` invokes the saved prechange compiler and current
compiler on identical version-5 packets, then launches both through the existing
native stream launcher. It executes 173 bitwise comparisons, including all 109
original input hashes, FP32/FP16/BF16 split partials, NaN/Inf/signed zero, tail and
empty components, the actual four-part control, two sums sharing scratch,
independent-plane controls, and non-default-stream/allocator churn.

Captured positive-extent split descriptors remain positive-extent contracts.
The empty direct component uses an explicit empty view matching its declared
stride; actual Dynamo empty/singleton specializations remain regression cases.
Runtime scalar/layout validation was not relaxed.

PTX barrier counts: shared/compensated single reduction 10→9, two reductions
20→19, plane/shared single 3→2, two reductions 6→5, independent planes 0→0.
Thus the between-reduction reuse barrier remains. The new 8-row SASS has the
same 121896 dynamic FADD count and 5184 BAR instructions instead of 5760.

Rotated prepared-component round medians, µs:

| Component | Before, rounds 1/2/3 | After, rounds 1/2/3 |
| --- | --- | --- |
| 8 rows, 9 parts | 4.163 / 4.169 / 4.153 | 4.131 / 4.118 / 4.133 |
| 64 rows, 9 parts | 23.530 / 23.523 / 23.476 | 23.478 / 23.491 / 23.025 |
| 64 rows, 4 parts | 22.277 / 22.325 / 22.340 | 22.313 / 22.253 / 22.305 |

These establish small/noisy cost, not a material acceleration or a new default
selection. The observed redundant barrier is not the explanation for the whole
gap. Nine parts are not universally better than four on CubeCL: the component
control does not justify replacing the existing regime's four-part decision.

## Validation and collection limitations

Eight CPU evidence-integrity controls reject incomplete capture, wrong GIL/native
Triton state, unmapped kernels, missing SASS identity and inconsistent dynamic
counts. Report/data provenance, selected generated code and regression results
are retained alongside the analysis.

The fresh post-fix full graph executes all 178 cases and passes every per-output
qualification against the appropriate efficient configured-Triton capture, with
unchanged tolerances/FP64 alternative. `post-fix-qualification.json` binds the
capture, reference captures, policy and selector by source/data SHA256. The
physical component's bitwise proof is separate from this graph qualification.
The fresh graph's CUPTI kernel-sum medians are 8.415 µs at 8×65539 and
28.321 µs at 64×65539. These fresh-versus-historical graph captures were not
rotated as a before/after pair, and final-plan selection differs at 64 rows;
they do not establish a graph speedup or slowdown caused by the barrier change.
New compiler identity reruns first-use selection, so graph results across
different selected plans need not be bitwise identical. The bitwise claim above
is specifically for identical semantic/physical packets.

Cold compiler/autotuner work under Nsight caused two collection problems:
the Triton native extension encountered a Nix linker-wrapper unbound-variable
failure, and a fresh CubeCL first-use tuner could not obtain its required CUPTI
events under Nsight injection. Both were resolved by completing the exact
unchanged warmup outside Nsight with the same cache identity, then collecting the
marked warm launches. No profiler/native import failure was suppressed, no empty
capture accepted, and no assertion weakened. Failed attempts remain separate.

The final-source regression/qualification disposition is recorded in `gates.json`,
`verification-runs.json`, and `review-disposition.md`. Those machine-readable
results distinguish full-suite results from the component proof. Nsight counter
collection is one replayed invocation per component; it is not a three-round
performance-selection test. No Compute Sanitizer result is claimed.

All **18 regression gates pass** on the final physical builder: Stage 0 C++ and
LLVM, Stages 1/1.5/2/3/4 and FP hardening, Stage 5 components/negative boundaries,
168-case corpus, 78 forward/9 backward precision matrix, frozen-order tests,
75 accumulation checks and the guarded long-row suite. Historical tracked
artifacts were restored after capturing the refreshed results under this packet.
Ruff, formatting, release Clippy/all targets with warnings as errors, and diff
checks pass. CubeCL paths retain Triton independence, no native Triton mapping,
and GIL-disabled CPython without `PYTHON_GIL` forcing. The user-authorized native
review accepted the bounded change with no actionable findings. The reviewer
checked source and retained evidence rather than rerunning GPU suites or Compute
Sanitizer. The direct old/current component proof covers four strategies;
coalesced columns additionally have current-source regression coverage and
source-level synchronization reasoning, not a direct old/current pair.

## Recommendation

Continue focused Stage 5 work. The next physical/codegen question is how to obtain
independent memory/accumulation work and a cheaper qualified collective without
the rejected four-chain candidate's slowdown. Any new candidate must qualify
per case/element against the efficient regime reference and retain protected
loads, conversion barriers and conservative FP. Do not restart a planner search
or assume a CubeCL compiler bug from generated CUDA verbosity.

`cwk` remains open. No compensation/default acceptance, next-stage expansion,
CUDA graphs, commit or push is part of this investigation.

## Reproduction

The component comparison needs the saved prechange `cubecl-stage0` executable
under `stage5/cache/long-row-diagnosis/before/`. It was copied before rebuilding,
with its SHA256 retained. If that ignored cache is discarded, rebuild revision
`49a903b` in a separate checkout before rerunning the comparison; the curated
before/after source/PTX evidence remains available without that executable.

```bash
export STAGE0_CARGO_PROFILE=release
export CUDA_PATH="$PWD/stage0/cuda13-toolkit/nvidia/cu13"
cargo build --release --locked --manifest-path stage0/Cargo.toml --features llvm,hip-probe
stage0/.venv/bin/python -m stage5.long_row_scratch_probe --measure
stage0/.venv/bin/python -m stage5.long_row_diagnose
stage0/.venv/bin/python -m stage5.long_row_diagnosis_checks
stage0/.venv/bin/python -m stage5.long_row_diagnosis_gates
```

For counter recollection, use the saved command JSONs with the compiler profiles
in `collection.json`; warm the identical command's Python portion outside Nsight
first. Export raw metrics with `ncu --import REPORT --csv --page raw` and SASS
with `ncu --import REPORT --page source --print-source sass` to the paths recorded
in the packet. The raw cache is required to rerun the analyzer against the
original byte hashes. `verification-runs.json` preserves the full post-fix
178-case graph command. Its numerical comparison is the existing
`reduction_policy.qualify_case(actual_row, reference_row)` for each ordered row,
selecting the existing efficient reference via
`long_row_analyze.reference_for_extent(actual_row["shape"][-1])`.
