# Stage 5 follow-up: long-row accumulation

## Current requirement clarification (2026-10-14)

The user clarified that CubeCL is intended as an alternative to Triton for
PyTorch Inductor. Existing mature implementations such as Triton, ATen, CUB and
CubeK should be treated as important references for behavior, algorithms and
performance, but not as unquestionable targets or ceilings.

For operations already handled by the normal Inductor/Triton path, CubeCL should
aim for comparable or better practical behavior and performance while preserving
PyTorch's operation contracts. Differences from Triton should be investigated
and explained, but matching Triton exactly is not itself the objective,
it is just the closest benchmark target.

Additional numerical accuracy is desirable when it comes at little or no cost,
but greater agreement with FP64 alone is not sufficient justification for a
materially slower default physical plan. Likewise, CubeCL should not deliberately
reproduce another implementation's numerical weakness when equal-or-better
performance can provide better accuracy.

FP64 and other higher-precision references remain useful diagnostics for
understanding numerical error; they are not the default optimization target.

This supersedes the earlier accuracy-first rationale without changing the
executed results or artifacts. The compensated long-row strategy remains useful
experimental evidence, but its suitability as the default must be judged against
matched alternatives on both numerical behavior and device performance. Its
current measured overhead is relative to the old CubeCL plan, not an equivalent
Triton kernel, so a matched comparison is still required.

Host-time comparisons remain qualified by the reference environment's GIL state.
Existing operation-specific precision barriers and ordinary contraction
restrictions remain in force; explicit FMA is a separate semantic operation, not
permission for implicit contraction.

The subsequent [matched reduction-reference investigation](STAGE5_REDUCTION_REFERENCE_REPORT.md)
now records identical-input Triton comparisons, compilation-history effects,
device costs and Nsight evidence. It leaves the default decision open; the
accuracy-first experiment below remains a historical executed result.

## Result

**The original long-row accuracy defect is corrected in the ordinary Stage 5
`first_use` path. Numerical qualification remains open:** one new finite-input
case exceeds the unchanged eager-relative tolerance because ATen itself is less
accurate than the compensated result. This report does not waive that failure,
claim universal eager agreement, or advance to Stage 6.

The selected physical plan uses FP32 compensated accumulation and a compensated
cooperative merge. Short reductions retain existing physical candidates; the
explicit `tuning="baseline"` mode retains the original order for diagnostics.
Stage 0–4 backends remain independent baselines. No PyTorch or CubeCL checkout
patch, new semantic operation, storage copy or relaxed-math permission is added.

## Reuse before invention and selection

The live installed ATen `Reduce.cuh:thread_reduce_impl` maintains independent
private accumulators, with `vt0=4` by default, then combines them. Its comment
identifies dependency-chain removal as the motive. Our captured Inductor→Triton
kernel similarly accumulates a logical tile then uses `tl.sum`; its physical
layout is not a promise of four accumulators per CUDA thread. We adapted the
independent-chain principle, keeping scalar accesses and explicit ownership.
Our private merge is balanced rather than ATen's final serial combine.

Executed comparison first used 77 inputs: the original sequential RNG history,
four independent seeds across 11 lengths, and 32 additional seeds. One ordinary
chain failed four cases, two chains failed five, and four chains failed one.
Four chains fixed the original case but failed seed 10017, whose sum is close to
zero. These variants are retained as diagnostic physical candidates and are
deliberately excluded from automatic selection.

The selected alternative follows Knuth's `TwoSum` and the compensated summation
structure described by [Ogita, Rump and Oishi](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf).
It uses the general addition residual rather than Neumaier's magnitude-dependent
residual; both recover lost low-order information, but this form avoids a
magnitude-dependent branch. We compensate private accumulation **and** the shared
merge. Compensation only before an ordinary shared tree would leave another
uncompensated cancellation boundary.

The final comparison adds 32 previously untested validation seeds, giving
109 eight-row cases. Eager-relative failures are 6/109 for one chain, 6/109 for
two, 2/109 for four, and 1/109 for compensation. All 109 compensated cases pass
the same numerical tolerance against FP64. These are two separate observations;
the latter does not turn the former into an eager-gate pass.

## Physical arithmetic and boundary

The semantic sum DAG, logical `outer`/`reduction` coordinates, runtime scalar ABI,
casts, masks and fused stores are unchanged. `PhysicalCandidate` independently
names the new `shared_tree_compensated` strategy:

```text
256 units, scalar loads, one workgroup per outer coordinate
each unit consumes r = unit + 256*k
private state: FP32 high sum and FP32 correction
shared state: 256 high sums + 256 corrections = 2048 bytes
tree strides: 128,64,32,16,8,4,2,1
uniform barrier after publication and every merge, plus before scratch reuse
outer output owner: unit zero; full epilogue: the existing parallel loop
phases: one per generated kernel; natural Inductor split sequencing unchanged
```

For `a + b`, the builder explicitly emits:

```text
t = a + b
z = t - a
error = (a - (t - z)) + (b - z)
correction = correction + error
```

At a tree merge, the left correction receives the high-sum residual and the right
correction. The final result is `high + correction`, rounded to the semantic/store
dtype by the existing replayed conversions. FP16/BF16 inputs still widen to FP32;
compensation is not FP64 accumulation and does not remove precision barriers.

If the ordinary high-sum addition becomes nonfinite, its correction is reset to
zero. The high sum retains ordinary addition's NaN/Inf behavior, avoiding the
unguarded `inf-inf` residual problem. This does not recover sums whose FP32
intermediates overflow, or establish every possible overflow-order equivalence.
NaN payload identity and universal signed-zero equivalence remain unclaimed.

The builder uses CubeCL operations throughout. All reduction-axis accesses remain
guarded; lazy semantic masks reuse the existing emitter. No barrier moves into a
lane-dependent branch or variable-trip-count private loop. Representative PTX
retains separate addition/subtraction and unconditional workgroup barriers.

## Policy, cache and ownership

`stage5.legality.reason` requires compensation when the logical reduction extent
exceeds 4096. This reuses the existing extent bucket: scalar256 chains below it
have at most 16 private additions. **This cutoff is a bounded experimental policy,
not a proved universal error bound.** Short-row numerical behavior is unchanged.
It is independent of input values, timings and Python backend names.

The legality signature records which side of the cutoff applies. Source hashes
invalidate old timing selections. With exactly one legal candidate, runtime now
uses that candidate rather than assuming it must be the frozen baseline. Neither
deterministic mode nor an old faster reduction can undo the long-row policy.
Entering a new short-row alignment class may compile new candidates; compatible
long rows reuse one graph and artifact. Tests warm both proof classes before
asserting reuse across 257,4096,4097,65539,131071,1048576.

The existing native Stage 5 launcher needed a bounded shared-memory allowance:
2048 bytes is accepted only for `(256,1,1)`; existing smaller configurations remain
valid. This changes no pointer/scalar ABI, allocator ownership, stream selection,
readiness wait, `record_stream`, synchronization or repeated-launch contract.

## Accuracy evidence and remaining reference gap

Original seed 6142, sequential shapes `64×31`, `64×257`, `8×257`, `8×65539`, row 7:

| Result | Value | Absolute error against FP64 |
| --- | --- | --- |
| Original CubeCL | 2.3123092651367188 | 7.8663e-5 |
| ATen FP32 | 2.3122177124023438 | 1.2890e-5 |
| Compensated CubeCL | 2.3122305870056152 | 1.5046e-8 |
| FP64 reference | 2.312230602051727 | Reference |

The ordinary reused Inductor graph now passes `rtol=atol=2e-5` and executes one
generated CubeCL kernel. An explicit FP32 CPU model matches GPU output exactly.

New seed 30003, `8×65539`, row 1:

| Result | Value | Absolute error against FP64 |
| --- | --- | --- |
| ATen FP32 | -1.08135986328125 | 7.3693e-5 |
| Compensated CubeCL | -1.0812861919403076 | 2.1530e-8 |
| FP64 reference | -1.0812861704098395 | Reference |

CPU `math.fsum` agrees with the recorded FP64 value for this row.
The eager-relative allowance is 4.1627e-5, so a result close to FP64 still fails
that comparison. [PyTorch's numerical guidance](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
allows order-dependent results; it does not automatically authorize changing this
experiment's gate. The strict comparison command still fails. An explicitly
named `--qualify-reference` diagnostic retains `selected_candidate_eager_gate_passes=false`
while checking FP64 accuracy separately. Normal runtime does not retry or detect
this case. The real compiled counterpart is retained as an explicit reference
counterexample, not silently included among eager-gate successes.

## Tests and regression status

[accumulation_checks.py](stage5/accumulation_checks.py) records 75 checks: three
storage dtypes; component extents 0,1,31,255,256,257,1023,1024,1025,4095,4096,4097,
65539,131071; output canaries; NaN/±Inf/±0; exact lost-unit/mixed-magnitude/subnormal
components; original history and held-out cancellation; dynamic policy crossings;
column and transposed input; one-kernel mean epilogue and lazy cat masking; RMSNorm
forward/backward with arbitrary upstream gradients for all three dtypes;
non-default-stream allocator pressure with original-pointer/current-stream/native
counter assertions; and malformed plan rejection before IR.

One isolated cache partition forces an actual Rust/C++/NVRTC invocation and native
launch. Its PTX matches the warm artifact byte-for-byte, and GPU output matches the
CPU model. Compiler options remain conservative; no fake compiler counter or
production-cache deletion is used. Profiler regions for row, column, transpose,
mean, lazy masks and each normalization backward contain one CubeCL kernel,
matching the actual native census. Training RMSNorm forward has two CubeCL
kernels per dtype, preserving the original saved-statistic boundary; each backward
has one. Native negatives reject excess/mismatched scratch before enqueue. A real
strict diagnostic exits 1 while preserving the completed qualification bytes;
mocked failure and non-integer policy controls pass independently.

All **17 regression gates pass**, including Stage 0 C++/LLVM, Stage 1/1.5,
complete Stage 2, Stage 3, Stage 4, FP hardening, Stage 5 controls/cache/compiler,
the **168-case Stage 5 corpus**, the 78-forward/nine-backward precision matrix,
frozen-order qualification, and the newly wired accuracy/harness suites. [Gate records](stage5/artifacts/accumulation/gates.json)
and this packet's `regressions` directory retain current results; historical
captures are restored byte-for-byte. The corpus performs 628 actual compiler
invocations and 1,946,887 native launches including tuning/warmup/profile trials;
these totals are not per-user-call kernel counts. The final focused run records
24 compiler invocations and 189,901 native launches, including cold candidates
and tuning. Its isolated fresh-compiler check requires one compiler invocation
and one native launch with PTX equality. Required review
completed after 23 model calls with no correctness blocker. Its gate wiring,
diagnostic warning, topology error, bound-ownership and retention findings were
addressed; the complete final 17 gates and focused checks were rerun. Rustfmt,
Clippy with all targets/warnings-as-errors, Ruff and diff checks pass.
[Review disposition](stage5/artifacts/accumulation/review-disposition.md) records the
small follow-ups; [validation hashes](stage5/artifacts/accumulation/validation.json)
bind the final sources and results.

## Cost and artifacts

The qualified same-process CUDA/CUPTI probe measures kernel events, excludes GPU
annotations and host enqueue gaps, and uses the same input/stream per comparison.
This is a bounded fixed-order diagnostic, not a broad performance claim:

| FP32 shape | Original kernel µs | Compensated kernel µs |
| --- | --- | --- |
| 64×31 | 2.015 | 2.206 |
| 64×257 | 2.174 | 2.399 |
| 8×65539 | 7.040 | 8.574 |
| 64×65539 | 21.883 | 22.439 |

The compensated row kernel has 36 physical registers/thread versus 39 for the
original, zero local bytes, and 2048 rather than 1024 shared bytes. These resource
facts come from production CUDA Driver API queries, not offline assembly tooling.
The short-row overhead supports keeping that existing policy. The long-row cost
buys the executed accuracy improvement; independent chains were insufficient.

[The evidence packet](stage5/artifacts/accumulation) preserves comparison outputs,
semantic replay, descriptors, plans, CubeCL IR, generated CUDA, PTX, manifests,
NVRTC options and physical resources. Raw profiler traces remain ignored locally
with paths/hashes in results. The same compensated descriptor/plan also constructs
IR and HIP source; this is an unexecuted target sanity probe, not ROCm support.

## Source changes and limits

| File | Responsibility |
| --- | --- |
| `stage0/src/descriptor3.rs` | Physical TwoSum/correction state and paired shared merge; diagnostic independent chains; old Stage 3 path retained |
| `stage0/src/descriptor5.rs` | Validate the bounded new strategies/counts/topologies before IR |
| `stage0/launcher.c` | Admit exactly the newly required 2048-byte scalar256 scratch |
| `stage5/plan.py` | Physical identities; omitted default count preserves frozen serialized plans |
| `stage5/legality.py` | Numerical eligibility before timing; cutoff included in proof signature |
| `stage5/runtime.py` | Select the sole legal candidate instead of assuming baseline |
| New accumulation probe/check/gate files | Independent order comparison, executed integration and preserved regressions |
| `stage5/precision_gates.py` | Small output-relative preservation/extra-command hook to reuse existing gate machinery |

These are experimental physical-backend/test changes, not general upstream
Inductor hooks. PyTorch and the local CubeCL checkout have zero additional patches.
The existing FP patch remains `9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2`.
Production remains PyTorch cu130 / CUDA 13.0 NVRTC on CPython 3.14.5t, GIL disabled
without `PYTHON_GIL` forcing, with Triton unavailable and no loaded native module.
No CUDA 13.1 offline diagnostic or Triton timing is substituted.

No general summation guarantee, FP32 overflow recovery, universal eager-relative
gate, new dtype, reduction rank/topology, split planner, fallback, portability
launcher or performance stage is claimed. `cwk` remains open for the numerical
reference/gate qualification. Next direction: settle that focused reference policy
before Stage 6; no next stage is started. Nothing is committed or published.


## Reproduction

```bash
export CUDA_PATH="$PWD/stage0/cuda13-toolkit/nvidia/cu13"
export STAGE0_CARGO_PROFILE=release
cargo build --release --locked --manifest-path stage0/Cargo.toml --features llvm,hip-probe
cc -std=c11 -Wall -Wextra -Werror -fPIC -shared stage0/launcher.c -ldl -o stage0/libstage0_launcher.so
stage0/.venv/bin/python -m stage5.accumulation_checks
stage0/.venv/bin/python -m stage5.accumulation_harness_checks
stage0/.venv/bin/python -m stage5.accumulation_gates
# Expected exit 1: the unchanged eager-relative diagnostic still finds seed 30003.
stage0/.venv/bin/python -m stage5.accumulation_probe --output stage5/cache/accumulation/strict.json
# Explicit classification of that failure, not a changed acceptance tolerance:
stage0/.venv/bin/python -m stage5.accumulation_probe --qualify-reference --output stage5/artifacts/accumulation/comparison.json
```
