# Stage 5 guarded adoption: column layout and long-row admission boundary

Evidence-retention correction (2026-10-19): a subsequent regression runner reused
this packet's `regressions/corpus.json` output path and overwrote that uncommitted
raw JSON. It could not be recovered and is no longer present here. The original
complete successful `regressions/corpus-final.log` and original failed capture in
`failures/corpus-initial.json` remain intact. The subsequent168-case/zero-failure
capture is explicitly retained as new evidence at
`../reference512-admission/regressions/corpus.json`; it is not a replacement for
the lost original measurements. The new runner now redirects output to its own
packet. Other column proof, trace, source and review files were not overwritten.

Current decisions live in [STAGE5_REPORT.md](../../../STAGE5_REPORT.md).
Base: `d8b0c5af`. No commit/push, new reduction topology, split/planner change,
numerical tolerance change, production warmup, hardware-state experiment or Stage6.

## Long-row partial: qualification retained, normal admission still blocked

The existing normal catalog contains `plane_shared_reference512`, but numerical
eligibility explicitly excludes it. Removing that exclusion alone is unsafe for
the current selector: it compares each partial buffer to the old scalar tree
before timing, whereas the retained qualification judges complete split9 sums
against the corresponding efficient reference/no-worse-FP64 policy.

[partial-admission.json](partial-admission.json) runs all148 existing exact inputs
on the original RTX3090. Every input hash and frozen complete output matches the
qualified packet. The unchanged normal per-partial comparison fails five cases:
`random_9241_1048576`, `random_101_1048576`, `random_7301_1048576`,
`new_heldout_40028`, `new_existing_finite_adversaries`.
For the adversary, reference512 preserves1,-1,0.25 in partial0 where the old tree
returns0. These are **selector-comparison failures, not new whole-reduction
qualification failures**. Original failures and reference policy remain intact.

The first attempt on RTX3080 stopped at the original RNG-history input hash,
before numerical admission comparison. Its source is retained in `failures/`;
the exception was `AssertionError: original_rng_history` at input hash checking.
The authoritative run uses RTX3090; no regenerated input was accepted by weakening
hash checks. `partial-check.log` retains that complete run.

No change was made to reduction eligibility or `qualified_for_runtime`; its
existing coarse compensation guard was not extended. Normal guarded graphs also
retain split32/4 at the principal workloads rather than manufacturing split9.
The next long-row integration task is to establish a graph-level admission and
validation boundary using the existing complete-reduction numerical policy.
It must resolve the partial-versus-whole mismatch and preserve split/consumer
scope. A local guard deletion is not that solution.

## Column layout: normal physical candidate

`bounded_column_pointwise` is a distinct scalar256 schema5 physical plan.
It translates the existing qualified CubeCL U64 FastDivmod/PlainLayout body into
the normal compiler; the historical probe and its artifacts remain unchanged.
There is no dependency on diagnostic packets or NumPy in normal execution.
`cubecl-std` is an explicit dependency of the normal compiler.

Initial admission is deliberately finite:

- FP16/BF16 only, exact complete column descriptors stored as source in
  `stage5/column_templates.json`.
- Shapes3×255,64×256,64×257, checked on every candidate launch.
- Both already-proved complete `B*N`/`N*B` extent forms; all other descriptor
  fields are exact, including SSA spelling, precision, strides and effects.
- Scalar256 assignment, lazy protected scalar reads, non-load replay, storage
  casts and U64 addressing are unchanged. Metadata remains7 i64 scalars/80bytes.
- Other dtypes/shapes/layouts/descriptor forms retain existing choices.

Python admission uses canonical JSON equality, so boolean/integer lookalikes
cannot disagree with native JSON admission. Native compilation independently
requires the complete supported descriptor and exact physical plan. All live
storage/layout/device/alias checks remain in `artifact.Kernel`; FastDivmod extras
join its existing immutable metadata/readiness/current-stream/lifetime mechanism.
The admission fact is part of the finite runtime proof signature, preventing
same-bucket unsupported extents from reusing an eligible selection. Static template
recognition runs once at construction; warm calls do not reread source files or
serialize full descriptors. Contract/template hashes enter artifact and tuning
identities. Existing persistent cache handling and selection criteria remain.

The existing first-use selector, baseline comparison and baseline/MAD selection
rule choose the plan. No forced winner, compiler-method substitution or broad
layout recognizer was added. Explicit baseline/deterministic/effect bypasses stay.

## Qualification and code identity

[column-cpu.json](column-cpu.json) contains12 CPU address/metadata proofs for
two storage dtypes × two extent forms × three shapes, before production GPU
source existed. The inline producer used existing `layout_model.verify` and
`params/signed_bits`; reproduction can rerun those helpers against the source
templates. The old broader66-case CPU proof remains unchanged evidence.

[column-controls.json](column-controls.json) records:

- All78 existing layout corpus cases preserve recorded input and output hashes
  and eager policy, with the new candidate only in admitted cases and the normal
  scalar baseline elsewhere. Existing exceptional-value cases are included.
- Four production descriptor variants match retained column IR/CUDA/PTX byte for
  byte when compiled with the retained entrypoint. Both extent spellings match.
- Twelve additional special-value/offset-output cases over all admitted forms and
  shapes preserve canaries and eager classes/zero signs.
-28 prelaunch rejections, including genuinely valid neighboring tensor layouts
  outside the admitted shape set, and four current-stream/offset controls.
- Six malformed/unqualified descriptor forms reject in Python and native code.
- Same-bucket proof signatures and candidate sets distinguish admitted/excluded
  extents. [Cache init](column-cache-init.json) and [fresh-process reload](column-cache-reload.json)
  pass, with zero compiler invocations on reload.

`frozen-identity/checks.json` regenerates all three retained uniform-loop partials
and original finals through the rebuilt normal compiler and checks byte-identical
IR/CUDA/PTX. No reduction emitter or numerical-policy source changed.

## Generated graphs and performance scope

All six admitted dtype/shape combinations run through ordinary `compile_backend`
and actual first-use tuning, plus two same-bucket unsupported-shape graph controls.
The benchmark compares real new ordinary generated one-kernel wrappers with the
retained pre-admission ordinary selected kernel from the composition packet.
The retained path uses the existing read-only physical loader and an explicit
output allocation outside the timing scope; it is not a host/wrapper-latency
comparison. The exact two proved extent spellings may differ between wrappers.

Six balanced rounds ×16 calls × two paths × six cases yield72 compressed CUPTI
traces/1,152 kernel durations. Every cell requires exactly16 expected kernel
events. The setup-only observer records the actual invoked generated wrapper;
no observer runs in timing. Raw [graphs](column-graphs.json) and traces retain all
rounds, input/output identities and coarse clock snapshots. No NCU replay or
synthetic warmup/primer/clock control was introduced.

In the final capture, five cases select the new plan and beat the retained
ordinary kernel in all30 paired rounds. Small BF16 keeps the scalar baseline;
its same-body timing variation is not attributed to the new mechanism.
FP16 medians are1.569vs1.632µs at3×255,1.632vs1.696µs at64×256/257.
Coarse clocks vary across the capture, especially BF16; no universal, intrinsic,
host-latency or Triton-parity gain is claimed.

`first-capture/` preserves a prior complete72-trace/1,152-duration run before
source formatting/build cleanup. All six cases chose the component there;
35/36 paired rounds beat retained ordinary. Neither capture is discarded.
The initial graph attempt failed before timing because the retained ordinary
lookup required one extent spelling; `failures/initial-graph-lookup.*` preserve it.
Lookup now admits only the existing two proved spellings; runtime admission and
arithmetic did not change to repair that harness failure.

## Validation and review

All18 established regression groups have passing results; the final complete
168-case corpus has zero failures. The first component census attempted the new
restricted plan on an unsupported generic fixture; it now records the rejection,
as for final9, and the affected group passes. The first complete corpus recorded
one `SystemError: <class 'UserWarning'> returned a result with an exception set`
on FP16 RMS backward. Its isolated unchanged repeat and the complete unchanged
168-case rerun pass. The source of that exception is unresolved; the failed full
run and both repeats are retained, with no exclusion/tolerance/warning change.
`regressions/gates.json` records the successful groups across these attempts,
and the capture files record restoration of historical tracked artifacts.

Required [review-mcp](review.json) found no correctness/safety bugs. Its example
metadata mismatch was fixed; post-review78/28-live/4-stream controls, four example
ABI lengths, four column and six frozen partial/final IR/CUDA/PTX identities pass.
The remaining minor findings are scoped in [the disposition](review-disposition.md).
The fix changes an unused example blob and a source comment; runtime, GPU bodies
and performance captures are unchanged. Ruff F, Rust formatting and the full
LLVM-enabled build pass. Clippy completes with the existing unused-Hip warning
and clone-on-copy in the translated frozen body; no warning-free claim.
No commit or push was performed.

Commands (select the named device explicitly; use the isolated Stage0 Python):

```text
cargo build --release --features llvm --manifest-path stage0/Cargo.toml --offline
CUDA_VISIBLE_DEVICES=<RTX3090 UUID> STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.adoption_checks
CUDA_VISIBLE_DEVICES=<RTX3080 UUID> STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.column_admission_checks controls
CUDA_VISIBLE_DEVICES=<RTX3080 UUID> STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.column_admission_checks cache
# Repeat cache in a fresh process for reload (retain the init result).
CUDA_VISIBLE_DEVICES=<RTX3080 UUID> STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.column_admission_checks graphs
```

CPU-before-source chronology belongs to the retained capture, not a fresh-tree
reproduction claim. The captured graph run changes no warmup policy. Existing
RMS/transpose diagnostics and parked cwk.5 remain unchanged; Stage5 remains active.
