# d39 — reuse-first transpose/layout trial

Current decisions: [STAGE5_REPORT.md](../../../STAGE5_REPORT.md).

**The existing component can be composed behind our compiler boundary, but this
single candidate is not competitive.** Ordinary wins all108 paired graph
comparisons: six rounds in each of18 materializing dtype/shape/function cases.
Stop this mechanism. No second layout algorithm, tuning, automatic eligibility,
production/reduction modification, synthetic warmup or Stage6 work follows.
Required independent review returned **No blockers**; seven minor observations
and parent checks are recorded in [disposition](review/disposition.md).

## First distinguish views from materialization

The original Stage5 `transpose` fixture passes `x.T` to `sin(x)+1`; it does not
request contiguous output. Fresh captures of both existing shapes33×257 and
256×513, plus aligned256×512, preserve transposed output strides. Inductor
absorbs the view: its descriptor loads/stores the same linear physical position.
There is no transpose copy to replace. All nine view controls pass the current
pointwise numerical policy and retain their strides.

The explicitly different workloads `(x.sin()+1).contiguous()` and
`x.contiguous()` require contiguous output. Inductor emits one kernel for each;
the former fuses the arithmetic and rearrangement. These are useful component
trials, not evidence of a missing copy or regression in the original corpus.
`capture.json` retains all27 initial layout/codegen captures. Its producer
snapshot records an initially loose FP32 descriptor-capture check; the later
GPU admission and nine view controls use the unchanged actual pointwise policy:
FP32 rtol/atol2e-6, FP16 .002/.001, BF16 .016/.008, plus value classes/signs.
There is no policy relaxation or performance inference from that initial check.

## One existing mechanism, runtime-free composition

The source is the pinned CubeCL
`cubecl-std/src/tensor/contiguous/perpendicular.rs::copy_perpendicular`.
It gathers storage vectors, transposes components in registers and stores
contiguous vectors. `upstream-perpendicular.rs` preserves the original file;
`component.diff` records the six mechanical changes to its kernel body:

- Use ordinary `#[cube]` expansion rather than its runtime launch generator.
- Replace concrete input/output tensors by existing virtual tensors.
- Translate indexed reads/writes to the virtual tensor's existing methods.
- Spell out Tensor's existing coordinate formula using virtual shape/stride.

The coordinate assignment, loop order, register transpose and worker ownership
are unchanged. The private concrete-Tensor kernel has no public runtime-free
composition entrypoint; the local translation reuses that actual body instead
of introducing a new transpose algorithm. No upstream checkout is edited.

`transpose_probe` expands the translated body into the same KernelBuilder →
CppCompiler<Cuda> → CUDA13.0 NVRTC (`--fmad=false`) → PTX → driver path. Its
virtual reader uses the existing typed vector-pointer/load pattern. Its writer
extracts each original storage element and replays the captured descriptor's
unchanged non-load/non-store operations through `descriptor2::emit`, including
every low-precision cast and the original scalar sin/add. It then packs the
results for the original vector store. The six exact captured semantic
templates are mandatory; this is not a general arbitrary-DAG adapter.

Widths follow the pinned CUDA128-bit I/O ceiling and the component's three
divisibility checks: aligned FP32 uses4, FP16/BF16 uses8; the tested odd layouts
require1. There is no width sweep or scalar fallback kernel. The diagnostic
uses the existing256-thread launch envelope and overrides only the worker-grid
calculation. Stock CubeDim can choose a smaller block for small worker counts;
this is a component-body trial, **not a benchmark of CubeCL's stock launcher or
its exact launch policy**. The number of active workers and their work remain
the component's `N/V`, with `M/V` batches each, for input `base[M,N].T`.

The existing Stage5 validator/enqueue path still owns live dtype/layout/span,
disjoint-storage, metadata readiness, current-stream and allocator-lifetime
handling. Additional guards admit only the eight predeclared shapes and their
required pointer alignment. All extents/intermediate coordinate products on
this domain are positive/bounded (at most131,328 scalar elements); no unsigned
physical coordinate changes the original signed-i64 semantic index meaning.
The whole-worker termination guard precedes all reads. Divisibility proves
every active vector is wholly in bounds. No redirected or eager masked access,
padding read or relaxed precision is introduced. Original views are rejected
by the candidate's exact-template boundary rather than silently materialized.

## Qualification and actual execution

Before the GPU adapter existed, `transpose_model.py` checked the literal full
coordinate/gather/register-transpose/store assignment against an independent
dense transpose:24 shape/dtype cases,1,215,843 scalar address identities, exactly
one owner per output. Arithmetic is unchanged pointwise replay, so this is an
address proof rather than a new FP reduction-order model. The chronological
CPU producer is retained; later formatting preserves its AST.

The GPU gate passes54 cases: both functions, three storage types, eight shapes,
and six additional special-value fixtures. All candidate outputs match the
retained ordinary PTX bitwise, including the tested NaNs and signed zeros, and
pass eager-relative policy. Finite dynamic range, ±0, subnormals, NaN, ±infinity
and large sin inputs are retained in `special-values.pt`. Input/output canaries
and input hashes are preserved.42 live invalid-argument checks, six nondefault
current-stream/aligned-offset controls and seven native pre-IR rejection checks
pass. This is the tested scope; no general concurrency or sanitizer claim.
The effects mutation is a second input rejected by exact-template admission,
not a separate effects-veto implementation. The inherited scheduling `baseline`
argument does not activate candidate tuning; its launch envelope is explicit.

Twelve qualified dtype/function/width artifacts retain IR, CUDA, PTX, manifests,
conservative NVRTC options and actual driver resource queries. The separate
eight-invocation NCU comparison uses the **actual timed ordinary winner** and
the qualified component, for FP32 fused/copy at aligned256×512 and odd256×513.
It PATH-discovers Linux NCU2026.3.1, records tool/section hashes, and preserves
production driver-loaded SASS and executed instruction counts. Replay latency
is excluded from warm timing. No clock/cache control is applied.

For the aligned FP32 component, SASS confirms1,024 executed128-bit input loads
and1,024 executed128-bit output stores per invocation. Both components have
zero executed shared barriers/shuffles in these captures. The fused component
uses40 registers; copy uses42. Fused driver local allocation is32 bytes/thread,
while copy is0; these finite counter invocations execute no local loads/stores,
so the allocation is not labelled observed spill traffic. PTX also contains
128-bit packed low-precision I/O; low-precision executed SASS was not collected.

The component's small grid and long worker loop survive the complete lowering:
aligned FP32 has one256-thread block/128 active workers/64 batches each;
aligned16-bit has64 active workers/32 batches. Odd256×513 has three blocks,
513 active workers and256 batches each. Ordinary uses far more blocks.
The aligned copy executes26,384 versus212,992 ordinary warp instructions yet
still loses the graph comparison; wide I/O and instruction savings do not
compensate for this mapping on the tested workload. No latency fraction is
assigned to a proxy, and no CubeCL/NVRTC correctness bug is established.

## Timing scope, evidence and stop

`timing-plan.json` predeclares six alternating paired rounds ×16 calls for all
18 cases. One process executes both real generated wrappers on identical input
bytes, verifies identical outputs and exactly16 named CUPTI kernels per capture.
All216 losslessly compressed raw traces/3,456 kernel events remain; their grids,
blocks, durations, qualified artifact and data identities are independently
checked. Actual wrappers and selected ordinary artifacts are retained.

[The full table](timing-table.md) includes pooled sample medians; `timing.json`
retains every round. These are complete-graph device kernel durations, not host
latency or NCU replay timing. The display-disabled RTX3080's coarse snapshots
range210/405 to1710/9251MHz SM/memory; **pooled costs are not stable intrinsic or
cross-state speed ratios**. External GPU activity is not excluded. All108 paired
rounds lose, which supports rejecting this candidate in this experiment without
claiming a universal stock-component result. No new Triton comparison is claimed.

The scalar-vector representation and frontend termination-finalization compile
failures, initial Cargo resolution/API errors, overlay-order failure after54
passing rows and an NCU harness import failure are retained under `failures/`
and the named build logs. The final gate repeats all54 cases/controls under the
correct overlay setup. GPU body/arithmetic are unchanged after qualification.
Only the counter harness was later bound to timed ordinary winners; the exact
timing producer snapshot and unchanged `graphs` function AST are checked.

Run the offline verifier with:

```sh
stage0/.venv/bin/python -m stage5.transpose_analyze
```

This verifier also checks the live probe binary at
`stage5/cache/reuse-first-rms/target/release/stage5-transpose-probe`. That is a
rebuildable cache dependency, not a retained executable. If it has been cleaned,
rebuild with the captured toolchain/source before verification; absence or a
different binary hash fails closed:

```sh
CARGO_TARGET_DIR="$PWD/stage5/cache/reuse-first-rms/target" cargo build --manifest-path stage5/transpose_probe/Cargo.toml --release --offline --locked
```

The SASS/clock parser targets this frozen NCU2026.3.1/RTX3080 packet and its
recorded CSV query order; it is not a general importer for new devices/tool
versions. Those hardcoded identities and the literal108-pair assertion do not
replace the individual trace, shape and invocation checks.

Use fresh output/cache paths for new experiments; historical phases are one-shot.
The verifier checks14,263 unchanged pre-pass tracked regular files, the exact
four-line evidence-retention addition to `.gitignore`, original upstream
body, CPU mapping, GPU identities/special values,12 qualified artifacts, all216
trace censuses and eight SASS totals against independent NCU metric rows. Native
release/locked/offline build, focused Ruff/format/Rustfmt and diff checks pass.
Production sources and all retained reduction packets remain byte-identical;
the historical18 production regression groups were not rerun or claimed here.
Stage5 stays active, cwk.5 stays parked and Stage6 stays deferred. A separate
broadcast/strided-pointwise reuse feasibility check is a reasonable next scope;
it is not started by this trial.
