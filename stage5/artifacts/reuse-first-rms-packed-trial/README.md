# 2b9 — one packed-storage-load RMS candidate

Current decisions: [STAGE5_REPORT.md](../../../STAGE5_REPORT.md).
This implements only the candidate proposed by reviewed [9e8](../reuse-first-rms-packed-load/README.md).
The scalar5nb, vector4wqc and source-first9e8 packets remain unchanged.

**The intended packed loads materialize and numerical qualification passes.**
FP32 generated graphs improve roughly3–4% over frozen wqc in all six paired
rounds at each of the three tested row counts. Ordinary remains faster in every
paired graph round across all nine cases. Low-precision changes are small and
mixed. Retain an explicit diagnostic; stop without another mechanism, tuning,
planner/default/production eligibility change or Stage6 work.

## Exactly one candidate

`stage5/rms_packed_probe` is an isolated copy of the frozen wqc probe with only
the input storage-access adaptation, type imports and distinct identities.
It locally adapts existing `descriptor5::reduction_vector_values`:

1. Compute the original scalar component0 address through unchanged signed-i64
   descriptor indexing. Preserve the pointer address space when reinterpreting
   it as four elements of the **original storage type**.
2. Issue one typed `LoadOp`, extract c=0..3 with `CompositeExtractOp`, seed the
   original scalar load ID, and replay only the original non-load prefix through
   unchanged descriptor2. Replaying the original load again would undo the change.
3. Assemble the four original FP32 squares and return them to the same actual
   CubeK PlaneReader/Sum components.

FP32 loads16bytes. FP16/BF16 load8bytes, preserving the original storage → FP32 →
storage → FP32 → FP32 cast chain and FP32 square. There is no low-precision
square, reassociation, new shuffle, inline PTX or vectorized output.
The entire1,356-byte `rms_components` body is byte-identical to wqc:32×8,
four ascending component chains, Lazy/Identity, ascending XOR1/2/4/8/16,
stock +0/ascending component fold, scalar output ownership and epilogue.
`candidate-source.diff` records the complete Rust difference.

The mechanism is `actual-cubek-rms-plane-packed-storage4-v1`; entrypoints begin
`cubecl5_rms_cubek_pack4_`. Manifest `storage_vector_width=4` is distinct from
`operand_vector_width=4` and overall scalar output `vector_width=1`.
The new runtime/cache identity binds its binary, adapter, unchanged semantic
emitter and the pre-implementation CPU gate. No ordinary catalog changed.

Exact three semantic templates and physical-plan admission remain enforced in
Python/native code. Before launch, the inherited runtime checks dtype/device,
shape/stride `[B,N]/[N,1]`, storage span, disjoint output, N in32/256/1024/4096,
B in1..65 and live pointer alignment modulo `4*element_size`. All admitted row
widths are divisible by4, so the original Branch bound `pos<N/4` proves an entire
packed vector stays within its row. Invalid lanes perform no load, return four
+0s and still enter the later plane reduction. No mask/clamped-load replacement.
The PyTorch-owned storage, InfoBuilder metadata/readiness/lifetime and native
current-stream launcher are unchanged.

The frozen control loads copied original wqc PTX through that same launcher.
It neither regenerates the frozen kernel nor writes the original directory.
Generated wrappers for this control call its explicit diagnostic runtime.
The candidate compiles through existing CppCompiler<Cuda> → CUDA13.0 NVRTC
(`--fmad=false`) → PTX → driver loading. No CubeK Client/allocator/runtime is used.

`environment.json` explicitly records release Stage0 compiler/metadata hashes.
The historical wqc inventory's corresponding hashes identify the existing debug
artifacts; they are not byte-identical to release. Both phase drivers default
to release, so that historical inventory alone does not establish its launch
profile. This discrepancy is retained, not silently normalized. The frozen PTX
itself is copied byte-identically. Launcher, FP patch, CUDA/NVRTC/builtins/headers,
Torch, device and math policy hashes match the old shader inventory. Current
comparisons are same-process; no cross-inventory latency ratio is claimed.

## Qualification before timing

- **CPU gate:** complete existing36 cases /868,800 output elements pass unchanged
  reference-or-better/eager/special-value policy before packed GPU source exists.
  `cpu.json` binds unchanged wqc references/model and its producer snapshot.
- **Uninstrumented GPU gate:** same36 cases pass the full policy, CPU-model
  comparisons, special values, input preservation and input/output canaries.
  All final output bytes, including NaNs and signed zeros in this corpus, match
  frozen wqc PTX. There are27 pre-launch live rejections, three nondefault-stream
  aligned-offset controls and12 native semantic/plan-envelope rejections before IR.
- **Intermediate parity:** qualification-only observers derive from the frozen
  wqc and new candidate source. They store the per-lane FP32 operands feeding Sum
  (including neutral padding), then all32 finalized sums per row. All36 pairs
  match bitwise. Their final outputs also match both uninstrumented paths.
  Full values are retained in `observer-values.pt`, hashes in `parity.json`.
  These observers are never timed; their additional buffer/stores and metadata
  are not part of the candidate or a second reduction mechanism.

`rms_packed_observe.py` records both parent and derived source hashes. Observers
use the existing storage/current-stream launcher with an internal observation
buffer between input and output. Extra gap/canary slots remain untouched.
The initial observer build used unsupported vector indexing; it was corrected
to the existing unrolled `.extract` API, with failed source/build retained.
An admission-run missing import and host-only NCU section-filename check are
also retained under `failures/`. None changed the candidate mechanism.
`gpu-initial.json` preserves the first successful gate; formatting the adapter
changed its cache identity, so all36 cases were requalified under the final
identity before parity, SASS capture and timing. No favorable timing replacement.

## Lowering evidence before timing

IR and CUDA contain a storage-vector load before scalar replay. Crucially,
the following target-code checks also pass for all three64×1024 kernels:

| Storage | PTX input load | Executed SASS input load | Input load warp instructions |
| --- | --- | --- | ---: |
| FP32 | `ld.global.nc.v4.f32` | `LDG.E.128.CONSTANT` | 512 |
| FP16 | `ld.global.nc.v4.u16` | `LDG.E.64.CONSTANT` | 512 |
| BF16 | `ld.global.nc.v4.u16` | `LDG.E.64.CONSTANT` | 512 |

The ledger excludes the first two64-bit metadata loads. Input loads are in the
reader region after metadata and before the first SHFL; scalar output-pass
loads occur later. Cold loop-remainder instructions are retained with zero
executions.512 warp loads equal64×1024/(4×32). This is actual production
driver-loaded SASS collected with Linux NCU2026.3.1, matching section assets,
kernel replay, `--cache-control none` and `--clock-control none`. Replay durations
are excluded from warm timing. `packing.json`, `counter-kernels.json`, the raw
report/CSV and `executed-sass.txt` predate the graph measurement.

FP32 executes39,680 warp instructions versus the retained wqc43,648; register
count remains36, with no static shared/local memory. This is supporting code
evidence, not an independent performance conclusion. No instruction-count or
register-count target drove another change.

## Timing scope and limits

Fresh comparisons use the display-disabled RTX3080 UUID
`GPU-510ddc27-95ac-c62f-d49b-242cc5183c72`, separate from historical RTX3090 results.
Three paths—ordinary `first_use`, frozen wqc PTX, packed candidate—run in one
process, six predeclared rounds covering all six permutations,16 calls per path
per round. Each capture requires exactly16 expected named CUPTI kernel events.
There are162 generated-graph captures and54 isolated captures for the three
64×1024 dtype cases:216 raw traces total. Actual wrappers and selected ordinary
plans are retained. No host enqueue intervals or profiler-replay timing is mixed.
All generated packed/frozen outputs match the qualified direct-call hashes.

`timing-table.md` contains pooled sample medians and all isolated round medians;
`analysis.json` retains every per-round median and win count. Ordinary wins all
six paired graph rounds in all nine cases. Packed wins all six versus wqc in
the three FP32 cases; low-precision comparisons are mostly sub-percent and mixed.
The BF16 B16 case also wins six rounds by a small amount; this is not a broad
competitive result or an accuracy-based reason for default promotion.

**Do not quote the primary FP32 isolated pooled median as a stable intrinsic
cost.** Its full record crosses fast/slow execution and coarse clock states;
the different paths transition in different rounds. No favorable prefix is
selected. Device snapshots across this pass include210/405 through1710/9251MHz
SM/memory. Coarse before/after snapshots cannot establish per-kernel clocks or
exclusive GPU use. Process-query output and every round remain in the packet.
There is no new synthetic warmup, memory primer, clock control or cwk.5 work.

## Reproduction and validation

The initial chronological CPU gate is deliberately one-shot. Later phases:

```sh
CUDA_VISIBLE_DEVICES=GPU-510ddc27-95ac-c62f-d49b-242cc5183c72 stage0/.venv/bin/python -m stage5.rms_packed_trial gpu
CUDA_VISIBLE_DEVICES=GPU-510ddc27-95ac-c62f-d49b-242cc5183c72 stage0/.venv/bin/python -m stage5.rms_packed_trial parity
stage0/.venv/bin/python -m stage5.rms_packed_trial_analyze
```

Do not rerun into this historical packet: use a new output/cache location for
new evidence. The source and both observer crates have release/locked/offline
build logs. Candidate Rust fmt, Python Ruff F/format and `git diff --check` pass.
The analyzer verifies sources, registry pins, source-derived observers, numerical
gate records/intermediate bytes, actual packed SASS and exact NCU counts,216
raw trace censuses/durations/wrappers, and13,789 pre-pass regular-file targets.
This includes all prior tracked sources/artifacts and the uncommitted9e8 packet,
excluding mutable Beads/report/ignore files. Both upstream source trees and
exercised compatibility copies are checked against9e8 source hashes.
Production sources are unchanged; the historical18 regression groups were not
rerun and are not claimed as new evidence. Stage5 remains active; Stage6 deferred.

Required review returned **Approve — no blockers**. Its five minor scoped
observations require no change to the tested candidate. Parent offline and
integrity checks pass again; no GPU code or timing changed after review.
See [review](review/review.md) and [disposition](review/disposition.md).
