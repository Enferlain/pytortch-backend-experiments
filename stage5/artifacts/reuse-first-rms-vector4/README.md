# wqc — actual vector4 PlaneReader/Sum trial

Current decisions live in [STAGE5_REPORT.md](../../../STAGE5_REPORT.md).
This implements the single follow-up selected by [hs6](../reuse-first-rms-attribution/README.md).
The scalar [5nb trial](../reuse-first-rms/README.md) and hs6 packet stay unchanged.

**Numerical qualification passes. No generated-graph comparison wins against
the ordinary path; the performance stop condition is reached.** FP32 graphs
improve on the scalar composition, while FP16/BF16 generally regress. Keep the
candidate explicit-only. No second mechanism, tuning sweep or promotion.

## Actual reused mechanism

The separate `stage5/rms_vector_probe` composes unchanged CubeK
`Reader::new` → `PlaneReader::{new,length,read}` → stock `Sum` Lazy/Identity
accumulation → `Sum::plane_reduce_inplace` → `Sum::to_output_parallel`.
The input precision is `(f32, Const<4>, f32)`. The latter method performs the
stock component fold, using CubeCL's ascending vector sum from +0. There is no
custom shuffle, tree or vector reduction.

The narrow new read adapter replays **unchanged** `descriptor2::emit` for each
of four original scalar operands, including storage casts and the FP32 square,
then constructs the existing CubeCL vector IR value. Logical scalar shape/stride
metadata stays `[B,N]` / `[N,1]`; input length is in vectors. Reader offsets are
converted back to original scalar positions before descriptor replay. The
metadata-only output remains `[B,1]`; the original scalar normalization epilogue
and `lane+32*k` output loop stay intact. Workgroups remain 32×8, eight rows per
block, with Lazy merge and all-lane results. The same uniform row guard and
Branch/+0 protected reads apply. No CubeK allocator, client or launcher is used.

CUDA13.0 NVRTC with retained conservative FP settings and `--fmad=false` compiles
the emitted CUDA through the existing native PyTorch-current-stream launcher.
PyTorch owns storage, metadata readiness and lifetime. The native compiler admits
only the three original semantic templates and exact mechanism/plan/entrypoint
envelope. Python also requires widths 32/256/1024/4096, rows 1..65 and input
alignment of four storage elements. Output alignment/layout validation remains
the original scalar contract. The physical plan's overall vector width remains
one; `operand_vector_width=4` identifies this internal reduction operand, with a
distinct mechanism/cache identity. No normal catalog/planner/runtime changed.

This is component vectorization through scalar semantic replay. **It does not
establish packed global vector loads.** The primary FP32 PTX retains four scalar
loads, and executed SASS has no 128-bit LDG. The scalar output pass also remains.
Changing either adapter/load or output mapping would be a separate mechanism.

CubeK and CubeCL retain the hs6 revisions and working-tree states. The new Cargo
lock matches the retained scalar probe's registry pins; only the root package is
renamed. The accepted build is release/locked/offline. `evidence-integrity.json`
binds actual compiler/tool binaries, component bodies, environment and captures.

## Numerical gate before GPU implementation

The exact CPU model rounds the original square to FP32, then accumulates four
component chains per lane at `128*k+4*lane+c`, ascending k. Each component uses
stock XOR offsets 1,2,4,8,16. Starting +0, it folds components c=0,1,2,3 in order,
then applies the original division, epsilon, rsqrt, multiply and storage cast.
The output uses each lane's own result; all-lane sums are checked.

The matched packet contains 15 vector-admissible cases from 5nb's frozen
reference, plus 21 new matched row-boundary cases at width1024 (rows
1/3/8/9/16/64/65 for FP32/FP16/BF16). The new references use the original
configured compiled-Triton precision policy, ATen and FP64 diagnostic values.
Their reference environment enables the GIL. It is used for numerical outputs,
not host performance comparisons. The 15 inherited reference cases retain their
original RTX3090 provenance; the new cases were captured on the RTX3080.
Every comparison uses the exact same saved inputs. All 51 historical odd-width
cases remain outside admission; there is no vector4 fallback for them.

The CPU gate passed **36 cases / 868,800 output elements before vector4 GPU
source existed**. The original producer snapshots and `cpu.json` retain that
chronology. After implementation, GPU qualification passes the same 36 cases,
the unchanged reference-or-better/eager tolerances, NaN/Inf/zero class and sign
checks, input preservation and input/output canaries. Thirteen cases match CPU
output bits; CPU rsqrt is not claimed as a GPU bitwise oracle. There are 27
invalid extent/odd-width/alignment/shape/stride/dtype/device/alias controls, all
rejected before launch, and three nondefault-stream aligned-offset controls.
Twelve mutated semantic/envelope cases reject before IR emission.

A later CPU reproduction writes `cpu-recheck.json` and a separate model file,
preserving the initial pre-implementation gate. The unchanged scalar control is
also numerically checked on all 36 new matched cases; its generated-graph output
hashes match those direct qualifications. Scalar control IR/CUDA/PTX matches the
frozen 5nb artifacts modulo entrypoint names. It is recompiled with the unchanged
original binary into this **new** packet, never into the historical packet.

## Timing and device-state limits

Fresh measurements use the display-disabled **RTX3080**,
`GPU-510ddc27-95ac-c62f-d49b-242cc5183c72`, selected explicitly through
`CUDA_VISIBLE_DEVICES`. These numbers are separate from the earlier RTX3090
capture. GPU/process/SM clock/memory clock/power snapshots bracket every block
of 16 calls. They are coarse observations, not per-instruction clock records or
proof of exclusive access; the process query reports no external compute rows.

**The device does not hold a single clock state.** The primary graph snapshots
show 210/405MHz SM/memory; later FP16/BF16 row-boundary graphs show
1710/9251MHz. Intermediate snapshots also show 375/405/450MHz SM values.
The isolated primary crosses states. No clocks were fixed and no synthetic
warmup, primer or video-playback manipulation was introduced. cwk.5 remains
parked; this is direct measurement provenance, not a new hardware investigation.

All scopes use CUPTI kernel durations, six alternating-order rounds and 16 calls
per path. The order reverses ordinary/scalar/vector4 between rounds; the scalar
control occupies the middle. Ordinary versus vector4 is the primary paired
comparison. This is not a fully permuted three-way schedule. Each actual
generated graph emits one kernel; 180 complete traces are retained and checked.
No profiler-replay durations, host latency or queued event intervals are mixed.

| Generated graph | Ordinary µs | Scalar CubeK µs | Vector4 CubeK µs |
| --- | ---: | ---: | ---: |
| FP32 64×1024 | 14.526 | 26.018 | 23.412 |
| FP16 64×1024 | 15.215 | 27.810 | 29.095 |
| BF16 64×1024 | 15.166 | 29.412 | 30.238 |
| FP32 16×1024 | 14.979 | 25.463 | 22.672 |
| FP32 65×1024 | 14.367 | 25.873 | 23.044 |
| FP16 16×1024 | 2.336 | 4.256 | 4.447 |
| FP16 65×1024 | 2.430 | 4.351 | 4.540 |
| BF16 16×1024 | 2.239 | 4.479 | 4.604 |
| BF16 65×1024 | 2.402 | 4.576 | 4.731 |

Vector4 loses **all six paired graph rounds against ordinary in every case**.
It wins all six graph rounds against scalar in the three FP32 cases, zero in
the FP16 cases and BF16 row-boundary cases, and one in primary BF16. This is a
scoped observed result under the recorded states, not a universal cost ratio.
Do not compare absolute latencies across clock regimes or GPUs.

The entire primary isolated record is below; no favorable prefix is selected.
Its pooled medians (6.196/10.256/12.254µs) do not qualify as a stable isolated
kernel-cost estimate because the rounds cross clock states.

| Isolated round | Ordinary µs | Scalar µs | Vector4 µs |
| --- | ---: | ---: | ---: |
| 0 | 2.369 | 4.098 | 3.542 |
| 1 | 2.368 | 4.103 | 3.551 |
| 2 | 2.367 | 4.093 | 12.043 |
| 3 | 10.054 | 16.521 | 12.466 |
| 4 | 13.089 | 26.091 | 23.437 |
| 5 | 14.525 | 26.116 | 23.442 |

## Actual executed code

The focused production driver-JIT capture uses the exact three timed FP32
primary entrypoints. The full NCU argv, executable/version/SHA and matching
section folder are in `counter-command.txt` / `ncu-provenance.json`.
Kernel replay uses `--cache-control none --clock-control none`; its durations
are excluded from timing. The raw report/CSV and executed SASS are retained.

| Executed resource / warp instruction | Ordinary | Scalar | Vector4 |
| --- | ---: | ---: | ---: |
| Registers/thread | 46 | 26 | 36 |
| Warp instructions | 35,840 | 47,680 | 43,648 |
| SHFL | 0 | 320 | 1,280 |
| BAR | 896 | 0 | 0 |
| IMAD | 2,944 | 12,864 | 8,000 |
| IADD3 | 3,456 | 6,400 | 4,800 |
| FADD | 2,944 | 2,432 | 3,648 |

The SASS sums agree with the raw metric. Vector4 reduces input-loop/index work
but needs four component collectives and a component fold. All component paths
still pack eight independent rows per block and retain scalar descriptor loads
and output work. Those facts support the observed tradeoff; they do not assign
individual microseconds or establish a separate causal contribution for each.
Neither fewer instructions nor higher register count is the acceptance gate.

## Verification, failures and decision

`rms_vector_trial_analyze.py` verifies the numerical/ABI records, 180 raw traces,
actual wrapper/kernel attribution, scalar-control code identity, driver-JIT
SASS metrics, dependency pins and **13,266** pre-pass file hashes. The complete
5nb and hs6 packets and frozen long-row sources are unchanged. Source, binary,
NVRTC, activity and profiler evidence are integrity-bound. Ruff F/format,
Cargo format and the isolated locked/offline release build pass.

Preserved setup failures include the first reference capture's missing
inventory arguments (outputs completed but the metadata write failed), a Rust
host build's division-op spelling and trait-method ambiguity, and the first
static rejection run's missing Python `copy` import. None changed the selected
mechanism or produced an unqualified vector4 GPU run. The reference attempt was
archived in full and a fresh capture was made on the selected RTX3080.

Required review returned **no blockers**; findings and affected rechecks are in
`review/disposition.md`, with the full response under `review/`. The analyzer
now also checks the absent 128-bit LDG suffix and empty process-query rows.
Graph project and
coverage tools remain unavailable (`Transport closed`); source verification is
direct and scoped. No graph completeness, generic vector bridge, training,
automatic eligibility or broad CubeK/custom ranking is claimed.

**Stop here:** actual vector4 reuse works numerically but gives no graph win
against ordinary. Keep the diagnostic and evidence; no alternate packing,
cooperative-row kernel, packed-load adapter, output-vectorization change,
planner/autotuner sweep or production promotion in this task. Stage 5 stays
active, Stage 6 deferred, and cwk.5 parked. No commit or push.
The remaining packed-load capability question is tracked separately as `9e8`,
unstarted; it begins with existing mature load machinery and adapter legality.

Reproduction (snapshot this fixed-output packet first):

```sh
CUDA_VISIBLE_DEVICES=GPU-510ddc27-95ac-c62f-d49b-242cc5183c72 stage0/.venv/bin/python -m stage5.rms_vector_trial cpu
CARGO_TARGET_DIR="$PWD/stage5/cache/reuse-first-rms/target" cargo build --release --locked --offline --manifest-path stage5/rms_vector_probe/Cargo.toml
CUDA_VISIBLE_DEVICES=GPU-510ddc27-95ac-c62f-d49b-242cc5183c72 stage0/.venv/bin/python -m stage5.rms_vector_trial gpu
CUDA_VISIBLE_DEVICES=GPU-510ddc27-95ac-c62f-d49b-242cc5183c72 stage0/.venv/bin/python -m stage5.rms_vector_trial admission
CUDA_VISIBLE_DEVICES=GPU-510ddc27-95ac-c62f-d49b-242cc5183c72 stage0/.venv/bin/python -m stage5.rms_vector_trial scalar_qualification
CUDA_VISIBLE_DEVICES=GPU-510ddc27-95ac-c62f-d49b-242cc5183c72 stage0/.venv/bin/python -m stage5.rms_vector_trial graphs
# Exact counter capture argv is retained; its report was exported with:
ncu --import stage5/artifacts/reuse-first-rms-vector4/primary-code.ncu-repz --csv --page raw
ncu --import stage5/artifacts/reuse-first-rms-vector4/primary-code.ncu-repz --page source --print-source sass
python -m stage5.rms_vector_trial_analyze
```

Rebuilding with a different toolchain/device state is a new capture; integrity
checks intentionally reject changed artifacts. The matched new reference phase
uses the installed reference interpreter and the separate `reference` command;
it is unnecessary for rechecking the frozen numerical packet.
