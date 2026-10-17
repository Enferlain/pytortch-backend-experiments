# hs6 — RMS selection and adapter attribution

Current decisions live in [STAGE5_REPORT.md](../../../STAGE5_REPORT.md).
This is a separate, host-only follow-up to the closed [5nb trial](../reuse-first-rms/README.md).
No GPU kernel was compiled, launched or timed here. Historical timings remain
conditional on their capture environment; no exclusive-GPU claim is added.

## Findings and decision

**5nb tested a deliberately fixed scalar composition, not an automatically
selected CubeK RMS implementation.** Its eight-plane packing follows CubeK's
conditional Plane routine heuristic. Its forced input vector width one bypasses
the ordinary input-width policy for the aligned primary shape. The same mature
policy leaves the two odd-width cases scalar. CubeK's API does not supply an
automatic Unit/Plane/Cube winner at this revision.

The adapter also exposes a redundant row-coordinate remainder through CubeK's
generic reader and CubeCL's generic virtual-tensor coordinate function. It
executes, but the identified region accounts for **1,536 / 47,680 = 3.22%** of
primary executed warp instructions. This fraction is an instruction attribution,
not a latency fraction or proof of a bound on potential speedup. Several other
verbose adapter expressions disappear before execution.

**Select one future reuse trial:** actual vector-width-four `PlaneReader + Sum`
for the aligned FP32 virtual operand at the existing primary width 1024. Keep
the existing 32×8 mapping, Lazy merge, protected reads, all-lane epilogue and
runtime boundary. Reuse `Sum::to_output_parallel` for the component fold; do not
invent a vector reduction. Model and qualify the changed arithmetic before GPU
implementation. This is a proposed experiment, not a qualified candidate or a
predicted speedup. No candidate implementation or new timing is part of hs6.
The proposed implementation is tracked separately as `pytortch-backend-experiments-wqc`.

## Which selection actually exists?

Source paths below are relative to `.stage0-sources/`. Hashes of every relied-on
file and target artifact are in `evidence.json`; `preservation.json` records
revisions and working-tree diff identities. CubeK remains
`4ca0235e7e4fddd82f67ec27bbc3d48c085be25a`; CubeCL remains
`6bba5fdf04206d9c3710dfbc46aa9850450ab405` with the retained patch.
All inspected CubeK bodies match their exercised compatibility copies.

| Boundary | Actual source / condition | What it establishes |
| --- | --- | --- |
| Public reduce | `cubek/crates/cubek-reduce/src/lib.rs::reduce` takes mandatory `ReduceStrategy`; `launch/strategy.rs` contains explicit `RoutineStrategy::{Unit,Plane,Cube}` | Caller chooses the routine. The adjacent optional-strategy/best-effort documentation is stale relative to the actual signature. |
| Routine dispatch | `launch/base.rs::prepare_reduce_launch` matches `strategy.routine` | It does not compare the routines or benchmark a winner. `autotune_level` gates shape-dependent fast paths here; its name does not establish an executed tuner. |
| Available catalog | `eval/benchmarks/strategy.rs::strategies` enumerates unit, independent-plane and plane-enabled cube, each with serial/parallel output-vectorization choices | Mature alternatives exist. This benchmark catalog is not an automatically selected RMS kernel. The negative source claim is bounded to the inspected reduce API/call chain and repository references listed in `selection-references.txt`. |
| Input vector width | `launch/utils.rs::generate_vector_size` calls CubeCL `tensor_vector_size_parallel` for stride-one reduction axes | Largest supported width dividing both the reduction extent and non-axis strides. |
| CUDA I/O policy | `cubecl/crates/cubecl-cuda/src/runtime.rs` sets `load_width=128`; IR properties convert bits to element counts | Under the default I/O width with no override, FP32 candidates are 4, 2, 1. This is a source/property model, not a query of a live CubeK client. |
| Plane packing | `routines/plane.rs::generate_blueprint` → `calculate_plane_count_per_cube(B*32,32,hardware)` | For B=64 and B=16, CubeK's constant eight-plane cap in the GPU branch yields eight planes per cube and grids eight/two when output width is one. The separate hardware cap is 32 and does not bind. |
| Merge | Inferred `PlaneStrategy { independent: true }` → `PlaneMergeStrategy::Lazy` | This matches the selected 5nb accumulation/collective structure, conditional on the caller selecting that strategy. |
| Bounds/idle | Inferred policy also depends on `AutotuneLevel`, divisibility, idle planes and `fuse_on_read` | 5nb deliberately retains Branch/+0 and its row-uniform guard. Stock Mask/None or fast-path selection is not substituted for its protected-load contract. |
| Result ownership | `components/global/plane.rs` writes through its writer only at lane x=0 | Stock scalar reduction output is not fused RMS output. 5nb composes the public reader/instruction body and keeps the sum available in all lanes. |

The host-only Rust probe links the actual pinned policy helper implementations,
using recorded CUDA32 / 1024-thread limits and the source-derived default I/O
width. It creates no CubeK runtime, allocator, client or tensor binding.
It is not a full `prepare_reduce_launch` or autotuner execution. Its helper
results and the source conditions above give this **conditional Plane/serial
output-vectorization** table for all three storage types:

| Shape | Actual virtual operand | Conditional stock input width | Planes / grid | Stock input iterations | 5nb input iterations |
| --- | --- | ---: | --- | ---: | ---: |
| 64×1024 | FP32 original cast chain + square | 4 | 8 / 8 | 8 | 32 |
| 64×255 | FP32 original cast chain + square | 1 | 8 / 8 | 8 | 8 |
| 16×4099 | FP32 original cast chain + square | 1 | 8 / 2 | 129 | 129 |

FP16/BF16 storage does not turn the virtual reduction operand into FP16/BF16;
using a low-precision square would violate the original descriptor. The probe
also checks zero/one-row plane-count floors, odd width and misaligned row stride.
It does not establish input address alignment or vectorized loads for a future
fused adapter; those remain separate implementation/ABI obligations.

The primary ordinary selected kernel has two warps cooperating on each row and
64 blocks; the trial has one warp per row and eight blocks. The primary scalar
input and output chains are twice as long per lane. The stock plane heuristic
does not consume reduction length or SM count when computing these eight planes.
The resulting packing is a plausible limitation for this workload. The odd
4099-width case retains only two blocks; width-four input cannot address it.
This source finding does not establish that a different routine would win.

## Ownership of the extra code

| Observed work | Earliest responsible boundary | Retained lowering / executed evidence |
| --- | --- | --- |
| Rank/coordinate walk to input batch offset | CubeK `ParallelReader::new`: loop over input rank, `output.coordinate(reduce_index,axis) * input.stride(axis)` | Present in 5nb IR/CUDA. NVRTC eliminates the two-axis loop; an active `row % B` remains. This is not a per-input-element rank walk. |
| Quotient/remainder formula | CubeCL `VirtualTensor::coordinate`: `(index / stride(dim)) % shape(dim)` | Metadata shape `[B,1]` and strides `[1,1]` come from the adapter. The first coordinate is `row % B`; the second is zero. The prior `row < B` guard makes the first mathematically equal to row, but that fact is not used to remove the remainder. |
| Wide coordinate arithmetic | 5nb compiler explicitly selects `AddressType::U64`; descriptor replay preserves signed-i64 indexing | PTX emits both `rem.u64` and an optimized `rem.u32` path selected by high-bit checks. Actual primary SASS uses the 32-bit reciprocal/correction path; the 64-bit remainder subroutine call executes zero times. Calling it an executed 64-bit division would be wrong. |
| Position → column → original position round trip | Our `ReplayExpand::coordinates` subtracts `row*N`, then unchanged descriptor replay adds it back | Present in CUDA; NVRTC cancels it. PTX input loads use reader batch offset plus lane/chunk index; output addresses use row*N directly. No retained timing cost is assigned to the canceled expression. |
| Input chunk/tail control | Actual `ParallelReader::read_plane` and `BoundChecks::Branch`, in the original 5nb scalar configuration | PTX unrolls four iterations with per-load tail predicates and a remainder loop. Actual SASS retains extensive wide index/control work. Removing guards or changing the arithmetic is not justified by counting it. |
| Collective shuffle loop | Stock Sum → CubeCL `plane_sum` lowering | Runtime loop with ascending XOR masks. Primary actual SASS executes 320 shuffles, 64 warps × five steps. |
| Division, epsilon and rsqrt inside CUDA output loop | Original semantic replay | NVRTC hoists them before the PTX output loop. They are not recomputed for each stored element. |
| All-lane output loop | RMS semantics and the selected one-plane-per-row composition | Each lane stores `lane+32*k`. This is required RMS output work, beyond the stock scalar-result reduce cost model. |

The exact primary remainder region is bounded in `sass-ledger.json`, including
all entries with zero execution counts. Its 1,536 executed warp instructions
include high-bit dispatch and reconvergence. The full component executes 47,680
warp instructions versus the ordinary path's 35,840. The retained packet binds
the actual SASS and both IR/CUDA/PTX sets. These observations do not partition
the measured microseconds among indexing, serial dependency chains, packing and
other effects. No changed-kernel intervention was run to claim that causality.

## Why the one selected follow-up is vectorized PlaneReader

Width four is supplied by a mature policy for the exact primary dimensions and
our FP32 operand; it was intentionally disabled in the first bounded trial.
It supplies four independent component accumulators per lane and cuts that
input loop to eight iterations. This has a stronger direct connection to the
scalar loop/index work than optimizing only the identified one-time remainder.
Actual load vectorization and speedup remain unproven through our replay adapter.

The candidate's order would be: for component c=0..3 and lane l, add rounded
FP32 squares at `128*k + 4*l + c` in ascending k; apply the stock plane sum to
each component; fold the four results with stock `Sum::to_output_parallel`.
The selected C++ `vector_sum` lowering starts +0 and folds components ascending.
This order differs from 5nb. Any future trial must model it exactly on CPU,
check all lanes against the existing RMS qualification packet and adversarial
cases under unchanged reference-or-better/eager/special gates (numerical tests
for admitted vector-aligned shapes, explicit rejection and retained scalar
controls for odd widths),
and stop before GPU implementation if it fails. Do not inherit scalar-order
qualification. No planner, output mapping, epilogue algebra or protected-load
change accompanies it. No guarantee of packed global loads is assumed.

Reject a GPU implementation that needs general descriptor-vector replay,
visibility patches or runtime changes. After qualification, compare independently
and in actual generated wrappers, with the existing rotated methodology and
device activity/clock provenance. Stop on a qualified tie/loss. The primary
width-four eligibility must be explicit; odd-width controls must remain covered
without inventing a second candidate. Automatic production eligibility stays
out of scope. Cube/cooperative-row alternatives remain deferred, not asserted
inferior or numerically qualified.

## Verification and limits

`inspect.py` checks the actual host helper output, dependency pins, source-copy
equality, PTX observations, actual SASS region and **13,249** preserved pre-pass
files. The complete 5nb packet and frozen long-row/production sources remain
byte-identical. It checks upstream revisions/diffs and binds its source and the
probe binary to the evidence manifest.
The verifier clears the previous successful validation marker before checking;
a later failure leaves `complete=false`. The retained setup-failure record is
also hash-bound. Required review returned no blockers; the small robustness and
wording findings and repeated checks are recorded in `review/disposition.md`.
Graph list/coverage tools returned
`Transport closed`; scoped direct reads/searches were used. There is no graph
generation/completeness claim. No GPU activity was needed for this attribution.

One setup failure is retained: the first build was requested before the host
manifest copy completed, and failed with a missing-manifest error. It executed
no probe or GPU work. The accepted build uses the copied retained lock with
only the root package renamed, `--locked --offline`, and no dependency updates.

Reproduce from the repository root (preserve the original packet before changes):

```sh
CARGO_TARGET_DIR="$PWD/stage5/cache/reuse-first-rms/target" cargo run --release --locked --offline --manifest-path stage5/rms_selection_probe/Cargo.toml
python stage5/artifacts/reuse-first-rms-attribution/inspect.py
```

The first command prints policy JSON; the verifier compares it with the retained
`host-policy.json`. Its default mode verifies the frozen evidence; `--capture`
is the explicit initial-snapshot operation, not a repair for drift. Review and
disposition are retained under `review/`. Stage 5 stays active; Stage 6 and cwk.5
remain deferred/parked. No new kernel, fallback, default promotion, commit or push.
