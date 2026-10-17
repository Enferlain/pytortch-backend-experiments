# 8x0 — selected reuse-first trial

Current decisions live in [STAGE5_REPORT.md](../../../STAGE5_REPORT.md).
This packet records selection evidence and an implementation boundary, not a
candidate implementation, numerical qualification or speedup. No backend or
upstream kernel source changed in this pass.
The bounded implementation is tracked as `pytortch-backend-experiments-5nb`.

## Decision

Try the actual CubeK **PlaneReader + Sum instruction** on the existing
`stage5.corpus.rms` inference workload. Start at FP32 **64×1024**; retain
**64×255** and **16×4099**, then FP16/BF16 storage, as qualification neighbors.
The latter shape is part of the existing normalization corpus; it does not
reopen the frozen long-row sum, split9 or partial topology.

The fresh baseline-only capture passes all nine dtype/shape cases with the
unchanged corpus tolerances. Each call emits **one** generated kernel, with one
FP32 sum and a `reduction`-scope output store. All three dtype descriptors and
their actual IR/CUDA/PTX are retained under `kernels/`; `baseline-capture.json`
binds every call to those files, input/output hashes and launch geometry.
The capture uses `tuning="baseline"` explicitly. It establishes the adapter's
semantic surface, not the current first-use winner or a performance baseline.

## Short component/workload map

Paths below are relative to `.stage0-sources/` unless prefixed `stage5/`.
They refer to local pinned source, not an untested latest upstream release.
`source-evidence.json` records revisions, paths, hashes, symbol locations and
the historical workload measurements used to prioritize this choice.

| Existing Stage 5 family | Actual available component | Adapter obligation and disposition |
| --- | --- | --- |
| `add`, `sin`, `fused` | CubeCL `Vector`, `VirtualTensorOperationsExpand`, `LinearLayout`; CubeK tile `procedural::Product`/`product_of` | Retain arbitrary descriptor arithmetic and every low-precision cast. Product is an actual multiplication recipe, not a ready-made implementation of `(x*y+x)/(y+2)`. Trial remains available; the existing large pointwise vector path is already near measured bandwidth limits. |
| `bias_row`, `bias_column`, `slice`, `stride_zero` | CubeCL `std/tensor/layout/{strided,linear}.rs`, view/virtual-tensor operations and contiguous index mapping | Prove broadcast/stride/offset maps and signed-i64 equivalence; do not insert a materialization by default. Good later layout trial, beyond this first adapter. |
| `transpose` | CubeCL `std/tensor/contiguous/perpendicular.rs::copy_perpendicular` and `launch_copy_perpendicular_ref` | Actual gather/register-transpose/store kernel. Shape/vector divisibility and fused `sin+1` must be handled, without replacing our PyTorch launcher or adding an unmeasured copy. Existing odd dimensions constrain the vectorized form; not the first trial. |
| `masked_cat` and protected producers | CubeK reduce `ReaderBoundChecks::Branch`; CubeCL virtual read adapters | Branch suppression is suitable for lazy access. `Mask` redirects a read and selects, so it cannot replace the protected-load contract. Preserve the existing masked-region replay. |
| `mean_epilogue` | CubeK `Reader`, `PlaneReader`, `Sum`, `Mean`, `GlobalFullPlaneReduce` | The stock components offer the right work and ownership structure for scalar-per-row results. Keep descriptor division/epilogue until its exact order qualifies; do not replace it by a reciprocal or silently use Mean's output conversion. Useful narrow neighbor, not another concurrent candidate. |
| **`rms_inference` — selected** | **CubeK `Reader::new`, `PlaneReader::{new,length,read}`, `<Sum as ReduceInstruction>::{null_accumulator,reduce,plane_reduce_inplace}`** | **Virtual reads replay input casts and square. The existing sum consumer replays division, epsilon, rsqrt, multiply and storage casts. Use Lazy/Identity local accumulation plus the stock all-lane plane sum.** |
| `rms_backward` / training normalization | Same readers and Sum; CubeK tile `Tile::{reduce_axis,store_rows,scale_rows,rescale_rows}` | Mature reduction/broadcast/row-scaling machinery exists. The inspected backward trace has three kernels versus two in the retained Triton record; it also contains a one-external-output fusion veto. Changing that scheduling contract would enlarge this trial, so defer it. |

Tile `procedural/normalization.rs` is normalization of separable filter factors
with a guarded reciprocal; its NaN fallback is not our RMSNorm epsilon/rsqrt
contract. Tile softmax's row sums/scaling and `Tile::reduce_axis` are additional
composition references, not drop-in RMSNorm kernels. The inspected crate exports
do not establish a ready-made RMSNorm entrypoint. This is a bounded map of these
families and source scopes, not an exhaustive inventory of CubeK.

Historical `results.json` records already show one RMS inference kernel and
2.113µs for the selected FP32 case. The RMS backward FP32 64×1024 record has
three kernels/6.941µs versus configured-default Triton's two/4.706µs. These dated
measurements motivate the map only; they cannot establish a present speedup or
be mixed with fresh trial samples. No new timing was collected here.

## Exact reuse and adapter boundary

CubeK is `4ca0235e7e4fddd82f67ec27bbc3d48c085be25a`; it declares CubeCL
`80bb36bf2eb1791f47e75dc4893da3046580ade6`. Our compiler remains CubeCL
`6bba5fdf04206d9c3710dfbc46aa9850450ab405` with the retained FP patch.
The old `research.json`'s CubeK revision/dependency describe an earlier snapshot.
Reuse the isolated compatibility-copy procedure in
`stage5.reduction_codegen_setup`, whose exercised readers and Sum bodies are
unchanged. Do not edit upstream checkouts, upgrade dependencies, regenerate old
evidence in place, or introduce CubeK's allocator/runtime/autotuner.

The stock `GlobalFullPlaneReduce::reduce_single` is the loop/composition
reference. Its full public `execute` subsequently restricts the writer to lane
zero. RMS inference needs the reduced value in every lane for its output loop.
Compose its **public lower-level Reader/PlaneReader/Sum components** and use the
all-lane returned accumulator; no private-visibility patch or extra
shared-memory broadcast is necessary for this boundary. Do not copy a new
shuffle tree or implement a custom sum.

The single initial mechanism is scalar-vector-width1, a 32-lane plane per row,
eight independent planes per block, Lazy merge. Eight planes follows CubeK's
GPU `calculate_plane_count_per_cube` cap for these row counts; it is a starting
configuration, not a measured optimum. Use `ceil(B/8)` blocks. Whole idle planes
must take a plane-uniform branch before any virtual read; the tail reader uses
`BoundChecks::Branch` and +0. All active lanes execute each collective.

The adapter's virtual input represents the descriptor's **FP32 reduction
operand**, including its original input/storage cast chain and square. This is
essential for FP16/BF16: using a low-precision `P::EI` square would be a semantic
change. Use `P=(f32, Const<1>, f32)` and a metadata-only virtual `[B,1]` reduced
shape to supply `Reader::new`'s coordinate mapping. Its output metadata does
not authorize an extra global result allocation/store. After the stock plane
sum, each lane replays the existing epilogue and stores its disjoint
`r=lane+32*k<N` coordinates.

Reuse `stage0/src/descriptor2.rs::{emit,index}` unchanged for semantic replay
from a separate diagnostic crate (the module has no `crate::` dependencies).
Provide a narrow `VirtualTensorOperationsExpand` adapter calling that emitter;
bind validated row/column coordinates to the captured descriptor. This part is
**proposed, not yet compiled**. Require positive contiguous rank2 tensors,
disjoint output, pure one-sum RMS descriptors, bounded known shapes, queried
CUDA plane width32 and the original signed-i64 semantic indexing. Do not widen
this into arbitrary Inductor DAG composition. Reject unsupported descriptors
before emission/launch.

Expand actual components into `KernelBuilder` → existing `CppCompiler<Cuda>` →
CUDA13.0 NVRTC (`--fmad=false`) → PTX → current-stream native launcher. Preserve
PyTorch-owned allocations, metadata readiness and storage lifetime. The
existing `stage5/codegen_probe` proves this runtime-free component route, but
does not by itself prove this new adapter or generated-graph integration.

## Qualification and stop conditions

Before any changed GPU arithmetic, model the scalar FP32 order exactly on CPU:
`a[lane]=+0`; for ascending `k`, add the already-rounded FP32 square at
`32*k+lane`, or the reader's +0 for a tail; then simultaneous XOR butterfly
offsets **1,2,4,8,16**. Check all lanes, not just lane0. Do not reuse the frozen
final's descending-order qualification. Replay the original epilogue without
algebraic changes. CPU/GPU rsqrt agreement must be qualified separately; a CPU
sum model alone cannot establish full GPU special-value semantics.

Use the existing nine RMS corpus cases and numerical thresholds, plus finite
dynamic-range/skewed/alternating inputs, zero and negative zero, subnormals,
square overflow, NaN and signed infinities. Include boundary widths
31/32/33,255/256/257,1023/1024/1025,4095/4096/4099 and whole/partial row blocks.
If existing general reduction-corpus inputs are used, recompute references for
their **squared RMS operands**; old raw-sum reference bytes are not RMS references.
Apply the existing per-element reference-or-better policy against a matched
efficient reference for the same RMS inputs and the unchanged eager-relative
gates. Obtain those reference outputs before GPU candidate implementation; a
tolerance-only comparison is insufficient. Preserve all cast barriers and
special-value classes/signs. No coarse compensated-reducer guard or tolerance
relaxation. Failure stops this arithmetic mechanism before timing.

After the CPU gate, require GPU model/order checks, canaries, genuine alias and
invalid-layout rejection, offset views and a nondefault current stream. Reuse
the original generated descriptor in a diagnostic scheduling/compile adapter;
retain actual generated wrappers and prove the candidate is the executed body.
First compare the candidate kernel independently; then compare real generated
graphs against ordinary first-use selection in one process, rotating order,
matching inputs and recording raw samples. Keep reference runtime/GIL and
host/device timing scopes distinct. No synthetic warmup or memory primers.
Archive IR/CUDA/PTX and a production executed-SASS capture if measured results
justify attribution. Barrier/register counts are explanatory evidence, not the
success criterion.

Success means the **actual reused components execute** with qualified semantics
and a stable measured graph improvement on the selected workload; report
neighbor regressions. A qualified tie/loss is still a useful negative reuse
trial, not permission to invent a second mechanism in this experiment. Stop and
report a concrete adapter/API/semantic incompatibility if it requires a generic
bridge, runtime ownership change, changed guards or relaxed semantics. Keep
automatic eligibility/defaults unchanged. Frozen reduction files, cwk.5 and
Stage6 remain outside this trial.

## Evidence and reproducibility

Run from the repository root:

```sh
STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python stage5/artifacts/reuse-first-selection/capture.py
stage0/.venv/bin/python stage5/artifacts/reuse-first-selection/verify.py
```

`capture.py` captures only the existing baseline, no candidate, tuning or timing.
`source-evidence.json` is a fixed, manually assembled selection snapshot of the
inspected sources/anchors and historical records. `verify.py` reproduces the
source/hash/anchor checks, protected-source/HEAD and upstream-diff comparisons,
baseline artifact/integrity binding, and exact current corpus tolerance policy;
it writes `validation.json`. No new reference behavior is inferred from hashes.
The required [review](review/review.md) found no blockers; its evidence-harness
findings and repeated checks are recorded in [disposition](review/disposition.md).
The graph service's project/list/search/coverage calls returned **Transport
closed**; source findings use direct file reads and scoped text searches.
No graph generation/freshness/completeness claim is made.
