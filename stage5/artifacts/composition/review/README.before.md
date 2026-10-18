# 68x — bounded composition of qualified RMS and column kernels

Current decisions: [STAGE5_REPORT.md](../../../STAGE5_REPORT.md).
Base commit `a2b4e4b4`; explicit diagnostic only. No production, eligibility,
planner, split, precision, warmup, hardware-state or Stage6 changes.

## Result

**The compatible components compose correctly, but enabling both loses all72
paired rounds against ordinary across the nine declared mixed graphs.**
The low-precision column improvement survives composition; the packed RMS cost
is much larger. Combining them does not recover that loss in this capture.
Column-only wins all48 low-precision paired rounds against ordinary. FP32 column
reuse wins2/24 rounds, while its three pooled medians are slower. Both beats
RMS-only49/72 times and column-only0/72.

The primary64-row captures have these median GPU kernel-duration sums in µs:

| Storage | Ordinary | RMS only | Column only | Both |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 22.994 | 31.005 | 23.276 | 31.256 |
| FP16 | 24.779 | 39.074 | 24.306 | 38.686 |
| BF16 | 24.689 | 39.812 | 24.163 | 39.307 |

All shapes/rounds remain in [timing.json](timing.json), with a complete
[table](timing-table.md) and audited per-kernel breakdown in [analysis.json](analysis.json).
These are the two actual kernel durations added per wrapper invocation; host
dispatch and inter-launch gaps are outside this timing scope. Coarse GPU snapshots
all report210MHz SM/405MHz memory. They do not establish each kernel's exact
clocks, exclusive GPU use, or performance at other execution states. No clock
control, synthetic primer, runtime-warmup patch or cwk.5 investigation was used.

## Which mechanisms can be combined

| Retained mechanisms | Relationship on this surface |
| --- | --- |
| Scalar CubeK RMS, operand-vector4 RMS, packed-storage RMS | Successive alternatives; packed2b9 already contains the actual vector4 reader/reducer and storage-load composition. Use its frozen qualified body once. |
| Generic pointwise layout and bounded column mapping | Alternatives for the same scalar256 kernel. Use the frozen qualified column body. |
| Packed RMS and column broadcast | Different kernels can coexist naturally in the two-output full graph tested here. |
| Uniform-loop long-row partial/grouped loads/reference work assignment | Already composed and frozen. These short unsplit RMS descriptors do not invoke that long-row path. |
| Constant9 owner-only final | Existing ordinary eligibility remains; these RMS graphs have no nine-partial final to replace. No split is introduced to create one. |
| Transpose component | Applies to explicit materialization; no such operation exists in this workload. |
| Pointwise vector plans and scalar256 column component | A fused within-kernel combination would require its own proof and qualification; it is not produced here. |

The selected workload is `return rms(r), x + bias`, with independent input
storages. Ordinary Inductor scheduling naturally emits two kernels. The RMS
inputs are exact retained reference inputs at16/64/65×1024; column shapes are
3×255,64×256,64×257. Each pair runs FP32/FP16/BF16. No graph breaks, fusion veto,
artificial barrier or intermediate materialization is inserted.

The natural serial `rms(x + bias)` fit control fuses the operations into a new
reduction descriptor. Existing exact RMS/column adapters do not admit it.
[Preflight](preflight.json) retains the fused descriptor and actual ordinary
selection. A qualified serial/fused component adapter remains an open capability;
this two-output result cannot answer its performance question. This pass does
not establish the behavior of every possible backend combination.

## Existing mechanisms and frozen contracts

The new loader translates the existing `rms_packed_frozen_runtime` read-only
PTX-loading pattern to the packed RMS kernel. It inherits the exact existing
PackedKernel and ColumnKernel `_validate` methods, using the existing
PhysicalKernel construction with `compile_ptx=False`. The original RMS constructor
always compiles, so invoking it directly would rewrite its retained packet.
The narrow loader follows the already-existing frozen-control solution instead.

All six component shader sets are copied byte-identically: IR, CUDA, PTX,
manifest, physical plan, NVRTC settings and shader semantic descriptor. No new
component GPU source or compilation is introduced. The incoming mixed-graph
descriptor is kept intact in the runtime object. PyTorch still owns pointers,
output allocation/current stream, metadata readiness and allocator lifetime.
Original scalar/plane assignment, arithmetic, storage casts, protected loads,
vector alignment, ownership and conservative FP policy remain unchanged.

The ordinary path uses unchanged `first_use` selection. Each mixed mode retains
the same scheduling/fusion machinery. The ablation census proves that ordinary
branches are the same selected bodies in ordinary/RMS-only/column-only, and that
both uses exactly the bodies observed in the corresponding single substitution.

## Extent-spelling boundary and preserved failures

The first full mixed-wrapper gate stopped before timing because the column
descriptor spelled its extent `N*B` where the retained template spelled `B*N`.
Every other descriptor field was identical. The original failure, raw descriptor,
tree diff and source are retained in `failures/`, [admission-failure.json](admission-failure.json)
and [admission-diff.json](admission-diff.json).

An explicit CPU proof checked all21 qualified positive/bounded signed-i64
column scalar tuples before extending the integration loader. It now admits
exactly the original complete column template and its reversed `numel` factors;
all other fields still require complete equality. The frozen adapters remain
untouched. This is a fixed two-form admission bridge, with no general expression
normalizer, scalar-slot remap or descriptor rewrite. The incoming multiplication
order is retained in runtime validation. [Proof](extent-alias-cpu.json).
No case, tolerance, numerical requirement or timing order was dropped or changed.
No additional semantic alias is introduced.

The original preliminary fit run imported a historical helper after selecting
the new cache, which reset its cache variable. Its source/result/log are retained
under `preliminary/`. The corrected authoritative fit imports dependencies first
and asserts the observed actual wrapper filename is under the isolated
composition cache. Preliminary results are not timing evidence. There were no
removed numerical failures or favorable timing-prefix replacements.

## Qualification and provenance

- Before the integration loader source existed: all36 existing RMS CPU cases
  pass the unchanged corresponding-reference/no-worse-FP64 per-element policy;
  all21 existing column address cases pass. `cpu-source.py` preserves that producer.
- The frozen loader passes all36 RMS cases and24 column finite/special cases
  byte-identically to prior qualified outputs, including input hashes and canaries.
  All33 live guards, six current-stream/offset controls and six caller-descriptor
  mutation/cache checks pass. [Gate](gpu.json). The pre-alias loader gate is retained.
- After the bounded extent bridge, all24 column cases pass again under the
  alternate descriptor spelling, preserving the incoming descriptor. Nine
  unqualified changes to extent operation, arithmetic or offsets are rejected.
  [Alias gate](alias-gpu.json).
- All nine complete generated graphs qualify across four modes on normal and
  special inputs:72 calls/144 outputs before timing. Ordinary RMS values and
  special column values/inputs are retained in `graph-values.pt`; component
  results also bind to the original full-corpus hashes. [Graph gate](graphs-gpu.json).
- Eight predeclared order-balanced rounds per graph, four paths,16 invocations,
  two kernels/invocation produce288 compressed raw traces and9,216 kernel
  durations. Every trace has the expected exact two-kernel sequence, geometry
  and bound PTX identities; no setup observer remains in the timing path.
- Actual executed wrapper filenames/hashes are recorded during setup. Copies,
  complete ordinary selected artifacts and all six frozen shader sets are
  retained. `sources.json` binds current sources; phase snapshots bind CPU and
  direct-GPU producers separately. The offline auditor validates these phases,
  per-element policy, raw trace sums/wins and16,972 prior file hashes.

No new SASS/counter collection was needed to compare unchanged component PTX.
The retained component packets contain their prior executed-SASS evidence; this
pass makes no new machine-instruction-count claim. It measures actual selected
kernel sequences and reports their separate durations.

## Reproduction and status

The preparation/CPU phases are chronological one-shot gates. Use a fresh packet
and cache for regeneration; do not overwrite this or historical packets.

```text
python -m stage5.composition_trial prepare
python -m stage5.composition_trial preflight
python -m stage5.composition_trial cpu
# The recorded extent proof precedes the exact alias-loader adaptation.
python -m stage5.composition_run gpu
python -m stage5.composition_alias_check
python -m stage5.composition_run graphs
python -m stage5.composition_analyze
```

The tested interpreter is `stage0/.venv/bin/python`, with Triton excluded and
GIL disabled. Set `CUDA_VISIBLE_DEVICES=GPU-510ddc27-95ac-c62f-d49b-242cc5183c72`.
[environment.json](environment.json) records actual Torch/CUDA/compiler/driver
identity. Production compilation stays CUDA13.0 NVRTC. Existing production
regression suites are not claimed rerun for this explicit diagnostic; prior
production sources and artifacts remain covered by the preservation ledger.

Retain the bounded negative combined result; keep default eligibility unchanged.
Required independent review is pending. No commit, push or remote sync.
