# 9e8 — packed storage loads at the RMS operand boundary

Current decisions live in [STAGE5_REPORT.md](../../../STAGE5_REPORT.md).
This is the source-first follow-up to closed [wqc](../reuse-first-rms-vector4/README.md),
not another implementation in that trial. Investigation date: 2026-10-18.

**Source-backed capability answer: a narrow adaptation is feasible.** Reuse the
existing descriptor5 packed storage load/extract/scalar replay pattern inside a
separate fixed-template RMS operand adapter. No new reduction, vector arithmetic,
generic DAG bridge, allocator, runtime, launcher or planner is required by this
proposal. This pass implements no GPU candidate and makes no speedup claim.

**Handoff status:** host/source checks pass. After review-mcp failed on provider
HTTP429 quota exhaustion, the user explicitly authorized native GPT-6.1 medium
review. It returned no blockers; parent source and host checks corroborate the
verdict. `9e8` is closed for this bounded capability investigation; follow-up2b9
is open and unstarted. See [native review](review/native-review.md) and the
preserved historical [failure record](review/failure.md).

## Where the scalar loads originate

The vector4 probe still registers scalar storage buffers using
`KernelBuilder::buffer(elem(storage_dtype))`. That is compatible with packed
access: the existing Stage 5 mechanism reinterprets a pointer at the desired
scalar element, preserving its address space. Changing the buffer ABI is not
necessary.

`ReplayExpand::__expand_read_method` for `VirtualTensorOperationsExpand<f32, Four>`
in `stage5/rms_vector_probe/src/main.rs:102` multiplies the reader position by4,
then invokes `descriptor2::emit` independently for components0..3. Each invocation
replays the scalar `load`, original casts and FP32 square, followed by
`CompositeConstructOp` of the four FP32 results. Thus the last representation
without a packed load is already our custom read adapter. `descriptor2.rs:328`
correctly emits `IndexOp` and scalar `LoadOp` for a scalar semantic load.

CubeK's `ParallelReader::read_plane` delegates through `ReaderBoundChecks` and
`View::read`. The plain view maps vector indices without changing their values;
the virtual tensor ultimately calls the adapter's read expansion. A vector
type/layout on that virtual value does not transform four scalar descriptor
loads into one storage-vector load. No CubeCL/NVRTC correctness defect has been
established.

| Boundary | Existing representation / observation | Evidence tier |
| --- | --- | --- |
| Frozen semantic template | One scalar input load; original scalar cast chain; FP32 multiply | Exact three templates, snapshotted |
| CubeK reader | `pos=32*k+lane`, vector offset `batch_offset+pos`, Branch with `pos<N/4` | Pinned source, exercised copy byte equality |
| wqc read expansion | Four scalar replay calls then FP32 vector construction | Current source plus retained wqc IR/CUDA |
| wqc CUDA13.0 PTX | Groups of four `ld.global.nc.f32`; no packed FP32 input load | Retained primary PTX; no new compilation |
| Existing descriptor5 mechanism | Storage-vector pointer/load; scalar extract; replay non-load semantic ops unchanged | Current source, not a new execution |
| Existing pointwise artifact | `float_4 const*` load lowers to `ld.global.nc.v4.f32` | Historical Stage 5 artifact, different kernel; capability precedent only |
| Proposed RMS adapter | Storage-vector load under same Branch, scalar extraction and replay | Source-feasible proposal; IR/PTX/SASS and execution unavailable |

The historical pointwise artifact does not prove the new RMS load survives
optimization or that FP16/BF16 will use one packed instruction. Its recorded
NVRTC flags include `--fmad=false` and compute86; it is not a fresh build.

## Mature mechanisms inspected and selected

- **CubeK ParallelReader/PlaneReader + Sum:** keep the actual wqc components,
  `BoundChecks::Branch`, Lazy accumulation, four FP32 component chains, stock
  ascending XOR shuffles and stock component fold. `Sum::null_input` is +0.
- **CubeCL VirtualTensor, View, PlainLayout and LinearViewLayout:** the existing
  view stack supports vector-indexed access but delegates the load to its
  backing implementation. Linear layouts can choose plain/strided/permuted
  mappings; this fixed dense template needs only the current plain mapping.
  Adding launch-time View/Tensor bindings is unnecessary and would expand scope.
- **Existing descriptor5 `reduction_vector_values`:** this is the closest reuse
  reference. It computes the original scalar address, reinterprets a pointer
  to a vector of the **storage dtype**, issues `LoadOp`, extracts each component,
  and calls unchanged `descriptor2::emit` for every non-load semantic operation.
  Its caller in `descriptor3.rs:787` supplies a whole-vector bounds branch.
  Its admission code rejects generic nested lazy regions and proves unit stride.
- **CubeCL IR/C++ machinery:** `PointerType`, `VectorType`, `ReinterpretCastOp`,
  `LoadOp`, `CompositeExtractOp` and `CompositeConstructOp` already express the
  required operations. `LoadOp` derives its result type from its pointer;
  C++ emission uses `reinterpret_cast` and a typed dereference. Vector alignment
  is scalar alignment times component count, emitted as `struct alignas(...)`.

Selected proposal: adapt that existing descriptor5 sequence locally in a new
isolated probe, specialized to the same exact three RMS templates. The current
probe includes only descriptor2. Importing the whole descriptor5 module would
also pull in descriptor3/4 and their unrelated build paths; refactoring the
production adapter is not necessary for this bounded trial. The proposal reuses
the same existing IR operations and replay sequence; it does not add inline PTX,
custom load instructions, a second reduction or a scalar fallback.

## One proposed adapter change

Within the existing valid-vector branch, compute the scalar position for
component0 using the existing signed-i64 descriptor index route. From that
scalar pointer, create an address-space-preserving pointer to four original
storage elements and load once. For c=0..3, extract the storage scalar, seed the
original load ID in a fresh values map, then replay **only the non-load prefix
operations** through unchanged descriptor2. Assemble the four resulting FP32
squares in the same c order and return them to the unchanged PlaneReader/Sum.

Seeding the map and replaying the entire prefix would be wrong: descriptor2
does not skip an operation merely because its ID already exists. It would emit
another scalar load. Likewise, this change must affect only the reduction input
read; the scalar output loop and its reload/epilogue remain the wqc behavior.

| Storage | Storage-vector width | Access size / required alignment | Returned operand |
| --- | ---: | ---: | --- |
| FP32 | 4 | 16 bytes | Four FP32 squares |
| FP16 | 4 | 8 bytes | Four FP32 squares |
| BF16 | 4 | 8 bytes | Four FP32 squares |

The 16-bit template's cast sequence remains storage → FP32 → storage → FP32 →
FP32 → FP32 square. Even apparently redundant casts are retained. A raw storage
load is not a low-precision multiply or a bit reinterpretation into FP32.
Widening FP16/BF16 to eight storage elements to reach128bits would change the
work assignment and is outside this proposal.

## Proof obligations and host verification

Keep the current exact templates and domain: rank2, dense `[B,N]` with strides
`[N,1]`, N in32/256/1024/4096, B in1..65, disjoint output, original dtype/device,
storage-span/current-stream/readiness/lifetime validation. The existing live
input pointer guard is modulo `4*element_size`, including tensor view offsets.
Allocation-base alignment alone is insufficient. Output remains scalar.

For a valid row r, CubeK computes `batch_offset=((r%B)*N)/4 = r*N/4`.
For lane l and chunk k, `pos=32*k+l`, so the proposed packed start is
`4*(batch_offset+pos) = r*N+128*k+4*l`. Component c therefore has exactly the
current scalar address `r*N+128*k+4*l+c` and exactly the existing component-chain
assignment. Since `pos<N/4` and N is divisible by4, all four addresses are in
the same row. Row bases preserve alignment. The current whole-row guard skips
invalid planes; lanes failing the inner Branch perform **no load**, receive
four +0 values and still participate in the later plane reduction.

Never move the load outside the Branch, replace it with Mask/clamping, admit
partial vectors or reinterpret unsigned wraparound as a valid signed address.
At B<=65 and N<=4096 every admitted scalar index is below266240, well within
signed-i64 bounds. Runtime sizes/pointers remain revalidated on every invocation.
Native compilation enforces exact semantic/plan identities; live pointer
alignment is a host validation obligation, not a compiler-known tensor fact.

`inspect.py` independently enumerates all260 admitted B/N pairs and verifies
11,600,160 scalar address identities, whole-vector bounds, skipped reader lanes
and row tails. Sixteen synthetic byte-address controls cover offset0..7 for
2/4-byte storage. Seven odd-width counterexamples show that the floor-chunk
reader would omit a remainder; a pitched-row counterexample loses alignment.
These are integer/model checks, **not calls to live GPU admission validators**.
The three exact template prefixes/cast chains and exercised CubeK source copies
are checked. Existing wqc evidence179 artifacts and13,708 previously tracked
regular-file targets are hash verified unchanged. Ten tracked directory links
are listed separately in provenance; mutable Beads/report/ignore files are
excluded. This pass does not rerun or claim the production regression suite.

## Separate implementation acceptance, not completed here

Tracked as `pytortch-backend-experiments-2b9`, open and unstarted.

A future single-candidate trial should retain the existing wqc CPU model and
reconfirm the complete36-case policy before writing the new GPU candidate.
There is no proposed arithmetic reassociation. New GPU qualification must prove
bitwise parity of extracted operands/reduction results and final outputs versus
the frozen wqc path on matched inputs, plus the existing full numerical and
NaN/Inf/signed-zero policy. Keep canaries, nondefault streams, aligned offsets,
misalignment/extent/layout/alias negatives and native template/envelope controls.
Do not weaken NaN-payload expectations silently if bitwise parity fails.

Use a distinct mechanism/entrypoint/manifest/cache identity describing storage
width4 separately from operand width4 and overall scalar output width1. Bind
the new source/compiler and prior gate identities. Keep all prior artifacts
and the frozen long-row partial unchanged. Compile through the same
CppCompiler<Cuda> → CUDA13.0 NVRTC → PTX → current-stream launcher.

Before timing, inspect new IR/CUDA/PTX/executed SASS: one storage-vector load in
the valid reader branch, exact scalar casts/squares, unchanged P4 accumulation,
shuffle/fold order and scalar output mapping. FP32 packed128bit and 16-bit
packed64bit access are the intended code-generation checks; packing is not
assumed. If that reuse path fails, stop and retain the failure rather than add
another mechanism. If it qualifies, measure isolated and actual generated
graphs in the established same-process rotated methodology against frozen wqc
and ordinary controls, retaining every raw round and device-state qualification.
Stop if actual graph benefit is absent. No tuning sweep, clock control, synthetic
primer, cwk.5 investigation, default promotion or Stage6 work follows implicitly.

## Reproduction and evidence limits

Run `python3 stage5/artifacts/reuse-first-rms-packed-load/inspect.py`.
It imports only the standard library, reads sources/artifacts and runs integer
checks. `--capture` was one-shot; normal verification cannot refresh snapshots
or preservation hashes. A failed initial capture encountered tracked directory
links and is preserved under `setup-failure/`; it never ran a compiler or GPU.

`sources.json` binds full source/artifact snapshots under `source/`.
`provenance.json` records HEAD c326854, CubeCL6bba5fdf with the unchanged retained
FP patch and clean CubeK4ca0235e. The exercised compatibility-copy bodies match
the inspected CubeK files. No upstream repository was modified.

The graph service is available. Its initial coverage generation was
2026-10-16T21:09:02Z, with stale/unavailable coverage for descriptor5 and the new
wqc sources. One stale snippet truncated the current helper; current source
lines488..551 were read directly. The graph's spurious artifact `aten::clone`
edge is not used as call evidence. The final coverage check refreshed to
2026-10-17T22:24:00Z and reports matching metadata without recorded issues for
the local implementation files. Both upstream source trees remain excluded;
IR/PTX and new packet files also lack code indexing. Direct source/artifact
reads supply material evidence, and full inspected files are snapshotted.
Graph query responses and coverage checks are retained here; no exhaustive
graph claim is made. The conclusion is source feasibility, not demonstrated
new lowering, execution safety, performance or automatic eligibility.
