# Stage 5: actual broadcast/strided layout composition

Current Stage 5 decisions live in [STAGE5_REPORT.md](../../../STAGE5_REPORT.md).
This is the bounded `s1a` trial, based on `8b96431`, on the display-disabled
RTX3080, CUDA13.0 production NVRTC and the pinned CubeCL checkout.

The existing layout components **can compose behind our boundary**. The one
candidate is numerically qualified on the declared surface. It does not establish
a generally faster replacement: it wins83/198 paired rounds against ordinary,
with regressions and sharply changing coarse clocks. Keep it explicit-only;
do not promote it, tune another candidate, narrow indices, or change the planner
in this pass. Required review completed; provenance findings are fixed and the
full qualification rerun passes, as detailed below.

## Component and boundary

The original corpus families are row broadcast `x[B,N]+y[N]`, column broadcast
`x[B,N]+y[B,1]`, stride2 slice `sin(x)+1`, and stride-zero expanded
`sin(x)+1`, in FP32/FP16/BF16. There are33 original cases and12 exact semantic
templates. The existing transpose-view result is not a materializing transpose;
this trial does not revisit that mechanism.

The native adapter invokes the actual public expanded constructors and methods:
CubeCL `LinearViewLayout`, `PlainLayout`, `PermutedLayout` and
`FastDivmod<usize>`. `PermutedLayout::to_source_pos` calls the actual upstream
`index_offset_contiguous_fastdivmod`; no device implementation is copied.
Plain layout handles literal-linear loads. Other loads normalize left-padded
broadcast dimensions to stride0 and use the upstream reverse-axis div/mod map.
The rank2 broadcast and stride-zero cases need the reference shape; the rank1
slice also fits PermutedLayout. The stock pitched StridedLayout host registration
assumes at least two dimensions; it is not the rank1 slice adapter used here.

Our adapter constructs these components inside the existing IR builder instead
of adopting CubeCL/CubeK allocation or launch ownership. It translates the private
host `find_params_u64` helper into Python because the public expanded component
accepts the parameters, while our existing InfoBuilder owns the scalar ABI.
Each reference axis adds the multiplier's signed-i64 bit representation and a
shift to that same immutable metadata buffer. Its cache key, readiness event,
current-stream wait, tensor `record_stream`, native launch and storage validation
remain in the original implementation. Buffers remain PyTorch-owned.

Assignment is unchanged scalar256, one active thread/output, same outer lazy
protected branch and scalar storage operations. Every non-load operation,
including cast barriers, arithmetic, sin and output indexing, goes through
unchanged `stage0/src/descriptor2.rs`. AddressIndex remainsU64, semantic indexing
signed-i64; no math relaxation, vectorization, new work distribution or reduction
change is included. Extra metadata sizes are80bytes for two-input rank2,
72bytes for one-input rank2, and56bytes for rank1. Runtime admission is restricted
to the exact12 templates and the declared scalar tuples, not a generic layout
implementation.

## Gates and chronology

1. [capture.json](capture.json) retains all33 original baseline generated-graph
   captures, layouts, hashes, typed semantic/physical packets and compiler identity.
2. [cpu.json](cpu.json) passed33 original cases/16,191,156 load-address identities
   **before the candidate GPU source existed**. The independent model compares
   the full upstream coordinate decomposition with literal descriptor indexing
   and checks each address against the original physical storage span.
3. [qualification-plan.json](qualification-plan.json) declares66 GPU cases:
   all originals plus31/32/33 neighbors where applicable.
   [cpu-neighbors.json](cpu-neighbors.json) covers all66/16,195,476 addresses.
   This extension ran after source creation but before any candidate GPU launch;
   an earlier attempt used system Python without NumPy and failed before calculation.
   That chronology is preserved, not rewritten as an earlier gate.
4. The physical domain is `0 <= dividend < 2^21`. `mul_hi <= dividend`, so
   `mul_hi + dividend < 2^22`: the upstream U64 FastDivmod addition cannot wrap.
   Host parameters are checked against exact independent quotient/remainder,
   including boundary dividend/divisor probes and signed-bit round trips.
5. [gpu.json](gpu.json):78 bitwise cases (66 finite cases plus12 special-value
   cases), existing pointwise tolerances, class/sign checks, unchanged inputs,
   canaries,90 invalid-live-input rejections before launch and12 nondefault-stream
   checks with offset8 storage views. [special-values.pt](special-values.pt)
   preserves the special inputs and actual/baseline/eager outputs.
6. [native-controls.json](native-controls.json):8 early-rejection controls for
   schema, entry, mechanism, plan, indexing, precision/index type, effects and
   boolean-vs-number semantic identity. No control reaches IR generation.

The first compiler attempt failed before GPU execution because child scopes were
still alive at CppCompiler's owned-context boundary. Releasing those scopes fixes
the adapter ownership. A subsequent harness attempt passed its first GPU comparison
but stopped while recording a missing artifact attribute. Both failures and the
failed compiler source are preserved under [failures](failures/). These were not
numerical failures or alternate candidates.

## Generated-wrapper measurements

[timing-plan.json](timing-plan.json) declares33 original graph cases,6 rounds,
all6 permutations of baseline/component/ordinary,16 calls/path/round.
Baseline is the retained scalar256 binary, checked by PTX hash; ordinary uses
the unchanged first-use selector; component uses the actual layout body.
The original Inductor wrapper collection is under [wrappers](wrappers/).
The post-review [wrapper inventory](wrapper-inventory.json) retains50 unique
cache-relative paths and hashes under `wrappers-by-path/`, with semantic/policy
bindings for all33 cases. It binds to the original timing-file hash. The original
profiling did not record the exact cache filename of each executed wrapper;
the supplement records matching generated sources, not invented execution-path
metadata. Future captures put the unique wrapper paths/hashes in `timing.json`.

[timing.json](timing.json), lossless [traces](traces/) and
[analysis.json](analysis.json) bind594 traces and9,504 CUPTI kernel durations
to exact entrypoints, input/output/PTX hashes, launch geometry and sample medians.
The analyzer rechecks the complete CPU map and retained special-value outputs.
These are GPU kernel durations for actual complete generated wrappers, not
host latency, queue-inclusive events, replay costs or full Triton comparisons.
No synthetic primer, runtime warmup change or clock/cache control was introduced.

The component wins83/198 rounds against ordinary and88/198 against matched
baseline. For example, FP32 row-broadcast3x255 pooled medians are
1.506/1.600/1.504us (baseline/component/ordinary); FP16 row-broadcast256x4099
is61.881/56.241/61.890us; FP16 column-broadcast256x4099 is
58.094/66.406/45.128us. All distributions, losing rounds and transitions remain.
Do not compare these different clock conditions as stable intrinsic kernel costs.

Coarse RTX3080 snapshots include SM/memory1710/9251MHz and210/405MHz, with
intermediate315/810,225/405 and255/405MHz. Most later snapshots are210/405MHz.
They bracket rounds, not individual target windows; they do not establish equal
clocks during each comparison. Restricted low-precision row-broadcast wins in
this capture do not justify promotion. The early abandoned timing attempt and
its plan, partial results, traces and source are retained in
`failures/first-timing/`; none is substituted into the admitted run.

That harness reset Dynamo between families, which changed the symbolic factor
order of `numel` and correctly failed exact descriptor admission. Matching the
original capture order without resets accepts the original descriptors. The
rejected semantic and isolated/sequence diagnostics are retained; no admission
rule or arithmetic was widened to bypass it.

## IR to actual executed code

Each baseline/candidate retains `cubecl.ir`, `kernel.cu`, production `kernel.ptx`,
`nvrtc.json`, driver resources and semantic/physical/identity packets. Timed
ordinary artifacts are copied under `timed-ordinary/`. The source/provenance
manifest binds the adapter and the relevant upstream components.

The exact PATH-resolved ncu binary/version/hash, matching section hashes,
environment and full command are in [ncu-command.json](ncu-command.json):

```text
ncu --target-processes all --profile-from-start off --replay-mode kernel
    --cache-control none --clock-control none --launch-count 12
    --section-folder <resolved-ncu-parent>/sections
    --section SourceCounters --section LaunchStats
    --metrics smsp__inst_executed.sum,launch__registers_per_thread
    --export stage5/artifacts/reuse-first-layout/layout-code --csv
    --log-file stage5/artifacts/reuse-first-layout/ncu-raw.csv
    stage0/.venv/bin/python -m stage5.layout_measure counter
```

The12 launches are four FP32 families in baseline/component/ordinary order.
Raw imported CSV [ncu-import.csv](ncu-import.csv) records, for IDs0/1,
`smsp__inst_executed.sum=1836352/1475640` and
`launch__registers_per_thread=24/18`. The actual loaded SASS and executed warp
counts are in [executed-sass.txt](executed-sass.txt); the full opcode ledger in
[sass-ledger.json](sass-ledger.json) sums exactly to every invocation's metric.
The `.ncu-repz` report is retained. NCU timing is not admitted as warm performance.

| FP32 target | Baseline -> component registers | Executed warp instructions | BSSY/BSYNC pairs |
| --- | --- | --- | --- |
| row broadcast256x4099 |24 ->18|1,836,352 ->1,475,640|32,792 ->0|
| column broadcast256x4099 |24 ->21|1,672,392 ->1,901,936|32,792 ->0|
| stride2 slice1048583 |22 ->20|1,671,303 ->2,326,683|32,769 ->32,769|
| stride-zero64x257 |22 ->21|37,600 ->40,678|1,028 ->514|

No executed local loads/stores appear in these12 captures. The baseline stride-zero
driver attribute still reports32 local bytes/thread; do not equate allocation
with executed spills. Neither path introduces shared memory or workgroup barriers.
Data loads/stores remain scalar. Component LDG instruction counts increase
because of metadata, not evidence of changed logical data requests: row broadcast
131,168 ->196,752; column131,168 ->262,336; slice98,314 ->196,621;
stride-zero1,548 ->3,604. These are executed warp instruction counts, not bytes.

The baseline CUDA expresses signed64 floor/mod semantics. NVRTC emits a guarded
U32 div/rem path when the operands' high words are zero, plus a signed64 fallback;
the retained SASS shows the reciprocal/integer path actually executing here.
The component replaces this with U64 mul-high/add/shift and removes the broadcast
division reconvergence. It also keeps the general layout decomposition:
column and stride-zero compute `(i/N)%B` instead of the already-bounded `i/N`;
slice computes `(i%N)*stride` instead of `i*stride`. Their extra coordinates and
metadata survive CUDA/PTX/SASS. The protected sin path retains its own control.
Lower register counts alone do not establish a win. This is a measured limitation
of this particular general-layout expression, not evidence that reuse is impossible.
No second candidate or U32 narrowing was attempted.

## Preservation, commands and review

`preservation.json` verifies15,011 pre-existing tracked regular files unchanged,
including production and retained reduction/transpose paths. The canonical report
and ignore-file originals are retained separately. Only new diagnostic sources,
this packet, one report row, narrow evidence ignore exceptions and Beads change.
No existing production regression suite is represented as rerun; this explicit-only
trial uses its scoped qualification gates. Stage6, eligibility and cwk.5 remain frozen.

```text
stage0/.venv/bin/python -m stage5.layout_trial
stage0/.venv/bin/python -m stage5.layout_model  # original gate, before GPU source
# CPU-neighbor extension chronology and declared plan are retained above.
CARGO_TARGET_DIR=.../stage5/cache/reuse-first-rms/target cargo build --release --offline --locked --manifest-path stage5/layout_probe/Cargo.toml
CUDA_VISIBLE_DEVICES=GPU-510ddc27-95ac-c62f-d49b-242cc5183c72 stage0/.venv/bin/python -m stage5.layout_qualify gpu
stage0/.venv/bin/python -m stage5.layout_qualify native
CUDA_VISIBLE_DEVICES=GPU-510ddc27-95ac-c62f-d49b-242cc5183c72 stage0/.venv/bin/python -m stage5.layout_measure graphs
stage0/.venv/bin/python -m stage5.layout_analyze
ruff check --select F stage5/layout*.py
cargo fmt --check --manifest-path stage5/layout_probe/Cargo.toml
git diff --check
```

Capture/CPU/GPU/timing phases intentionally refuse to overwrite completed packets;
the CPU first gate intentionally refuses to run after GPU source creation.
The analyzer is repeatable. Reproduction needs a fresh packet and the recorded
phase ordering, not deletion of old failures. Nothing was committed or pushed.

## Review disposition

Required [review-mcp](review.json) returned with no numerical/ownership blocker,
two medium provenance findings and bounded hardening observations. Both medium
findings are addressed; see [the complete disposition](review-disposition.md).
Native review remains disabled.

The final adapter retains the shared `compiler_identity.adapter` digest and adds
a distinct `layout_adapter` digest; it also validates the CPU gate's capture hash
before compilation. Corrected identities produce different entrypoint names.
The native emitter/binary, mapping, math, precision, assignment and ownership
remain unchanged. [post-review-gpu.json](post-review-gpu.json) reruns all78 cases,
90 live controls and12 stream checks successfully. The analyzer verifies all12
PTX bodies are byte-identical after replacing their entrypoint names; all78 input,
output, scalar and classification records equal the original qualification.
The original GPU/timing/NCU packets and phase snapshots are preserved; final
source snapshots are separately hash-bound in `post-review-sources.json`.

The counter path now loads the already-qualified PTX without invoking the probe
or NVRTC. A focused rerun verifies all156 original candidate files unchanged
([counter-readonly.json](counter-readonly.json)). No performance numbers from
that rerun are used. SASS subgroup matching now uses opcode boundaries.
The final audit still verifies9,504 original CUPTI durations, all16,195,476 CPU
address identities and15,011 frozen files, plus the post-review identity/PTX,
wrapper and read-only loading checks. Original analysis outputs are retained
under `pre-review-analysis/`. No second GPU mechanism or timing sweep was added.
