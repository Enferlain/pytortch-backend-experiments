# l1m — proved column coordinate, existing layout components

Current decisions: [STAGE5_REPORT.md](../../../STAGE5_REPORT.md).
This is one user-authorized follow-up to the qualified
[s1a generic layout trial](../reuse-first-layout/README.md), based on `c5457e92`.
Only column broadcast is specialized. No reduction, production/default,
precision, split, runtime warmup, vectorization or planner changes are included.

## Result

The experiment supports the proposed direction on this bounded surface:
**an already-proved coordinate can bypass redundant generic reconstruction while
still using actual CubeCL components**. It passes the full existing numerical and
runtime gate. The redundant `%B` work and associated metadata loads disappear
through executed machine code, rather than merely looking simpler in source.

The12 actual column graphs win68/72 paired rounds versus frozen generic and43/72
versus ordinary. Low-precision cases show clearer benefits; FP32 and the large
FP16 graph still trail ordinary. Coarse clocks vary substantially, so these
captures do not establish a universal or clock-normalized intrinsic speedup.
Keep the candidate explicit-only. This pass adds no second GPU mechanism or
automatic eligibility. Required review status is recorded below.

## Proof and actual component selection

Inside the original protected region, positive `B,N` and `0 <= i < B*N` imply
`q = floor(i/N) < B`. Thus `q%B == q`. The column input has shape `[B,1]`,
stride `[1,1]` and relative offset0, so its scalar physical address is simply `q`.

The initial plan named `FixedDimLayout`. Source/type-check trials established
that its `IntoDyn` interface accepts tuples of2..12 components, and converts
coordinates through `Sequence<i32>`. It could only be used here with an internal
coordinate narrowing, which would mix another mechanism into this trial.
The failed scalar/tuple1 compiler attempts and their sources are retained in
`failures/`. Neither produced or executed a GPU candidate.

The selected implementation instead invokes actual existing
`FastDivmod<usize>::div` and `PlainLayout::to_source_pos`. PlainLayout is the
identity map for an already-valid scalar position and needs no IntoDyn conversion.
It is already used in the original adapter for literal-linear input loads.
This is a component/API correction for the same proved address transformation,
not a new division algorithm, custom layout implementation or candidate sweep.

The authoritative [CPU gate](cpu.json) passed all66 existing finite/neighbor cases,
16,195,476 load-address identities, before GPU source creation. Of these,
3,249,687 column-input addresses are specialized. It compares against literal
descriptor indexing and the frozen generic coordinate map under the original
U64 FastDivmod no-wrap domain. `preliminary-cpu.json/log` retain an earlier passing
run before adding a non-model fingerprint helper; the authoritative run also
preceded GPU source. Its exact source is `cpu-source.py`.

After that gate, the only model-file edit marks public imports as re-exports;
the runtime checks complete normalized Python AST equality against the snapshot.
The [API-specific CPU gate](cpu-component-api.json) checks all21 declared column
cases against the selected PlainLayout identity before the first candidate GPU
launch. Its chronology is explicit: the source/type API investigation had begun.

## Frozen contracts and qualification

The exact3 captured column semantic templates are admitted by Python and native
code. Scalar tuples come from the same predeclared qualification plan. The
original signed-i64 semantic evaluation and U64 physical addresses remain;
there is no new coordinate narrowing. The scalar256 work assignment, protected
branch and non-load descriptor replay are byte-identical in the Rust source
sections checked by the analyzer. The [emitter diff](emitter.diff) shows the
bounded change and the entry/identity/admission plumbing.

The existing metadata ABI remains7 i64 scalars/80bytes. The runtime inherits
the frozen `GenericKernel._validate` unchanged, retaining layout/span/alias
checks and original extra scalar serialization. The existing PyTorch-owned
storage, immutable metadata events, current-stream waits, allocator lifetimes
and native launches remain on the Stage0/2/5 path.

Candidate identities keep the shared adapter digest, separately name the column
adapter, and bind CPU/API gates, native binary/emitter and semantic emitter.
Cached candidate directories validate all retained file hashes and load existing
PTX without regeneration; incomplete directories fail visibly. The frozen generic
path directly loads retained post-review s1a PTX, never its compiler.

[gpu.json](gpu.json) passes all78 existing cases bit-for-bit against frozen
generic, with unchanged eager-relative tolerances, special-value classes/zero
signs, input hashes and storage canaries.24 cases exercise the specialized column
body;54 use frozen row/slice/stride-zero PTX.90 invalid-live-input rejections
occur before launch, and12 offset/nondefault-stream checks pass.
[special-values.pt](special-values.pt) preserves all12 special inputs and outputs;
[native-controls.json](native-controls.json) retains8 early rejection controls.

A harness import-order failure stopped before GPU execution and is retained in
`failures/import-order.log`. No numerical failure was removed or weakened.

## Generated graphs and timing limits

[timing-plan.json](timing-plan.json) was declared in the CPU phase:12 original
column cases (four shapes in each of FP32/FP16/BF16),6 complete permutations of
generic/component/ordinary,16 calls per path/round. All paths execute one actual
generated-wrapper kernel. Generic and component have the same scalar256 plan;
ordinary retains its unchanged first-use selection, including existing alternatives.

Original33-case descriptor construction order is replayed during setup so
Inductor retains the exact captured symbolic descriptors. Non-column component
and generic setup calls load frozen PTX and are not timed. Existing setup calls
are temporarily observed to record the exact generated wrapper caller filename;
the observer is removed during timing. Every case/path binds a cache-relative
retained wrapper and SHA256, not a post-hoc matching-source heuristic.

[timing.json](timing.json),216 lossless traces under `traces/`, and
[analysis.json](analysis.json) bind3,456 CUPTI kernel durations to names, geometry,
input/output/PTX hashes and actual wrapper sources. These are GPU kernel durations
of complete generated wrappers, not host latency, queue-inclusive timing, NCU
replay costs or a full Triton comparison. No synthetic primer, new warmup or
clock/cache control is introduced.

| Case | Generic / specialized / ordinary pooled median, us | Specialized wins vs generic / ordinary |
| --- | --- | --- |
| FP32 256x4099 |406.622 /405.889 /398.658|4/6 /0/6|
| FP16 256x4099 |66.402 /53.950 /45.118|6/6 /0/6|
| BF16 256x4099 |67.269 /55.989 /60.036|6/6 /6/6|
| FP16 64x257 |9.472 /9.173 /9.605|6/6 /6/6|

All12 cases, all rounds, losses and transitions are retained. Coarse RTX3080
SM/memory snapshots include1710/9251MHz,210/405MHz and intermediate states.
The first FP32 case crosses states; its pooled median is not a stable intrinsic
cost. Snapshots bracket rounds rather than individual target windows. They do
not establish equal clocks per path or exclusive device ownership. The large
low-precision improvements versus generic are observed in every paired round;
do not turn that observation into an unconditional production speedup claim.

## IR to executed SASS

The first arithmetic difference is at component expansion: the constructed IR
has one `math.u_mul_hi` versus generic's two. Both raw IRs still read all7 scalar
slots; unused metadata disappears during compilation before emitted CUDA. This
trial selects a simpler existing component using our proof. It does not add or
demonstrate automatic symbolic-bound inference inside CubeCL's optimizer.

The unchanged80-byte metadata ABI is optimized within the shader: generated CUDA reads
only scalar slots0,5,6 (numel and the N multiplier/shift), versus generic slots
0,2,3,4,5,6. The generic second quotient/remainder on B and its reconstruction
are absent. Actual U64 mul-high/add/shift performs the remaining quotient.
This is not a claim about a literal SASS division-instruction count.
All3 dtypes retain their original data arithmetic and storage casts.

Full tool path/version/hash, matching section hashes, exact argv and device UUID
are in [ncu-command.json](ncu-command.json). The command uses:

```text
ncu --target-processes all --profile-from-start off --replay-mode kernel
    --cache-control none --clock-control none --launch-count 9
    --section-folder <resolved-ncu-parent>/sections
    --section SourceCounters --section LaunchStats
    --metrics smsp__inst_executed.sum,launch__registers_per_thread
    --export stage5/artifacts/column-layout/column-code --csv
    --log-file stage5/artifacts/column-layout/ncu-raw.csv
    stage0/.venv/bin/python -m stage5.column_layout_measure counter
```

The9 launches follow the predeclared FP32 shapes3x255,64x257,256x4099, each in
generic/specialized/ordinary order. Raw [ncu-import.csv](ncu-import.csv), IDs6/7,
record `smsp__inst_executed.sum=1901936/1246096` and
`launch__registers_per_thread=21/16`. The primary comparison therefore has34.5%
fewer executed warp instructions and registers21→16. The opcode ledger
[sass-ledger.json](sass-ledger.json) sums exactly to each invocation's metric.

| FP32 shape | Generic / specialized / ordinary warp instructions | Global-load warp instructions |
| --- | --- | --- |
|3x255|1,392 /912 /1,224|192 /120 /96|
|64x257|29,884 /19,604 /26,286|4,118 /2,576 /2,062|
|256x4099|1,901,936 /1,246,096 /1,672,392|262,336 /163,960 /131,168|

Generic and specialized execute no BSSY/BSYNC in these captures; ordinary retains
its division-path reconvergence. All9 execute no local loads/stores. Data reads
and stores remain scalar; lower LDG counts reflect removed metadata instruction
work, not fewer logical tensor elements or measured DRAM byte savings. No shared
memory or workgroup synchronization is introduced. The actual driver SASS and
`.ncu-repz` are retained; NCU timings are not admitted as warm performance.

## Reproduction and review

Sources and hashes are under `sources/` and `sources.json`; the exact timing-phase
source is retained separately because a later edit only removes an unused import.
The analyzer checks the timed `graphs` function AST against current source.
Timed ordinary artifacts are retained under `timed-ordinary/`.
`preservation.json` verifies16,514 earlier regular files, including the entire
s1a packet and all production/reduction/transpose paths. The report/ignore originals
are separately retained; the current changes add only this diagnostic, its packet,
one canonical report row, narrow ignore exceptions and Beads records.

```text
stage0/.venv/bin/python -m stage5.column_layout_trial  # before GPU source
# API-specific gate chronology is retained in cpu-component-api.json.
CARGO_TARGET_DIR=.../stage5/cache/reuse-first-rms/target cargo build --release --offline --locked --manifest-path stage5/column_layout_probe/Cargo.toml
CUDA_VISIBLE_DEVICES=GPU-510ddc27-95ac-c62f-d49b-242cc5183c72 stage0/.venv/bin/python -m stage5.column_layout_qualify gpu
CUDA_VISIBLE_DEVICES=GPU-510ddc27-95ac-c62f-d49b-242cc5183c72 stage0/.venv/bin/python -m stage5.column_layout_qualify native
CUDA_VISIBLE_DEVICES=GPU-510ddc27-95ac-c62f-d49b-242cc5183c72 stage0/.venv/bin/python -m stage5.column_layout_measure graphs
CUDA_VISIBLE_DEVICES=GPU-510ddc27-95ac-c62f-d49b-242cc5183c72 stage0/.venv/bin/python -m stage5.column_layout_analyze
ruff check --select F stage5/column_layout*.py
cargo fmt --check --manifest-path stage5/column_layout_probe/Cargo.toml
git diff --check
```

CPU/GPU/timing generation refuses to overwrite completed phase packets; the CPU
gate refuses to run after GPU source creation. Reproduction needs a fresh packet
with the recorded phase order. The analyzer is repeatable. No existing production
suite is claimed as rerun for this explicit-only diagnostic.

Required review-mcp is blocked, with no review verdict. The original pending
request could not be recovered after a session refresh (`exec cell 220 not
found`); it produced no retained result. The resubmitted request returned
HTTP429/code1308: five-hour GLM quota exhausted, with provider-reported reset
`2026-10-18 17:17:39` (timezone unspecified). The raw result is [review.json](review.json).
Its transport-level `isError: false` does not mean that review succeeded; the
result explicitly reports the model API failure. Native review remains disabled,
so the review workflow stops here and `l1m` remains open. Review was requested
for descriptor/admission and ownership guarantees, including caller-mutation
isolation, as well as evidence provenance and conclusions. These have no
independent review verdict yet. No tested source or capture was changed while
waiting. No commit, push or remote sync.
