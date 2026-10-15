# Frozen reference512 code-generation investigation

Current decisions belong in [STAGE5_REPORT.md](../../../../STAGE5_REPORT.md).
The user rejected the previous stop recommendation. Stage 5 remains active;
Stage 6, automatic eligibility and qualification/precision policies are unchanged.

## Retained change

`reference512_diagnostic` now expresses the existing four-chain work as a
**workgroup-uniform chunk loop**, following CubeK's reader mechanism. For chunk
base `q=0,2048,...<E`, thread `t` still owns `q+t+512*j`, `j=0..3`. All four
protected pending loads precede accumulator updates. An update guard `q+t<E`
preserves even the original thread-dependent loop's inactive-update set.
The merge, decreasing XOR shuffles, shared exchange, root publication, owner,
nine-way logical split and fixed final are unchanged. Semantic coordinates
remain signed i64. No new load intrinsic or inline assembly was needed.

Four mechanisms were qualified and measured separately before combination:

| Probe | Registers | 8×1048576 partial µs | 64×65539 partial µs | Decision |
| --- | ---: | ---: | ---: | --- |
| Original frozen binary | 48 | 72.65 | 26.67 | Retained as byte-exact control |
| Uniform chunk loop | 40 | 41.60 | 21.51 | Retained in the existing diagnostic |
| Owner-local result, one barrier | 48 | 73.59 | 26.63 | Safe on proved owner-only scope; no useful isolated win |
| Bounded u32 physical loop counter | 36 | 44.10 | 22.07 | Useful attribution, slower than uniform loop |
| Uniform loop plus counter narrowing | 46 | 41.71 | 22.77 | Secondary regression; combination rejected |
| Efficient retained Triton | 29 | 44.78 | 21.32 | Same-process reference in uniform-loop trial |

The original-binary and Triton controls were remeasured within each trial;
`analysis.json` retains those separate distributions. The table's two control
rows quote the uniform-loop trial. Three rotated rounds each contain 384 warm
CUPTI kernel events per path/case. These are prepared-partial durations; final
execution, host intervals and cold Nsight replay are excluded. Do not treat the
nearby secondary medians as a broad speedup claim over Triton.

The 48→40 register change admits three rather than two resident 512-thread
blocks per RTX3090 SM. At the primary case's 72 blocks, that residency limit alone
cannot explain the speedup: the within-block code expression also changed.
Fewer registers alone are not a tuning objective: the isolated 36-register
counter probe loses to the retained 40-register expression; their combination
allocates 46 registers and loses residency again.

## Existing mechanisms inspected and executed

The local CubeK commit is `4ca0235e7e4fddd82f67ec27bbc3d48c085be25a`; its guide
and reduce subtree match the earlier `a680b701` revision. CubeCL is the existing
`6bba5fdf04206d9c3710dfbc46aa9850450ab405` checkout with conservative FP hardening.
CubeK normally pins a different CubeCL revision. The original attempted local
build failure is retained in `cubek-original-build.log`.

`stage5.reduction_codegen_setup` creates an isolated source copy, redirects its
CubeCL dependencies to our checkout, adapts five unrelated tile routines to the
local f32 constructor API, and exposes the private bounds-reader module for this
probe. `cubek-compatibility.patch` records every adjustment. Initial unexecuted
reader-setup artifacts under `cubek-reader-*` and the initial-error log are retained;
the executed controls are the `cubek-branch`, `cubek-mask`, `cubek-cube-*` and
`cubek-plane-branch` packets used by the current harness. The exercised
reader, Sum instruction, virtual-tensor adapter and Plane/Cube reducer bodies
are unchanged. Neither upstream source checkout is modified by this setup.

The separate Rust probe expands the actual `ReaderBoundChecks::Branch` and
`::Mask`, `TensorArgs`/`init_tensors`, `reduce_kernel_virtual` and Plane/Cube
components into KernelBuilder IR, then uses the same `CppCompiler<Cuda>`,
CUDA13.0 NVRTC (`--fmad=false`) and CUDA driver. Device pointers, metadata tensors
and streams belong to PyTorch. No CubeK Client, allocator, autotuner or launcher
is instantiated. `metadata-checks.json` verifies ten diagnostic metadata packets
against CubeCL's own InfoBuilder; the runner now uses InfoBuilder directly.

For the reader/codegen controls only, input rows are padded into `[B*9,E]` with
`E=ceil(N/9)`. This adds 40 zero elements in the primary case and 512 in the
secondary case. Each reader keeps four private chains and writes each thread's
private merge. The Cube component deliberately uses 512 units `(32,16)` to
compare code generation at the same cooperation, bypassing the host Routine;
normal GPU Routine inference caps at eight planes. The Plane control uses
`(32,8)`, eight separate output rows per block. These controls have different
arithmetic/output work and are **not frozen-topology replacement candidates**.
Their eager-relative checks are harness sanity checks, not qualification under
our per-element reduction policy.

Prepared primary medians are about 42.4 µs for the Branch reader, 42.3 for Mask,
46.2 for the Cube sum and 304 for the Plane sum. The Plane control has only nine
blocks and much longer per-thread chains; that duration cannot be attributed to
its instruction syntax alone. Actual register counts are 40/52 for the readers,
34/36 for Cube Branch/Mask, and 36 for Plane Branch, all without local memory.
`comparisons.json`, the generated IR/CUDA/PTX and both comparison SASS captures
retain the complete evidence.

## Bounds, load predication and reconvergence

Existing CubeCL `core/src/io.rs::{read_masked,read_checked}`, checked-pointer
postprocessing, memory `LoadOp`, asynchronous checked shared copies and the
inline-assembly facility were inspected. The checked/masked scalar helpers
redirect reads to a valid address and select a fallback; they do not guarantee
non-evaluation. Checked asynchronous copies stage into shared memory and do not
fit the frozen scalar accumulation boundary. Ordinary scalar LoadOp has no
predicate operand. The existing branch representation already lowers to
predicated scalar reads when its surrounding loop is expressed appropriately.
A new target-specific load mechanism is therefore unnecessary for this fix.

Executed SASS confirms the difference between CubeK's two reader modes:

| Input case / padded reader rows | Branch active scalar-load threads | Mask active scalar-load threads |
| --- | ---: | ---: |
| 8×1048576 | 8,388,648 | 8,404,992 |
| 64×65539 | 4,195,008 | 4,718,592 |

These are predicated-on thread instruction counts for 32-bit input LDG, excluding
64-bit metadata loads. `input-load-counts.json` and the source-view SASS retain
the raw counts. Mask issues reads for inactive coordinates to index zero; Branch
suppresses those accesses. Similar warm durations do not make Mask suitable for
our lazy-access contract or effectful generated producers. No mask substitution
is installed.

The boundary trace is:

1. Original physical IR RangeLoop starts at thread ID. CUDA uses
   `for(int64_t r=t; r<E; r+=2048)` around grouped protected loads and updates.
2. NVRTC PTX retains per-thread trip conditions and peels/unrolls that loop.
   The actual driver-JIT SASS uses two reconvergence regions for loop-control
   paths. The input reads themselves are already predicated scalar LDG.
3. CubeK's uniform `i=0..ceil(length/CUBE_DIM)` reader controls emit predicated
   Branch loads with no BSSY/BSYNC. The retained Triton uniform chunk loop likewise
   has predicated scalar reads without those regions.
4. Translating the uniform chunk expression, while preserving each original
   chain/update guard, eliminates all executed BSSY/BSYNC in our FP32 component.
   Load count, shuffle count and both existing publication barriers remain fixed.

Representative executed warp counts from `mechanisms-sass.txt`:

| 8×1048576 | BSSY | BSYNC | BAR | SHFL | Active input-load threads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen | 2304 | 2304 | 2304 | 6048 | 8,388,608 |
| Uniform | 0 | 0 | 2304 | 6048 | 8,388,608 |
| Owner-local only | 3456 | 3456 | 1152 | 6048 | 8,388,608 |
| Narrow only | 1152 | 1152 | 2304 | 6048 | 8,388,608 |
| Triton | 0 | 0 | 1152 | 10368 | 8,388,608 |

The secondary capture independently confirms zero reconvergence for Uniform and
exactly 4,194,496 active input-load threads for every frozen-topology mechanism.
The owner-local probe's extra reconvergence shows why removing a barrier need
not improve machine control flow. These counter changes are evidence of the
mechanism, not an additive latency decomposition.

## Ownership, integer cost and composition boundary

The codegen controls also expose an ownership difference. CubeK's plane reducer
commits from lane zero of each plane. Its plane-enabled cube reducer uses
`worker_pos=UNIT_POS_Y`: every lane of plane zero scans the shared plane sums
and can reach the writer, while equivalent lanes publish each plane's shared
slot. Our frozen component instead publishes one shared value per warp and
stores the partial only from unit zero. Triton replicates the inter-warp merge
across warps and likewise predicates the final store to its owner. These
patterns cannot be interchanged for an arbitrary generated sink: repeated
writer invocation could duplicate epilogue effects.

CubeK's generic `plane_sum` lowers through CubeCL's ascending-offset polyfill;
our qualified body uses literal descending XOR masks. That arithmetic difference
prevents direct reuse of the stock collective for the frozen topology. Both use
the existing CubeCL shuffle lowering and have zero local-memory spill bytes in
these captures. CUDA local variable verbosity alone is not evidence of spills.

The owner-only partial kernel does not need to publish its result back to every
unit. The isolated owner probe keeps the first-plane result in a local SSA value
and omits root shared publication/reload and its barrier. It admits exactly one
reduction and only outer/owner stores. It rejects multiple reductions or
reduction-domain consumers. Existing broadcast and scratch-reuse barriers remain
in normal fused paths. CubeK's final cube barrier instead protects its next
writer iteration from overwriting shared values still being scanned; it is not
a generally removable root-broadcast barrier. The safe owner probe was not
combined or installed because its isolated benefit was negligible.

All semantic metadata, signed floor/modulo behavior, split-row coordinates,
input offsets and pointer construction remain i64. The u32 probe narrows only
the physical induction variable: Stage5 runtime validation proves
`0<=E<=2^20`, thread ID is at most 511 and the last 2048-step increment stays below
`2^21`. It widens back to i64 before semantic replay. This removes carry/control
work and changes NVRTC loop generation, but does not provide a universally
better allocation. Exact register-liveness attribution to each C++ variable
is unavailable; the isolated probes establish sensitivity, not a unique
register-by-register decomposition.

The executed uniform-loop SASS makes the induction-variable change concrete:
`UMOV UR4,URZ` / `UMOV UR5,URZ` initialize the chunk counter in uniform registers;
`IADD3 R22,...,R21,UR4` adds the lane ID, and `UIADD3 UR4,...,0x2000` plus its
carry updates the four-times-unrolled chunk counter. The frozen SASS instead
compares its per-thread induction state under BSSY B0/B1. This is an observed
uniform-register/control-flow promotion, not an inference from fewer C++ locals.
Metadata N and extent remain paired 64-bit values. Signed ceiling division uses
wide multiply/carry/select instructions; block quotient/remainder by nine feed
`N*row + E*split + coordinate`. `LEA`/`LEA.HI.X` scale that i64 offset by four
and add the original input pointer. Per-chain offsets 512/1024/1536 and paired
extent predicates feed the same protected scalar loads. The retained IR, CUDA,
PTX and SASS expose each boundary of that address/control chain.

Integer instruction totals also resist proxy optimization: Uniform executes
more IADD3 and ISETP at the primary shape while substantially reducing time.
Triton itself executes more IMAD there than Uniform. Generic i64 split division,
modulo/sign adjustment, metadata reads, pointer carries and predicates remain
visible in IR/CUDA/PTX/SASS. They are not all proved unnecessary for arbitrary
Inductor-generated indices. No blanket unsigned or 32-bit semantic rewrite was
introduced.

The compiled virtual-tensor controls demonstrate kernel-body invocation without
CubeK runtime ownership. Blueprint/Routine separates structural choices from
launch adaptation, and ReduceArgs/VirtualTensor can host generated readers and
writers. However stock Plane/Cube reducers change the frozen accumulation and
shared-merge topology; CubeK's unsigned tensor-coordinate arithmetic cannot
replace arbitrary signed-i64 Inductor semantics by assumption. Arbitrary
prologue/epilogue composition would require an adapter preserving our semantic
precision barriers, lazy/effectful accesses, split9 mapping and result visibility.
This pass reuses the uniform-loop mechanism behind our existing boundary;
it does not claim a general Inductor-to-CubeK fusion adapter is implemented.

## Validation and reproduction

Each of Uniform, Owner, Narrow and Uniform+Narrow passes 152 exact partial/final
checks **before its timing run**: all 148 existing corpus cases, non-default-stream
repeat and three special-value cases. Canaries, dtype/storage neighbors and the
finite adversaries remain intact. No tolerance or qualification policy changes.
The frozen arithmetic and earlier failures are preserved in the prior packet.
Additional checks cover ten physical-counter boundaries through `E=2^20`,
rejection above that bound before launch, nonzero input storage offsets,
owner-probe refusal for normalization and multiple reductions, all six legacy
partials and three finals with identical CUDA/PTX, and installed Uniform/frozen
CUDA identities in all three dtypes. The normalization capture has 18 phases;
all 36 retained arrays match the frozen packet byte-for-byte. These checks are
retained in `checks.json`; the established 18 regression gates in `gates.json` all pass.
Historical packets are restored byte-for-byte by the gate runner.

## Fresh fixed-final graph comparison

The first unpinned reference graphs let Triton's final autotuner drift between
XBLOCK8 and XBLOCK1. Those captures and the resulting identity-check failure are
preserved and excluded as a group from fixed-final timing conclusions;
`collection-corrections.json` records the exclusion. The replacement harness
uses PyTorch's existing `autotune_lookup_table` keyed by generated source, pinning
the retained partial `(XBLOCK1,R0_BLOCK2048,16 warps)` and final `(XBLOCK1,2 warps)`.
The FP32 reference PTX executable text matches both retained kernels exactly
after debug-only sections/locations are removed. No reference arithmetic or
configuration sweep is introduced. The initial wrapper overlay-order failure
is retained separately; setup was corrected before replacement capture.

Three fresh C/T,T/C,C/T rounds each capture seven calls. All three CubeCL graph
captures reproduce the frozen 178 recorded outputs/input hashes, preserve the
eager-relative failures, and pass the existing corresponding-efficient-reference
per-element policy. `graph-analysis.json` verifies partial/final identities,
input hashes, kernel census and raw trace hashes.

| FP32 case / scope | CubeCL round medians µs | Triton round medians µs |
| --- | --- | --- |
| 8×1048576 partial | 45.23 / 45.37 / 45.60 | 47.46 / 47.06 / 46.55 |
| 8×1048576 fixed final | 2.21 / 2.15 / 2.21 | 1.28 / 1.28 / 1.28 |
| 8×1048576 graph sum | 47.43 / 47.53 / 47.80 | 48.74 / 48.34 / 47.84 |
| 64×65539 partial | 21.36 / 25.29 / 25.28 | 23.73 / 23.79 / 23.65 |
| 64×65539 fixed final | 2.34 / 2.34 / 2.43 | 1.28 / 1.28 / 1.28 |
| 64×65539 graph sum | 23.70 / 27.60 / 27.74 | 25.01 / 25.07 / 24.90 |

The primary gap is substantially reduced. The secondary remains variable and
loses at graph level in two rounds; no broad competitiveness/default claim is
made. Both backends' finals are fixed within this comparison and unchanged by
the mechanism. The remaining final-kernel cost is outside this pass's frozen-final
scope. Continue Stage 5 on these measured remaining issues; do not substitute
another arithmetic topology or move to Stage 6.

With review-mcp unavailable, the user explicitly authorized a native adversarial
review. It found no retained-kernel correctness defect, but identified harness
provenance gaps. The gate runner now snapshots all pre-existing regular files
and symlink identities in its protected artifact roots, including untracked and
ignored files. It restores changed/deleted/replaced objects and relocates newly
created evidence into a separate packet outside historical discovery. CubeK
setup removes the generated cache tree before extracting the pinned archive.
The follow-up review accepted the fixes with no unresolved findings;
`native-review.json` records scope, findings and limitations.

The retained CPU fixture exercises the runner's actual main function for both
success and deliberate exit-7 failure, including file/symlink replacements,
broken links, historical additions and output exclusion. A current-tree check
snapshots all 14,195 protected files with no changes; the clean CubeK rebuild
reproduces the same 1,140-file tree and compatibility patch. All generated kernels
and original measurement evidence remain unchanged, so GPU captures were not
repeated. `review-provenance-checks.json`, `review-protected-scope.json` and
`review-setup-checks.json` retain these checks. The original pre-review hash
manifest is preserved separately; the current manifest includes the reviewed
source and disposition. Review task `pytortch-backend-experiments-cwk.4` is closed.
No commit, push, default promotion or Stage 6 work is performed.

Run the bounded CPU fixture from the repository root:

```sh
PYTHONPATH=. stage0/.venv/bin/python \
  stage5/artifacts/reduction-structure/codegen/review-provenance-checks.py
```

Run GPU commands sequentially from the repository root:

```sh
export STAGE0_CARGO_PROFILE=release
export CUDA_PATH="$PWD/stage0/cuda13-toolkit/nvidia/cu13"
cargo build --release --locked --manifest-path stage0/Cargo.toml --features llvm,hip-probe
stage0/.venv/bin/python -m stage5.reduction_codegen_setup
stage0/.venv/bin/python -m stage5.reduction_codegen_compare --compile
# Each run qualifies before measuring; combination follows isolated trials.
stage0/.venv/bin/python -m stage5.reduction_codegen_mechanisms uniform --build
stage0/.venv/bin/python -m stage5.reduction_codegen_mechanisms owner --build
stage0/.venv/bin/python -m stage5.reduction_codegen_mechanisms narrow --build
stage0/.venv/bin/python -m stage5.reduction_codegen_mechanisms uniform_narrow --build
ncu --target-processes all --profile-from-start off --launch-count 12 \
  --section-folder /home/imi/tools/nvidia/nsight-compute-2026.3.1/sections \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section InstructionStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export "$PWD/stage5/cache/codegen/mechanisms" \
  --csv --log-file "$PWD/stage5/cache/codegen/mechanisms.csv" \
  stage0/.venv/bin/python -m stage5.reduction_codegen_capture
ncu --import stage5/cache/codegen/mechanisms.ncu-repz --page source --print-source sass
ncu --import stage5/cache/codegen/mechanisms.ncu-repz --csv --page raw
stage0/.venv/bin/python -m stage5.reduction_codegen_analyze
stage0/.venv/bin/python -m stage5.reduction_codegen_gates
```

NCU actions retain their launch order in `mechanism-capture.json`: several
mechanisms deliberately have the same entrypoint. The analyzer binds by action
index and verifies every SASS warp-instruction sum against its raw NCU row;
aggregating those rows by kernel name would incorrectly mix frozen and changed
code. NCU uses the production NVRTC/driver-JIT route; its cold replay durations
are not substituted for warm timing. The direct Triton comparator loads the
retained cubin, preserving Triton's actual assembler output and avoiding a
reassembly by a different ptxas.
