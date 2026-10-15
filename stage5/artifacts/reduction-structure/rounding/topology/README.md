# Work assignment and arithmetic topology

Current status belongs in [STAGE5_REPORT.md](../../../../../STAGE5_REPORT.md).
This is the bounded follow-up from `b74c15c`, with exactly one CPU-modeled
topology and one GPU candidate. The earlier merge-only rejection remains intact.

## Decision

**Numerically qualified on this corpus; unsuitable as a broad long-row default.**
The full reference-derived topology fixes heldout_10017 and the existing finite
adversaries together. FP32 performance remains materially unattractive at larger
row counts and at one million elements, after retaining protected-load batching.
Recommend stopping further long-row optimization here. Keep `cwk` open for the remaining practical
FP32/codegen gap; do not promote this plan or automatically start another topology,
planner or Stage 6. The favorable low-precision results remain scoped evidence.

Explicit `tuning="reference512_diagnostic"` selects the single plan for diagnosis.
It remains explicitly excluded from automatic eligibility at every extent despite qualifying on this bounded numerical corpus. The semantic descriptor, logical split machinery, final reduction, precision barriers, conservative FP permissions and PyTorch storage/current-stream boundary remain unchanged. Compensation is retained for
continuity/diagnosis, not newly accepted.

The existing source-hash cache policy invalidates persisted tuning selections
after this change. The first subsequent `first_use` run can therefore incur
re-validation/re-timing even though the legal candidate set and default policy
are unchanged. Existing empirical candidates can be reselected by timing; no
promise of preserving a previously cached timing winner is made.

Like the existing diagnostics, `reference512_diagnostic` uses the frozen baseline
when a generated kernel's logical reduction extent is ≤4096. For a split graph
this is the partial extent, not necessarily the original input row length. Direct
fixed-packet qualification exercises the reference-derived plan at every corpus
length separately.

## Reconstruction from retained artifacts

The efficient configured reference is `R0_BLOCK=2048, num_warps=16, XBLOCK=1`.
The FP32 artifact key is
`22FRMEHQY7GOBHBJC6OITADZEDJIOXOLC3DDHYLIPDIIOU5CHWAA` under the parent
`selected/` packet. FP16/BF16 keys are
`WX636FQBE4IA5O4I7KTAU4GRMOSMVSCXSWFKVOQUVUNCWOX7DZ2A` and
`Z24LN6YAEP5LPULWC5FB4RYYRC2KFNYCEA4HR45UJTFE6RUE33GQ`.
Their retained LLVM/PTX agree on the assignment and collective structure;
`../../triton-8-sass.txt` records executed FP32 machine instructions.

For output row `b`, split `p`, input length `N`, let `E=ceil(N/9)` and
`block=9*b+p`. One block has 512 threads / 16 warps. Each thread `t` maintains
four FP32 chains `a[j]`, initially positive zero:

```text
r = t + 512*j + 2048*k, j=0..3, k increasing
input coordinate = b*N + p*E + r
load only if block<9*B AND r<E AND p*E+r<N; otherwise protected +0
update a[j] = RN32(a[j]+loaded) only when block<9*B AND r<E
private = RN32(RN32(a[0]+a[1]) + RN32(a[2]+a[3]))
warp = XOR butterfly distances 16,8,4,2,1, RN32 after each add
shared[(t>>5)] = warp result; byte address (t>>3)&60
uniform publication barrier
reload shared[t&15]; every half-warp has the 16 warp sums
inter-warp XOR butterfly distances 8,4,2,1, RN32 after each add
only thread 0 stores partial[block]
```

LLVM lines 8–98 establish coordinates, predicates, four scalar loads and FP32
chain updates; lines 100–155 establish adjacent private pairing, decreasing XOR
butterflies, shared byte addresses and publication. The output predicate follows
at lines 156–162. PTX `.reqntid 512`, scalar predicated loads, `add.rn.f32`,
`shfl.sync.bfly`, shared exchange and owner store independently confirm it.
The SASS capture confirms execution rather than only an unused source pattern.
No vector memory operation or compensation is inferred from LLVM's four-value
private vector representation.

The reference final (key `PLVMBVHWVSQKEG44V3HNSW54ALZS25L5BLDYTBYXBMIXR6LWQXSA`
for FP32) loads nine partials in lanes 0–8, neutral-fills through lane 15,
then uses XOR distances 8,4,2,1; thread 0 owns the output. The CPU reconstruction
models that final separately to check against retained reference outputs. The
candidate uses the **unchanged CubeCL scalar256 shared-tree final**. With nine
nonzero inputs its owner arithmetic has the same effective 8,4,2,1 halving order.

`stage5.reduction_topology_model` is entirely CPU FP32 arithmetic; CUDA is used
only to regenerate the original exact RNG input bytes and recorded eager/FP64
diagnostics. `stage5.reduction_topology_checks` independently implements the
owner's scalar recurrence with `struct.pack/unpack` rounding. No CPU RNG sequence
is substituted for the original CUDA RNG history.

## Numerical evidence and invariants

`qualification.json` has 148 distinct cases / 1349 elements, all passing the
unchanged `reference-or-better-v1` policy by reference tolerance. This comprises
the original 109 exact cases, six dtype/storage neighbors, 32 later held-out
seeds and the existing five-row finite adversary. Every input hash agrees with
the earlier packet. All 108 applicable retained 65K reference output tensors
are reproduced bitwise, and the captured adversarial partial buffer matches.
Other lengths use their existing corresponding-efficient reference regime for
qualification, not a claim that they were compiled with this same layout.
The CPU packet's `gpu_candidate_implemented=false` field describes that model
capture only: it contains no GPU candidate construction or execution. The later
GPU implementation and its qualification are recorded in `gpu-qualification.json`.

For heldout_10017 element 4, model/candidate/reference return
`0.1167755126953125`; FP64 is `0.1167629364572349`. The finite adversaries return
`1,-1,0.25`, preserving the small terms. At coordinates 0,256,512, thread 0 owns
the cancelling large terms in chains 0/1, and thread 256 owns the small term in
chain 0. The full assignment explains why adjacent pairing now works.

`gpu-qualification.json` records 149 direct checks including a non-default-stream
repeat, exact model/GPU partial and final byte matches, original input pointers,
output canaries, and NaN/Inf/zero-sign classifications for all three dtypes.
The candidate retains the existing full-epilogue broadcast: one warp leader
publishes each warp sum, the first warp reduces the 16 shared values, thread 0
publishes the root, and a second barrier makes it readable by all units. These
publication/inter-reduction barriers are preserved; Triton's redundant shared
stores and layout replication are not copied into the ownership contract.
All owner arithmetic matches the modeled topology on the corpus.

Three real fullgraph/dynamic Inductor captures each contain 178 cases, all
qualifying against corresponding efficient references. Their eager-relative
failure records remain separate and unchanged in meaning: original_rng_history
and heldout_30003 also fail eager comparison in the reference. They are not
silently relabelled eager passes. See `graph-analysis.json` and `analysis.json`.

## Performance and compiler trace

Prepared partial components use three rotations and 384 CUPTI kernel events
per plan/case/round. The logical split is nine and the final packet is fixed;
the final's execution is excluded from these numbers. Existing load-batch and
compensated paths run in the same process on identical input/storage/stream.
Medians in microseconds, candidate / old load-batch / compensated:

| Case | Candidate | Load-batch | Compensated |
| --- | ---: | ---: | ---: |
| FP32 8×65539 | 4.227 | 3.904 | 4.128 |
| FP32 64×65539 | 27.236 | 24.010 | 23.381 |
| FP32 8×131071 | 6.209 | 4.355 | 5.536 |
| FP32 8×1048576 | 72.365 | 44.006 | 60.194 |
| FP16 8×65539 | 2.592 | 3.778 | 3.714 |
| FP16 64×65539 | 14.563 | 14.338 | 15.016 |
| BF16 8×65539 | 2.690 | 3.809 | 3.742 |
| BF16 64×65539 | 14.161 | 14.224 | 15.149 |

All four FP32 cases regress versus load-batch under the existing at-least-2/3
rounds plus disjoint median±3MAD rule. FP16/BF16 gains at 8 rows and larger lengths
do not erase those failures; the complete 12-case samples are retained in
`component-timings.json`.
In that JSON the key `baseline` denotes `plane_shared_load_batch`, as confirmed
by each embedded plan; it does not denote `shared_tree_reduction_baseline`.
The tables here use the unambiguous label “Load-batch”.

Fresh generated-graph comparison rotates C/T, T/C, C/T in separate environments.
Each case has seven independent CUPTI captures per round with the established
activity warmup. Input hashes, partial/final identities, fixed nine-way split,
efficient reference keys and kernel census are checked. FP32 64×65539 partial
medians are `26.619/26.892/26.715` versus Triton `21.310/21.275/21.325` µs;
8×1048576 is `72.664/73.939/72.333` versus `44.712/43.868/43.940` µs.
FP32 8×65539 varies across candidate rounds (`7.197/4.478/4.281` µs versus
`4.704/4.740/4.680`); do not turn that unstable case into a universal ratio.
Separate final and full graph sums are retained, including other noisy rounds.
These are device measurements; Triton's GIL-enabled environment does not support
unqualified host comparisons.

IR → CUDA → PTX → executed SASS evidence is in `kernels/`, `ncu-commands.json`,
`candidate-*-sass.txt` and `analysis.json`. Grouped protected pending loads precede
four independent updates in physical IR and CUDA. CUDA has literal decreasing
XOR shuffles. NVRTC still peels/unrolls the FP32 loop into predicated scalar loads
and grouped updates with 64-bit index/predicate bookkeeping. This is not a return
to the old nested protected-load/update IR. Executed SASS retains no spill
loads/stores, and the production driver query reports zero local bytes.

At 8×65539 the new FP32 partial executes 2304 BSSY, 2304 BSYNC, 2304 BAR and
6048 SHFL warp instructions. The old load-batch capture has 1152/1296/1152/3240;
the efficient reference has 0/0/1152/10368. More or fewer shuffles alone do not
predict the measured performance. The first five candidate shuffles execute
across 16 warps; the last four execute only in the first warp, followed by root
publication. The candidate uses 48 registers/thread (FP16 63, BF16 56), 64 shared
bytes and two resident 512-thread blocks/SM. Reference FP32 uses 29 registers,
64 shared bytes and no spills. Broader cooperation, conservative address/control
lowering, scratch publication and residency remain differences. Their separate
causal shares are **not isolated**; no proxy-metric optimization or second
candidate is attempted.

NCU uses Linux 2026.3.1 with matching section assets; it captures the actual
CUDA13.0 NVRTC/driver-JIT route. Cold replay durations are not warm timing samples.
`collection-corrections.json` preserves exclusion/replacement of the initial
FP32-8 NCU capture, conservatively discarded because a CPU-model refresh also
generated CUDA RNG inputs during collection startup. Warm timing runs were
sequential. No CUDA13.1 offline binary substitutes for executed code.

## Verification and reproduction

`checks.json` records independent scalar arithmetic, 12 Python/Rust invalid plan
neighbors, six native geometry rejections, and unchanged automatic ineligibility.
The native launcher admits the new 512×1×1 geometry only with 64 shared bytes;
all other ownership/readiness/current-stream/allocator protections are inherited.
Final source hashes, required review disposition and regression status are
recorded with this packet after verification.

All **18 established regression gates pass**, including the full 168-case corpus,
constructor precision, frozen-order qualification, accumulation and long-row
checks. Rustfmt, all-target release Clippy with warnings denied, touched-Python
Ruff F checks and diff whitespace checks pass. The additional independent scalar,
Python/Rust/native negative controls pass. Full style lint also reports existing
import/SIM/E731 findings and is not claimed as passing. `legacy-code-identity.json`
checks exact CUDA/PTX for all six old load-batch/compensated partials across three
dtypes; the direct GPU probe checks the three unchanged finals. The regression
runner restores **4219 historical tracked artifacts** byte-for-byte, excluding
only the intentionally edited forwarding README; prior failure data is intact.
Required review is recorded separately in `review.md` and `review-disposition.md`.
It returned with no blocking correctness or benchmark-validity findings. All
minor naming/cache/scope notes are explicitly resolved in the disposition.

Run GPU commands sequentially, from the repository root:

```sh
export STAGE0_CARGO_PROFILE=release
export CUDA_PATH="$PWD/stage0/cuda13-toolkit/nvidia/cu13"
cargo build --release --locked --manifest-path stage0/Cargo.toml --features llvm,hip-probe
cc -std=c11 -Wall -Wextra -Werror -fPIC -shared stage0/launcher.c -ldl -o stage0/libstage0_launcher.so
stage0/.venv/bin/python -m stage5.reduction_topology_model
stage0/.venv/bin/python -m stage5.reduction_topology_probe
stage0/.venv/bin/python -m stage5.reduction_topology_checks
stage0/.venv/bin/python -m stage5.reduction_topology_benchmark
# graph-commands.json contains the six ordered, separate-environment captures.
# ncu-commands.json contains prewarm, collection and export commands.
stage0/.venv/bin/python -m stage5.long_row_normalization --impl cubecl --cubecl-policy reference512_diagnostic --output-dir stage5/artifacts/reduction-structure/rounding/topology --cache-dir stage5/cache/reduction-topology/norm
stage0/.venv/bin/python -m stage5.reduction_topology_analyze
stage0/.venv/bin/python -m stage5.reduction_topology_gates
```

No default eligibility, qualification policy, compensation policy, Stage 6,
commit or push is part of this pass. Historical failures and packets remain
preserved. This establishes the topology's tested numerical viability and its
remaining cost gap; it does not establish a broadly competitive default.
