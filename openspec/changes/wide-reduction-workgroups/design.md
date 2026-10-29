# Design

## Context

See proposal.md (Why) for the measurements. This section covers the current constraints.

- **Plan catalog** (`compiler/plan.py`).
  - Reduction plans run 64, 128 or 256 units.
  - The only 512-unit plan is `plane_shared_reference512`, a diagnostic sum topology that is
    never admitted by default.
  - Welford's eight-wide plans live in `welford_catalog()`, at 128 and 256 units.
- **Legality.**
  - Rust `plan.rs` admits `plane_shared` at `[64|128|256, 1, 1]`.
  - Four-accumulator ("combined") plans are fixed at 256 units, or 512 for reference512.
  - Python `legality.py` checks vector alignment and target topology (`plan.units` against the
    device limit). It has no rule tying workgroup size to extent.
- **Launcher** (`launcher.c`, `stage5_launch`). It accepts more than 256 units only for
  `[512,1,1]` with exactly 64 bytes of shared memory (reference512). Shared memory above 1 KiB is
  allowed only up to 256 units at 12 bytes per unit. This rejected Welford at 512 units in the
  4z8.35 experiment ("unsupported launch geometry").
- **Emitter.** `plane_shared` keeps one shared slot per plane: 16 slots at 512 units, at 4 bytes
  for a sum and 12 for Welford. In the experiment, sum, max and softmax kernels at 512 units
  compiled and ran correctly with no emitter change.
- **Long-row sum rule** (`legality.numerical_reason`). Kernels whose reductions are all sums,
  over rows longer than `LONG_REDUCTION_THRESHOLD` (4096), may only use `shared_tree_compensated`
  at 256 units. It is labeled an "experimental long-row continuity policy" from Stage 5, never
  compared with Triton, and it appears in no spec.
- **CubeK.** `cubek-reduce` caps a cube at 8 planes (`NUM_PLANE_MAX`, 256 units). Its routines
  are single-pass, with no epilogue that re-reads the row. So its sizing rule does not address
  this problem, and there is no pattern to reuse. Triton is the reference: for inner reductions,
  `num_warps = R0_BLOCK // 128`, capped at 16 for rows up to 8192. That gives 512 threads from
  `R0_BLOCK = 2048` on.

## Goals / Non-Goals

**Goals:**

- Long rows keep their rows resident in L1 across passes, as Triton does.
- Every new candidate stays inside the accuracy contracts of its reduction type.
- The long-row sum rule is derived from Triton's band, not inherited.

**Non-Goals:**

- New reduction topologies. The 512-unit plans reuse `plane_shared` and, if needed,
  `shared_tree_compensated` code paths.
- Changing the selection rule (`6cl`).
- Online softmax.

## Decisions

### 1. Add `plane_shared` at 512 units, widths 1, 2 and 4, to the shared catalog

- **Why `plane_shared`.** It reduces across a plane with shuffles and keeps one shared slot per
  plane, so 512 units need 16 slots. `shared_tree` would need 512.
- **Why these widths.** Widths 1, 2 and 4 are the ones the experiment ran.
- **Combined plans stay at 256 units.** These are the four-accumulator plans. Their topology rule
  is part of their sum qualification, and the experiment's slow cases did not need them.
- **Alternative: a separate catalog for long rows**, so short-row cases keep their manifest
  candidate lists. Rejected: Triton applies 16 warps from 2048 elements, which is not only long
  rows. Decision 2 already keeps short rows' candidate sets unchanged; only the recorded
  rejected list grows.

### 2. Offer 512-unit plans only for rows of at least 2048 elements

- **Rule.** A selection policy in `candidates()`, like the Welford lane policy, rejects 512-unit
  plans below 2048 elements, with a recorded reason. It is not part of `reason()`, which stays
  the launch contract.
- **Why 2048.** That is where Triton reaches 16 warps. Below it most units would hold no element,
  and every extra candidate costs first-use tuning time.
- **Signature.** The legality signature gains the fact `extent >= 2048`, so cached candidate sets
  stay correct across shape buckets.
- **Alternative: offer at every extent and let timing decide.** Rejected: it adds tuning cost on
  every short row for a plan that cannot win there.

### 3. Launcher geometry

- **New rule.** `stage5_launch` accepts `[512,1,1]` with shared memory up to 12 bytes per unit
  (6 KiB). Every other block stays at most 256 units, as today. The reference512 case
  (64 bytes) is inside the new rule.
- **Bound.** The bound keeps the existing per-unit limit ("one 12-byte slot per unit"), so a
  compensated sum (8 bytes per unit) and Welford (12 bytes per plane) both fit.
- **Alternative: general blocks up to the device limit.** Rejected: no plan needs them, and the
  launcher's limits are part of its safety contract.

### 4. Welford through the existing lane policy

- **float32.** The widest admitted vector for float32 Welford rows is 4. A `plane_shared` 512×v4
  plan has 2048 lanes (Triton's `TRITON_LANES`) and already counts as Triton-shaped. No policy
  change is needed.
- **16-bit rows.** The widest vector is 8, so a Triton-shaped 512-unit plan must be 512×v8 (4096
  lanes, more than Triton's 2048).
  - It is added to `welford_catalog()` and judged by the band.
  - If it falls outside the band, the policy instead admits 512×v4 (2048 lanes) as
    Triton-shaped for 16-bit rows longer than `TRITON_LANES`.
  - The band test decides between these two, and both are prepared in the tasks.
- **Fused sums.** A kernel fusing a sum with Welford keeps scalar lanes (bead `4z8.32`).
  `plane_shared` 512×v1 is scalar and is admitted through decision 2.

### 5. Long-row sums: measure, then derive the rule

- **Measurement.** The band test (`welford_band.py`, with its Triton reference) gains float32
  cases:
  - `x.sum(-1)` and the sum consumer, at 2048×8192 and 256×32768, zero-mean and offset data;
  - the case generator asserts that the rows compile unsplit.
- **Forced plans.** Each case runs with each plan forced: `shared_tree_compensated` 256,
  `plane_shared` 512×v1/v2/v4, and the existing 256-unit plans.
- **Rule.** The plans in band become the long-row set in `numerical_reason`. The rejection reason
  is renamed to name Triton's band.
- **Compensated plan at 512 units.** If only compensated plans are in band,
  `shared_tree_compensated` gains a 512-unit workgroup, since its scratch fits the decision 3
  bound. Otherwise no compensated plan is added.
- **Why measure first.** Triton itself reduces an 8192 row as 2048 lanes with 4-element chains
  and no compensation. A plain `plane_shared` 512 plan has 16-element chains (vector lanes fold
  into one accumulator, bead `4z8.32`), so whether it is in band is an empirical question.
- **Alternative: keep the rule and accept the gap.** Rejected: the standard is Triton's speed and
  accuracy, and this rule was never judged against either.

### 7. Welford vector lanes as independent accumulators (added during group 2)

The pinned band (results, task 2.2) gave two findings:
- On 16-bit 2048×8192 rows, 512×v8 is the only plan in band.
- On float32 2048×8192 rows with affine inputs, no plan is in band. Before this change the
  admitted plans there were 256×4 and 256×v4, so the gap is not new.

**Cause (to be confirmed by the band).** The two backends combine a row's elements in different
orders:
- Triton's looped Welford keeps `R0_BLOCK` (2048) independent accumulators. Each runs
  sequentially over its strided elements, and they are merged once at the end, within a thread
  and then across lanes.
- Our vector plans pair-merge the lanes of each loaded vector and fold the result into a single
  per-unit accumulator at every step.
- CubeK's `Sum` keeps one accumulator per vector lane and merges them at output time, the same
  as Triton.

**Decision.**
- A vector `plane_shared` plan gets a variant, `private_merge = "lanes"`. It keeps one Welford
  accumulator per vector lane across the loop and merges the lanes in order after the loop,
  before the plane and shared steps.
- It is a separate plan identity: existing plans, their band results and the manifest records
  of other kernels stay unchanged.
- The variant is offered for Welford-only kernels (fused sums keep scalar lanes, bead `4z8.32`).
- The band decides the policy for float32 rows longer than `TRITON_LANES`, and whether the
  variant also replaces the folding plans for 16-bit rows.

**Policy for long 16-bit rows.** Only Triton-shaped plans with the most lanes are kept, the same
rule as for one-pass rows. That leaves 512×v8; 256×v8 and the four-accumulator plans were
outside the band there.

**Alternatives considered.**
- Folding the lanes with Triton's order inside the existing plans. Rejected: it changes every
  vector Welford kernel at once, including in-band cases at 1280 and 4096.
- One warp per row with vector chains, which already has Triton's lane count. Rejected: it is
  too few threads for long rows, and too slow.

### 8. Two-step variance where Inductor applies it (added during group 4)

**Finding.** Triton's kernel for bfloat16 `var_mean` at 300×4096 is not Welford: it is a sum for
the mean and a sum of squared deviations. `lowering.use_two_step_variance` lowers a variance this
way for CUDA inputs of a 16-bit dtype with `config.triton.use_two_step_variance_min_numel` (1024)
≤ N ≤ `use_two_step_variance_threshold` (32768) when the reduction is not split (and, on every
backend, for N up to `unroll_reductions_threshold`, 8). It returns False when the output has one
element. The 16-bit check holds only under `is_triton()`, which for CUDA reads
`config.cuda_backend == "triton"`; our backend sets its own name, so these rows lower to Welford.

The rule sees the dtype of the tensor `var_mean` receives. LayerNorm and GroupNorm upcast to
float32 before calling `var_mean` (`_refs._normalize`, `_refs.native_group_norm`), so they keep
Welford in both backends; only direct 16-bit `var`, `var_mean` and `std` calls are affected. The
16-bit LayerNorm gaps of task 4.2 are Welford against Welford (decision 9).

**Decision.**
- At install, the backend wraps `lowering.use_two_step_variance` and `lowering.is_triton`. The
  wrapper sets a context variable for the duration of the call; while it is set, `is_triton`
  answers True for a CUDA node when `config.cuda_backend` is a CubeCL backend name. The global
  config is never changed, so concurrent compiles on the free-threaded build are unaffected, and
  `is_triton` calls outside the rule keep their answer. Calls nested inside it see True: today
  only `_make_reduction_inner`'s argreduce check, whose answer the rule does not use.
- The split check inside the rule then runs under the real configuration and choices handler
  (`CubeCLChoices`), the same decision that later splits (or not) the two sums. A test asserts
  the two-step sums compile unsplit wherever the rule chose two-step.
- The kernel becomes a sum and a dependent sum over the same row, a shape the backend already
  compiles (softmax's max and sum).
- Consequences:
  - Unsplit direct 16-bit variances of 1024 to 32768 elements no longer produce Welford kernels.
  - Decision 7's 16-bit findings were measured on LayerNorm, whose reference stays Welford; they
    stand. The 16-bit Welford policy keeps governing LayerNorm and GroupNorm rows.
  - The long-row sum rule (compensated only above 4096) was measured on float32 inputs. The band
    decides the plan set of the 16-bit two-step kernels.

**Alternatives.**
- Reimplementing the condition in the backend. Rejected: it would drift from the pinned PyTorch.
- Patching `config.cuda_backend` around the call. Rejected: a global change visible to concurrent
  compiles, and the split check would run under a different backend than the sums it predicts.

### 9. Welford per-block cost (added during group 4)

**Finding.** LayerNorm keeps Welford in both backends in every dtype (decision 8). At 300×4096
both launch 300 blocks of 512 threads with the same DRAM traffic. On float32 LayerNorm ours
executes 54% more instructions and stalls 1.8 times as long at barriers (task 4.2). The 16-bit
LayerNorm gaps at 300×4096 (4 to 14% with the fastest plan) and the smaller ones at 2048×8192
are the same comparison, Welford against Welford. On the 2048×8192 LayerNorm with affine parameters, our kernel also
keeps 3 blocks per SM against Triton's 2, so three rows plus the weights exceed L1 (DRAM reads
125 against 75 MB).

**Decision.**
- Attribute the extra instructions and barrier waits from SASS and ncu against Triton's kernel:
  the lane merges, the shared tree and its synchronizations, the tail guards and the index
  arithmetic. Fix them in the emitter, keeping each plan's combine order (the band verdicts
  stand only if the order is unchanged, or the band is rerun).
- Reach Triton's residency on the long affine row. The mechanism (a per-launch occupancy cap
  through shared memory, or a register budget) is chosen by measurement. A launcher limit it
  needs is a change to decision 3's bound, recorded there.

- After the fixes, re-judge by timing the "most lanes" rule for float32 rows of at most 4096
  elements, where every shaped plan is in band (task 4.2; kept, task 6.4).

**Outcome (measured, results 6.1–6.3).**
- **Per-element pass on vectors** (`vector_elements`): 128-bit loads and stores, as Triton's second
  loop. Per-element stores join the vector alignment checks.
- **Exact vectors:** with one chain and a constant extent that is a multiple of the width, the
  ragged tail is not emitted. Kept: equal or faster at every workload shape (4096×640: 29.5 against
  31.8 µs).
- **First-element peel** (Triton's `is_first`): lanes plans only. Bit-identical to the unpeeled
  kernel for finite inputs (an infinite first element starts m2 at 0, as Triton's does, instead
  of NaN). An extension to the folding and scalar plans was measured without gain and reverted.
- **Residency:** not occupancy (registers and blocks per SM matched Triton) but the cache hint.
  The CubeCL patch gains `retained_load`, lowered to `ld.global.L1::evict_last`, and the emitter
  marks the reduction pass's loads of buffers read again. Triton's L2 hints measured slower and are
  not adopted.
- **Rejected: exact vectors only where no load is retained.** The pair costs about 11% against
  retained loads alone (7% against Triton) at 300×4096 (about 1.5 waves) only; at workload shapes it is best or tied. Bead `4z8.38` keeps the
  case and its inline-PTX hypothesis.

### 6. Tests and timing

- **Existing plan tests.** The every-plan tests (`test_reduction_combiners.py`,
  `test_reduction_welford.py`, `test_multi_output.py`) iterate the catalog, so they cover the new
  plans automatically. The eligibility filters are checked so 512-unit plans are not skipped.
- **New tests.**
  - The extent rule of decision 2.
  - The launcher accepting 512-unit Welford geometry and refusing larger blocks.
  - The long-row candidate sets.
- **Timing.** `welford_timing.py` gains:
  - the `softmax` and `sum_consumer` kinds, and LayerNorm without affine;
  - the shapes 300×4096 and 2048×8192, float32 and bfloat16.
  It reports per-kernel CUPTI times as today. Runs cover the 3080 and the 3090.

## Risks / Trade-offs

- [Selection noise picks a 256-unit plan where 512 is faster.] → The paired selection rule
  (`4z8.28`) already compares per round; the timing gate catches a lost case. Pinned per-plan
  ncu comparisons are used for diagnosis (memory `ab-codegen-changes`).
- [512-unit plans raise register pressure.] → At 512 units each thread may use 128 registers
  before occupancy drops below one block per SM. The kernels measured use 24 to 60. The timing
  gate and ncu cover regressions.
- [The 16-bit Welford choice falls outside the band at both 512×v8 and 512×v4.] → The 512-unit
  plan is then not Triton-shaped for those rows, and the bfloat16 LayerNorm gap at 2048×8192
  (about 1.2×) is recorded on a bead. The float32 and sum gains stand.
- [The two-step 16-bit kernels fall outside the band under plans the sum policy admits.] → The
  band decides their plan set (decision 8), as decision 5 did for float32 sums.
- [Manifest churn.] → Candidate lists change only for corpus cases with rows of at least 2048
  elements. The changes are reviewed by matching records on content, then re-recorded.

## Migration Plan

- **Caches.** No artifact or descriptor format changes. The native digest changes with
  `plan.rs` and `launcher.c`, so caches rebuild.
- **Rollback.** Revert the commits; the manifest is re-recorded with them.

## Open Questions

- None for decision 9 (outcome recorded there); the SDXL LayerNorm shapes are attributed in task 6.1.
- The measured choices (decision 4's 16-bit Welford plan, decision 5's long-row sum set,
  decision 8's two-step plan set) are decided by the band test.
