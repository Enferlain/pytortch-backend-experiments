# Design

## Context

See proposal.md for motivation. The facts that shape the approach:

### Inductor (pinned 2.14.1)

- **Lowering.** `var_mean_helper_` upcasts the input to its computation dtype (float32 for
  float16/bfloat16), then calls `WelfordReduction.create(..., "welford_reduce")`. It chooses
  the two-step `sum` lowering (`var_mean_sum_`) only when `use_two_step_variance` holds. For a
  non-Triton device that means a reduction of at most `unroll_reductions_threshold` elements.
  Every realistic norm therefore reaches us as `welford_reduce` with a float32 dtype.

  Triton is different: for float16/bfloat16 inputs with an unsplit reduction of at most 32768
  elements (`config.triton.use_two_step_variance_threshold`), it takes the two-step sum path.
  The reference data shows this as a looped two-pass kernel for var_mean at 300×4096. So
  Triton's band for those cases comes from two-pass arithmetic, while we receive Welford
  (decision 7).
- **Three output buffers.** `WelfordReduction` realizes three buffers (mean, m2, weight). Each
  node's body calls `ops.reduction(dtype, dtype, "welford_reduce", value)`, gets a 3-tuple and
  stores one component. An unused weight buffer is removed by the scheduler.
- **Splits** (`create_multilayer`):
  - If the extent divides evenly by the split, the first stage is `welford_reduce` over block
    ranges.
  - Otherwise it is `welford_combine` over (x, constant 0, constant 1), so masked elements carry
    weight 0.
  - The second stage is always `welford_combine` over the three partial buffers.
- **Combine** (`ir.py` `welford_combine_fn`, `triton_helpers.welford_combine`): Chan's parallel
  update. `delta = mean_b − mean_a`, forced to 0 when the means are equal (Triton) or both are
  the same infinity (`ir.py`). `w = w_a + w_b`. `r = w_b / w`, forced to 0 when `w == 0`
  (Triton). The result is `(mean_a + delta·r, m2_a + m2_b + delta²·w_a·r, w)`.
- **Triton's element update** (`triton_helpers.welford_reduce`) takes the first element as is,
  then applies `delta = x − mean; w' = w + 1; mean' = mean + delta / w';
  m2' = m2 + delta·(x − mean')`, per lane in a looped (`red`) kernel. Lanes are then combined with
  `tl.reduce(…, welford_combine)`.
- **Persistent kernels.** A persistent (`per`), non-cooperative Triton kernel uses
  `welford_reduce_fallback` instead: `mean = sum(x)/N`, `m2 = sum((x − mean)²)`, `weight = N`.

### Triton at real shapes (this machine, Triton venv)

| Case | Kernels |
| --- | --- |
| GroupNorm 1×320×64×64 | split into 5: looped `welford_reduce` writing mean, m2 and weight, then a persistent `welford_combine` writing mean and rstd |
| GroupNorm 2×64×8×8 | one persistent two-pass kernel writing mean, rstd and the normalized output |
| LayerNorm 4096×1280 bf16 | one looped Welford kernel writing mean, rstd and the normalized output |
| `var_mean` 8×300000 | masked split: first stage `welford_combine` over (x, 0, 1) |

Norm kernels write two or three outputs. Our fusion rule admits one, so each output becomes its
own kernel here (proposal, out of scope).

### Our emitter (`descriptor3.rs`)

- `Combiner` has `identity`, `lift`, `combine`, `plane`, `butterfly`, `output`, `arity` and
  `supports`. Accumulators, shared scratch (one array per component) and private merges are
  already slices.
- Validation rejects `arity() != 1`.
- A reduction op has one operand (`args[0]`), checked to be an explicit float32 conversion, and
  one result (`ready[id] = output(...)`).
- Identity values are combined into accumulators on purpose: inactive batched lanes, plane lanes
  beyond the plane count, `planes`/column lanes beyond the extent. So the combine must be neutral
  on the identity.
- `plane()` for non-sum types is a shuffle-XOR butterfly, where every lane computes
  `combine(own, partner)`. That gives the same bits on every lane only if the combine is
  commutative bit for bit. Max/min are, Welford isn't: `mean_a + delta·w_b/w` and
  `mean_b − delta·w_a/w` round differently.
- Division follows the kernel-wide NVRTC `--prec-div` setting (Triton-like approximate division
  by default, exact with Inductor's division-rounding switch), as every other `truediv` does.
  Welford's divisions use the same `FDivOp`.

## Goals / Non-Goals

**Goals:**
- `welford_reduce` and `welford_combine` compile under every non-sum-only strategy, unsplit and
  split, static and dynamic, at an accuracy no worse than Triton's band.
- Every unit sees one result per row.
- No byte change to sum, max or min kernels.

**Non-Goals:**
- Kernels with more than one external output, and therefore the Triton performance floor on
  norms. Both belong to the follow-up change.
- A compensated or extended-precision Welford. It would cost more than Triton without a shown
  need.
- Welford for integer inputs. Inductor never produces it.

## Decisions

### 1. Two combiners, one combine function

`Combiner::WelfordReduce` and `Combiner::WelfordCombine` share their combine, plane step,
output and identity `(0, 0, 0)`. They differ in arity of input and `lift`:
- `WelfordReduce` lifts one value to `(x, 0, 1)`.
- `WelfordCombine` takes its three operands as they are.

Both have accumulator arity 3, all float32.

`supports` follows max/min: everything except `shared_tree_compensated`,
`plane_shared_reference512` and final9.

- *Alternative: lower `welford_combine` to three sums plus pointwise math.* That isn't possible,
  because the combine isn't separable.

### 2. Combine is Triton's guarded form

```text
delta = mean_a == mean_b ? 0 : mean_b - mean_a
w     = w_a + w_b
r     = w == 0 ? 0 : w_b / w
mean  = mean_a + delta * r
m2    = m2_a + m2_b + delta * delta * w_a * r
```

Triton runs this, except that the m2 term is evaluated as `delta * (delta * (w_a * r))`. That is the same number of operations, but an empty lhs then contributes exactly 0 even when `delta²` overflows. Triton's left-to-right product gives inf·0 = NaN there. One guard is added: when `w_a == 0` the ratio is exactly 1. Under approximate division (`--prec-div=false`, as Triton's `div.full`), `w_b / w_b` can be 1 − 1 ulp. With the identity on the left, that drifted constant rows' means, and later combines turned the drift into m2: variance 5e-17 instead of 0 on bfloat16 2×1048579, where Triton gives 0. It costs one compare and select per combine. The `mean_a == mean_b` guard covers equal infinities (the `ir.py`
guard) and is exact for equal finite means. The `w == 0` guard makes the identity neutral for
finite operands: combining with `(0, 0, 0)` returns the other operand unchanged, since r is 0
or 1 and delta·0 is 0. The emitter combines identities into accumulators throughout, so this is
required, not cosmetic.

For an infinite mean, combining with an identity lane gives inf·0 = NaN. Triton's masked lanes
do the same: its `tl.where(mask, …, 0)` lanes enter `tl.reduce`. Because our lane and tree
layout differs from Triton's, whether a given row ends up inf or NaN can differ. So the spec
promises only a non-finite result for rows holding ±inf, not Triton's exact pattern.

### 3. Element update: lift-and-combine, not Triton's incremental formula

The private loop accumulates with `combine(acc, lift(x))`, which is the same code path as every
other strategy step. Algebraically it is the Welford update. Its rounding differs from Triton's
`mean + delta / w'` by about one rounding per step: `delta · (1 / w')` against `delta / w'`, and
`delta² · w_a / w'` against `delta · (x − mean')`.

Accuracy is gated by the spec's band test. If the lift-and-combine form falls outside Triton's
band, `Combiner` gets an `accumulate(acc, x)` hook with Triton's incremental formula, used only
in the private loops. That is an additive fallback; the test decides.

- *Alternative: start with the incremental formula.* That needs a second update path through
  four private-loop shapes (plain, vectorized tail, multi-accumulator batch, final-load batch)
  before there is evidence it matters.

### 4. Plane step: butterfly, then broadcast lane 0

For combiners that aren't bitwise commutative (Welford), `plane()` runs the existing XOR
butterfly and then `plane::BroadcastOp` of each component from lane 0. Lane 0's result is the
one the shared-memory publish, the `independent_planes` writer and the leader stores already
use, so the stored statistics are unchanged by the broadcast. It only makes the other lanes
agree, which fused reduction-scope epilogues (normalized elements) need.

The shared-tree and plane-shared paths already end by reading `shared[0]`, so all units agree
there. A butterfly whose result only lane 0 consumes (the reference512 and final-plane probes)
is sum-only, so they never meet Welford.

- *Alternative: a combine that is symmetric in its arguments.* Making Chan's update bitwise
  commutative would cost extra operations on every step. The broadcast is 3 shuffles per plane
  step.

### 5. Tuple results through `reduction_component`

The reduction op keeps one id, and the Rust side records its full component vector. A new
scalar op `reduction_component {args: [reduction_id], index: k}` yields component k:
- For `welford_*`, k is 0..2.
- For sum/max/min it is invalid, and validation rejects it.

`CubeCLKernel.reduction` returns a 3-tuple of these for Welford types. Validation:
- **Operands:** `welford_combine` has exactly three operands and every other type exactly one.
  Each operand must be an explicit float32 conversion, so the existing `args[0]` check is
  extended to all operands.
- **Closure:** `closure()` gathers the bodies of every operand, not only `args[0]`.
- **Arity:** the arity check admits 1 and 3.

The 4-reductions-per-kernel limit counts reduction ops, not components. Shared memory is sized
by the largest arity in the kernel, which the code already does.

- *Alternative: three ops with an `output` field each, one per component.* This would duplicate
  the reduction in the descriptor and rely on CSE in the Rust emitter to avoid emitting it three
  times.

### 6. Numerical comparison in tuning and tests

- **Tuning.** `tuning.compare` treats Welford kernels like sum kernels (tolerance) for mean and
  m2 outputs, and exactly for weight outputs. `legality.EXACT_REDUCTION_TYPES` is unchanged.
- **Long rows.** The long-row compensated rule already applies only when every reduction is a
  sum, so Welford rows over 4096 elements keep the ordinary plans. That matches Triton, which
  doesn't compensate either.
- **References.** The Triton reference data comes from a generator next to
  `extrema_reference.py`: seeded inputs stored as SHA-256 hashes, plus Triton's outputs and fp64
  statistics. It covers `var_mean` rows and GroupNorm/LayerNorm statistics.
- **Band test.** As in `triton-floor-band`: compare at matched split and check stability across
  seed halves.

### 7. Persistent rows: Welford by default, two-pass only on evidence

Triton uses two-pass for rows that fit one block's registers. Two-pass is usually at least as
accurate as Welford, and with offset data it can be clearly better. We default to Welford
everywhere: one pass, and no second read of the row, because our kernels reload instead of
keeping rows in registers.

The band test includes short rows with offset data. If Welford is outside Triton's band there,
the front end emits Inductor's own `welford_reduce_fallback`. That is two dependent `sum`
reductions, which the emitter already compiles (softmax's max→sum precedent). It is used for
the rows where Triton uses two-pass arithmetic: persistent rows, and low-precision unsplit rows
up to 32768 elements (two-step variance). The thresholds come from Inductor's heuristics and
the reference data's per-case algorithm record.

That fallback is a measured addition, not part of the default plan.

Measured (task 4.3): it was needed. On float32 37×129 rows with offset data, Welford's variance error was about 150× Triton's two-pass; the fallback brings it into the band. The fallback is used exactly where `InductorChoices.should_use_persistent_reduction` holds for the kernel, which is Triton's own rule.

Two-pass kernels are compiled with correctly rounded division (`exact_division`, `--prec-div=true`). Under approximate division, ptxas turns `sum / N` by a constant into `sum · fl(1/N)` and fuses that multiply into every `x − mean`. A constant row then got variance of about 1e-17·v² instead of Triton's exact 0. With `div.rn` the mean of a constant row is exact and nothing is left to contract. The division runs once per row, so the cost is negligible. A per-element division fused into the same kernel would also become exact, which is slower but more accurate; norm epilogues multiply by rstd rather than divide.

### 8. Triton-shaped plans for Welford: many lanes, balanced merges

After decisions 3 and 7, a few cases stayed outside the band. The largest was the saved mean of float16/bfloat16 LayerNorm: about 2 ulp against Triton's 0.7 ulp, 2.6–3.5× in mean absolute error. The float32 Welford statistics were 1.1–1.3×. The normalized output and rstd were in the band.

The cause is the plan's shape. Triton's looped Welford uses R0_BLOCK = 2048 lanes over 512 threads (4 contiguous lanes each) and combines lanes in balanced trees. With float16-quantized inputs, every combine then has equal power-of-two weights (ratio exactly ½), so the mean is nearly exact. Our default plans walk 5–10 strided elements per thread in sequence (ratios 1/3, 1/5, …) before the tree. A float32 simulation of both shapes under correct rounding shows equal errors, which confirms that it is the hardware arithmetic combined with the weights, not the formula.

Our existing multi-accumulator plans (`shared_tree_multi_accumulator`, `plane_shared` and `plane_shared_load_batch` with private accumulators and a halving or adjacent merge) have Triton's shape. They are barred today by sum-specific diagnostic rules. For Welford those rules don't apply, since they record sum qualifications. Group 5 admits them for Welford kernels, adds 8-accumulator variants (2048 lanes at 256 threads, Triton's lane count), and measures accuracy and speed per plan. The outcome, whether it closes the gap or not, is recorded here.

**Outcome (accuracy).** It closes the gap. Three pieces were needed:

1. **Pair-tree vector lanes.** In the vector path, Welford combines a load's contiguous lanes as a balanced pair tree before one combine into the accumulator. Adding them sequentially gave weights 1, 2, 3, 4 and inexact ratios.
2. **An eight-wide plane plan.** `plane_shared` with vector width 8 (128 or 256 threads, 16-byte loads, so 16-bit storage only) reaches Triton's lane count. It is Welford-only: sums keep their qualified plan set and the manifest is unchanged.
3. **A selection rule** (`legality.welford_plan_reason`). Welford row kernels use either `plane_shared` at the widest vector the storage allows, or the four-private-accumulator plane plans. Column plans keep their own layout. The tuning fallback is the widest legal vector plan, else `plane_shared` ×4. A fallback outside the band (the old 256-thread single-chain tree) would win whenever candidates can't be told apart on time (bead 6cl).

Measured per plan, against Triton at matched settings (ratio of mean absolute error, both seed halves):

| Case | before | after |
| --- | --- | --- |
| bfloat16 LayerNorm 4096×1280 mean | 3.5 | 1.00 |
| float16 LayerNorm 4096×1280 mean | 2.6 | 1.00 |
| float32 300×4096 offset variance | 1.2 | 0.98 |
| GroupNorm 1×320×64×64 mean | 1.15 | 0.90–0.96 |

Across the whole 66-case sweep, no compiled case is outside the band (`testing/results/welford-band.json`). The six cases that don't compile are dynamic GroupNorm indexing (4z8.22) and the multi-dimension small GroupNorm (4z8.7).

The single-chain plans really are less accurate here: one lane per thread walking the row (5–10 elements) gives 1.1–3.5×. The four-accumulator plans alone fix float32 rows, but not float16-quantized LayerNorm rows, where only the contiguous pair-tree lanes reproduce Triton's power-of-two weights.

**Outcome (speed).** Measured on the RTX 3080 (non-display, three alternating rounds; `testing/results/welford-timing.json`), comparing the rule's selection with an unrestricted selection (rule off):

- **Unsplit rows** (var_mean 37×129 and 300×4096, LayerNorm 4096×1280): no measurable cost.
- **GroupNorm 1×320×64×64:** 14–18% slower (47 vs 40 µs float32, 30 vs 26 µs bfloat16).
- **Split var_mean 8×300000:** 44% slower (61 vs 42 µs).

Those are exactly the cases where the unrestricted choice (per-row warps or the single-chain tree) is 1.15–1.3× outside Triton's accuracy band. We keep the rule: accuracy must not be measurably worse than Triton, and the speed question is redone in 4z8.21. There, norms become single multi-output kernels and the plan picture changes. Overall CubeCL is 1.2–3× slower than Triton on these graphs. The main cause is one kernel per output, each repeating the reduction, which is recorded on 4z8.21 and not gated here.

First-use tuning also compares each candidate against the baseline. For Welford that is a coarse corruption guard: the finite pattern must match, the weight exactly, and other outputs within 1% of the largest output magnitude. Variance of rows with a large mean is ill-conditioned, so correct plans legitimately differ beyond a sum tolerance (on 37×129 rows with mean 1e3 the old check refused correct candidates and failed the compile). Accuracy is the band test's job.


## Risks / Trade-offs

- **[Sum/max/min bytes change]** Threading multi-operand reductions and component ops through
  validation and closure could touch shared code. → Group 2 is a pure refactor gated on manifest
  PASS and the max/min contract tests before any Welford code lands.
- **[Identity not neutral]** One missed guard in the combine turns empty lanes into NaN. →
  Scenario tests with extents smaller than the cooperation width, every strategy. A hand-authored
  descriptor test combines only identity lanes.
- **[Performance hidden by single-output kernels]** Three kernels per norm will lose to Triton.
  → Recorded rather than gated. Floor and fusion move to the follow-up bead, which blocks the
  scoreboard matrix (`4z8.11`) together with this one.
- **[Register pressure]** Three-component accumulators triple private state, and the
  multi-accumulator plans quadruple that again. → First-use tuning measures the candidates.
  Plans exceeding resource limits fail compilation and are recorded as rejected, as today.
- **[Weight above 2^24]** Weight isn't exact past 2^24 elements. → It can't arise per kernel
  (extent ≤ 1,048,576), and split partial counts stay below 2^24 for any extent Inductor accepts
  here. This is stated in the domain note.

- **[Coarse candidate guard]** The Welford corruption guard in first-use tuning is relative to the largest output in the kernel. A bug confined to rows of much smaller magnitude could pass it. → It is only a guard against gross codegen errors. The hand-authored descriptor tests check every plan against fp64 per row, and the band test checks the selected plans. A per-row bound would need each row's statistics, which a kernel's output doesn't carry.
- **[Column-layout Welford]** For Welford reductions over a non-contiguous dimension, no vector plan is legal. The fallback is then the four-accumulator plane plan, a strided layout, and `coalesced_columns` replaces it only if timing separates them. → Correct, but possibly slow. Column-layout Welford (BatchNorm-style statistics) gets its own accuracy and timing check when a workload needs it; until then this is a recorded follow-up.

## Migration Plan

Each task group lands as one commit after the full suite, the manifest check and a Sonnet
review. Group 2 (multi-operand and component plumbing, no Welford yet) is a pure no-op on
generated code. Rollback is a revert of the affected commit. Artifact keys include the compiler
binary digest, so caches regenerate.

## Open Questions

- Whether lift-and-combine stays inside Triton's band on long rows, or needs the incremental
  `accumulate` hook (decision 3). The band test in group 4 answers it.
- Whether short offset rows need the two-pass path (decision 7). Same test.
