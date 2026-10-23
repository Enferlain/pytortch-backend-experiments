# Design

## Context

See proposal.md for motivation. The facts that shape the approach:

- **Front end.** `CubeCLKernel.reduction` (`inductor/reductions.py`) converts the value to float32
  and emits a `reduction` descriptor op carrying `reduction_type`, `source_dtype` and
  `accumulator_dtype`. A low-precision result is rounded to its semantic dtype before fused
  consumers see it. Admission is checked twice before fusion: `semantic_checks.check_operation`
  (sum only, float types) and `reduction_checks` (at most four reductions per kernel, one
  reduction domain).
- **Descriptor schema 3** already carries `reduction_type` and `accumulator_dtype` per reduction
  op. The Rust validator (`descriptor3.rs` around line 258) rejects anything but
  `sum`/`float32`. No schema bump is needed for single-value types.
- **Emitter.** `descriptor3.rs::build_physical_codegen` implements every reduction strategy in
  one function. Sum shows up in three forms:
  - identity constants `0.0`
  - `Add` (and `compensated_add`) in private accumulation and in the shared-memory trees
  - `plane::FSumOp`, or a hand-written shuffle-XOR tree (`reduce(..., &[16, 8, 4, 2, 1])`) whose
    combine is an add, for cross-lane steps
- **Tuning.** Candidates come from `plan.catalog()` filtered by `legality.reason`. First-use
  tuning checks each candidate against the baseline with `tuning.compare`, which allows a
  tolerance for schema-3 kernels and skips the signed-zero check for them.
- **CubeCL ops.** The audited CubeCL source already has `cmp::FMaxNanOp`/`FMinNanOp`. On
  `__CUDA_ARCH__ >= 800` they emit PTX `max.NaN.f32`/`min.NaN.f32`, the same instruction Triton's
  `tl.maximum(..., PropagateNan.ALL)` compiles to. They use `__hmax_nan` for half, and an
  `isnan`/`fmaxf` fallback below sm_80. CubeCL also has `math::IsNanOp`.
- **Measured behavior** (pinned PyTorch, RTX 3090):
  - Eager and Triton both propagate NaN in max/min reductions and in `maximum`/`minimum`.
  - Triton orders −0 < +0 everywhere: `amax(+0,−0)` is +0 and `amin` is −0, for any order.
  - Eager's reduction zero sign depends on order, and eager `relu(-0.0)` returns −0.
  - `max.dim` values go through max-with-index (first occurrence), which belongs to the arg family.
  - `config.strict_signed_zero` (default False) only switches Triton to compare/select reductions.

## Goals / Non-Goals

**Goals:**
- A combiner abstraction that the next reduction types (argmax, Welford, prod/any, online
  softmax) plug into without restructuring the strategies again.
- Max/min results bitwise identical to Inductor+Triton apart from NaN payload, at Triton's
  instruction cost.
- No byte change to any existing sum kernel.

**Non-Goals:**
- Multi-value accumulators in use. The data path is shaped for tuples, but only 1-tuples are
  admitted. Welford (`4z8.15`) turns on arity 2–3 together with its own combine type and tests.
- New strategies or retuning. Max/min reuse the existing strategy shapes, and only eligibility
  changes.
- Ordered signed zero below sm_80. The audited fallback (`fmaxf`) doesn't order zeros. The
  inventory records sm_80+ as the domain and no workaround is added.
- Honouring `config.strict_signed_zero`. With ordered `max.NaN` the default already gives
  order-independent zero signs, so the strict variant changes nothing observable for us.

## Decisions

### 1. Combiner as a Rust enum, not a trait object or a descriptor-side program

`enum Combiner { Sum, Max, Min }` lives next to the strategies. Each variant provides:
- `identity(scope, dtype)`
- `combine(scope, a, b)`
- `plane(scope, v, widths)`
- `arity()`
- `supports(strategy)`

The descriptor names the type (`reduction_type`), and the compiler maps it to a variant.

- *Alternative: a trait mirroring CubeK's `ReduceInstruction` and implemented with `#[cube]`.*
  Our emitter builds IR directly on a `Scope`, not through `#[cube]` kernels, so a trait would add
  generic plumbing for three variants. We reuse CubeK's design (identity, ordered combine,
  `fuse_accumulators` for the plane step, explicit coordinate tracking later for arg types), not
  its type machinery.
- *Alternative: describe the combine as an op list inside the descriptor.* This is more general,
  but it moves numerical semantics out of the audited compiler into data and makes per-type
  validation harder.

### 2. Sum keeps its exact instruction sequence

`Combiner::Sum` returns exactly what the code emits today: `0.0` constants, `Add`, and the existing
`FSumOp` or add-tree. Strategies are refactored so that each of those sites calls the combiner,
with no reordering. The manifest check (structural digests of the 168-case corpus) is the proof.
A digest change means the refactor reordered something and is a bug.

### 3. Max/min combine with `FMaxNanOp`/`FMinNanOp`; plane step is a shuffle-XOR tree

Max/min combine with `FMaxNanOp`/`FMinNanOp` and use the identities −inf and +inf. For the
plane step, max/min use the existing shuffle-XOR tree shape, with the combiner's `combine` in
place of the add.
- *Alternative: CubeCL's plane max op.* Its CUDA lowering uses the plain (NaN-ignoring) max, which
  would break NaN propagation across lanes.
- *Alternative: compare/select (`a > b || isnan(a) ? a : b`).* It's order-dependent for zeros and
  costs more instructions than one `max.NaN`.

Because `max.NaN` is commutative and associative on every non-NaN input (zeros ordered), the
result doesn't depend on strategy, split, private-accumulator count or tree order. That's what
lets the spec require bitwise agreement with Triton.

### 4. Strategy eligibility is declared by the combiner

`Combiner::supports(strategy)` is the one source of truth. The Python side gets the same table
(`legality.reason` rejects with "strategy X does not support reduction type Y") so that rejected
candidates are recorded, never compiled. Ineligible for max/min:
- `shared_tree_compensated`: compensation is a sum error term.
- `plane_shared_reference512` and `independent_planes_final9`: frozen qualified sum arithmetic
  with exact-template admission.

Every other reduction strategy, `coalesced_columns` included, is eligible. The Rust validator
enforces the same table, so a hand-authored descriptor can't bypass it.

### 5. Accumulators are slices, arity checked at the boundary

Private accumulators, shared-memory slots and plane values become `Vec<IrValue>` of length
`arity()` per logical accumulator. Shared memory is sized as slots × arity. Only arity 1 passes
validation in this change. Threading tuples now, while the strategies are being touched anyway,
avoids a second pass over the same code for Welford. The cost is a few `[0]` indexings. Without
a client the tuple path is exercised only by arity-1 tests. That's acceptable because
validation keeps arity > 1 unreachable.

### 6. Split partials combine with the same reduction type

For max/min, Inductor's second stage is again `max`/`min` over the partials, with masked
padding defaulting to the identity, so no new code path is needed. A tuple type's second stage
is a different type (`welford_combine`) that it registers as its own combiner later. The
`final_plan.py` constant-final template (`reduction_type: "sum"` hard-coded) stays sum-only and
checks the type instead of assuming it.

### 7. Front-end admission and tuning comparison

- **Admission.** `reductions.py` and `semantic_checks.py` accept `max`/`min` with float inputs.
  The value is converted to float32 as for sum: exact for float16 and bfloat16. The
  low-precision rounding of the result stays and is exact, because the max of rounded inputs is
  representable. The four-per-kernel limit is reworded to count reductions, not sums.
- **Tuning comparison.** `tuning.compare` uses exact comparison for max/min: bitwise equality
  except NaN payload, including signed zero. Any first-use candidate that differs from the
  baseline is a codegen bug, not reassociation.

### 8. Pointwise `maximum`, `minimum`, `isnan`

These are added to the shared pointwise op table (`descriptor2.rs` op arity and emission, and
`semantics.py`):
- float `maximum`/`minimum` → `FMaxNanOp`/`FMinNanOp` in float32 compute
- `isnan` → `IsNanOp` returning bool

Integer `maximum`/`minimum` in index arithmetic keep their current path. `fmax`/`fmin` need no
op of their own, but Inductor's decomposition (`where(isnan(b) | (b < a), a, b)`) needs a bool
`bitwise_or`. It is added next to the existing bool `bitwise_and` (`effects.py` predicate types,
`descriptor4.rs` `or`), admitted on the general pointwise path for bool operands only. Their zero
signs follow the decomposition, as Triton's do, since both run it. `clamp_min`/`clamp_max` lower to `maximum`/`minimum` and are covered with them.

## Risks / Trade-offs

- **[Refactor changes sum code]** Threading a combiner through about 1,100 lines of strategy code
  could reorder an instruction. → Do the refactor as its own task group, before any max/min code,
  and gate it on manifest PASS plus the full suite.
- **[Fusion exposes new combinations]** Admitting max/min lets Inductor fuse kernels such as
  softmax's max followed by a sum that depends on it. → Dependent reductions in one kernel
  are rejected or accepted exactly as the current emitter already decides for sums. A softmax
  graph is included in the tests to record the outcome. If it's rejected, it moves to `4x9.10`
  with a recorded reason, not into this change.
- **[Performance]** The shuffle tree for max/min may cost more than `FSum` does for sum. →
  Spec'd floor against Triton with a recorded measurement. Triton also uses a shuffle-based
  `tl.reduce`, so parity is expected. A gap is investigated with SASS before any change.
- **[Pre-sm80 GPUs]** Signed-zero ordering isn't guaranteed there. → Documented domain limit,
  stated in the inventory note, and not tested.
- **[Contract test drift]** The Stage-gate inventory lists "max rejected" as a ported assertion. →
  Update that entry to say capability added, superseded by acceptance tests, with this change as
  the reason.

## Migration Plan

Each task group lands as one commit after the full suite, the manifest check and a Sonnet review.
The refactor group lands first and is a pure no-op on generated code. Rollback is a revert of the
affected commit. Artifact keys include the compiler binary digest, so caches regenerate
automatically.

## Open Questions

- Whether `coalesced_columns` (column reductions) needs a different identity placement for max
  than for sum. It is answered by its own test in the max/min group and doesn't change the
  approach.
