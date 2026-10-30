# Design

## Context

See `proposal.md` for motivation and `docs/research/sk8/README.md` for the measurements. This
design adds what the spike left open: how tile kernels get into the production compiler route.

- **Spike kernels ran on CubeCL's runtime.** `tile_proto` and `fused_probe` launch through a
  `Client`, the tile `Launcher` and `TileArg`. Production compiles in a separate process with no
  CUDA runtime and launches through `launcher.c` from `cubecl_torch/runtime/launch.py`.
- **The compiler route builds kernels with `KernelBuilder`** (`native/src/reduction/emit.rs`):
  buffers, then `int64` scalars, compiled with `CompilationOptions::default()`.
- **Grid constants are not a kernel property.** `cubecl-cpp`'s metadata pass emits `info_st` by
  value only when `supports_features.grid_constants` is set; `cubecl-cuda`'s runtime sets it and
  the default leaves it off. Off, every kernel takes one `info` pointer, our ABI. Bead `f05.9`
  describes what the spike's runtime produced, not what our route would.
- **The spike's `sums` buffer does not exist in production.** Both kernels stored the row sum in an
  extra buffer and read it back. A real sum consumer has no such output.
- **Two legality layers:** `inductor/legality.py` and `native/src/plan.rs`. A candidate must pass
  both, and `PhysicalCandidate.from_dict` accepts only catalog members.
- **Our FP patch does not touch `cubecl-tile`**, so the copy starts from the pinned source.

## Goals / Non-Goals

**Goals:**

- Tile kernels in the production route, with no change to the launcher, metadata or runtime.
- Inductor's fused loop bodies, emitted by our emitter, as the only memory access in those kernels.
- Each gap sk8 found closed in our copy, or the plan held out of selection with a recorded reason.

**Non-Goals:**

- Deriving tile levels from Inductor's config, or from Triton's rules. Stage 1 states one layout.
- Using tile for matmul, attention, softmax or pointwise kernels.
- Removing any existing plan, even if tile wins everywhere. That is a later decision.
- Bitwise equality with Triton, which would also need its thread ownership (sk8).

## Decisions

### 1. Copy the crate whole, unmodified first

`cubecl-tile/src` goes to `cubecl_torch/native/cubecl-tile`, with `Cargo.toml` rewritten to path
dependencies on the pinned checkout and an `UPSTREAM.md` recording the revision and each local
change. The first commit adds it unused, so the manifest can show no generated code moved before
anything else changes. Upstream's `tests/` stay out: they need the `cubecl` crate's test runtime.

Alternatives: extending `cubecl-fp.patch` keeps the code in the checkout but puts a growing fork
into a patch meant for upstream candidates, and into every CubeCL rebase. Copying only the modules
we use makes merges from a fast-moving upstream harder, with no effect on generated code.

### 2. Build tile kernels with our `KernelBuilder`; no `Launcher` or `TileArg`

A tile kernel is built like every other kernel: our buffers and `int64` scalars. The kernel-form
`Partitioning` is constructed in the builder, its runtime extents taken from our scalars instead of
a launch argument. Reduction state lives in shared memory; all memory reads and writes go through
our emitter (decision 3). Nothing else is registered, so the artifact has the existing ABI and
metadata, and the runtime, guards and `launcher.c` do not change. This retires `f05.9`.

Alternative: keep `TileArg` and `Launcher`, in our copy taking hardware properties in place of a
`Client` (it uses one only for plane size, limits, barrier support and vector sizes). That keeps
more upstream code in use, but `TileArg` registers per-operand scalars and tensor metadata in
CubeCL's typed layout, while our metadata library pushes only `i64` scalars and buffer lengths. It
would need a typed metadata contract through compiler, artifact and runtime, for operands we do
not use. It stays the fallback if decision 2 fails group 2 of the tasks, since a client-free launcher
is a small change in code we own, and the metadata contract is what would then be designed.

### 3. Inductor's body runs in recipes; hints and row sum come from our side

Reads go through procedural recipes that call `semantic::emit`, as `fused_probe` does:

- **Hints:** the recipe marks row loads retained in the reduction pass and streaming in the
  storing pass, through the descriptor ops' own flags, as the emitter does. No DSL hint syntax is
  needed, since no memory tile is read. This settles `f05.5` for stage 1.
- **Row sum:** after the slots drain, each unit reads the sum from shared memory into a value and
  passes it to the emitter as the reduction's result. There is no extra buffer, and the sk8 pitfall
  of a second `const __restrict__` binding cannot occur.
- **Stores:** per-row and per-element stores are the emitter's, reused from the reduction emitter's
  per-element pass (vector stores included) rather than the DSL's `product` and `copy_from`, which
  panic on a runtime axis. This settles `f05.10`.

### 4. One layout: CubeK's `InSlots`

One row per cube, 16 planes × 32 units, planes taking 128-wide steps in runs
(`Levels::planes_across`), a cube-level block reducer with 16 registers, slots drained by writer
order. It gave 308-line kernels and about 31 ms compile; the cyclic combine unrolled to 1,188 lines
and 67 ms. The block is `[512,1,1]`, already legal in `launcher.c`, with a grid of one cube per row.
Multi-row cubes wait for a later stage.

### 5. Plans: one strategy, with width as the alignment class

New strategy `tile_row_sum` at `[512,1,1]`, `plane_width=32`, vector widths 1 and 4, plus 8 for
16-bit rows, with existing `PhysicalCandidate` fields only. A step is `32 × width` elements wide. A
width above 1 needs a row of whole steps, because a vector line cannot be masked, so legality adds
an `extent % step == 0` fact to `signature()`. Width 1 serves any row, masked where it does not
divide. Rows below 2048 are rejected as a selection policy, as for other 512-unit plans; the plan
stays structurally legal, which is what lets short rows be pinned and validated.

Structural rule, in both layers: exactly one reduction, a sum; no mask region; not part of a split.

The new fact only adds to vector legality. A plan with a width above 1 still goes through the
existing checks in `structural_reason` and `plan.rs`: row base and stride alignment, unit stride,
storage width and pointer alignment. `tile_row_sum` takes no shortcut around them.

Alternative: a separate strategy per width. That adds names without adding a rule.

### 6. Numerics: an explicit summation order; keep the plan out if it misses

The order becomes a compile-time setting in our copy, chosen by the kernel builder and selecting
code at expansion, with no runtime branch: how vector lanes, units and planes are combined. It
defaults to upstream's order, so behavior only changes where a kernel asks for it. `tile_row_sum`
asks for Triton's: one accumulator per vector lane combined as a tree, units combined xor 16→1
(`space/share.rs`), planes combined as a tree where the slots are read. `order_model.py` predicts
that reaches 7.04e-6 / 0.191 against Triton's 6.96e-6 / 0.187. Each choice goes through
`welford_band` on GPU, compared with the setting on and off.

Alternative: changing the lowering outright. It is less code, but it specializes a generic
mechanism for one numerical target: any later tile kernel would inherit Triton's order, and each
upstream merge would conflict with it.

`numerical_reason` rejects uncompensated plans on float32 rows over 4096, because they were outside
the band. Tile plans get a branch in that rule only for classes measured in band. If float32 long
rows miss, the tile plan stays rejected there with its reason, `shared_tree_compensated` keeps
those rows, and 16-bit rows and shorter rows still get tile plans. Compensation is not a tile option
in stage 1: it is not a monoid, so it needs the multi-value cell of `8im`.

Alternative: bit-for-bit Triton order. That needs its thread ownership, which the layout does not
give, and the band does not ask for it.

### 7. Bounds: redirect the coordinate before evaluating

In our copy, `read_checked` and `read_masked` compute the in-bounds test first and evaluate the
recipe at a redirected coordinate, then select. Memory tiles already work this way, it needs no
branch, and it fixes every recipe, not only ours.

This makes purity part of the recipe contract, since an out-of-range read still evaluates the
recipe once at the redirected coordinate. A recipe may load but must not write or have any other
side effect. Upstream documents recipes as memory-free, so our copy's docs change to say this.
Stage 1 meets it: stores stay outside recipes, and kernels with RNG, mutation or indirect ops never
reach a tile plan, because `structural_reason` keeps them on the frozen baseline.

A zero-length row has no coordinate to redirect to, and the launch guard allows a row length of 0.
The walk over such a row should have no steps, so no recipe runs; task 3.2 tests that.

Alternative: pass the predicate into recipes so the emitter's guarded loads apply. That adds a
branch per element and leaves a recipe that forgets it unsafe.

### 8. Validation hook: a test-only compile option

The kernel's memory recipes gain CubeCL's `validate_index` under a compile option that defaults to
off, keyed into the artifact, so a validating kernel never shares a key with a production one.
`validate_index` reports with a device print, so the first use confirms the report reaches the test
through `cuLaunchKernel`. If it does not, the check counts violations into an extra test-only
buffer instead. A negative control runs with the redirect disabled and must report.

### 9. Judge by the existing machinery

Selection, band tests and pinned `ncu` timing are the existing ones (memories `ab-codegen-changes`,
`triton-floor-band`). Plans are pinned to compare them, never compared through first-use selection.

## Risks / Trade-offs

- [`Partitioning` may not build outside `Launcher`; its levels or `planes_across` may assume state
  the launcher sets] → Group 2 is a gate on one real kernel before anything else is built on it;
  the fallback is in decision 2.
- [The reducer's destination tile is memory-backed in the spike] → Its code reads only the tile's
  space and width, so group 2 uses a shared-memory or procedural destination; if that fails, the
  fallback applies.
- [The DSL's unit and plane indices may not match a `[512,1,1]` block as they do `cube_dim` from
  levels] → Group 2 checks outputs and the plane ownership against `CUBE_DIM`, and adds a
  `[32,16,1]` launcher geometry only if the block must keep that shape.
- [Order fix misses the band on float32 long rows] → Decision 6: the plan stays rejected there. Sk8's
  fused RMSNorm gain on those rows is then not delivered in stage 1, which would be flagged.
- [Scalar recipes lose where vector loads matter] → Width 4 and 8 recipes come in group 3; pinned
  timing decides, and a plan that loses is not a candidate.
- [The copy drifts from a fast-moving upstream (187 commits in two months)] → `UPSTREAM.md` keeps
  merges deliberate. Maintenance cost is one of the things stage 1 measures.
- [First-use tuning gets slower with more candidates] → Long rows only, about 31 ms per kernel;
  compile time is recorded in the results.

## Migration Plan

Additive. The first build invalidates every artifact key, once; caches rebuild on use. Rollback is
removing the strategy from `catalog()`, after which selection is as before, or reverting the crate.
The bead `f05.8` stays open but outside this change; `8im` still depends on `f05.11`.
