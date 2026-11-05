# Design

## Context

- **How a plain sum reaches CubeK today.** The bridge (`cubecl_torch/native/bridge`, CubeCL's
  CUDA client inside the torch process) calls `cubek_reduce::reduce` with `TensorArgs` for plain
  row sums only (`bridged.py`, `bridge_sum_rows`). Everything else uses native plans.
- **The native op emitter.** It writes semantic ops straight into a CubeCL `Scope`
  (`semantic::emit`, `semantic::index`). Index expressions are in terms of the output coordinate
  (`linear`) and the reduction coordinate (`reduction`); `analysis::bind_coordinate` binds the
  latter.
- **The ecosystem crates.** CubeK's reduce and CubeCL are built from `sources/`, with the same
  crate paths in both Rust crates.
- **Burn's fused reduce** (`burn-cubecl-fusion/src/optim/reduce`):
  - `FusedReduceArgs` implements CubeK's `ReduceArgs`.
  - `read_input` runs `fuse_on_read` and `write_output` runs `fuse_on_write`.
  - The launch (`reduce_kernel_fused`) calls `init_tensors::<FusedReduceArgs>` and then
    `reduce_kernel_virtual`.
  - Shapes and strides come from a reference layout (`ref_shape`, `ref_stride`).
  - `tune.rs` tunes over CubeK's Unit, Plane and Cube routines, each with `BlueprintStrategy::Inferred`.
- **What Inductor records for RMSNorm**, captured 2026-11-04 from `CubeCLKernel` on this build:
  - Forward: one row kernel (`sum(x²)`, a per-row `rstd` store, a per-element store).
  - Input gradient: one row kernel (`sum(g·w·x)` and a per-element store).
  - Weight gradient:
    - at 64×1024, one column reduction of `g·x·rstd[r]` over 64 rows;
    - at 8192×1280, a split first stage (on the RTX 3080, 32 splits: 40960 outputs over 256
      rows; `floor_div`/`mod` in its load indices; per-output store only), then a plain column
      sum over the splits. The split count is Inductor's choice per GPU.
  - Only the weight gradient is the plain `FusedReduceArgs` shape, so it is the first proof.
- **Triton's comparable mechanism.** Inductor's Triton codegen turns such indices into
  block-pointer shapes and strides with `BlockPatternMatcher` (`codegen/block_analysis.py`,
  `match_mod_div_block_expr`).

## Goals / Non-Goals

**Goals:**

- Answer the architecture question concretely: can the recorded body run in CubeK's read and
  write hooks? Either it runs at the Triton floor, or the missing ecosystem abstraction is named
  and fixed in `sources/`.
- One mechanism for the whole class: any prologue, sum and per-output epilogue that the native
  emitter can express. Nothing specific to RMSNorm.

**Non-Goals:**

- Per-element passes over the row (`reduce_broadcasted`), several reductions, Welford and
  non-sum reductions. They are refused with a reason here and are the follow-on changes.
- Changing native plans, the native tuner (architecture step 2) or `mapping="inductor"` (zah.10).

## Decisions

1. **Take Burn's structure; emit with our emitter.**
   - From Burn, copied and changed as needed: the `ReduceArgs` implementation, its state, and
     the launch that calls `init_tensors` and `reduce_kernel_virtual`.
   - `read_input` and `write_output` call `intrinsic!(|scope| …)` and emit the recorded ops with
     `semantic::emit` into the scope CubeK's kernel is being expanded in.
   - *Rejected: translating our ops into Burn's `FuseOp` IR and using `burn-cubecl-fusion`.*
     It would pull in Burn's fusion engine, and that IR lacks some of what our ops carry:
     division precision, wrapping integer ops, index arithmetic.
   - *Rejected: a new op interpreter.* It would be a second emitter for the same ops.

2. **The bridge crate links the native crate as a library.** The native crate already builds an
   `rlib`. A small public surface is added: emit a body, emit an index, validate a description.
   Both crates name the same `sources/cubecl` paths, so cargo builds one CubeCL.
   - *Rejected: moving the emitter into a third crate now.* That is a mechanical move to make
     later, once it is clear the emitter outlives the native plans.

3. **The body travels as comptime configuration.**
   - The canonical JSON of the body (ops, buffers, reduction coordinate binding) is the comptime
     value, hashed into the kernel id. It is parsed once per expansion.
   - CubeK caches compiled kernels per body and per blueprint, as it does for Burn's
     `FuseBlockConfig`.

4. **Buffers are a sequence of typed global buffers.** The input argument holds every buffer the
   body reads or writes, each with its storage type defined at launch (Burn's `GlobalArgs`
   pattern, copied minimally). The emitter receives them as its `arrays`. The output argument
   holds no buffer of its own. `write_output` stores through the recorded store ops and their
   indices, so non-affine store indices need no layout for CubeK's output.

5. **CubeK gets the real layout, recovered as Inductor recovers it for Triton.**
   - At codegen, the Python side runs Inductor's own `BlockPatternMatcher` on the index of each
     load that feeds the reduction operand. It records the result as a virtual shape and strides
     over (output dimensions…, reduction dimension).
   - **No reference is chosen.** Loads that span every output and reduction dimension (`g` and
     `x` in the weight gradient) must agree on that layout, and the agreed layout is the one
     given to CubeK. Loads over fewer dimensions (`rstd[r]`, a broadcast) don't compete. When
     full-size loads disagree, the candidate is refused with both layouts in the reason.
   - That is deliberate. Picking among disagreeing layouts is physical policy. "Largest load" or
     a similar rule in `cubecl_torch` would quietly become permanent. When a case needs it, take
     Burn's reference selection (`engine/launch/plan.rs` `ReferenceSelection`; `layout.rs`: a
     choice costs the traffic of the inputs that disagree with it), or find the abstraction
     CubeK's reads need and add it in `sources/cubek`.
   - The split weight gradient (32 splits on the RTX 3080) becomes shape [32, 1280, 256], strides [327680, 1, 1280], reduce
     axis 2.
   - `shape_input`/`stride_input` return that layout. CubeK's vectorization and its
     parallel-or-perpendicular choice therefore see that the reduction axis is strided.
   - In `read_input`, CubeK's offset is decoded back into coordinates using that layout. The
     output coordinate becomes `linear`, the reduction coordinate is bound, and each load's
     recorded index is evaluated as written. Burn handles non-reference tensors the same way.
   - When no layout is found, the candidate is refused with that reason. No guessed layout is
     used.
   - *Rejected: a contiguous [outer, extent] virtual tensor.* It runs correctly, but CubeK
     would then choose its mapping for a memory order that isn't there.

6. **Vector lanes.**
   - When CubeK's vectorization gives `SizeIn > 1`, the vector runs along the layout's
     contiguous axis.
   - In the first version the body is emitted once per lane and the results are assembled into
     the vector. Loads along that axis are then lane-adjacent, so coalescing is kept.
   - If ncu shows the per-lane form below Triton, the body is emitted on vector-typed values,
     as the native compiler's `reduction/vector.rs` does.
   - The decision is recorded with the measurement.

7. **Selection: CubeK's routines with inferred blueprints.**
   - The new kind adds three candidates to the existing tuner: Unit, Plane and Cube, each with
     `BlueprintStrategy::Inferred`, the set Burn tunes over.
   - No forced blueprint, no `mapping="inductor"` and no per-class admission rule.
   - If inference misses the Triton floor, the cause is fixed in `sources/cubek` (zah.10's
     work), not routed around here.
   - The CubeCL tuner replacing our tuner for these candidates is architecture step 8.

8. **Numerics and accumulation.**
   - The reduction's accumulator dtype is the recorded `accumulator_dtype`. CubeK's `In` is the
     operand dtype after the prologue's casts (float32 for the bfloat16 cases).
   - Contraction and division precision come from the same numerics policy the plain bridge
     passes.
   - No compensation is added. The band decides (see the spec).

9. **Measurement uses the existing toolkit.**
   - Before our version, Burn's fused reduce is run on the same computation and shapes on CUDA
     and timed, to know what the mechanism gives upstream.
   - **Two oracles, kept apart.**
     - *Replay and index correctness* (did we read and write the same elements?): inputs made of
       small integers, so every sum is exact in any order. Each output is compared exactly with a
       float64 reference, with a guard region around the outputs. The descriptor-compiled result
       may serve as a structural cross-check here, since exact inputs make it order-free.
     - *Numerical acceptance:* only the band against Triton and a high-precision reference.
       Native plans are not a numerical oracle; step 1 found some of them outside Triton's band.
   - The Triton floor is a diagnostic gate. A miss is traced (ncu source and stall views, CubeK's
     inferred blueprint) and fixed in `sources/cubek` or `sources/cubecl`, or recorded on a bead.
     It is never closed here with a forced blueprint or an admission rule.
   - Ours is then measured with the existing harnesses:
     - accuracy: the band, with the weight-gradient cases added to the reference cases;
     - speed: pinned ncu timing (memory `ab-codegen-changes`);
     - first call: `cubek_first_call.py`, extended to these graphs.
   - Inductor+Triton in the Triton venv is the floor.

## Risks / Trade-offs

- **[Risk] The emitter's `arrays` model doesn't match a launch argument's expanded buffers**
  (raw pointers against CubeCL arrays). → Adapt at the bridge boundary. If CubeCL lacks a way to
  hand an expanded buffer to scope-level code, add it in `sources/cubecl` and note it in
  `sources/NOTES.md`.
- **[Risk] Linking both crates duplicates or conflicts CubeCL features** (`cubecl-cpp` default
  features off in the native crate). → Check the dependency tree in the first build task. Align
  the features in both manifests.
- **[Risk] CubeK's routines assume the virtual offset is affine in its shape and strides.** →
  That is what decision 5 provides. Non-affine parts stay in our recorded indices, never in
  CubeK's layout.
- **[Risk] Per-lane emission multiplies the body's instruction count.** → Measured (decision 6).
  Vector-typed emission is the planned fallback.
- **[Trade-off] Three more candidates per covered kernel** add compile and timing to first use
  on the native tuner. → Accepted under step 2's decision. First-call cost is measured and
  reported, not optimized here.

## Migration Plan

Additive. The new candidates are offered only where the coverage reason admits them. Withdrawing
the kind's catalog entries restores today's selection. The corpus manifest changes only for the
kernels where it is offered; that diff is reviewed before re-recording.

## Open Questions

- How the reference layout is chosen when full-size loads disagree (decision 5). No first-proof
  case needs it. It is answered from Burn's reference selection or a CubeK abstraction when a
  case does, not with a rule in `cubecl_torch`.

- Should the bridge also reach CubeK's `shared_sum` or split routines for the weight gradient's
  second stage (a plain column sum over 32)? Answered by measurement; it doesn't change the
  mechanism.
