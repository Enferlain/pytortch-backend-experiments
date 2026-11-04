# Proposal

## Why

On the 3080 qualification, 137 of 187 reduction kernels have only native plans. The ecosystem path
(CubeK reduce through the runtime bridge) takes nothing but a plain row sum, because the bridge
hands CubeK a tensor rather than the kernel's computation. This is step 3 of `docs/architecture.md`
(bead zah.11). Burn's `FusedReduceArgs` shows that CubeK's `ReduceArgs` hooks can carry a fused
read and write computation. The question that decides the architecture is whether the body
`CubeCLKernel` records from Inductor can be replayed there the same way. If it can't, the reason
names the abstraction missing from CubeK or CubeCL.

## What Changes

- **New bridged reduction kind.** A bridged reduction whose `ReduceArgs` implementation replays
  the recorded semantic ops:
  - the operand's loads and per-element ops in `read_input`;
  - the per-output ops and stores in `write_output`.
  It emits through the native compiler's existing op emitter into the scope CubeK's kernel is
  expanded in, as Burn's `fuse_on_read`/`fuse_on_write` do. One kernel launch, built from
  `sources/cubek` and `sources/cubecl`.
- **Real layout for CubeK.** The virtual tensor CubeK reduces presents the operand's real layout
  (shape and strides recovered from Inductor's index expressions, Burn's reference-layout
  mechanism). CubeK's own blueprint inference then sees the actual memory order.
- **First proof: RMSNorm's weight gradient**, as Inductor emits it.
  - At 64×1024 it is one column reduction of `g·x·rstd[r]`.
  - At 8192×1280 it is a split first stage over a `floor_div`/`mod` index, then a plain column sum.
  - Both are run in float32 and bfloat16, against Inductor+Triton: Triton's band, ncu kernel time
    and first-call cost.
- **Burn first.** Burn's own fused reduce is run end to end on the same computation before ours is
  built, so we know what the mechanism yields upstream.
- **Selection.** The new kind enters selection beside existing plans, as bridged plans do now.
  CubeK infers its blueprint. No new forced blueprint or admission rule is added for it; a gap is
  fixed in `sources/` or recorded against zah.10.
- **Out of scope here:** the RMSNorm forward and input-gradient kernels (`sum(x²)` and
  `sum(g·w·x)` with a per-element pass over the row). They need Burn's `reduce_broadcasted`
  composition (architecture mechanism 2) and are the follow-on change, designed from this one's
  results.

## Capabilities

### New Capabilities

- `fused-reductions`: lowering an Inductor reduction kernel's recorded body (prologue,
  reduction, per-output epilogue) through the CubeCL ecosystem's reduce. Covers what is offered,
  the layout given to CubeK, numerics, and the Triton floor for these kernels.

### Modified Capabilities

None in `openspec/specs/`. The unarchived `torch-runtime-bridge` change's `runtime-bridge` spec
says a fused body gets no bridged candidate ("Fused body" scenario). That scenario is narrowed to
bodies outside `fused-reductions`' coverage, in that change's spec.

## Impact

- **`cubecl_torch/native`:**
  - The op emitter (`semantic::emit`, index lowering) is exposed to the bridge crate.
  - `bridge/` gains the replaying `ReduceArgs` and its launch.
  - The bridge crate depends on the native crate as a library.
- **`cubecl_torch/inductor`:**
  - `bridged.py` gains a launcher for the new kind.
  - `legality.py` gains its coverage reason.
  - `compiler/plan.py` gains its bridged catalog entries.
- **`sources/cubek`, `sources/cubecl`:** changed directly if the replay needs something they
  don't provide.
- **Testing:**
  - Band, timing and first-call harnesses gain the weight-gradient cases.
  - The corpus manifest changes only where the new candidate is offered.
- **Not affected:** native plans and their generated code. The Triton reference venv is unchanged.
