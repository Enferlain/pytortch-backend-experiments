# cubecl_torch: target architecture

Status: **draft for review** (2026-11-04). It settles how we approach the project, so that later
changes are measured against one statement instead of fragments in memories, beads and reviews.
Sources: the discussion of 2026-11-04 (bead zah.11), the external review
[`external_pilot_110426.md`](external_pilot_110426.md), and the bd memories `settled-approach`,
`work-direction-2026-11-04`, `decision-verification-is-a-diagnostic`,
`feedback-fix-the-system-not-route-around`, `upstream-changes-are-fine` and
`pre-consolidation-is-history`. The module tables in
[`cubecl_torch/README.md`](../cubecl_torch/README.md) describe the code as it is; this document
says where it is going.

## The goal

Use the CubeCL ecosystem where PyTorch's Inductor uses Triton today, without losing to Triton on
accuracy, speed, memory use or first-call cost. "The ecosystem" is CubeCL, CubeK, and Burn where
it shows how they are meant to be driven. Standalone custom ops come on top, at any time.

> `cubecl_torch` is an Inductor codegen backend. It translates Inductor-scheduled computation
> into native CubeCL-ecosystem programs while preserving PyTorch's semantic and runtime
> contracts. It does not maintain its own physical GPU scheduler: CubeK and CubeCL own physical
> mapping, blueprint inference and tuning, and missing capabilities are fixed there, in our copy
> under `sources/`. Triton is the differential oracle and the accuracy and speed floor, not the
> source of production policy.

Short form: **consume Inductor's semantic schedule and facts; let the CubeCL ecosystem choose the
physical realization.** Inductor is upstream of us; Triton's implementation of Inductor is beside
us.

## Why this needs writing down

The native plan catalog, its admission rules, and a first-use tuner that compiles, output-checks
and times every candidate were not designed for `cubecl_torch`:

- Stage 5, an experiment, chose "named physical alternatives plus empirical selection" to explore
  performance while keeping numerics as the frozen baseline had them.
- The consolidation (2026-10-20) carried it over unchanged, on purpose: it was a behaviour-preserving
  refactor proved by byte-identical output.
- Every change since added strategies and rules to it (Welford, multi-output, wide workgroups,
  cubecl-tile, the bridge), because "add a strategy and its admission rules" was the established
  way to add a capability. Nobody re-decided the structure.

The pieces of the architecture that did exist (purpose, scope, CubeCL/CubeK first, Triton as the
floor) agree with each other. The layer between Inductor's kernel and the GPU program was never
defined, and the inherited catalog filled it. Today's problems sit there: the rule soup, the
3–15× first-call cost, plans admitted outside Triton's band, and fixes that kept converging on
Triton's structure one patch at a time.

## What Inductor gives us, and what it decides for Triton

Inductor produces two kinds of output. We consume the first and not the second:

1. **The semantic schedule and facts: what to compute.** Fused kernel bodies and which operations
   share a kernel; iteration spaces, reduced axes, index expressions, masks; dtypes and upcasts;
   layouts, strides, alignment and divisibility; access-pattern hints (`INNER`, `OUTER`); memory
   planning and kernel order; mutation and other effects.
2. **Choices made on Triton's behalf: how to realize it on Triton.** Block sizes, warp counts,
   pipeline stages, persistent or looped rows, config lists and their autotuning.

Inductor draws this line itself. `SIMDKernel.should_use_persistent_reduction` is left to
subclasses ("defined in subclass") and Triton's decides it; Inductor's Halide backend receives the
same schedule and hands realization to Halide's autoschedulers.

**The grey zone.** Some decisions made with Triton in mind are already in the graph we receive:
split reductions (`num_splits` uses the SM count, skipped when a backend declares
`REDUCE_TO_SINGLE_ELEMENT`), fusion (the backend's scheduling can refuse), Welford as a tuple
reduction (`TUPLE_REDUCTION`). We consume these by default; they are part of the computation and
split reductions set accumulation order. Where the ecosystem would do better, the channel is
Inductor's: declared `BackendFeature`s and the scheduling hooks, not re-deciding inside our
backend.

## Who owns what

Triton does three jobs under Inductor; Inductor makes a fourth set of decisions itself.

| Job | Inductor + Triton | Owner here |
| --- | --- | --- |
| Operations, fusion, split and tuple reductions, memory planning | Inductor | Inductor; we declare capabilities |
| Kernel program for a fused body | Inductor writes a Triton block program | `cubecl_torch` translates the body; the program is a CubeK routine or CubeCL kernel |
| Physical mapping: threads, planes, vector loads, partials, reduction tree | the Triton compiler, through its layouts | CubeK routines (blueprints) and cubecl-tile |
| Configs, tuning, launch, caching | Inductor's heuristics, Triton's autotuner and launcher | CubeK blueprint inference, CubeCL's tuner and runtime |
| Numerics switches (`emulate_precision_casts`, division rounding, FMA contraction) | Inductor config | Inductor config, honoured by `cubecl_torch` |
| Streams, storage lifetime, aliasing, launch validation | Inductor's Triton launcher | `cubecl_torch`, PyTorch's runtime contract |

CubeCL is lower level than Triton: nothing in CubeCL itself maps a block program to threads. The
ecosystem fills that layer with CubeK routines and cubecl-tile, and Burn shows how to drive them.
Stage 5 filled it with its own structures, which is where the per-scenario rules come from. They
are the price of owning structures, not inherent to the problem.

What stays ours: the Inductor integration, PyTorch's runtime contract (the runtime bridge), the
qualification and measurement harnesses, and our changes to the ecosystem in `sources/`.

## Capability-directed lowering

For each scheduled kernel `cubecl_torch` asks a small question: *which ecosystem mechanism can
express this program?* It does not ask the large one (topology, vector width, accumulators, merge
order, launch shape); that belongs downstream. Real fused kernels have several regions (RMSNorm:
`x*x → sum → rsqrt → *x`), so routing is by what a mechanism can express, not by a kind label.

Mechanisms, most native first:

1. **CubeK reduce with the body in its read and write hooks.** CubeK's `ReduceArgs` lets the
   caller define how operands are read and results written; Burn's `FusedReduceArgs`
   (`.stage0-sources/burn/crates/burn-cubecl-fusion/src/optim/reduce/args.rs`) runs fused code
   there and calls `reduce_kernel_virtual` (`sources/cubek/crates/cubek-reduce/src/launch/base.rs`).
   For us the body is what `CubeCLKernel` already records from Inductor.
2. **Elementwise and reduction blocks composed**, as Burn's `reduce_broadcasted`
   (`burn-cubecl-fusion/src/optim/reduce_broadcasted`).
3. **CubeK matmul and attention; cubecl-tile.**
4. **Plain CubeCL kernels** for pointwise work.
5. **None fits:** extend the ecosystem in `sources/`, as deeply as needed. A new local physical
   strategy is the last resort and is classified like any other rule.

Whether 1–4 cover Welford, multiple outputs, argmax ties and masked bodies is not known; it is
established by running them, not by reading.

## Selection, accuracy and tuning

**Why Triton needs no rule soup.** Triton has one template per kernel kind: each thread
accumulates its share in a loop, then a tree merges the partials. Configs only change parameters
of that one structure, so accuracy is a property of the template, and choosing a config needs no
output check. Inductor gives one config per known reduction class and a short list (about six)
for the default class and persistent rows, which it times. Selection and tuning work when
candidates share a structure; they become rule soup when candidates are different kernels, as our
catalog's are.

So:

- **Qualify structures, choose parameters.** A CubeK routine with its blueprint parameters is one
  structure; it is qualified offline against Triton's band, and the selector only picks
  parameters.
- **Selection is the ecosystem's, and how it selects is open.** It can be blueprint inference,
  tuning, cached tuning, a hybrid, or something better we build into CubeK/CubeCL; what is fixed
  is that it lives in the ecosystem and meets Triton's floor, including first-call cost.
- **For now, the default is CubeK's blueprint inference,** a rule applied at compile time, as
  Triton's heuristic is, because the tuner does not yet meet the first-call floor. This is the
  current migration state, not a permanent rule. Where inference falls short of the accuracy
  floor, a blueprint forced from Inductor's config is the stopgap (`mapping="inductor"`), and the
  fix goes into CubeK's inference (bead zah.10). The mapped cube showed what inference needs:
  enough partials per row, merged as trees.
- **No runtime verification or costly measurement by default.** The per-candidate output check and
  CUPTI multi-round timing are development diagnostics. Offline qualification
  (`testing/qualification.py`, `testing/results/qualification-<gpu>.json`) replaces the runtime
  check; a plan outside the band on any GPU is not admitted.
- **CubeCL's tuner is our tuner, and we change it until it meets our needs.** Measured on Burn's
  default path it costs 2.3–2.6 s per sum on a first call (compiling all 5–6 candidates first) and
  its picks are noisy (single-launch event timing). Those are fixed in `sources/cubecl` (batched or
  kernel-duration timing, candidates pruned by blueprint inference or priority groups, the
  persistent tune cache), as deeply as it takes, never routed around by writing another beside it.
  `inductor/tuning.py` shrinks to glue. Once the tuner meets the floor, whether it runs on the
  default path is decided by measurement, not by today's default.

## Triton's role, and how we compare

Triton is the differential oracle, and the place to look when we lose: its PTX/SASS, mapping, load
pattern and reduction tree explain why, and the difference becomes a general ecosystem capability.
A CubeCL kernel physically unlike Triton's that is in its band, at least as fast, and no worse in
memory and first-call cost is a success.

The comparison is per kernel: the same fused kernel from Inductor, generated once by Triton and
once by us, same inputs and launch context, judged on the accuracy band, pinned kernel time under
ncu, memory traffic, first-call cost and launch cost, at matched settings and without tolerance
margins. Whole workloads (scoreboard: SDXL, Anima, optimizer) come on top to catch integration
effects.

## Classifying rules

Every project-local rule is one of three kinds, recorded where it lives:

| Kind | Where it lives | Examples today |
| --- | --- | --- |
| **PyTorch contract** | stays in `cubecl_torch` | bridged launches don't `record_stream` by default, as PyTorch ops don't (memory `bridge-record-stream-decision`); bridged launches refuse misaligned addresses a vectorized load would fault on (`inductor/bridged.py`) |
| **Physical policy** | moves to CubeK/CubeCL | which blueprint a reduction gets; lanes-first, plane-tree and unit elements already moved (`sources/NOTES.md`) |
| **Triton-parity experiment** | temporary, with a deletion criterion | `mapping="inductor"` (`bridged._inductor_blueprint`): deleted when CubeK's inference offers an in-band structure (zah.10); `legality.cubek_speed_reason`: deleted per plan when it qualifies; the uncompensated native plans refused for float32 sums over 2048+ elements (`legality.LEGACY_FLOAT32_SUM_MIN`, from the plan qualification): a restriction on the frozen legacy path, deleted with those families; Welford's use of Inductor's persistent heuristic (`inductor/codegen.py:863`): physical policy in our semantic layer, to classify |

The full sweep of `inductor/legality.py`, `inductor/reduction_policy.py`, `inductor/tuning.py` and
`native/src/plan.rs` has not been done yet.

## How changes are made

- **Name the mechanism.** A new strategy, rule or tool says which mechanism it implements and where
  that mechanism lives in Triton, Inductor, CubeK, CubeCL or Burn. If it can't, it doesn't go in
  without the user's agreement.
- **Say whether it moves toward or away from this target,** in the OpenSpec design step.
- **Second fix of the same kind: stop** and look for the mechanism. Check beads (with notes),
  memories, results and history for earlier encounters first.
- **Run upstream end to end before designing around it.** When an ecosystem mechanism falls
  short, determine why before creating any parallel project-local mechanism. Then extend, replace,
  extract (from Burn, for example) or restructure the ecosystem machinery in `sources/` as deeply
  as necessary; the existing implementation is not something to preserve for its own sake. What
  is ruled out is a redundant system beside it, built without understanding why the existing one
  isn't suitable ("it's slow, so we write our own"). No change is shaped for upstream acceptance;
  real upstream issues go in `sources/NOTES.md`.
- **Baseline first.** Reach Triton's baseline Triton's way before improving on it, and don't go
  deep on one topic before the whole picture exists.
- **One measurement toolkit.** Pinning, the band, ncu timing and first-call timing written once.
  Pinning has five copies today (`cubek_band.pin`, `tile_band.pin`, `timing` pinned select, test
  helpers, `qualification.py`); the next change that touches one folds them together.

## Where we are

| Piece | Today | Target |
| --- | --- | --- |
| Front end | `CubeCLScheduling(SIMDScheduling)`, registered with `register_backend_for_device`; admission and fusion in `inductor/admission.py` | unchanged in kind; fusion declared from ecosystem capabilities |
| Kernel body | recorded by `CubeCLKernel`, compiled by our Rust descriptor compiler (`native/src`) in a separate process | the same body replayed in CubeK hooks or a CubeCL kernel, in-process through the bridge |
| Physical plans | native catalog: 42 plans (`compiler/plan.catalog`), 11 Welford, 3 tile; legality in `inductor/legality.py` and `native/src/plan.rs` | **frozen**: no new strategies; reference and fallback; families retire one at a time |
| Ecosystem path | runtime bridge (`native/bridge`, `runtime/bridge.py`, `inductor/bridged.py`): plain row sums, 7 candidates (`plan.bridge_catalog`); the mapped cube within 2% of Triton on every class of 4096+ elements except bf16 300×4096 on the 3090 (1.09×), in band on all 40 classes measured. For float32 only that forced, Triton-like blueprint is in band; CubeK's own inferred blueprints are not, so "CubeK passes" means "CubeK configured like Triton passes" | CubeK provides and selects in-band structures itself, so qualification confirms structures instead of becoming a per-blueprint admission list; every capability the native path has, through mechanisms 1–4 |
| Selection | first use compiles, output-checks and times ~30 candidates: 0.8–3.9 s first call against Triton's 0.19–0.65 s | selection in the ecosystem at Triton's first-call floor (for now blueprint inference, tuning opt-in until the fixed tuner earns its place); no runtime check |
| Qualification | 3090 table: 1613 pinned runs, 103 outside the band in 14 cases, 3 default selections outside | both GPUs, on the current build, gating admission |
| Ecosystem source | `sources/cubecl`, `sources/cubek` at upstream 15135eec / 74921bf4 plus our commits | grows with every capability we need |

## Migration

In order; the corpus manifest, the suite and the Triton band gate each step.

1. **Qualify and refuse.** Qualification tables for both GPUs on the current build; plans outside
   the band on either are refused (temporary; deleted when the family retires).
2. **First call without runtime tuning or checks.** Ecosystem path: blueprint inference, forced
   Inductor blueprint as the stopgap. Native path, while it remains: a fixed choice among its
   qualified plans, recorded as temporary Triton parity. This must not grow into a class-to-plan
   rule table, which would be the catalog again. Target: Triton's first-call cost.
3. **Fused-reduction bridge.** `CubeCLKernel`'s recorded body → `sources/cubek` `ReduceArgs` read
   and write, as Burn's `FusedReduceArgs` → `sources/cubecl` machinery; RMSNorm's `sum(x²)` first,
   against Triton. Whatever is missing is changed directly in the layer it belongs to.
4. **CubeK offers the in-band structures itself;** `mapping="inductor"` is deleted (zah.10).
5. **Classify every rule** (table above), moving physical policy into `sources/`.
6. **Expand by semantic capability:** prologue, epilogue, several reductions (softmax, LayerNorm),
   Welford, multiple outputs, argmax ties, masks, split and cooperative reductions; Burn mined for
   each.
7. **Retire native families** once the ecosystem path covers their cases and meets the gates; the
   native path stays as the differential reference until then.
8. **Fix CubeCL's tuner** in `sources/cubecl`, then measure whether it belongs on the default path.

## Open questions

- How much of the native descriptor compiler (`native/src/semantic.rs`) survives as the translator
  from Inductor's body into the hooks, and how much CubeCL's own expansion replaces? Step 3
  answers it.
- Which `BackendFeature`s and fusion limits do we declare to Inductor once routing is by
  capability, and do they change what Inductor schedules?
- What is the fallback when no mechanism fits and the native catalog has retired?
- Is first-call cost the largest gap on real workloads? Steady-state step time on SDXL and Anima
  (scoreboard) and the cache-warm first call have not been re-measured against this plan.
