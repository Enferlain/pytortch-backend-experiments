# Design

## Context

See `proposal.md` for the motivation, and the three spec deltas for the required behaviour.

**What runs today.** Compiling and running the corpus through `stage5.backend` loads these modules:

| Area | Modules (lines) |
|---|---|
| Inductor classes, a five-level subclass chain | `stage1/backend` (241) → `stage2/backend` (502) → `stage3/backend` (323) → `stage4/backend` (523) → `stage5/backend` (158) |
| Capability checks | `stage1/capabilities`, `stage2/capabilities`, `stage3/capabilities`; plus `node_reason`/`pre_fusion` in `stage4/backend` |
| Selection and policy | `stage5/choices`, `index_masks`, `legality`, `plan`, `reduction_policy`, `final_plan`, `column_plan`, `reference512_plan`, `reference512_oracle` |
| Runtime chain | `stage0/bridge` → `stage1/runtime` → `stage2/runtime` → `stage3/runtime` → `stage4/runtime` → `stage5/artifact`, `stage5/runtime` |
| Support | `stage1/plan`, `stage1/precision`, `stage1/evidence`, `stage2/indexing`, `stage3/plan`, `stage5/timing`, `stage5/diagnostics`, `fp_hardening/patches` |
| Overlay | `stage1/overlay`: copies the installed `torch/_inductor` and edits it, pinned to wheel commit `5c48869…` |

**Dependencies on Inductor.** Outside the Inductor classes, only two modules depend on Inductor:
`stage5/runtime` (first-use tuning uses Inductor's `PersistentCache`) and `stage5/timing`
(Inductor's benchmarker). Nothing else below the Inductor classes imports Inductor or Dynamo.

**Native side.**
- `stage0/launcher.c` exports `stage0_launch` through `stage5_launch`, plus resource queries.
- The Rust binary `cubecl-stage0` reads a schema-5 descriptor packet and replays semantic
  schemas 2–5 into CubeCL IR. It builds against path dependencies into
  `.stage0-sources/cubecl@6bba5fdf`, which carries the audited FP patch stack
  (`stage2/cubecl-ieee.patch`, `fp_hardening/cubecl-fp.patch`).

**Artifacts.**
- Stored under the fixed `stage5/cache/artifacts/<key>/`, which is untracked in git.
- Each holds `descriptor.json`, `cubecl.ir`, `kernel.cu` and `kernel.ptx`, plus integrity digests.
- The cache key includes digests of Stage 5 source files, the launcher binary and the patch.

**Constraints.**
- Historical verifiers (`stage5/regressions.py`, `stage5/final_gates.py`, `stage4/verify.py` and
  others) import stage modules directly and must keep working.
- The isolated 3.14t venv (`stage0/.venv`) has no Triton and gets its packages through symlinks.
  `.pth` files are excluded, so an editable install won't work there; code runs with the
  repository root on `sys.path`. `pytest` 9.1 is available in that venv.

## Goals / Non-Goals

**Goals:**
- Organise modules by responsibility instead of by stage, with the dependency direction enforced:
  front ends → descriptor compiler → runtime → native.
- Keep the generated code identical, proved by content comparison over the corpus.
- Make the custom-op path demonstrably independent of Inductor.

**Non-Goals:**
- Flattening every class in the subclass chain in this change (see Decision 3).
- Pruning unused descriptor schemas or launcher entry points (follow-up, after equivalence holds).
- Moving or editing the CubeCL checkout, changing toolchains, or porting to Windows.

## Decisions

### 1. Package layout: `cubecl_torch/` at the repository root

```text
cubecl_torch/
  runtime/      KernelArtifact contract, artifact cache, native launch, contract-driven
                validation, stream/storage/lifetime, provenance/FP-state check, tracing
  compiler/     descriptor schema, indexing/expression model (stage2/indexing), physical-plan
                representation (stage1/3/5 plan types), precision rules, Rust/NVRTC adapter,
                descriptor -> launch-contract binding
  inductor/     backend entry, Inductor classes, capabilities, candidate generation, legality
                and admission, reduction policy, final/column/reference512 plans and oracle,
                first-use tuning and selection cache, overlay installer
  ops/          torch.library registration helpers + one proof op
  native/       Rust crate (copied from stage0/), launcher.c, CubeCL patch files, build script
  testing/      equivalence harness, manifest check, contract inventory, GIL/Triton guard
```

Dependency rule:

- `runtime/` imports nothing above it: not `compiler/`, `inductor/` or `ops/`, and not
  `torch._inductor` or `torch._dynamo`.
- `compiler/` imports only `runtime/`.
- `inductor/` and `ops/` use `compiler/` and `runtime/`.

A test enforces the rule (specs: kernel-runtime layer independence; descriptor-compiler layer
dependency).

The split follows what the code does today. Stage 2's runtime evaluates descriptor buffers and
scalars on every launch (`stage2/runtime.py`), and Stage 5's runtime and artifact code import the
reference512, column and final plan policies and hash their sources. Leaving those in `runtime/`
would make it an Inductor-descriptor runtime, which a CubeK-produced kernel couldn't use.

- *Alternative: rename `stage5/` into the package.* Rejected. The stage modules import each other
  by stage name, and the historical verifiers depend on them.
- *Alternative: a separate repository.* Rejected for now. The equivalence reference and the
  scoreboard live here.

### 2. Copy, don't move: the stages stay frozen

Live modules are copied into the package and their imports rewritten. `stage*/` and
`fp_hardening/` stay as they are, so the old gates still run and act as the equivalence reference.
Native sources are copied the same way, and `cubecl_torch/native` becomes the live build. Its
Cargo path dependencies point at the same `.stage0-sources/cubecl` checkout, so compiler inputs
are identical.

- *Trade-off:* duplicated code until the stages are archived. Acceptable, because the stages are
  read-only history.

### 3. Preserve the class structure; regroup by responsibility

The five-level `CubeCLKernel`/`CubeCLScheduling`/`CubeCLOps` chain overrides methods in subtle ways
(precision casts, lazy masks, reductions, RNG, indirect indexing, physical plans). Collapsing it
into single classes is the riskiest part of a refactor whose main requirement is identical output.
So this change keeps the inheritance links but places each level in a module named for what it
adds (`inductor/pointwise.py`, `inductor/semantics.py`, `inductor/reductions.py`,
`inductor/effects.py`, `inductor/physical.py`). Flattening happens later, under the
artifact-manifest check, where it's useful.

- *Alternative: flatten now.* Rejected. It mixes behaviour risk into a structural change.

### 4. Kernel artifact contract

`runtime/` defines `KernelArtifact`, which is everything needed to cache, validate and launch a
kernel, with no reference to descriptors:

- the PTX bytes and entry point;
- the launch shape: grid rule, block shape and dynamic shared memory;
- an ordered parameter list (tensor pointers, metadata buffers, scalars);
- tensor guards (device, dtype, layout/stride/offset rule, extent rule);
- compiler provenance (CubeCL pin and patch state, compiler identity, toolchain).

Guards and parameters are expressed as small data rules the runtime evaluates, not as descriptor
expressions. For the descriptor compiler, `compiler/` builds those rules from the descriptor:
the evaluation `stage2/runtime.py` does today moves into a binding step there. The runtime only
applies the resulting rules.

The contract is kept minimal on purpose. It's shaped by one producer, and is expected to change
when the first CubeK-produced kernel arrives (cube dimensions, plane configuration, CubeCL launch
metadata). A runtime test launches the Stage 0 programmatic `sin(x)+1` PTX through a hand-written
contract, so the seam is exercised from day one (spec: kernel-runtime, producer-independent
contract).

- *Alternative: keep descriptor evaluation in the runtime.* Rejected. The runtime would be
  unusable for any non-descriptor producer.

### 5. Cache identity split

The runtime artifact key covers producer inputs (for the descriptor compiler: descriptor packet
and physical plan) and compiler provenance (compiler binary, launcher, FP patch state). Selection
and admission policy sources (final, column and reference512 plans and oracle, reduction policy,
legality) leave the runtime key and go into the selection-cache identity owned by `inductor/`.
Editing a policy then re-runs selection without invalidating unchanged compiled artifacts.
Generated IR/CUDA/PTX is unaffected, because keys aren't inputs to code generation.

### 6. Structural equivalence by artifact content, not cache keys

Cache keys include source and binary digests, so they differ by construction. The harness runs
each of the 168 corpus cases in fresh processes, once on the Stage 5 backend and once on the
package backend. The Stage 5 side records regions through a hook on its compiled-kernel
constructor; the package reports them directly.

For each case, the harness builds an **ordered list of region records**: region identity (kernel
name order within the graph), role, descriptor digest, physical plan, and IR/CUDA/PTX digests. It
compares the lists with multiplicity, never as sets. It also compares:

- eligible candidate sets;
- bitwise outputs under `baseline`;
- selection under injected deterministic timings. Both benchmarkers are replaced by the same
  timing table, and the chosen candidate and selection-cache key must match. Real `first_use`
  winners aren't compared, because they depend on timing noise.

A passing run is saved as the structural manifest (spec: inductor-backend, manifest
reproducibility). Scoreboard parity compares region counts and rejection reasons in fail-fast and
inventory modes.

- *Alternative: compare digest sets.* Rejected. Two regions sharing an artifact would collapse,
  hiding a mapping regression.

### 7. Porting historical contracts: classify, then port

The frozen gates hold negative and runtime contracts that the corpus doesn't cover. Running them
against the package isn't practical, because they import stage modules directly. Instead,
`testing/contract_inventory.md` lists every assertion in these scripts:

- Stage 5: `components`, `controls`, `proof_controls`, `descriptor_controls`, `launch_controls`,
  `cache_probe` (init/reload), `compiler_compare`;
- Stage 4: `verify`, `runtime_checks`, `verify_bounds`, `counter_guard_probe`, `components`;
- Stage 4's `regressions` (stages 0–3).

Each assertion is classified as either:

- **live contract**: bad layout, dtype, device or extents; corrupt cache; resource mismatch;
  malformed descriptor; stream lifetime and offsets; rejection reasons. These are ported into
  package tests.
- **historical**: experiment-specific probes or superseded stage behaviour. These are recorded
  with the reason they aren't ported.

This keeps coverage of real contracts without re-creating the full Stage 5 gate cost.

### 8. Launch host-overhead parity

Moving descriptor binding out of the launch path changes per-launch host code, which identical
IR/CUDA/PTX doesn't cover. A microbenchmark launches the same prepared corpus kernels through both
implementations in alternating rounds. The package's median warm per-launch host time must not
exceed Stage 5's by more than the measured noise. This is the exploration-tier version of the
scoreboard's host-overhead concern.

### 9. Cache location

`runtime` resolves its cache root in this order:

1. an explicit configuration call;
2. the `CUBECL_TORCH_CACHE_DIR` environment variable;
3. a per-user default: `$XDG_CACHE_HOME/cubecl_torch`, or `%LOCALAPPDATA%` on Windows.

Paths are built with `pathlib` only, with no Linux-specific assumptions.

### 10. First-use tuning stays in the Inductor front end

The tuning cache (built on Inductor's `PersistentCache`) and the benchmarker depend on Inductor,
so they move to `inductor/` unchanged, with a hook for injecting deterministic timings (Decision
6). The proof custom op uses a fixed physical plan. An Inductor-independent tuner for custom ops
is follow-up work.

### 11. Tracing is opt-in and bounded

Stage 4 appends a full graph string to a module-global `TRACE` list on every compile, which grows
without bound in long training runs. In the package, tracing is off by default and enabled with
`CUBECL_TORCH_TRACE=1` (equivalent to `CUBECL_STAGE5_TRACE`). Generated code doesn't change.

### 12. Proof custom op from a hand-authored descriptor

`ops/` registers one op, `cubecl_torch::sin_plus_one` (the Stage 0 vertical case). Its schema-5
descriptor is written directly in Python and compiled by `compiler/`. It has a fake
implementation for tracing and a fixed scalar pointwise plan. Tests check eager use without
Inductor, a mixed compiled graph (supported ops → op → supported ops) where it stays one opaque
extern call, and concurrent calls from several threads. A general descriptor-authoring API is
follow-up work, to be designed with the first CubeK-based op.

### 13. Project metadata

A root `pyproject.toml` declares the `cubecl_torch` package, ruff settings and the pytest
configuration. Tests run with `stage0/.venv/bin/python -m pytest cubecl_torch/testing`. They
aren't installed into the venv. The native build is `cubecl_torch/native/build.sh`, which
verifies the CubeCL patch state, then runs `cargo build --release` and builds the launcher.

## Implementation notes (deviations found while applying)

- **Front-end launch hook.** Vector and final-plan legality (`stage5/legality.reason`) is
  re-checked before every launch and depends on tensor pointers and the queried target. It is
  not expressed as a data rule: `LoadedKernel` accepts an optional `launch_check(scalars,
  tensors)` callable that the Inductor front end attaches for those plans. It is not part of
  the cached contract. All other guards (dtype, device, exact layout, storage span, extents,
  output numel, disjointness, positive scalars, column shapes via a finite `table` rule) are
  data rules.
- **Metadata export.** The copied native crate gains one export, `cubecl_torch_metadata`,
  which serialises CubeCL `InfoBuilder` words for 0..33 scalars (the Stage 0 programmatic PTX
  has none). It shares the existing serialiser; code generation is unchanged.
- **Schemas.** The compiler accepts semantic schemas 3 and 4 only, the ones a schema-5 packet
  replays. Stage 2-only schema-2 descriptors are rejected as malformed.
- **Precision edits** (`stage1/precision`) are Inductor source edits used only by the
  overlay, so they live in `inductor/precision.py`, not `compiler/`.
- **Tracing.** The class chain's module `TRACE` lists became a `TraceLog` view onto the
  runtime's bounded, opt-in trace; events that stringify graphs are built only when tracing
  is on. Rejection messages are unchanged (including their stage wording) for scoreboard
  parity.
- **Selection identity.** Selection-cache keys hash compiler provenance, which differs by
  construction between Stage 5 and the package. Selection equivalence compares the key data
  without provenance, plus the chosen candidate.
- **Nondeterministic outputs.** Float `atomic_add` with colliding indices (embedding backward)
  is order-nondeterministic within Stage 5 itself (three Stage 5 runs, three gradient
  digests). The harness runs each `baseline` case twice per implementation; outputs that do
  not repeat bitwise within an implementation are compared with the Stage 4 tolerance
  (2e-5) instead of bitwise.
- **Generated kernel names** in the wrapper are `cubecl_kernel_<n>` instead of
  `cubecl_stage5_<n>`; region identity in the harness is call order.

## Risks / Trade-offs

- **The layer split and the cache-identity split are real code changes, not file moves.**
  Binding now happens in `compiler/`, and keys are computed differently. → The structural
  equivalence gate (byte-identical IR/CUDA/PTX, bitwise outputs, injected-timing selection), the
  ported live contracts and the host-overhead parity check together cover the behaviour that can
  change.
- **The artifact contract is shaped by a single producer.** → Keep it minimal, exercise it with
  the Stage 0 PTX test, and expect to revise it with the first CubeK op rather than over-design it
  now.
- **The overlay edits Inductor source by string matching**, tied to one wheel commit. → Copy the
  installer verbatim, keep its pinned-commit and diff-hash checks, and test that the installed
  `torch/_inductor` files are unchanged.
- **Hidden reliance on stage-relative paths.** Examples: `ROOT / "column_templates.json"`,
  `stage2/cubecl-ieee.patch`. → Grep for `Path(__file__)` and `ROOT /` in every copied module;
  package data files move with their code. Missing files fail loudly at import.
- **A native rebuild in a new directory changes the binary bytes.** → Equivalence compares
  generated IR/CUDA/PTX, not binary digests. Each environment records its compiler identity.
- **Duplicated code drifting** between `stage*/` and the package. → The stages are frozen history;
  fixes go only to the package. AGENTS.md/CLAUDE.md will say so.
- **The dirty-source check uses git on the CubeCL checkout.** → Make the source path a parameter,
  and test it against a temporary copy or a fake git repository, never the real checkout.
- **Equivalence runtime:** about 168 cases × 2 backends × first-use compilation. → Run it in
  background mode. It's a one-time gate; later changes use the manifest check.

## Migration Plan

1. Build the package next to the stages (no behaviour change for existing scripts).
2. Run the structural equivalence, contract-inventory tests, host-overhead parity and scoreboard
   parity checks, then record the manifest.
3. Switch `scoreboard/run_one.py` to `cubecl_torch.inductor`.
4. Update AGENTS.md/CLAUDE.md and STAGE5_REPORT.md: the package is the live backend, and the stages
   are history.

Rollback is to revert step 3. The stages are untouched throughout.

## Open Questions

- The final package name (`cubecl_torch` is a working name). Renaming only touches imports.
- Whether a later change should vendor the CubeCL checkout as a git submodule or keep the
  `.stage0-sources` convention. This doesn't affect this change, which keeps the existing checkout.
