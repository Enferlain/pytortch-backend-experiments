# Design

## Context

See proposal.md (Why) and the research notes on bead `4z8.21` (2026-10-27). Current state that
shapes the approach:

- **Where the single-output rule is enforced.** It is a contract checked at every layer:
  - fusion: `admission.fusion_allowed`, "one disjoint external output required";
  - codegen: `CubeCLKernel.codegen_kernel`, which requires one output name and no in-place
    buffers;
  - "the last buffer is the only write": `compiler/descriptor.py`, `tuning.AutotunedKernel`,
    and in Rust `semantic.rs::validate_description` and `reduction/analysis.rs`;
  - reduction stores must be all per-row or all per-element: `compiler/descriptor.py`,
    `compiler/binding.py`;
  - the `output_numel` guard on the last tensor: `binding.py`, `runtime/launch.py`;
  - first-use selection's scratch and `compare()`, on the last tensor only;
  - the pointwise vector adapter's "requires one store" rule: `legality.structural_reason`.
- **What already works with several outputs.**
  - `plan.rs`'s `outputs == 1` counts rows per workgroup, not output buffers.
  - The emitters already iterate over every store and address `arrays[buffer]`:
    `pointwise/physical.rs`, and `reduction/emit.rs::stores()`.
  - The descriptor already lists buffers with an `access` field, and every store names its
    buffer.
- **The buffer ceiling is 6 everywhere** (5 for reductions):
  - Python: `MAX_BUFFERS`, fusion's "one to four external inputs", pre-fusion
    `max_inputs` 4 and 5, `supported_input_counts` up to 5;
  - Rust: validators 5 and 6, `cubecl_torch_metadata` ≤ 6;
  - launcher: `stage5_launch`'s `void *args[7]`.
- **What the probe showed (rules forced open in a scratch run).**
  - Inductor's scheduler fuses into Triton's kernel shapes: LayerNorm 1, var_mean 1, GroupNorm 3.
  - `emit()` already de-duplicates structurally, through `semantic_cache` keyed by the full op
    record. So `welford_reduce` appears once. The masked split first stage still had three
    `welford_combine` reductions.
- **No storage-reuse buffers reach us.** `backend.py` sets `inplace_buffers=False`, so Inductor
  never hands our kernels an `in_out_ptr`-style buffer.
- **Largest Triton kernels in the target workloads:**
  - GroupNorm+SiLU forward and backward: 9 buffers (6 in, 3 out);
  - LayerNorm backward: 7;
  - in-place AdamW: 7;
  - functional Adam: 8.

## Goals / Non-Goals

**Goals:**

- Several outputs per kernel, through every layer, with single-output descriptors and
  generated code byte-identical (manifest).
- One reduction per distinct reduced value per kernel, masked operands included.
- A total buffer limit that covers the measured training kernels, with room to spare.

**Non-Goals:**

- Outputs that alias inputs or each other (`4z8.9` adds that contract).
- Fusion decisions of our own. We take Inductor's scheduler fusion and only stop rejecting it.
  Triton's kernel shapes come from that same scheduler.
- New physical plans for multi-output kernels. The existing catalogs apply. Any timing gap is
  diagnosed per kernel first.

## Decisions

### 1. Buffer order stays "inputs, then outputs"

The descriptor keeps its canonical order: every read buffer, then every write buffer (one or
more). The contract changes from "only the last buffer is written" to "a non-empty prefix of
reads, then a non-empty suffix of writes".

- **Validation.** Each store targets a write buffer, each load a read buffer, and each write
  buffer is stored at least once. A reduction kernel stores each output exactly once: a
  per-row and a per-element store to one buffer would race. A pointwise kernel may store an
  output several times, which each unit runs in program order at its own element. The live
  contract `test_nested_lazy_load_and_predicated_store` (a fill, then a predicated store under
  nested masks) depends on that; agreed with the user 2026-10-28 during task 1.4.
- **Why.** A single-output descriptor is unchanged byte for byte: same order, same access
  strings. So manifest digests and cached artifacts of existing kernels stay valid. The runtime
  argument order (`input_names + output_names`, as `call_kernel` writes) also stays.
- **Alternative: free per-buffer access in any order.** More general, but it changes nothing we
  need and makes every validator handle interleaving.
- **Alternative: a new schema number.** Rejected: schema 3 and schema 4 descriptors with one
  output are already valid instances of the widened contract.

### 2. Total buffer limit of 16

A kernel may have up to 16 buffers in total, inputs plus outputs.

- **Pre-fusion.** The per-node input limit becomes 15. A node needs one output, so 15 inputs is
  the most that fits the total.
- **Fusion.** "One to four external inputs" becomes "at most 16 external buffers". The lower
  bound (at least one external input) stays.
- **Every other layer moves to the same constant.** `MAX_BUFFERS`, the Rust validators,
  `cubecl_torch_metadata`, and `stage5_launch` (17 pointers including the info buffer).
- **Why 16.** It covers the measured maximum (9) with headroom for wider optimizer variants and
  backward kernels. The cost is fixed: a 17-entry pointer array, and metadata words that already
  scale with the buffer count. CUDA's 4 KB parameter space is far away (17 pointers are 136
  bytes).
- **Alternative: no limit, as in Triton.** Triton has none, but every layer here validates
  counts explicitly, and a bound keeps the native ABI fixed-size.
- **Not touched.** The historical `stage4_launch` and the Stage 1–4 metadata entry points stay
  as they are (kernel-runtime: live historical contracts are preserved).
- **New rejection reason.** It names the limit ("more than 16 buffers per kernel"). The
  scoreboard's reason table (`scoreboard/families.py`) and the coverage inventory map it to this
  change's family.

### 3. Fusion admission

`fusion_allowed` drops the single-output rule. It keeps:

- the RNG/indirect separation;
- storage dtypes;
- the input lower bound;
- the reduction-domain rules;
- the four-reductions bound.

It adds the buffer limit, and keeps kernels with `rand` or `atomic_add` single-output. Both
are effect kernels, whose replay and bypass rules assume one destination.

Inductor computes `outputs` as today: writes not consumed inside the fusion, plus graph outputs,
plus buffers with users outside it.

### 4. Codegen: several output names, and canonical masked regions

**Output names.** `codegen_kernel` accepts one or more output names. It still rejects
`args.inplace_buffers`, which `inplace_buffers=False` keeps empty: a non-empty value is a
contract violation, not a feature gap.

**Masked regions.** `masked()` builds its body under a scoped CSE, so two nodes masking the same
load produce records whose nested ids differ, and `emit()`'s cache never matches them. The
change adds a canonical key for a masked record: the body's ids are renamed by position before
hashing (alpha-renaming). A second identical masked region then reuses the first region's value.
Everything downstream (the `welford_combine` operands, then the reduction itself) then keys
equal, so the existing record cache de-duplicates the reduction, as Triton's
`cse.reduction_cache` does.

- **Alternative: key the reduction on its inputs' structure, Triton-style.** Not used: Triton's
  key is the operand CSE variable, so equal operands matter in both designs, and ours lack
  them only because of the masked naming.
- **Task 3.1 confirms the cause before the fix** by dumping the three operand records of the
  split first stage.

### 5. Reduction emission: one pass for per-element stores, one leader branch for per-row stores

`reduction/emit.rs::stores()` emits each store with its own dependency closure: per-row stores
inside an `if unit == 0` branch, and per-element stores each in their own loop over the extent.
With several stores:

- **Per-element stores share one loop.** All per-element stores share one strided loop over the
  extent, with the union of their closures replayed once per element. This is Triton's single
  second loop. Without it, LayerNorm with weight and bias saving its output twice, or a kernel
  with two normalized outputs, reloads the row once per store.
- **Per-row stores share one leader branch.** All per-row stores share one `unit == 0` branch
  with the union of their closures. This is CubeK's `IndicesWriter` shape: one reduced value,
  several writers, one commit.
- **Single-store kernels are unchanged.** A group of one store emits exactly the IR it does
  today, so single-store kernels keep their code (manifest).
- **Order.** Per-row stores come before the per-element loop, as in Triton's LayerNorm, where
  rstd is stored before the second loop.

`analysis.rs`'s rule "an outer store must not depend on a varying reduction coordinate"
already applies per store and stays.

### 6. Pointwise emission

`pointwise/physical.rs` already gathers each store's lane values and emits one vector store per
store op. `legality.structural_reason`'s "vector adapter requires one store" becomes: every
store is direct-linear and aligned, like every load. `scalar.rs` and `column.rs` need no
change: column plans match exact templates, which are single-output.

### 7. Launch contract: one extent guard per output

`binding.py` emits one `output_numel` guard per write buffer, keyed by buffer index:

- **Per-row** reduction stores have the outer extent.
- **Per-element** reduction stores have outer × extent.
- **Pointwise** stores have `numel`.

The guard for the last buffer keeps today's serialized form (no index field), so a
single-output artifact's guards and digest are unchanged. Each other output gets the same
guard kind with an explicit buffer index (`runtime/artifact.py` documents both forms).
`runtime/launch.py` checks each guard against its own tensor. `launch()` still returns the
last tensor; callers ignore the return value.

Mixed store scopes (`{"outer", "reduction"}`) become legal in `descriptor.py` and `binding.py`.

### 8. First-use selection and comparison per output

`tuning.select` allocates scratch for every write buffer, runs the baseline into it, and checks
each candidate output by output. The comparison class is chosen per store from the reductions
its value depends on, walking the dependency closure the way `legality.stored_components`
already walks casts:

| Store depends on | Comparison |
| --- | --- |
| Only max/min, or the Welford weight component | bit-exact |
| Any Welford mean or m2 | Welford corruption guard |
| Any other sum | sum guard (or the qualified-plan guard where `qualified_for_runtime`) |
| No reduction (pointwise kernel) | pointwise tolerance |

`exclusion()` records the failing output's index. These classes are today's kernel-level
predicates (`exact_reduction`, `welford_weight_output`, `welford_reduction`) applied per store,
so a single-output kernel is classified exactly as now.

Plans that are single-output by construction keep their checks:

- `reference512` (`reference512_plan.validate` requires one output);
- `final9` (`final_plan.semantic_reason` requires two buffers);
- column templates.

### 9. Plan policy for kernels mixing a sum with a Welford reduction

**Today.** The Welford plan rule and the Welford catalog apply only when every reduction is
Welford. A kernel that also reduces a sum keeps the sum-qualified plans and the sum baseline. Its
Welford accuracy has not been measured.

**Approach.** First measure that mix with the band test (`welford_band.py` extended with mixed
cases) under the current policy. If any Welford output is worse than Triton, the policy becomes
the intersection: a plan must pass the sum qualification and `welford_plan_reason` (evaluated
against the Welford part), and the fallback is the first plan legal under both.

- **Why measure first.** The intersection may empty the candidate set on some shapes.
  Measuring shows whether it is needed at all. Norm kernels after this change remain
  Welford-only (LayerNorm, GroupNorm and var_mean have no sum), so the mix is mainly user code.
- **Where it is recorded.** The result goes into `reduction-combiners` memory and this design.

### 10. Tests

- **New `test_multi_output.py`:**
  - kernel counts and outputs per kernel, against Triton's structure, for the scenarios of
    `multi-output-kernels`;
  - eager or fp64 agreement;
  - the split first stage holds one reduction;
  - hand-authored descriptors for every malformed case;
  - a 16-buffer kernel compiles, and a fusion that would make 17 is rejected;
  - per-output launch guards;
  - selection with a corrupted second output (injected through a test plan).
- **Existing tests updated:**
  - `test_float_arithmetic.py`: `adam_kernels` becomes 1.
  - `test_admission_paths.py`: the pointwise case is admitted.
  - `test_rejections.py`: its second sample reason (LayerNorm backward over the input limit)
    gets a new sample.
  - Contract tests for the "exactly one final output" messages.
- **Accuracy.** `welford_band.py` gains the multi-output cases and the mixed sum and Welford
  cases, run on both GPUs.
- **Timing.**
  - `welford_timing.py`, `float_arithmetic_timing.py` and `rounding_remainder_timing.py` are
    re-run on the 3080 and the 3090.
  - Their per-kernel breakdowns are compared with Triton's.
  - The Welford plan rule's cost on the now single-kernel split `var_mean` is revisited, as the
    welford-reductions notes asked.

## Risks / Trade-offs

- **Fused regions change kernel count in the manifest corpus.** → Run the manifest after the
  fusion group and review every changed case. Each must be a graph with a multi-output fusion.
  Any single-output region that changes is a bug.
- **One emitted kernel replaces several, so first-use selection picks one plan for several
  outputs.** A plan that was best for one output alone may lose. → Times are judged per kernel
  against Triton's same fused kernel, which also uses one config.
- **Longer kernels may raise register pressure.** For example, LayerNorm with statistics and
  output, or GroupNorm's second stage with three outputs. → Check the per-kernel timing. If it
  falls below the floor, look at resources (`stage5_resources`) and SASS before plan work.
- **Masked de-duplication could merge regions that differ.** → The canonical key covers the
  whole body, the mask operand and the fill. It only renames ids defined inside the body. Unit
  tests cover two masked regions that differ only in fill or index.
- **A higher buffer limit admits regions that were rejected before, exposing later gaps.** For
  example, GroupNorm backward now reaches other rejections, or kernels the timing hasn't seen.
  → This is intended (prerequisite removal). The census is re-run and new reasons are filed as
  beads, not fixed here.
- **Merged per-row and per-element store groups change emission order for multi-store kernels
  only.** → Single-store IR is unchanged by construction. A Rust unit test pins it, and the
  manifest confirms it.

## Migration Plan

Task groups are ordered so that each one leaves the tree working and the manifest explained:

1. The buffer contract and limit, with no fusion change. The manifest is unchanged.
2. Several outputs through codegen, the compiler and the runtime, behind the still-active
   fusion rule. Hand-authored descriptors test it. The manifest is unchanged.
3. Masked-region de-duplication and store grouping. The manifest is unchanged.
4. Lift the fusion rule, with selection per output. The manifest changes are reviewed and
   re-recorded.
5. Accuracy and timing.

Rollback is per group (git revert of the group's commit). Cached artifacts are content-keyed, so
old and new coexist.
