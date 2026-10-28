# Tasks

Each group closes the same way:

1. Run the quality gates and checks:
   - style: ruff format, ruff check, `cargo fmt` and clippy (`-D warnings`);
   - the full suite (`stage0/.venv/bin/python -m pytest cubecl_torch/testing -n 4`, output
     saved to a file);
   - the manifest check (`stage0/.venv/bin/python -m cubecl_torch.testing.manifest_check check --jobs 6`).
2. Run a Sonnet (medium) review of the group's diff, then verify and fix its findings.
3. Commit the group.

Never rebuild the native pieces while a GPU job or test run is in flight. Afterwards, clean
`/tmp/pytest-of-$USER` and scratch caches.

## 1. Buffer contract and a 16-buffer limit (no fusion change)

- [x] 1.1 Widen the descriptor contract (design decision 1): a non-empty prefix of read buffers, then a non-empty suffix of write buffers.
  - Each store targets a write buffer and each load a read buffer.
  - Each write buffer has exactly one store.
  - At most 16 buffers.

  Apply this in `compiler/descriptor.py` (`MAX_BUFFERS = 16`), in `tuning.AutotunedKernel`'s input check, and in Rust `semantic.rs::validate_description` and `reduction/analysis.rs`. Error messages name the buffer and the problem. Verify with:
  - Rust unit tests and new contract tests for each malformed case of the `multi-output-kernels` "Descriptor buffer contract" requirement (store into an input, unwritten output, two stores to one output, 17 buffers), each with nothing written to the cache;
  - `cargo test`.
- [x] 1.2 Raise the native bounds to 16 buffers: `cubecl_torch_metadata` in `native/src/lib.rs`, and `stage5_launch` in `native/launcher.c` (17 pointers including the info buffer; `void *args[17]`). Leave `stage4_launch` and the Stage 0–4 metadata entry points unchanged. Rebuild with `native/build.sh`. Verify with:
  - a hand-authored pointwise descriptor with 15 inputs and one output, compiled, launched and compared with eager;
  - the runtime contract tests.
- [x] 1.3 Raise the front-end limits (design decision 2):
  - Pre-fusion `Path.max_inputs` becomes 15 for both paths.
  - `CubeCLKernel.supported_input_counts` covers 0..15.
  - The fusion check becomes "at most 16 external buffers", inputs plus the kernel's outputs. The lower bound of one external input stays.

  The new rejection reason names the limit. Update the mappings for that reason in `scoreboard/families.py` and `testing/coverage_inventory.py` (`4z8.23`'s row). Give `test_rejections.py` a new second-reason sample, since LayerNorm backward no longer exceeds the limit. Verify with:
  - LayerNorm and GroupNorm (with weight and bias, then SiLU) forward and backward compiling with no input-count or buffer-count rejection;
  - a test where fusing two nodes would make 17 buffers, which is rejected with that reason while the region still compiles;
  - `test_coverage_inventory.py`.
- [x] 1.4 Close the group. Manifest PASS with no changed case is required: single-output descriptors are unchanged, and regions that were rejected only for the input count are outside the corpus. If the manifest reports a change, explain it before going on.

## 2. Several outputs through codegen, compiler and runtime (fusion rule still active)

- [x] 2.1 Let `CubeCLKernel.codegen_kernel` accept one or more output names in the descriptor's write suffix. Keep rejecting a non-empty `args.inplace_buffers`, with a message stating that `inplace_buffers=False` should keep it empty. Allow mixed store scopes (`{"outer", "reduction"}`) in `compiler/descriptor.py` and `compiler/binding.py`. Verify with:
  - a hand-authored reduction descriptor storing mean (per row) and `x - mean` (per element), which compiles under every eligible plan and equals fp64;
  - a single-output descriptor whose JSON is unchanged byte for byte.
- [x] 2.2 Add per-output launch guards (design decision 7). `binding.py` emits one `output_numel` guard per write buffer:
  - the last buffer keeps today's serialized guard;
  - every other output gets an explicit buffer index.

  `runtime/launch.py` checks each guard against its own tensor, and `runtime/artifact.py` documents both forms. Verify with:
  - the `kernel-runtime` scenario "Extent of a non-final output", as a runtime contract test;
  - an unchanged digest for single-output artifacts (`test_runtime_provenance.py` / the manifest).
- [x] 2.3 Lift the pointwise vector adapter's "requires one store" rule in `legality.structural_reason`: every store must be direct-linear and aligned, like every load. Verify with:
  - a hand-authored three-output pointwise descriptor (Adam-shaped, 5 inputs) under `vector_pointwise`, `scalar_pointwise` and the baseline;
  - each output equal to eager bit for bit.
- [x] 2.4 Close the group (manifest PASS with no changed case required).

## 3. One reduction per value and one pass per store kind

- [x] 3.1 Confirm the cause of the three `welford_combine` reductions in the masked split first stage: dump the operand records of 8×300000 `var_mean` with the fusion rule forced open in a scratch run, and record the result in design decision 4. Then add the canonical key for `masked` records: body ids are renamed by position before the record is hashed in `emit()`. Verify with unit tests showing that:
  - two identical masked regions share one value;
  - regions that differ only in fill, mask or index stay separate;
  - with the rule forced open in a scratch run, the split first stage has one `welford_combine`.
- [x] 3.2 Group stores in `reduction/emit.rs::stores()` (design decision 5):
  - all per-row stores in one `unit == 0` branch;
  - all per-element stores in one loop over the extent;
  - each group replays the union of its dependency closures once;
  - per-row stores come first.

  A group of one store emits today's IR. Verify with:
  - a Rust unit test pinning single-store IR;
  - a hand-authored descriptor with two per-element stores and two per-row stores, which equals fp64 under every eligible plan and has one extent loop in the generated source.

  Done 2026-10-28. Single-store IR is pinned by the manifest check (every corpus kernel, PASS) rather than a separate Rust IR fixture; the loop count is checked in `test_multi_output.py`.
- [x] 3.3 Close the group (manifest PASS with no changed case required).

## 4. Fusion with several outputs, and selection per output

- [x] 4.1 Make `tuning.select` handle every output (design decision 8):
  - scratch for every write buffer;
  - per-output comparison classes from each store's dependency closure;
  - `exclusion()` naming the failing output.

  A single-output kernel keeps its current class. Verify with:
  - the `multi-output-kernels` scenarios "Comparison per output", "Weight beside mean and m2" and "Corrupted second output", the last through an injected faulty plan in a test;
  - a check that user outputs are untouched until the final launch.
- [x] 4.2 Drop "one disjoint external output required" in `admission.fusion_allowed` (design decision 3). Kernels with `rand` or `atomic_add` stay single-output and keep their existing reason. Update `test_admission_paths.py` (the pointwise case is admitted) and `test_float_arithmetic.py` (`adam_kernels == 1`). Add `test_multi_output.py` covering the kernel-structure scenarios of `multi-output-kernels`, checked against Triton's structure from the probe:
  - Adam, frexp (int64 exponent), sum plus amax;
  - LayerNorm, var_mean unsplit and split, GroupNorm;
  - the random-number case staying separate.

  Each output is checked against eager or fp64. Verify that the new and updated tests pass.
- [x] 4.3 Run the manifest check. Confirm that every changed case, region and file belongs to a graph with a multi-output fusion, and that no region whose kernels all write one output changed. Record the explanation in a results note under `cubecl_torch/testing/results/`, then re-record the manifest (`manifest_check record --jobs 6`). Verify that `check` passes after recording.
- [x] 4.4 Update the coverage inventory: "multi-output kernels" becomes implemented, with a domain note (several outputs per kernel, up to 16 buffers, effect kernels single-output, no aliasing outputs). `INPLACE_BUFFERS` stays missing and names its bead (`4z8.31`). Regenerate the markdown and verify `test_coverage_inventory.py`.
- [x] 4.5 Close the group (suite, reviewed manifest, review, commit).

## 5. Accuracy against Triton

- [ ] 5.1 Extend the Triton reference (`welford_reference.py`, Triton venv) and `welford_band.py`:
  - the multi-output kernel cases (var_mean sweep; LayerNorm and GroupNorm statistics and normalized output in one kernel);
  - the mixed cases of the `mean-variance-reductions` scenario "Welford fused with a sum", at short, medium and long rows with zero-mean and offset data.

  Verify that the reference regenerates byte-identical data from its seed.
- [ ] 5.2 Run the band sweep on both GPUs. Record the results under `cubecl_torch/testing/results/`. If any mixed case's Welford output is worse than Triton, apply design decision 9's intersection policy (sum qualification and `welford_plan_reason` together, with a fallback legal under both) and rerun. Record the outcome in design decision 9. Verify that no case is worse than Triton.
- [ ] 5.3 Close the group.

## 6. Speed against Triton, and integration checks

- [ ] 6.1 Re-run on the RTX 3080, then confirm on the RTX 3090 (warm clocks, alternating rounds, per-kernel breakdowns):
  - `welford_timing.py`, adding the multi-output norm cases;
  - `float_arithmetic_timing.py` (whole Adam step);
  - `rounding_remainder_timing.py` (frexp).

  Revisit the Welford plan rule's cost on the now single-kernel split `var_mean` (bead notes, welford-reductions 7). For any case slower than Triton beyond the short-kernel exception:
  - inspect resources and SASS of the slower kernel;
  - fix it inside this change, or, if that takes much more effort, record it on a bead with its per-kernel breakdown.

  Record the results under `cubecl_torch/testing/results/`. Verify against the `multi-output-kernels` and `reduction-kernels` timing scenarios and the `float-arithmetic` "Whole Adam step" scenario.
- [ ] 6.2 Re-run the scoreboard `cubecl` inventory for SDXL, Anima and the optimizer. Record which regions changed status, and each workload's new first rejection, on `4z8.21` and `4z8.2`. File newly exposed reasons as beads; don't fix them here. Afterwards check `du -sh /tmp` and clean up the caches.
- [ ] 6.3 Close `4z8.21` and `4z8.23`. Leave a note on `4z8.9` saying that the multi-output contract is in place and that aliasing outputs remain. Update the `reduction-combiners` memory with the multi-output facts: the per-store comparison classes, the masked canonical key, and the mixed-kernel policy outcome. Update `cubecl_torch/README.md` where it states the buffer or output limits. Close the group (suite, manifest, review, commit).

## Workflow follow-up

- Archive the change after the last group is committed, so `multi-output-kernels` joins `openspec/specs/` and the `reduction-kernels`, `float-arithmetic`, `mean-variance-reductions` and `kernel-runtime` deltas apply.
- Next, per the work direction: the transcendental family (`4x9.3`). User-visible mutation (`4z8.9`) is unblocked by this change; its place in the order is the user's call.
