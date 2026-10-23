# Tasks

Each group closes the same way:
1. Run the full suite (`stage0/.venv/bin/python -m pytest cubecl_torch/testing -n 6`, output saved
   to a file) and the manifest check (`python -m cubecl_torch.testing.manifest_check check --jobs 6`).
2. Run a Sonnet (medium) review of the group's diff, then verify and fix its findings.
3. Commit the group.

## 1. Inventory classification

- [x] 1.1 In `cubecl_torch/testing/coverage_inventory.py`, apply these reclassifications. Regenerate `coverage_inventory.md` and verify that `test_coverage_inventory.py` passes.
  - `TUPLE_REDUCTION` moves to the `scan` family. Note: only gates multi-output scans in the pinned Inductor.
  - `REDUCE_TO_SINGLE_ELEMENT` becomes `n/a`. Note: not advertised, as Triton's default; Inductor splits instead; `4x9.16`.
  - The `fmax` reduction type and the `fmaximum` op move to `reduce-softmax`. Note: only producer is the two-pass softmax fallback.
- [x] 1.2 Update the beads. Verify with `bd show` on each.
  - Narrow `x3c` to the remainder/fmod contract, noting that the extrema half is delivered by this change.
  - Remove `fmax` from `4x9.7`'s open items and add it to `4x9.10`.
  - Record in `4x9.6` that this change implements it.
- [x] 1.3 Close the group (suite, manifest, review, commit).

## 2. Combiner refactor (no change to generated code)

- [x] 2.1 In `cubecl_torch/native/src/descriptor3.rs`, add `enum Combiner { Sum }` with `identity`, `combine`, `plane`, `arity` and `supports(strategy)`. `Sum` must reproduce the current constants, `Add`/`compensated_add` and `FSumOp`/add-tree exactly. Map the descriptor's `reduction_type`/`accumulator_dtype` to a combiner in validation, rejecting unknown types with an error that names them. Verify with `cargo test` in `cubecl_torch/native` and the rebuilt binary (`native/build.sh`).
- [x] 2.2 Thread the combiner through every strategy in `build_physical_codegen`: shared tree, multi-accumulator, plane-shared, load-batch, reference512, independent planes, final plane and columns. Private and shared accumulators become slices of length `arity()`, and shared memory is sized as slots × arity. Validation admits only arity 1. Verify that the manifest check PASSes with no changed case, region or file, and that the `cubecl_torch/testing` suite passes.
- [x] 2.3 Mirror the strategy-eligibility table on the Python side (`inductor/legality.py`). An ineligible strategy is rejected with "strategy <s> does not support reduction type <t>" and recorded among rejected candidates. Make `compiler/final_plan.py` check the reduction type instead of assuming sum. Verify with a unit test of the eligibility table (sum: every current strategy eligible) and with `test_contracts_tuning.py` unchanged.
- [x] 2.4 Close the group (suite, manifest PASS required, review, commit).

## 3. Float extrema pointwise ops

- [x] 3.1 Add float `maximum`/`minimum` (→ `FMaxNanOp`/`FMinNanOp` in float32 compute) and `isnan` (→ `IsNanOp`, bool result) to the pointwise op table: `inductor/semantics.py`, `semantic_checks.py` type rules, and `native/src/descriptor2.rs` arity and emission. Integer index `maximum`/`minimum` keep their current path. Verify that `torch.maximum`, `torch.minimum`, `clamp`, `clamp_min`, `torch.fmax`, `torch.fmin` and `torch.isnan` compile for float32/float16/bfloat16 without an unsupported-capability error.
- [x] 3.2 Add a Triton reference generator: a script run with the Triton venv that writes seeded inputs (with ±inf, NaN, ±0) and Triton outputs to `cubecl_torch/testing/data/`, with input hashes. It must be reusable by group 4 for reductions. Verify that the script regenerates byte-identical data from the same seed.
- [x] 3.3 Add contract tests for the float-extrema pointwise scenarios:
  - NaN operands
  - (−0, +0) in both orders
  - bitwise agreement with the recorded Triton outputs, apart from NaN payload
  - `fmax`/`fmin` with one NaN
  - `isnan` against eager

  Inspect the SASS of one `maximum` kernel and confirm it contains `FMNMX.NAN`, or record what it contains. Verify that the tests pass.
- [x] 3.4 Set `maximum`, `minimum` and `isnan` to `implemented` in the coverage inventory. The domain note says: float32/16/bf16, NaN-propagating, −0 < +0 on sm_80+; int64 index values as before. Regenerate the markdown and verify `test_coverage_inventory.py`.
- [x] 3.5 Close the group (suite, manifest, review, commit).

## 4. Max/min reductions

- [ ] 4.1 Add `Combiner::Max` and `Combiner::Min`:
  - identity −inf/+inf
  - combine `FMaxNanOp`/`FMinNanOp`
  - plane step: the shuffle-XOR tree with the combiner's combine
  - ineligible for `shared_tree_compensated`, `plane_shared_reference512` and `independent_planes_final9`

  Add a Rust unit test of `Combiner::from_descriptor` and `supports`, and a `legality.candidates()` test showing the sum-only strategies recorded as rejected for max. Verify with a hand-authored max descriptor that compiles under each eligible strategy, and is rejected under the compensated strategy with an error naming strategy and type, with nothing written to the cache.
- [ ] 4.2 Admit `max`/`min` with float inputs in `inductor/reductions.py` and `semantic_checks.py`. Reword the per-kernel limit in `reduction_checks.py` to count reductions rather than sums. Integer `max` still raises an error naming the dtype. Gate the sum-specific rules in `legality.numerical_reason` on `reduction_type == "sum"`: long rows admit only `shared_tree_compensated`, which max/min can't use, so otherwise no candidate is legal. Make `tuning.compare` require bitwise equality apart from NaN payload, signed zero included, for max/min kernels. Verify that `x.amax(-1)`, `x.amin(-1)`, `x.max()` and `(x * 2).amax(-1) + 1` compile and match eager, apart from the sign of zero.
- [ ] 4.3 Add contract tests for the `reduction-kernels` and `float-extrema` reduction scenarios. Verify that they pass.
  - Admitted and rejected types and dtypes.
  - Split max with ragged extent, static and dynamic (8×65539, 4×262147, 2×1048579).
  - Fused epilogue.
  - First-use candidate set excludes ineligible strategies.
  - Baseline policy.
  - NaN first, middle and last.
  - Mixed zeros in any order, including split shapes.
  - Infinities.
  - Bitwise agreement with group 3's Triton reference data, extended to reductions across float32/16/bf16 and static/dynamic.
- [ ] 4.4 Turn the `max` case in `test_contracts_effects.py::test_unsupported_capabilities_rejected_without_compiling` into an acceptance test. Update the gate contract inventory entry for that assertion, citing this change as the reason. Verify `test_contract_inventory.py` and `contract_inventory` regeneration.
- [ ] 4.5 Compile a softmax graph (`torch.softmax(x, -1)`) and record whether the dependent max→sum kernel is accepted. If rejected, record the rejection reason in `4x9.10`, with no code change. Verify with a test that asserts the recorded outcome.
- [ ] 4.6 Set `max` and `min` to `implemented` in the coverage inventory, and update the `reduction` op's note. The domain note: float32/16/bf16 inputs, float32 accumulation, NaN-propagating, ordered zeros on sm_80+, every non-sum-only strategy, split and dynamic. Regenerate the markdown and verify `test_coverage_inventory.py`.
- [ ] 4.7 Close the group (suite, manifest PASS, review, commit).

## 5. Integration checks

- [ ] 5.1 Time `x.amax(-1)` and `x.amin(-1)` for float32 and bfloat16 at short (37×129), medium (300×4096) and long or split (8×65539, 2×1048579) shapes, under CubeCL and under Inductor+Triton (Triton venv), with CUDA events and repeated rounds. Record the results under `cubecl_torch/testing/results/`. Verify that each CubeCL time is within Triton's run-to-run spread or faster. Otherwise inspect the SASS and record a bead before going on.
- [ ] 5.2 Run the full Stage 5 equivalence (`python -m cubecl_torch.testing.equivalence run --jobs 6`) and confirm 336/336. Rerun the workload scoreboard's `cubecl` inventory mode for SDXL, Anima and the optimizer, and record which regions changed status (the optimizer clamp regions are expected to move past `maximum`). Note the new first wall per workload in `4z8.2` or `4z8.15`.
- [ ] 5.3 Close `4x9.6` and `4x9.7`. Keep `bd remember` findings up to date: refine the combiner design note and the Triton-band result in `cubecl-coverage-checklist-2026-10-22`. Close the group (suite, manifest, review, commit).

## Workflow follow-up

- Archive the change after the last group is committed, so the `reduction-kernels` and `float-extrema` specs join `openspec/specs/`.
