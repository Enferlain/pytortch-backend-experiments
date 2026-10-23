# Tasks

Each group closes the same way:
1. Run the full suite (`stage0/.venv/bin/python -m pytest cubecl_torch/testing -n 6`, output saved
   to a file) and the manifest check (`python -m cubecl_torch.testing.manifest_check check --jobs 6`).
2. Run a Sonnet (medium) review of the group's diff, then verify and fix its findings.
3. Commit the group.

## 1. Beads and Triton reference data

- [x] 1.1 Create the bead for multi-output reduction kernels under `4z8` (`4z8.21`). It lifts "one disjoint external output required" for reduction kernels so that mean, rstd and the normalized output share one kernel, and it checks the Triton performance floor for norms. Make `4z8.11` depend on it. Record in `4z8.15` that this change implements Welford with single-output kernels. Verify with `bd show` on each.
- [x] 1.2 Add `cubecl_torch/testing/welford_reference.py` next to `extrema_reference.py`, run with the Triton venv. It writes seeded inputs (stored as SHA-256), fp64 statistics and Inductor+Triton outputs to `cubecl_torch/testing/data/`. Cases:
  - `var_mean(x, -1, correction=0)` at 37×129, 300×4096, 8×300000 and 2×1048579, for float32, float16 and bfloat16, with zero-mean and offset (mean 1e3, std 1) data, static and dynamic.
  - GroupNorm (1×320×64×64, 1×640×32×32, 2×64×8×8) and LayerNorm (4096×1280) mean/rstd/output in training mode.
  - Rows with NaN and ±inf.

  Also record, per case, whether Triton's kernel was persistent (two-pass) or looped (Welford), and its split count. Verify that the script regenerates byte-identical data from the same seed.
- [x] 1.3 Close the group (suite, manifest, review, commit).

## 2. Multi-operand reductions and component results (no change to generated code)

- [ ] 2.1 In `cubecl_torch/native/src/descriptor3.rs` validation:
  - Check the operand count per combiner. Every current type has one operand.
  - Apply the explicit-float32-conversion check to every operand.
  - Add the `reduction_component {args: [reduction_id], index}` op. It is rejected for every current type, with an error naming type and index.
  - Make `closure()` gather the bodies of all operands.
  - Keep the component vector of each reduction in the emitter, not only `output()`.

  Verify with Rust unit tests for the operand-count and component errors (`cargo test` in `cubecl_torch/native`), the rebuilt binary (`native/build.sh`), and manifest PASS with no changed case, region or file.
- [ ] 2.2 In `inductor/reductions.py`, factor the descriptor emission so that a reduction can return several component values. Current types still return one value with an unchanged descriptor. Verify with an unchanged manifest and with `test_reduction_combiners.py` and `test_reduction_extrema.py` passing.
- [ ] 2.3 Close the group (suite, manifest PASS required, review, commit).

## 3. Welford combiners in the compiler

- [ ] 3.1 Add `Combiner::WelfordReduce` (one operand, lift `(x, 0, 1)`) and `Combiner::WelfordCombine` (three operands). Shared properties:
  - accumulator arity 3, identity `(0, 0, 0)`
  - the guarded combine from design decision 2
  - `output` exposing all three components
  - sum-only plans ineligible

  Validation admits arity 3 for these types only. Add Rust unit tests for `from_descriptor`, `arity` and `supports`. Verify with `cargo test`.
- [ ] 3.2 Make `plane()` for Welford run the butterfly and then broadcast each component from lane 0 (`plane::BroadcastOp`). Sum, max and min keep their current plane step. Verify with an unchanged manifest and the max/min contract tests.
- [ ] 3.3 Add hand-authored Welford descriptor tests in a new `test_reduction_welford.py`, comparing against a NumPy/fp64 reference within a float32 tolerance:
  - `welford_reduce` and `welford_combine` compile under every eligible strategy and are rejected under the compensated strategy with an error naming strategy and type, with nothing written to the cache.
  - Extents smaller than the cooperation width (identity-only lanes) give no NaN.
  - A reduction-scope store of `x - mean` sees one mean per row under the `independent_planes` and `plane_shared` strategies.
  - Operand-count and component mismatches are rejected.

  Verify that the tests pass.
- [ ] 3.4 Close the group (suite, manifest, review, commit).

## 4. Front-end admission and the numerical contract

- [ ] 4.1 Admit `welford_reduce` and `welford_combine` with float32 dtype:
  - In `inductor/reductions.py`, return the 3-tuple of `reduction_component` values.
  - Update `semantic_checks.py` admission and the legality mirror (`REDUCTION_TYPES`; `SUM_ONLY_STRATEGIES` stays as it is).
  - In `tuning.compare`, use a tolerance for mean/m2 outputs and exact comparison for weight outputs.

  Other Welford dtypes raise an error naming type and dtype. Verify that `torch.var_mean(x, -1)`, `x.var(-1)`, `x.std(-1)`, `F.group_norm` and `F.layer_norm` compile for float32/float16/bfloat16 without an unsupported-capability error, and that the existing `welford_reduce` rejection test case becomes an acceptance case.
- [ ] 4.2 Add contract tests for the `reduction-kernels` scenarios:
  - Admitted Welford, and types still rejected (without `welford_reduce` in the list).
  - Masked split (8×300000) and unmasked split (GroupNorm 1×320×64×64), static and dynamic.
  - Operand/component mismatch from the front end's point of view.
  - First-use candidate sets exclude the sum-only plans.

  Verify that they pass.
- [ ] 4.3 Add the `mean-variance-reductions` accuracy band test against group 1's reference data: error against fp64 at matched split, checked across seed halves (`triton-floor-band`). Record the results under `cubecl_torch/testing/results/`. If a case is outside Triton's band, apply the design's fallback and rerun before going on:
  - Long rows: decision 3, the incremental `accumulate` hook in the private loops.
  - Short offset rows: decision 7, `welford_reduce_fallback` on the rows where Triton uses two-pass (persistent, or low-precision two-step variance).

  Record which fallback, if any, was needed and why, in `design.md` and the results file.
- [ ] 4.4 Add the remaining `mean-variance-reductions` scenarios: exact weight (ragged split, 1048579 elements), NaN first/middle/last, constant rows (variance exactly 0, also split), empty partials, ±inf rows non-finite, and the fused normalization epilogue's one value per row. Verify that they pass.
- [ ] 4.5 Set `welford_reduce` and `welford_combine` to `implemented` in the coverage inventory. Domain note: float32 result (inputs upcast by Inductor), one reduction dimension, extent ≤ 1,048,576 per kernel, every non-sum-only strategy, split and dynamic, single external output per kernel until the multi-output bead. Regenerate the markdown and verify `test_coverage_inventory.py`.
- [ ] 4.6 Close the group (suite, manifest PASS, review, commit).

## 5. Integration checks

- [ ] 5.1 Time `torch.var_mean(x, -1)` for float32 and bfloat16 at 37×129, 300×4096 and 8×300000 under CubeCL and under Inductor+Triton (Triton venv). Use CUDA events, repeated rounds and a per-kernel breakdown, recording device, clocks and display state. Record the results under `cubecl_torch/testing/results/` and on the multi-output bead. Any per-kernel gap beyond the short-kernel exception gets a SASS inspection, recorded on that bead. Verify that the results file exists and the bead notes cite it.
- [ ] 5.2 Run the full Stage 5 equivalence (`python -m cubecl_torch.testing.equivalence run --jobs 6`) and confirm 336/336. Rerun the scoreboard's `cubecl` inventory mode for SDXL, Anima and the optimizer, and record in `4z8.2` and `4z8.15` which regions changed status and each workload's new first rejection. Verify that the run directory and bead notes exist. Afterwards check `du -sh /tmp` and clean up caches.
- [ ] 5.3 Close `4z8.15`. Update the `reduction-combiners` memory (`bd remember --key reduction-combiners`) with the Welford facts: arity 3, components, plane broadcast, which fallbacks were needed. Close the group (suite, manifest, review, commit).

## Workflow follow-up

- Archive the change after the last group is committed, so `mean-variance-reductions` joins `openspec/specs/` and the `reduction-kernels` delta is applied.
- Next: the multi-output reduction kernel change (bead from 1.1), then the coverage census (`4z8.19`) and the all-reasons reporting (`4z8.20`).
