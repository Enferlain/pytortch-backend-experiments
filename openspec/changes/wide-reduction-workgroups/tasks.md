# Tasks

Each group closes the same way:

1. Run the quality gates and checks:
   - style: ruff format, ruff check, `cargo fmt` and clippy (`-D warnings`);
   - the full suite (`stage0/.venv/bin/python -m pytest cubecl_torch/testing -n 4`, output
     saved to a file);
   - the manifest check (`stage0/.venv/bin/python -m cubecl_torch.testing.manifest_check check --jobs 6`).
2. Run a Sonnet (medium) review of the group's diff, then verify and fix its findings.
3. Commit the group.

Never rebuild the native pieces while a GPU job or test run is in flight. Compare code
generation choices with plans pinned and Nsight Compute, not through first-use selection
(memory `ab-codegen-changes`). Afterwards, clean `/tmp/pytest-of-$USER` and scratch caches.

## 1. 512-unit `plane_shared` plans: geometry, legality, catalog

- [x] 1.1 Launcher geometry (design decision 3): `stage5_launch` accepts `[512,1,1]` with shared memory up to 12 bytes per unit. Every other block stays at most 256 units, under the existing shared rules. Update the comment that states the limits, then rebuild with `native/build.sh`. Verify with runtime tests:
  - a 512-unit Welford-sized launch (192 bytes) and a compensated-sized one (4 KiB) are accepted;
  - `[1024,1,1]`, `[512,2,1]` and 512 units with 6145 bytes are refused with "unsupported launch geometry".
- [x] 1.2 Rust legality (`native/src/plan.rs`): `plane_shared` admits `[512,1,1]` at widths 1, 2, 4 and 8. Combined (four-accumulator) plans stay at 256 units. Verify with a `plan.rs` unit test covering the admitted and refused shapes, and `cargo test`.
- [x] 1.3 Catalog and extent rule (design decisions 1 and 2):
  - add `plane_shared` 512×v1/v2/v4 to `catalog()`;
  - add the selection policy in `legality.candidates()` that rejects 512-unit plans for rows under 2048 elements, with a recorded reason;
  - add the fact `extent >= 2048` to `legality.signature()`.

  Verify with:
  - legality tests: the candidate sets at 1024, 2048 and 8192, a recorded reason below 2048, and different signatures at 2047 and 2048;
  - the every-plan tests in `test_reduction_combiners.py` and `test_multi_output.py` running the new plans at ragged extents and empty lanes (check their eligibility filters include them);
  - sum, max, min and softmax kernels at 512 units matching their references.
- [x] 1.4 Re-measure the 4z8.35 experiment cases with the plans now in the catalog. Pin each plan and use ncu on the 3080: float32 and bfloat16 softmax and the sum consumer at 2048×8192, against the 256-unit plans and Triton. Record the results in `testing/results/wide-reduction-workgroups.md`. The sum consumer stays under the long-row sum rule until group 3, so measure it pinned.
- [x] 1.5 Close the group. The manifest is expected to change only in the candidate and rejected lists of corpus cases with rows of at least 2048 elements. Review the changes by matching region records on content, record the review in `testing/results/wide-reduction-workgroups.md`, then re-record. Any change to a chosen artifact or a descriptor stops the group until explained.

## 2. Welford at 512 units

- [x] 2.1 Check that float32 Welford-only kernels admit `plane_shared` 512×v4 (2048 lanes) under `welford_plan_reason` at rows longer than 2048 (design decision 4). Two consequences of group 1 come with it:
  - `tuning.welford_fallback` now picks 512×v4 for float32 rows of at least 2048;
  - at exactly 2048 elements, 512×v4 is the only Triton-shaped plan left, so selection is bypassed.

  Both are covered by the band in 2.2. Verify with a policy test in `test_reduction_welford.py`, and with LayerNorm statistics plus the normalized output at 2048×8192 compiled and launched with the plan pinned, meeting the mean-variance accuracy contract.
- [x] 2.2 For 16-bit Welford rows, add `plane_shared` 512×v8 to `welford_catalog()`. Then extend the band (`welford_reference.py`, `welford_band.py`) with LayerNorm at 2048×8192 and at a 2048-element row (4096×2048):
  - float32 and bfloat16 (and float16);
  - with and without affine parameters;
  - only the new cases added to `data/welford_triton.pt`, generated with the Triton venv (`--new`); existing entries are kept.

  Run the band on both GPUs with each admitted Welford plan pinned. Result: 512×v8 is the only plan in band on long 16-bit rows (design decision 7). Tighten `welford_plan_reason` for rows longer than `TRITON_LANES`: keep only Triton-shaped plans with the most lanes. Verify with:
  - a policy test (16-bit 8192: only 512×v8 among row plans; float32: the widest-lane plans);
  - the 16-bit band cases in band on both GPUs with first-use selection.

  Record the results in `testing/results/wide-reduction-workgroups.md`.
- [x] 2.3 Add the lane-accumulator variant (design decision 7): `plane_shared` 512×v4 and 512×v8 with `private_merge = "lanes"`, Welford-only. Each vector lane keeps its own Welford accumulator across the loop; the lanes are merged in order after it. Implement it in the reduction emitter, `plan.rs` (admitted private merge for vector plane plans, Welford only) and the Python catalog and legality. Verify with:
  - `cargo test`, and a `plan.rs` test admitting the variant for Welford and refusing it for sums;
  - the every-plan Welford tests (`test_reduction_welford.py`: special rows, constant rows, ragged tails) launching the variant;
  - LayerNorm at 2048×8192 under the variant matching fp64.
- [x] 2.4 Run the pinned band again on both GPUs with the variant included. Set the policy for float32 rows longer than `TRITON_LANES` to the plans in band, and for 16-bit rows decide between folding and lanes by the band. Verify with:
  - the full Welford band on both GPUs with first-use selection, no case outside;
  - policy tests pinning the outcome.

  Record the results in `testing/results/wide-reduction-workgroups.md`. If no plan is in band for float32, stop and report with the per-plan data.

## 3. Long-row sums judged against Triton

- [x] 3.1 Extend the band with float32 long-row sums (design decision 5):
  - `x.sum(-1)` and the sum consumer at 2048×8192 and 256×32768;
  - zero-mean and offset data;
  - a check that each case compiles unsplit;
  - the Triton reference for the new cases only.

  Verify with the band tool running them with first-use selection on the 3080.
- [x] 3.2 Run the new cases with each plan forced, on both GPUs: `shared_tree_compensated` 256, `plane_shared` 512×v1/v2/v4, and the 256-unit sum plans. Record which plans are in Triton's band in `testing/results/wide-reduction-workgroups.md`.
- [x] 3.3 Replace the long-row rule in `legality.numerical_reason` with the measured set; the rejection reason names Triton's band. If only compensated plans are in band, admit `shared_tree_compensated` at `[512,1,1]` in `plan.rs` and the catalog, and include it in the 3.2 measurement. Verify with:
  - legality tests for the long-row candidate sets of a sum-only kernel, a softmax kernel (unchanged) and a fused Welford-and-sum kernel (unchanged);
  - the band on both GPUs with first-use selection, no case outside;
  - the spec scenario "Policy recorded".

## 4. Timing coverage and the performance floor

- [ ] 4.1 Extend `welford_timing.py` with:
  - the `softmax` and `sum_consumer` kinds, and LayerNorm without affine;
  - the shapes 300×4096 and 2048×8192, float32 and bfloat16;
  - the same per-kernel CUPTI reporting and selection details as the existing cases.

  Verify with the tool running these cases on the 3080 for one round.
- [ ] 4.2 Run the full timing (`--rounds 3`) on the 3080 and the 3090. Record medians against Triton, per-kernel breakdowns and selected plans in `testing/results/wide-reduction-workgroups.md` and the JSON files. For each case slower beyond the spread, either fix it in this change, or record it on a bead with its per-kernel breakdown (the spec scenario "Row timing"). Verify with the record.
- [ ] 4.3 File the online-softmax follow-up bead (Triton's two-pass softmax) with the 2048×8192 pass counts and DRAM traffic from `4z8.35`. Link it from the results file.

## 5. Integration

- [ ] 5.1 Update `cubecl_torch/README.md` (plan catalog and legality sections: 512-unit plans and the extent rule; the long-row sum rule), and the coverage inventory if a row changes. Verify with `test_coverage_inventory.py` and the contract inventory check.
- [ ] 5.2 Run the full suite, the manifest check and the Welford band on both GPUs one last time. Close `4z8.35` with the results, and update the `reduction-combiners` memory with the 512-unit plans and the long-row rule. Verify with the suite passing, the manifest at PASS, and no band case outside.

## Workflow follow-up

- Archive the change after review; sync the `reduction-kernels` delta into the main spec.
