# Tasks

Each group closes the same way:

1. Run the quality gates and checks:
   - style: ruff format, ruff check, `cargo fmt` and clippy (`-D warnings`);
   - the full suite (`stage0/.venv/bin/python -m pytest cubecl_torch/testing -n 4`, output
     saved to a file);
   - the manifest check (`stage0/.venv/bin/python -m cubecl_torch.testing.manifest_check check --jobs 6`).
2. Run a Sonnet (medium) review of the group's diff, then verify and fix its findings.
3. Commit the group.

Never rebuild the native pieces while a GPU job or test run is in flight. Compare plans pinned
under Nsight Compute, not through first-use selection (memory `ab-codegen-changes`). Afterwards,
clean `/tmp/pytest-of-$USER` and scratch caches.

## 1. Copy the crate (bead f05.2)

- [ ] 1.1 Copy `.stage0-sources/cubecl/crates/cubecl-tile/src` to `cubecl_torch/native/cubecl-tile` at the pin, with a `Cargo.toml` using path dependencies on the checkout and `UPSTREAM.md` recording revision `907676dc` and an empty change list. Add it to `native/Cargo.toml` and update `Cargo.lock`. Verify: `build.sh` passes with `--locked`, and `diff -r` against upstream shows only `Cargo.toml` and `UPSTREAM.md`.
- [ ] 1.2 Add the copy-traceability test: the copy matches the recorded revision except for files listed in `UPSTREAM.md`, skipped cleanly if the checkout is absent. Verify: the test fails on a scratch copy with an unlisted edit.
- [ ] 1.3 Accept the digest change once. Verify: the manifest check reports no changed case, region or file, and a test confirms a changed compiler binary produces a new artifact key.

## 2. Gate: one real kernel through the production route (decisions 2 and 4; beads f05.3, f05.9)

- [ ] 2.1 Build RMSNorm-with-weight as a new compiler route, scalar, in `KernelBuilder`: kernel-form `Partitioning` from our scalars, CubeK's `InSlots` layout, recipes calling `semantic::emit`, row sum carried as a value (no extra buffer), stores by the emitter. Verify with a route-level test that the generated source takes one `info` pointer and no `__grid_constant__`, and that it compiles without creating a CUDA context.
- [ ] 2.2 Launch it through `runtime/launch.py` with `launcher.c` unchanged, pinned at 2048×8192. Verify: outputs match float64 within the sum plans' reference test, and DSL plane ownership is correct under a `[512,1,1]` block; if it is not, add the `[32,16,1]` geometry and its refusal tests.
- [ ] 2.3 Measure the pinned kernel under `ncu` on the 3080 against Triton and `shared_tree_compensated`. Verify: results recorded in `testing/results/cubecl-tile-stage-1.md`, within the spread of sk8's 216.8 µs.
- [ ] 2.4 Decide the gate. If 2.1 to 2.3 hold, update bead `f05.9` as retired by decision 2. If not, stop, record what failed, and bring the fallback (client-free `Launcher`, typed metadata contract) back to the user before continuing, since it changes the runtime contract.

## 3. Recipes: bounds and vectors (decisions 7 and 8; beads f05.6, f05.7)

- [ ] 3.1 Add the validation compile option, off by default and part of the artifact key, and confirm reports reach the test (or count into a test-only buffer). Verify with the unfixed `read_checked` at 2048×4099: 29 indices past `x`, matching sk8's range.
- [ ] 3.2 In our copy, test bounds first and evaluate at a redirected coordinate, in `read_checked` and `read_masked`; log it in `UPSTREAM.md`. Verify: zero reports at row lengths 31, 100, 4099 and 65539, the negative control still reports, and 2048×8192 generated code is unchanged apart from the masked path being absent.
- [ ] 3.3 Add a vector `evaluate` (default one lane at a time) in our copy, and vector recipes through the emitter's vector loads at widths 4 and 8. Verify: correct outputs at whole-step rows, unit tests that a partial row refuses a vector line, and pinned `ncu` numbers for width 1 against 4 on the sum consumer and RMSNorm.
- [ ] 3.4 Run the sum consumer at 2048×4099 with width 1 under PyTorch allocations next to a guard buffer. Verify: canaries intact, zero validation reports.

## 4. Fused bodies (decision 3; beads f05.5, f05.10)

- [ ] 4.1 Generalize the route from RMSNorm to any admitted descriptor: `sum`, `mean`, the sum consumer, per-row and per-element stores, runtime scalars and weights, 16-bit storage. Use the reduction emitter's per-element pass, and extract shared code rather than duplicating it. Verify with tests comparing each kernel against the reference at 2048×8192, 2048×4099, 300×4096 and 7×2049.
- [ ] 4.2 Confirm the load hints. Verify: generated source shows retained loads in the reduction pass and streaming loads in the storing pass, and pinned `ncu` DRAM reads match sk8's 70 MB on the sum consumer.
- [ ] 4.3 Runtime extents. Verify: one compilation across 2048×8192, 1024×8192 and 2048×4096 (compiler invocations counted), correct outputs at each, and identical bytes across three fresh compiler processes.

## 5. Summation order (decision 6; bead f05.4)

- [ ] 5.1 Apply the three order changes in our copy, one commit-sized step each, logged in `UPSTREAM.md`: lane tree over per-lane accumulators, units xor 16→1, tree plane combine in the slots read. Verify: `order_model.py` matches the GPU output bit for bit at each step.
- [ ] 5.2 Add tile plans to the `welford_band` harness and run them pinned on float32 and 16-bit inputs, normal and offset, at 2048×8192 and 256×32768, for sum and sum consumer. Verify: results recorded per class; they decide 6.2.
- [ ] 5.3 Pin the timing effect. Verify: the order changes cost no time outside the spread on the 3080, or the cost is recorded and judged against the rule that speed and accuracy may not be traded against Triton.

## 6. Catalog and legality (decisions 5 and 6; bead f05.11)

- [ ] 6.1 Add `tile_row_sum` to `catalog()` and `native/src/plan.rs`, with the structural rule in both layers, and the `extent % step` fact in `signature()`. Verify: tests for accepted and refused descriptors in both layers (max, Welford, two reductions, mask region, split, unaligned width 4), each with its reason, and different signatures at 4096 and 4099.
- [ ] 6.2 Add the selection policy (rows below 2048) and the numerical branch for classes measured in band in 5.2, recording reasons. Verify: the candidate sets at 1024, 2048 and 8192, and for float32 over 4096 per 5.2's result.
- [ ] 6.3 Force each admitted class through the band (the `forced_band` pattern from memory `fused-sum-and-division-policy`). Verify: no admitted plan outside it.
- [ ] 6.4 Run the acceptance for each family: index validation at row lengths 31, 100, 4099 and 65539 with zero reports, before the family is admitted. Verify: a test in the suite, with the kernels in a private artifact cache.

## 7. Acceptance and records

- [ ] 7.1 Time the sum consumer and RMSNorm with tile plans pinned against Triton and the best existing plan at 2048×8192 and 2048×4099, plus the census shapes that have a sum-only kernel, on the 3080 and confirmed on the 3090. Verify: results in `testing/results/cubecl-tile-stage-1.md`; any gap is fixed, or the plan is held out and recorded on a bead.
- [ ] 7.2 Record compile time per kernel and first-use tuning cost against before. Verify: numbers in the results file.
- [ ] 7.3 Review the manifest changes by matching region records on content, then re-record. Verify: every change is a candidate list, a rejected list or a tile win, and non-tile IR, CUDA and PTX digests are unchanged.
- [ ] 7.4 Update `cubecl_torch/README.md` (the copied crate, how to merge upstream, the tile plans) and the coverage inventory if a family's status changes. Verify: `test_coverage_inventory.py` passes.
- [ ] 7.5 Update beads and memory: close the stage's beads, record what stage 1 showed about upkeep and whether a stage 2 is worthwhile, and note whether `4z8.37`, `4z8.39`, `4z8.40`, `6cl` or `9hg` moved. Verify: `bd show` on each.
