# Tasks

Each group closes the same way:

1. Run the gates:
   - style: ruff format, ruff check, `cargo fmt` and clippy (`-D warnings`) for `native/` and the
     bridge crate;
   - the full suite (`-n 4`, output to a file);
   - the manifest check.
2. Get a Sonnet (medium) review of the group's diff, then verify and fix its findings.
3. Commit the group.

Also:
- One GPU job at a time. Never rebuild native pieces while one runs.
- Time on the 3080 with a clock warm-up, and confirm on the 3090.
- Delete scratch caches and ncu/nsys reports when done.

The reference graphs are the RMSNorm backward weight gradient:
- 64×1024 (one column reduction);
- 8192×1280 (a split first stage and a column-sum second stage);
- each in float32 and bfloat16.

## 1. Burn's fused reduce end to end (design 9)

- [ ] 1.1 Build a small CUDA program on the `.stage0-sources/burn` checkout, with its own CubeCL pin and fusion enabled. It computes `(g·x·rstd[:, None]).sum(0)` at both shapes and dtypes. Confirm from Burn's fusion logging (or the compiled kernel names) that it runs as one fused reduce. Verify: the program prints results matching a float64 reference, and its kernel census shows the fused reduce kernel.
- [ ] 1.2 Time Burn's fused kernel and Inductor+Triton's kernels for the same computation under ncu at fixed clocks on the 3080, and record Burn's selected routine. Verify: the table is in `cubecl_torch/testing/results/fused-reduction-bridge.md`, with the Burn and CubeCL revisions used.

## 2. Layout and coverage on the Python side (design 5)

- [ ] 2.1 At codegen, record the reduction operand's virtual layout (output dimensions plus the reduction dimension, with strides) from Inductor's `BlockPatternMatcher` on the reference load's index. Record nothing when it doesn't match. Verify: unit tests on the captured semantics (64×1024 and 8192×1280, both dtypes). The split stage gets shape [32, 1280, 256] and strides [327680, 1, 1280]; a gather-style index gets no layout.
- [ ] 2.2 Add the fused bridged kind's coverage reason in `legality.py`. It admits one `sum` reduction, loads and per-element ops before it, per-output ops and stores after it, and a recorded layout. Each refusal names the part not covered (a reduction-scope store, several reductions, a non-sum reduction, no layout). Verify: tests for each refusal, including RMSNorm forward's reduction-scope store (the spec's "Per-element pass over the row").
- [ ] 2.3 Narrow the "Fused body" scenario in `openspec/changes/torch-runtime-bridge/specs/runtime-bridge/spec.md` to bodies outside `fused-reductions`' coverage. Verify: `openspec validate torch-runtime-bridge` and `openspec validate fused-reduction-bridge` pass.

## 3. The replaying `ReduceArgs` (design 1–4, 6, 8)

- [ ] 3.1 Expose the native crate's body, index and description-validation emitters as a small public API. Link the native crate from the bridge crate, and align CubeCL features so `cargo tree -d` shows one CubeCL. Verify: both crates build, clippy is clean, and the native crate's tests and `native_matrix` output are unchanged.
- [ ] 3.2 Copy Burn's `FusedReduceArgs`, state and `reduce_kernel_fused` launch into the bridge, and replace `fuse_on_read`/`fuse_on_write` with emission of the recorded ops through `intrinsic!`. Inputs are a sequence of typed global buffers. The body is comptime configuration. The offset is decoded through the recorded layout. The body is emitted per lane for `SizeIn > 1`. Any CubeCL or CubeK change it needs goes into `sources/` with a `sources/NOTES.md` entry. Verify: a bridge-level test runs the four reference kernels with each CubeK routine (Unit, Plane, Cube, inferred) and compares every output with the descriptor-compiled result, plus a guard region showing nothing outside the outputs is written.
- [ ] 3.3 Add the `bridge_fused_reduce` entry point. It takes the body, the layout, the buffers, the routine and the numerics policy (contraction, division precision) as the plain bridge does. Verify: a PTX check shows the policy's division instruction in an epilogue with a division, and the plain sum bridge's tests still pass.

## 4. Selection (design 7)

- [ ] 4.1 Add the fused kind's catalog entries (Unit, Plane, Cube, each inferred) and its launcher in `bridged.py`. The launch check comes from the frozen plan's contract, as for plain sums. Verify: for each reference graph under `first_use`, the candidate set includes the three fused candidates (spec scenarios "Split weight gradient" and "Unsplit column reduction"), and RMSNorm forward and input gradient show the refusal reason. The compiled results match eager within the band test.
- [ ] 4.2 Re-record the corpus manifest after reviewing the check's diff. Only kernels the coverage reason admits may change, and each change only adds candidates or selections. Verify: the reviewed diff is summarized in the group's commit message.

## 5. Measurement against Triton (spec "Triton floor")

- [ ] 5.1 Add the reference graphs to the band's reference cases and run the band with each fused candidate pinned on the 3080 and 3090. Verify: the verdicts are in `results/fused-reduction-bridge.md`, and every candidate is inside the band or its failure is recorded with a cause.
- [ ] 5.2 Pinned ncu timing of each fused candidate against the Triton kernels, at fixed clocks, on the 3080 and confirmed on the 3090. Record CubeK's inferred blueprint and vector size per case. Decide per-lane against vector-typed emission from these numbers (design 6). Verify: the table and decision are in the results file. Any slower-than-Triton case beyond noise has its cause found (ncu source and stall view) and is fixed in `sources/` or recorded against a bead.
- [ ] 5.3 First-call cost of the reference graphs with the fused kind offered against withdrawn, and against Triton, using `cubek_first_call.py` extended to these graphs. Verify: the table is in the results file.

## 6. Records

- [ ] 6.1 Update `docs/architecture.md`'s "Where we are" table with what the ecosystem path now covers and the measured result. Add bead notes on zah.11: the step 3 outcome and the missing abstractions found, if any. Add beads for any gap left open, and record the follow-on change for row kernels with a per-element pass (`reduce_broadcasted`). Verify: the notes and beads exist (`bd show zah.11`).

## Workflow follow-up

- Archive after review, once torch-runtime-bridge is archived (its `runtime-bridge` spec must exist in `openspec/specs/` first).
