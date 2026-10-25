# Tasks

Each group closes the same way:
1. Run `ruff format`, `ruff check`, `cargo fmt` and clippy (`-D warnings`).
2. Run the full suite (`stage0/.venv/bin/python -m pytest -q -n 4 cubecl_torch/testing`, output
   saved to a file) and the manifest check
   (`stage0/.venv/bin/python -m cubecl_torch.testing.manifest_check check --jobs 6`).
3. Run a Sonnet (medium) review of the group's diff, then verify and fix its findings.
4. Commit the group.

## 1. Dtype table and shared op table (no behaviour change)

- [x] 1.1 Add the dtype table (design 2):
  - In `native/src/ir.rs`: `elem` and `constant` for int8, int16, int32, int64, uint8 and bool, with a `ConstantValue::UInt` arm and a range check per width.
  - In Python: an `INTEGERS` table (name, signed, width) in `compiler/descriptor.py`. The compiler layer may not import the Inductor layer, so `semantic_checks.INTEGERS` and `semantics.DTYPES` derive from it. Group 3 adds a value-dtype tuple distinct from storage `DTYPES` when descriptors start carrying the new dtypes.

  Admit nothing new yet: the value dtypes accepted on each path stay as today. Verify with a Rust unit test of `elem`/`constant` per width, including out-of-range constants rejected, and with the unchanged suite.
- [x] 1.2 Introduce `OP_RULES` in `semantic_checks.py` (design 3) for the ops admitted today, with today's classes, and type `node_reasons` from it. The table must reproduce today's admission exactly. Today int64 `minimum`/`maximum` and bool `bitwise_and`/`bitwise_or` are admitted only on some paths, so their `operation_type` hooks (`rng_type`, `predicate_type`) stay until 2.3 and 3.2 make the table uniform. After that, `extra_ops`/`operation_type` remain only for `reduction`, `rand`, `load_seed`, `indirect` and `store_reduction`. Verify that the suite passes and that `test_rejections.py` messages are unchanged.
- [x] 1.3 Restructure `semantic.rs::validate_ops` into a per-op rule table with the same classes, reproducing today's checks. Verify that `cargo test`, the `native_matrix` self-comparison and the manifest check report no change.
- [x] 1.4 Close the group (manifest PASS with no changed case).

## 2. One op vocabulary

- [x] 2.1 Fold the effects path's integer ops into the shared layer (design 1):
  - Python: `effects.CubeCLKernel.operation` and `CubeCLOps._default` emit `add`, `mul`, `minimum`, `maximum`, `bitwise_and` and `bitwise_or` with int64/bool operands, in place of `iadd`, `imul`, `imin`, `imax`, `and` and `or`.
  - Rust: `semantic.rs` validates and emits them by operand dtype, and `effects.rs` loses its integer arms.

  Verify with `cargo test` and the suite.
- [x] 2.2 Run the manifest check and list every changed case. For each one, confirm that only the descriptor and artifact-key digests differ and that the IR, CUDA and PTX bodies are identical (entrypoint-normalized). Write the list and comparison to `cubecl_torch/testing/results/integer-rename-manifest.md`, then re-record the manifest. Verify that a second check PASSes.
- [x] 2.3 Admit bool `bitwise_and`/`bitwise_or` on the reduction path through the shared table. Verify that `torch.fmax(a, b).sum(-1)` and `torch.fmax(a, b) * 2` each run as one kernel (one native launch per warm call) and match eager. Add both as tests, and update the note on `4x9.1`.
- [x] 2.4 Close the group.

## 3. Integer values and arithmetic

- [x] 3.1 Admit the integer value dtypes (int8, int16, int32, int64, uint8, bool) on every path through `constant`, `to_dtype`, `index_expr`/`value_expr` and `masked`.
  - Casts follow design 4: to bool as `ne(x, 0)`; float64 scalars still floating-only.
  - Integer `index_expr` computes in int64, then casts.
  - Retire the effects path's int32 `constant`/`cast`/`index` arms in favour of the shared ones. Re-check the RNG descriptors with the manifest; any changed case goes into 2.2's file.
  - Admit int64 storage in ordinary pointwise kernels. It was admitted only on the RNG and indirect paths, although the inventory and this spec treat int64 storage as supported. Schema 4 already accepts it. Reduction kernels keep rejecting int64 buffers (`4z8.25`), and the fusion check keeps such nodes out of them.
  - Add the cast tests from the spec's "Casts between bool, integer and float" scenario (eager parts).

  Verify that those tests pass.
- [x] 3.2 Add integer `add`, `sub`, `mul` and `neg`, comparisons (signed, unsigned and bool), `where`, `maximum`/`minimum` and `masked` results, in Python `OP_RULES`/`operation` and in Rust validate/emit (design 4). Add the contract tests for the spec's "Integer value dtypes keep their width", "Wrapping integer arithmetic" and "Integer and bool comparisons, selection and extrema" scenarios, every width with its limits. Verify that they pass.
- [x] 3.3 Add the logical and bitwise ops (`logical_*`, `and_`, `or_`, `xor`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`), with the aliases normalized in Python (design 5) and `BoolNot` for bool negation. Bool `bitwise_not` emits `BoolNot`, never C++ `~` (which promotes a bool to int); the eager comparison of `~(p > 0)` covers it. Add the contract tests for "Logical operations" (1-D and 2-D padding masks, the logical expression; `avg_pool2d` with padding gets past `and_` but stops at `Min()` in its index, bead `4z8.18`) and "Bitwise operations". Verify that they pass.
- [x] 3.4 Update `coverage_inventory.py` with the implemented ops and integer domain notes:
  - The integer domain notes for add, sub, mul, neg, comparisons, where, maximum/minimum and to_dtype.
  - The logical and bitwise ops become implemented.

  Regenerate `coverage_inventory.md` and verify `test_coverage_inventory.py`.
- [x] 3.5 Close the group.

## 4. Shifts, division and bitcast

- [x] 4.1 Add a Triton reference generator, `cubecl_torch/testing/integer_reference.py`, run with the Triton venv, on the `extrema_reference.py` pattern. It records seeded inputs (as hashes) and Triton outputs to `testing/data/integer_triton.pt` for:
  - shifts over the spec's shift amounts and every width
  - float-to-integer casts of NaN, ±inf and out-of-range values
  - `int_truediv`

  Verify that a rerun regenerates byte-identical data.
- [x] 4.2 Add the shifts (`bitwise_left_shift`, `bitwise_right_shift`, `lshift`, `rshift`) with the eager guard (design 4). Add the contract tests for "Shifts follow eager for every shift amount", asserting eager equality and listing each case where the recorded Triton output differs. Also add the NaN/out-of-range cast cases against the Triton data. Verify that they pass.
- [x] 4.3 Add `truncdiv`, `floordiv` and integer `mod` with the zero-divisor guard. Inductor emits `mod` for integer `fmod`, and `remainder` for integer `torch.remainder`, lowered here as eager's sign adjustment of `mod` (Inductor's generic form needs `signbit`, an extrema op). Float `mod`/`remainder` stay rejected (`x3c`). Add the contract tests for "Integer division and remainder", including zero divisors and the dtype minimum divided by −1 (kernel completes; nonzero-divisor elements equal eager). Verify that they pass.
- [x] 4.4 Add `int_truediv` (two casts and `truediv`) and `to_dtype_bitcast` (the `bitcast` descriptor op with a same-size check in Python, `descriptor.py` and Rust; design 4). Add the contract tests for "True division of integers" (against the Triton data) and "Bit reinterpretation" (float32↔int32, bfloat16/float16↔int16, specials). Add a Rust test that rejects a size-mismatched bitcast. Verify that they pass.
- [x] 4.5 Inspect the SASS of one int64 `floordiv` kernel and one guarded shift kernel (see `cubecl_torch/README.md`, Inspecting CubeCL kernels). Record whether the division pair compiles to one routine and what the guard costs. If floordiv runs two division routines, switch to Triton's complement form (design 4 alternative) and rerun 4.3's tests. Record the finding in the design's Context or on the bead.
- [x] 4.6 Set the remaining int-family ops to implemented in the coverage inventory, with domain notes: shifts (eager guard), division (zero divisor unspecified, no trap), `int_truediv`, `to_dtype_bitcast`. Split integer `mod` from float `mod` in the notes. Regenerate the markdown and verify `test_coverage_inventory.py`.
- [x] 4.7 Close the group.

## 5. Out-of-scope rejections and integration

- [ ] 5.1 Add rejection tests for what stays out of scope. Each must raise the unsupported-capability error with a reason naming the dtype or op, and leave no cached kernel:
  - an integer sum reduction
  - int32 tensor storage
  - an op-level bool `add` in a hand-written descriptor (Inductor lowers aten bool `add`/`mul` to `logical_or`/`logical_and`; add those as acceptance cases)
  - float `fmod`

  Verify that they pass, and that `scoreboard/families.py` maps each reason to its family with none unmapped (`test_rejections.py`).
- [ ] 5.2 Time the four representative kernels from design 7 under CubeCL and Inductor+Triton on the RTX 3080, and confirm on the RTX 3090, with clock warm-up and repeated rounds. Write `cubecl_torch/testing/results/integer-timing.json`. Verify that each CubeCL time is within Triton's spread or faster. Otherwise inspect the SASS and file a bead before going on.
- [ ] 5.3 Rerun the census inventory for all ten scoreboard workloads (`--on-reject record`, results to `scoreboard/results/int-inventory`). Record on `4z8.19` and `4x9.1`:
  - which `int` reasons are gone
  - which remain
  - each affected region's new set of reasons

  Clean up the run's caches. Verify that no reason the spec covers remains.
- [ ] 5.4 Beads and memory:
  - File the integer-reduction follow-up bead: integer sum/max/min combiners; needs `4x9.1`, blocks `4x9.9`'s integer cases.
  - Close `4x9.1`.
  - Update the `bd remember` key `cubecl-coverage-checklist-2026-10-22` with the integer contract (eager-defined shifts, zero divisor unspecified, width-carrying values).

  Close the group.

## Workflow follow-up

- Archive the change after the last group is committed, so the `integer-arithmetic` spec joins `openspec/specs/`.
