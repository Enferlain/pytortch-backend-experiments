# Tasks

Each group closes the same way:
1. Run `ruff format`, `ruff check`, `cargo fmt` and clippy (`-D warnings`, all three feature sets).
2. Run the full suite and the manifest check. The suite command is
   `stage0/.venv/bin/python -m pytest -q -n 4 cubecl_torch/testing`; save its output to a file. The
   manifest command is `stage0/.venv/bin/python -m cubecl_torch.testing.manifest_check check --jobs 6`.
   Never rebuild the native pieces while either runs.
3. Run a Sonnet (medium) review of the group's diff, then verify and fix its findings.
4. Commit the group.

## 1. CubeCL patch: single-rounding fma folding and rounded multiply/divide (no front-end change)

- [x] 1.1 Extend `.stage0-sources/cubecl` at the pin (design 2):
  - `FmaOp`'s `const_eval` uses `mul_add` for f32/f64.
  - New `FMulRnOp` (`math.f_mul_rn`) and `FDivRnOp` (`math.f_div_rn`):
    - Constant folding: IEEE `*` and `/`.
    - Simplification: `x / 1` only.
    - C++ printing: `__fmul_rn`/`__fdiv_rn`, scalarized over vectors.
    - LLVM lowering: plain `fmul`/`fdiv`, no fast-math flags.

  Regenerate `cubecl_torch/native/cubecl-fp.patch` from the working tree diff. Add the newly
  patched files to the list in `.stage0-sources/README.md`. Rebuild with `native/build.sh`.
  Verify that `build.sh` accepts the tree and that clippy passes for all three feature sets.
- [x] 1.2 Add Rust tests in the native crate:
  - An `FmaOp` with three constant operands, chosen so single and double rounding differ, folds
    to the single-rounding value.
  - `FMulRnOp`/`FDivRnOp` fold to IEEE results, including a zero divisor.
  - Neither is rewritten by `inst_combine` when contraction is allowed.

  Verify with `cargo test`.
- [x] 1.3 Run the manifest check and the `native_matrix` self-comparison. Verify that the manifest
  PASSes with no changed case: generated code is unchanged although the compiler digest moved.
  Close the group.

## 2. Triton reference data

- [x] 2.1 Add `cubecl_torch/testing/float_arithmetic_reference.py` on the `integer_reference.py`
  pattern, run with the Triton venv. It records seeded inputs (as SHA-256) and Inductor+Triton
  outputs into `testing/data/float_arithmetic_triton.pt` for every float-arithmetic spec scenario
  that names Triton:
  - `abs`, `square` and `relu` over special values and every listed dtype
  - `add`/`sub` with alpha
  - `addcmul` and `addcdiv` with value 1 and value ≠ 1
  - the plain division alongside `addcdiv`
  - `lerp` with scalar and tensor weights

  The rounding-sensitive float32 inputs are constructed from seeds and checked in float64 in the
  generator (design 5). Verify that a rerun regenerates byte-identical data, and that the data file is small enough
  to track (no multi-megabyte tensors). Close the group.

## 3. `abs`, `square` and `relu`

- [ ] 3.1 Add the `abs` descriptor op (design 1, 3, 4):
  - Python: an `OP_RULES` row.
  - Rust: a `VALUE_OPS` row, plus emission — `FAbsOp` for floats, wrapping compare-and-select for
    signed integers, identity for unsigned.

  Add a Rust test that bool `abs` is rejected and that int32 `abs` emits no C++ `abs` call. Verify
  with `cargo test`.
- [ ] 3.2 Add the `square` and `relu` decompositions in `CubeCLOps`, with their decomposed
  `OP_RULES` rows. Exclude the decomposed rows from the Python/Rust table comparison if one exists.
  Verify that `torch.square`, `torch.relu` and `torch.abs` graphs compile on the pointwise,
  reduction and indirect paths (fused into a sum, applied to a gather).
- [ ] 3.3 Add contract tests in `testing/test_float_arithmetic.py` for the spec's "Float absolute
  value" (special values; the lerp scenario waits for group 4), "Square" and "Relu" scenarios,
  against the recorded Triton data and eager. Also add the `integer-arithmetic` scenario "Every
  width with its limits" for integer `abs`. Verify that they pass.
- [ ] 3.4 Set `abs`, `square` and `relu` to implemented in `coverage_inventory.py`, with domain
  notes (float sign-clearing `abs`; integer `abs` wraps at the minimum; `relu` as `maximum`, +0.0
  for −0.0). Regenerate the markdown and verify `test_coverage_inventory.py`. Close the group.

## 4. `fma`, `mul_rn` and `div_rn`

- [ ] 4.1 Add the three descriptor ops (design 2, 4):
  - Python: `OP_RULES` rows, floats only.
  - Rust: `VALUE_OPS` rows, `fma` with arity 3.
  - Emission: `FmaOp`, `FMulRnOp`, `FDivRnOp`.

  Add a Rust test that an int32 `fma` is rejected naming the op and the dtype, and one that the
  generated CUDA calls `fma`, `__fmul_rn` and `__fdiv_rn` (group 1 tests folding only, not
  printing). Verify with `cargo test`.
- [ ] 4.2 Add contract tests for these spec scenarios, against the Triton data and eager:
  - "add with alpha"
  - "Single rounding is observable" (contraction on, and `emulate_precision_casts` on)
  - "Constant operands" (a descriptor-level test with three constant `fma` operands)
  - "addcmul with a value"
  - "addcdiv with a value" and "addcdiv with value 1"
  - "Plain division keeps its policy"
  - "Weight test in lerp"

  Verify that they pass.
- [ ] 4.3 Add the integration tests:
  - The spec's "Adam-style step": every region compiles through CubeCL, bitwise eager.
  - "Fused into a reduction": one kernel each, checked by launch count.
  - "Integer operand to fma".

  Verify that they pass.
- [ ] 4.4 Set `fma`, `mul_rn` and `div_rn` to implemented in the coverage inventory, with domain
  notes (single rounding; never contracted; IEEE regardless of the division policy). Regenerate the
  markdown and verify `test_coverage_inventory.py`. Close the group.

## 5. Integration: code, speed and workloads

- [ ] 5.1 Inspect the SASS of one `addcmul(value≠1)` kernel and one `addcdiv` kernel (see
  `cubecl_torch/README.md`, Inspecting CubeCL kernels). Record two things:
  - The `mul_rn` product stays a separate rounded multiply, not fused into the `fma`.
  - `div_rn` runs one IEEE division routine per element, as Triton's `tl.div_rn`.

  Record the finding in the design's Context. If either fails, fix the emission before timing.
- [ ] 5.2 Add `testing/float_arithmetic_timing.py` on the `integer_timing.py` pattern, timing the
  spec's three kernels under CubeCL and Inductor+Triton:
  - The RTX 3080 first, then confirmation on the RTX 3090.
  - Clock warm-up and alternating rounds.
  - Results to `cubecl_torch/testing/results/float-arithmetic-timing-{3080,3090}.json`, plus a
    summary `float-arithmetic-timing.md`.

  Verify that each CubeCL time is within Triton's spread or faster. Otherwise inspect the SASS and
  either fix the gap or file a bead before going on.
- [ ] 5.3 Rerun the census inventory for the scoreboard workloads (`--on-reject record`, results to
  `scoreboard/results/basic-inventory`). Record on `4x9.2` and `4z8.19`:
  - which `fma`/`div_rn` reasons are gone
  - what now stops the affected SDXL and Wan 2.2 regions

  Clean up the run's caches. Verify that no `basic`-family reason remains, except float floor
  division, which stops at `floor` (`x3c`).
- [ ] 5.4 Beads and memory:
  - Close `4x9.2`.
  - Add the `abs`-in-lerp and `square`-in-var notes as resolved.
  - Update the `bd remember` key `cubecl-coverage-checklist-2026-10-22` with the basic contract
    (explicit rounding honoured whatever the global numerics; `relu` follows Triton's signed
    zero).
  - Record the two new CubeCL ops and the `FmaOp` folding fix as upstreaming candidates on the
    `cubecl-pin-and-upgrade` memory.

  Close the group.

## Workflow follow-up

- Archive the change after the last group is committed, so the `float-arithmetic` spec joins
  `openspec/specs/` and the integer `abs` requirement joins `integer-arithmetic`.
