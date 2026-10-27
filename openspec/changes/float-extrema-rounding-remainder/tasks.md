# Tasks

Each group closes the same way:
1. Run `ruff format`, `ruff check`, `cargo fmt` and clippy (`-D warnings`, all three feature sets).
2. Run the full suite and the manifest check, and save the suite output to a file.
   - Suite: `stage0/.venv/bin/python -m pytest -q -n 4 cubecl_torch/testing`.
   - Manifest: `stage0/.venv/bin/python -m cubecl_torch.testing.manifest_check check --jobs 6`.
   - Never rebuild the native pieces, or edit `.stage0-sources/cubecl`, while either runs.
3. Run a Sonnet (medium) review of the group's diff, then verify and fix its findings.
4. Commit the group.

## 1. CubeCL patch: half-to-even `RoundOp` folding (no front-end change)

- [x] 1.1 In `.stage0-sources/cubecl` at the pin, make `RoundOp`'s `const_eval` use
  `round_ties_even()` for f16, bf16, f32 and f64 (design 3).
  - Inspect the other backends' `RoundOp` lowering (LLVM, SPIR-V, WGSL) and record in the
    design's Context whether any rounds half away from zero.
  - Regenerate `cubecl_torch/native/cubecl-fp.patch` from the working tree diff.
  - Rebuild with `native/build.sh`.

  Verify that `build.sh` accepts the tree and that clippy passes for all three feature sets.
- [x] 1.2 Add a Rust test in `native/tests/fp_rewrites.rs`: `RoundOp` with the constants 0.5,
  1.5, 2.5, −2.5 and −0.5 folds to 0.0, 2.0, 2.0, −2.0 and −0.0. Verify with `cargo test`.
- [x] 1.3 Run the manifest check and the `native_matrix` self-comparison. Verify that the
  manifest passes with no changed case: generated code is unchanged although the compiler digest
  moved. Close the group.

## 2. Reference data

- [ ] 2.1 Add `cubecl_torch/testing/rounding_remainder_reference.py`, on the
  `float_arithmetic_reference.py` pattern and run with the Triton venv (design 7). It records into
  `testing/data/rounding_remainder_triton.pt`:
  - seeded inputs (as SHA-256);
  - Inductor+Triton outputs for every scenario of the `float-remainder`, `float-rounding`,
    `float-sign-exponent` and `float-extrema` (isinf) deltas that names Triton;
  - eager outputs for the cases where the specs say Triton is wrong.

  Construct special values explicitly, and check large-quotient and subnormal inputs in float64
  in the generator. Pin the device with `torch.cuda.set_device`: Inductor+Triton faulted on
  `cuda:1` tensors without it. Verify that a rerun regenerates byte-identical data and that the
  file is small enough to track. Close the group.

## 3. Remainder: `fmod`, float `mod` and float `remainder`

- [ ] 3.1 Add the `fmod` descriptor op and float `mod` (design 1):
  - Python: `OP_RULES` rows. `fmod` takes floats. `mod`/`remainder` widen to float32, float16
    and bfloat16 and keep rejecting bool, with reasons naming the op and the dtype.
  - Rust: `VALUE_OPS` rows. Float `fmod` and `mod` emit `FRemOp`; integer `mod` keeps its C
    remainder.

  Add a Rust test that a bool `fmod` is rejected, and a Python test that the generated CUDA calls
  `fmod` and never emits the floor-modulo formula. Verify with `cargo test` and the new test.
- [ ] 3.2 Extend `CubeCLOps.remainder` to floats as eager's form (design 2). Verify that
  `torch.remainder`, `a % b` and `torch.fmod` graphs compile on the pointwise, reduction and
  indirect paths.
- [ ] 3.3 Add `testing/test_float_remainder.py` with the contract tests for every
  `float-remainder` scenario except float floor division (group 4) and timing (group 6). Run
  them against the recorded data and eager:
  - large quotients;
  - the cases where Triton is wrong;
  - agreement with Triton elsewhere;
  - infinite divisors;
  - contraction on;
  - fused into a reduction;
  - applied to a gather.

  Change the float `fmod`/`remainder` rejection assertions in `test_contracts_fp.py` and
  `test_integer_arithmetic.py` to acceptance. Add the `rem_*` and `loaded_rem` audit-route cases
  at contraction on (design 7). Verify that they pass.
- [ ] 3.4 Set `fmod`, `remainder` and `mod` to implemented in `coverage_inventory.py`, with domain
  notes: exact truncating `fmod`; `remainder` as eager, where Triton's `frem` is inexact; float
  `mod` as `fmod`. Update `scoreboard/families.py` if its reject pattern no longer matches.
  Regenerate the markdown and verify `test_coverage_inventory.py`. Close the group.

## 4. Rounding, `isinf` and float floor division

- [ ] 4.1 Add the `floor`, `ceil`, `trunc`, `round` and `isinf` descriptor ops (design 1, 3):
  - `OP_RULES` and `VALUE_OPS` rows, floats only, with `isinf` returning bool.
  - Emission via `FloorOp`, `CeilOp`, `TruncOp`, `RoundOp` and `IsInfOp`.

  In `CubeCLOps`:
  - integer `floor`/`ceil`/`trunc`/`round` return the operand;
  - add the four `*_to_int` compositions.

  Add a Rust test that a bool `floor` is rejected. Verify with `cargo test`.
- [ ] 4.2 Add `testing/test_float_rounding.py` with every `float-rounding` scenario except timing:
  - special values and ties;
  - round with decimals;
  - constant tie, folded with no rounding call in the CUDA;
  - ops-level `*_to_int` through a test op on the `int_truediv` pattern;
  - fused into a reduction.

  Verify that they pass.
- [ ] 4.3 Add the `float-extrema` isinf scenarios to `test_float_extrema.py`, and the
  "Rounding integers is the identity" scenario to `test_integer_arithmetic.py`. Add the float
  floor-division scenarios of `float-remainder` to `test_float_remainder.py`, recording eager's
  divergent values. Verify that they pass.
- [ ] 4.4 Set `floor`, `ceil`, `trunc`, `round`, the four `*_to_int` ops and `isinf` to
  implemented in the coverage inventory, with domain notes (half-to-even `round`; integer
  operands are the identity; float floor division follows Inductor's lowering). Regenerate and
  verify `test_coverage_inventory.py`. Close the group.

## 5. Sign and exponent ops

- [ ] 5.1 Add the `CubeCLOps` decompositions `sign`, `signbit` and `copysign` (design 4), with
  their decomposed `OP_RULES` rows (excluded from the Python/Rust table comparison). Verify that
  `torch.sign`, `torch.signbit` and `torch.copysign` graphs compile for float and integer
  operands.
- [ ] 5.2 Add the `nextafter`, `ldexp` and `frexp` decompositions (design 5) with their
  `OP_RULES` rows. `frexp` returns the `(mantissa, exponent)` pair. Verify that each graph
  compiles, including `frexp` with one output and with both outputs used.
- [ ] 5.3 Add `testing/test_float_sign_exponent.py` with every `float-sign-exponent` scenario
  except timing, against the recorded data and eager:
  - include a `nextafter` operand computed in the kernel, not only a loaded one (design Risks);
  - include bfloat16 `ldexp` results in float32's subnormal range.

  Add the "Integer sign and sign bit" scenario to `test_integer_arithmetic.py`. Verify that they
  pass.
- [ ] 5.4 Set `sign`, `signbit`, `copysign`, `nextafter`, `ldexp` and `frexp` to implemented in
  the coverage inventory, with domain notes: eager where Triton is wrong (`nextafter` in the
  dtype, subnormal `ldexp`/`frexp`); `ldexp` with an integer exponent only. Regenerate and verify
  `test_coverage_inventory.py`, with the `extrema` family showing no missing item. Close the
  group.

## 6. Integration: code, speed and workloads

- [ ] 6.1 Inspect the SASS of four kernels (see `cubecl_torch/README.md`, Inspecting CubeCL
  kernels):
  - `remainder` and `fmod`: one `fmodf` routine per element and no floor-modulo sequence;
  - `round`: a single `FRND` instruction;
  - `ldexp` and `nextafter`: register count and instruction count compared with Triton's.

  Record the findings in the design's Context. If any is unexpected, fix the emission before
  timing.
- [ ] 6.2 Add `testing/rounding_remainder_timing.py` on the `integer_timing.py` pattern (design
  8). It times every spec's timing scenario under CubeCL and Inductor+Triton:
  - the RTX 3080 first, then confirmation on the RTX 3090;
  - remainder with small and large quotients;
  - results to `results/rounding-remainder-timing-{3080,3090}.json` and a summary
    `rounding-remainder-timing.md`.

  Verify that each CubeCL time is within Triton's spread or faster. Otherwise fix the gap first
  (for remainder, the fast path of design 2), or file a bead if fixing it takes much more effort.
- [ ] 6.3 Rerun the census inventory for the scoreboard workloads (`--on-reject record`, results
  to `scoreboard/results/extrema-inventory`). Record on `x3c` and `4z8.19` which extrema reasons
  are gone and what now stops the affected regions. Clean up the run's caches. Verify that no
  `extrema`-family reason remains.
- [ ] 6.4 Beads and memory:
  - Close `x3c`, noting the float floor-division divergence from eager (design 6).
  - Update the `bd remember` key `cubecl-coverage-checklist-2026-10-22` with the extrema
    contract: exact `fmod`; eager `remainder`; half-to-even `round`; eager where Triton
    mishandles subnormals or half-precision steps; floor division as Inductor.
  - Record the `RoundOp` folding fix, and `FModFloorOp`'s inexactness, as upstreaming notes on
    `cubecl-pin-and-upgrade`.

  Close the group.

## Workflow follow-up

- Archive the change after the last group is committed. The three new specs then join
  `openspec/specs/`, and the deltas join `float-extrema` and `integer-arithmetic`.
