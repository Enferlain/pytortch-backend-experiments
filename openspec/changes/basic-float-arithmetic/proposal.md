# Proposal

## Why

Six common pointwise ops are still rejected: `abs`, `square`, `relu`, `fma`, `mul_rn` and
`div_rn`. These make up the coverage checklist's `basic` family (bead `4x9.2`).

The fused ones matter most. On CUDA, Inductor lowers each of these to `fma`, which rounds once,
to match eager CUDA:
- `add`/`sub` with `alpha`
- `addcmul` and `addcdiv`
- both `lerp` forms

The `addcmul`/`addcdiv` lowerings also need `mul_rn` or `div_rn`, a multiply or divide that must
round on its own. So does float floor division.

These ops appear in every Adam-style optimizer step, in GroupNorm and other norm backwards, and
in residual mixing. The 2026-10-25 census finds:
- `fma` in 14 SDXL regions (51 nodes) and 2 Wan 2.2 regions
- `div_rn` in one SDXL region

`lerp` with a tensor weight also needs `abs` (`weight.abs() >= 0.5`). `var`/`var_mean` over rows
of 8 or fewer elements need `square` (bead note on `4x9.2`).

## What Changes

- **Fused multiply-add, `fma(x, y, z)`.**
  - Computes `x * y + z` with a single rounding, as Triton's `tl.fma` and eager CUDA do, whatever
    the contraction setting.
  - Constant folding in the CubeCL IR (`FmaOp`) rounds twice today, so a folded `fma` would differ
    from the executed one. The patch stack fixes that.
- **Rounded multiply and divide, `mul_rn(x, y)` and `div_rn(x, y)`.**
  - Each rounds its own result to nearest even, as Triton's `libdevice.mul_rn` and `tl.div_rn`
    do.
  - `mul_rn` is never contracted into a following add, even with contraction on.
  - `div_rn` is IEEE division even under the default `div.full` division policy (4z8.17).
  - CubeCL has no such ops. The patch stack adds them, printing as `__fmul_rn`/`__fdiv_rn` in the
    CUDA C++ backend.
- **Absolute value, `abs(x)`.**
  - On floats it clears the sign bit (NaN stays NaN, −0 → +0), as Triton's `fabs`.
  - On signed integers it wraps at the dtype minimum, as eager. `abs(INT_MIN)` is undefined in
    C++, so it is built from defined operations.
  - On unsigned integers it is the identity.
- **`square(x)` and `relu(x)`.** These decompose in the front end, as Inductor's generic op
  overrides and Triton do:
  - `square` becomes `mul(x, x)`.
  - `relu` becomes `maximum(x, 0)`, with the zero in `x`'s dtype. So it inherits the
    NaN-propagating, ordered-signed-zero `maximum` of the `float-extrema` spec.
- **Admitted everywhere.** All six ops are admitted on every kernel path (pointwise, fused,
  reduction, RNG, indirect) through the one op table, with float and integer domains as stated
  above. `fma`, `mul_rn` and `div_rn` take floating operands only, as Inductor emits them.
- **Accuracy and speed held to the Triton floor.** Every op's result equals Triton's bit for bit
  on recorded inputs (special values included), and equals eager where Inductor's lowering
  intends to match eager (`addcmul`, `addcdiv`, `lerp`, `add` with `alpha`). Representative
  optimizer-step kernels are timed against Triton.
- **Coverage inventory.** The six `basic` items become implemented, with domain notes.
- **Out of scope:**
  - Float floor division needs `floor` (extrema family, `x3c`), so it still stops there after
    `div_rn` is admitted.
  - Transcendental ops are `4x9.3`.

## Capabilities

### New Capabilities

- `float-arithmetic`: the numerical contract of basic floating-point value ops in compiled
  kernels (absolute value, square, relu, fused multiply-add, and explicitly rounded multiply and
  divide), measured against Triton (the floor) and eager CUDA.

### Modified Capabilities

- `integer-arithmetic`: adds integer absolute value, which wraps at the dtype minimum.

## Impact

- **Python front end:**
  - `cubecl_torch/inductor/admission.py`: `OP_RULES` rows.
  - `cubecl_torch/inductor/codegen.py`: `CubeCLOps` `square`/`relu` decompositions; `fma`,
    `mul_rn`, `div_rn` and `abs` through `operation`.
- **Descriptor compiler:** `cubecl_torch/native/src/semantic.rs` (`VALUE_OPS`, validation and
  emission of `abs`, `fma`, `mul_rn`, `div_rn`), plus Rust tests.
- **CubeCL patch stack** (`cubecl_torch/native/cubecl-fp.patch`):
  - Single-rounding constant folding for `FmaOp`.
  - New `FMulRnOp`/`FDivRnOp` with C++ printing, vector unrolling and LLVM lowering.
  - Recorded with the pin as the upgrade procedure requires.
  - The rebuilt compiler binary changes every artifact key, but generated code for existing
    kernels must not change; the manifest check verifies this.
- **Tests and data:**
  - A Triton reference generator and recorded data, on the `extrema_reference.py` pattern.
  - Contract and admission tests.
  - An FP-rewrite test that a constant-folded `fma` equals the executed one.
- **Results:** timing results under `cubecl_torch/testing/results/`; a census rerun for the
  SDXL and Wan 2.2 `fma` regions.
- **Coverage inventory:** `cubecl_torch/testing/coverage_inventory.py` and its generated markdown.
