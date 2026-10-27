# Proposal

## Why

The coverage checklist's `extrema` family (bead `x3c`) still has 16 rejected items:
- `fmod`, and float `remainder`/`mod`
- rounding: `floor`, `ceil`, `trunc`, `round` and their four `*_to_int` forms
- `sign`, `signbit`, `copysign` and `isinf`
- `nextafter`, `ldexp` and `frexp`

Float floor division (`a // b` on floats) also stops here. Inductor lowers it through `div_rn`,
`floor`, `isinf` and `isnan`, and `floor` is missing. Clamp-style optimizer code and index
arithmetic over float scales reach these ops.

`x3c` exists because these contracts are subtle. A pre-proposal probe on the RTX 3080 (pinned
PyTorch, Triton 3.8) compared Inductor+Triton with eager:

| Op | Inductor+Triton | Eager |
|---|---|---|
| float32 `1e10 % 3` | 256 | 1 |
| float32 `3e38 % 1.1` | 9.7e30 | 0.44 |
| float32 `1e7 % 0.1` | −0.049 (wrong sign) | 0.051 |
| `-0.0 % 2` | +0.0 | −0.0 |
| `nextafter(0, 1)` float32 | FLT_MIN (flushes subnormals) | smallest subnormal |
| `nextafter` float16/bfloat16 | never moves (steps in float32, rounds back) | one step in the dtype |
| `ldexp`, `frexp` of a subnormal | wrong (e.g. `frexp(1e-40)` exponent −2³¹+1) | exact |
| `frexp(-0.0)` mantissa | +0.0 | −0.0 |

Float `remainder` goes through Triton's `%`, which lowers to LLVM `frem`. NVPTX expands `frem`
inexactly. `fmod` (`libdevice.fmod`), `round`, `sign`, `copysign` and float floor division match
eager in the probe.

Triton is the floor, not the target. Where it is wrong as above, this backend follows eager.

## What Changes

- **Remainder.**
  - `fmod(a, b)` is the exact truncating remainder (CUDA `fmod`, the routine Triton's
    `libdevice.fmod` calls), equal to eager and Triton.
  - Float `remainder(a, b)` is eager's form: `r = fmod(a, b)`, plus `b` when `r` is nonzero and
    its sign differs from `b`'s. It is exact where Triton's is not (large quotients, signed zero,
    results outside `[0, b)`).
  - Float `mod` (Inductor's C remainder) is admitted as the exact `fmod`.
  - All three stay exact with FMA contraction on. CubeCL's `FModFloorOp` (`a − floor(a/b)·b`) is
    not used: it is inexact for large quotients and breaks under contraction (bead note on `x3c`).
- **Rounding.**
  - `floor`, `ceil`, `trunc` and `round` (half to even, as eager and `libdevice.nearbyint`)
    use CubeCL's rounding ops.
  - The four `*_to_int` forms are the rounding op followed by the existing cast.
  - The CubeCL patch fixes `RoundOp`'s constant folding. It folds with half away from zero
    while the executed `rint` rounds half to even, so a folded `round(2.5)` is 3 and an executed
    one is 2.
  - Float floor division compiles through Inductor's own lowering and matches Triton bit for bit.
- **Sign and classification.**
  - `sign` follows Inductor's generic form, as Triton, for floats and integers.
  - `signbit` reads the sign bit for floats and tests `< 0` for signed integers.
  - `copysign` combines the bits of its operands.
  - `isinf` uses CubeCL's `IsInfOp`.
- **Representation ops.**
  - `nextafter` steps in the operands' own dtype and keeps subnormals, as eager.
  - `ldexp` scales by a power of two with one rounding, as true `ldexp`, including subnormal
    inputs and results.
  - `frexp` returns eager's mantissa and exponent for subnormals and signed zeros. It is a
    two-valued op: Inductor emits it as two pointwise nodes over one value.
- **Admitted everywhere.** Every op is admitted on every kernel path (pointwise, fused,
  reduction, RNG, indirect) through the one op table, with the dtype domains above.
- **Accuracy and speed held to the Triton floor.**
  - Every op's non-NaN result equals Triton's bit for bit on recorded inputs, special values
    included, except where Triton is shown above to be wrong; there it equals eager.
  - The `remainder`, `fmod`, floor-division and rounding kernels are timed against Triton.
- **Coverage inventory.** The 16 items become implemented, with domain notes.

## Capabilities

### New Capabilities

- `float-remainder`: float `fmod`, `remainder`/`mod` and floor division in compiled kernels: their
  results for every sign combination, zero and infinite operands, signed zeros and large
  quotients, against eager and Triton.
- `float-rounding`: `floor`, `ceil`, `trunc`, `round` (half to even) and their `*_to_int` forms,
  including constant folding consistent with execution.
- `float-sign-exponent`: `sign`, `signbit`, `copysign`, `nextafter`, `ldexp` and `frexp`. These
  are ops on a float's sign and exponent, exact by definition and specified at the bit level,
  subnormals and signed zeros included.

### Modified Capabilities

- `float-extrema`: adds float `isinf` next to `isnan`.
- `integer-arithmetic`: adds integer `sign` and `signbit`, and states that rounding ops on
  integer operands are the identity, as eager.

## Impact

- **Python front end:**
  - `cubecl_torch/inductor/admission.py`: `OP_RULES` rows, and the float rows for
    `mod`/`remainder`.
  - `cubecl_torch/inductor/codegen.py`: `CubeCLOps` decompositions (float `remainder`, `sign`,
    `signbit`, `copysign`, `nextafter`, `ldexp`, `frexp`, `*_to_int`).
- **Descriptor compiler:** `cubecl_torch/native/src/semantic.rs` (`VALUE_OPS`, validation and
  emission of `fmod`, `floor`, `ceil`, `trunc`, `round`, `isinf`), plus Rust tests.
- **CubeCL patch stack** (`cubecl_torch/native/cubecl-fp.patch`): `RoundOp` constant folding
  rounds half to even. Recorded with the pin as the upgrade procedure requires. Generated code
  for existing kernels must not change; the manifest check verifies this.
- **Tests and data:**
  - A Triton reference generator and recorded data, on the `float_arithmetic_reference.py`
    pattern.
  - Eager reference data for the cases where Triton is wrong.
  - Contract and admission tests.
  - The existing rejection tests for `fmod`/`remainder` (`test_contracts_fp.py`,
    `test_integer_arithmetic.py`) change to acceptance tests.
- **Results:** timing results under `cubecl_torch/testing/results/`; a census rerun for the
  scoreboard workloads.
- **Coverage inventory:** `cubecl_torch/testing/coverage_inventory.py` and its generated markdown.
