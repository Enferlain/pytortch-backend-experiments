# Design

## Context

See proposal.md (Why) for the probe results that motivate the contracts.

Current state:
- The op table is shared by both sides: Python `admission.OP_RULES` and Rust
  `semantic.rs` `VALUE_OPS`, dispatched on dtype and used by pointwise, fused, reduction, RNG and
  indirect kernels.
- `CubeCLOps` in `inductor/codegen.py` already decomposes ops into admitted descriptor ops:
  - `square` and `relu`;
  - integer `remainder`, as eager's sign adjustment of the C remainder.
- `to_dtype_bitcast` is admitted with eager-exact bits for every float dtype. A float16/bfloat16
  value held in float32 compute is rounded to its dtype first, as Triton does.
- Integer ops wrap at their width. int16 and int32 values exist, so bit-level float work
  needs no new value type.
- float16/bfloat16 values compute in float32 and round once at their store, as in Triton.

CubeCL at the pin already has the ops this change needs:
- `FRemOp`, printed as C++ `fmod`: CUDA's exact `fmodf`, the routine Triton's `libdevice.fmod`
  calls.
- `FloorOp`, `CeilOp`, `TruncOp` and `RoundOp`, printed as `floor`, `ceil`, `trunc` and `rint`.
  `rint` rounds half to even, matching `libdevice.nearbyint` under the default rounding mode.
- `IsInfOp`, printed as `isinf`.

There is one inconsistency: `RoundOp`'s constant evaluator uses Rust `round()`, which rounds half
away from zero.

`RoundOp` in the other backends (checked in group 1):
- LLVM lowers it to `llvm.roundeven` (ties to even).
- WGSL's `round` is ties to even by the WGSL spec.
- SPIR-V maps it to GLSL `Round`, whose tie direction is implementation-defined (`RoundEven`
  would be exact). This is not used in production; it is recorded as an upstreaming note.

`FModFloorOp` (`a − floor(a/b)·b`) is inexact for large quotients. Under contraction its
expansion fuses into an `fma` and fails the FP audit route's `loaded_floor` case (bead note on
`x3c`).

Generated code, checked in group 6. Method: SASS for sm_86, from our `kernel.ptx` (NVRTC 13.0)
assembled with `ptxas -arch=sm_86` 13.1 and read with `cuobjdump -sass`; Triton's cached cubins
read the same way. Counts are for our scalar plan against Triton's two-element kernel:
- `fmod`/`remainder`: CUDA's `fmodf` is inlined, with no call and no floor-modulo sequence (its
  `FRND`s are `fmodf`'s own truncations). Ours is 112 instructions and 14–15 registers per
  element. Triton's `fmod` (libdevice) has the same shape, at 160 instructions for two elements.
  Triton's `frem`-based `remainder` adds a called division slow path (`CALL`).
- `round`: one `FRND` per element, as Triton.
- `ldexp`: 56 branch-free instructions and 11 registers, against Triton's libdevice `ldexpf` at
  112 for two elements.
- `nextafter`: 48 instructions and 12 registers, against Triton's 104 for two elements.

CubeK has nothing reusable for these ops. Its only rounding is in CPU reference code, and its
remainders are integer `div_mod` for index math.

The timing probe below was a quick check, not a qualified result. On the RTX 3080 at 4M float32
elements, eager `torch.fmod` (CUDA `fmodf`), Triton's `remainder` and a plain Triton add all ran
at 78–79 µs. The memory-bound kernel hides `fmodf`'s instruction cost, for small and large
quotients alike.

## Goals / Non-Goals

**Goals:**
- Every extrema-family op exact, or equal to Triton where Triton equals eager, with no new
  CubeCL ops: reuse CubeCL's existing ops, and build the rest from admitted descriptor ops.
- One implementation per op for all three float dtypes.

**Non-Goals:**
- Replacing Inductor's float floor-division lowering with eager's algorithm (decision 6).
- `ldexp` with a float exponent: Inductor's fallback needs `pow` (`4x9.3`).
- Fixing `FModFloorOp` upstream. It is unused here; recorded for upstreaming only.

## Decisions

### 1. Direct descriptor ops only where CubeCL has the op

New `VALUE_OPS` rows with Rust emission:

| Op | Operands | Emits |
|---|---|---|
| `fmod` | float | `FRemOp` |
| `mod` | float (integer `mod` keeps its C remainder) | `FRemOp` |
| `floor`, `ceil`, `trunc`, `round` | float | `FloorOp`, `CeilOp`, `TruncOp`, `RoundOp` |
| `isinf` | float | `IsInfOp` |

Everything else is a `CubeCLOps` decomposition into admitted ops (decisions 2–5). This follows
the `square`/`relu` precedent.

Decompositions in Python reuse ops that are already admitted and tested on every kernel path. The
emitter and the Python/Rust table comparison stay unchanged for them.

Alternative: new CubeCL ops printing CUDA's `copysignf`, `nextafterf`, `ldexpf` and `frexpf`.
Rejected:
- float16/bfloat16 `nextafter` must step in its own format, so it needs bit-level code anyway.
- `frexp` has two outputs, which a CubeCL op does not express well.
- The patch stack grows with no accuracy or speed gain.

### 2. Remainder is eager's form over the exact `fmod`

Float `remainder` extends the existing integer decomposition:

```
r = mod(a, b)                        # FRemOp: exact fmod
ne(lt(r, 0), lt(b, 0)) and ne(r, 0)  # eager's signs_differ(r, b)
where(cond, add(r, b), r)
```

- This is eager CUDA's kernel step for step. `< 0` rather than `signbit` gives the same result
  because `r` is nonzero, and a NaN `r` yields NaN either way.
- `add(r, b)` is a plain add with no multiply, so contraction cannot touch it.
- For float16/bfloat16 the add runs in float32 and rounds once at the store, as eager's half
  add does.
- Inductor's generic `remainder` (`mod` plus a `signbit` test) is not used. Its `mod` lowers to
  Triton's inexact `frem`, and the decomposition here is already exact.

`fmodf` contains intentional `fma`s that NVRTC's `--fmad=true` leaves exact; the bead's audit
note found `loaded_rem` and `rem_*` exact at contraction on. The contract tests and the audit
route at contraction on confirm it. `FModFloorOp` is never emitted.

Speed: no fast path (an fma-based quotient for small quotients) is built up front. The context
probe shows the memory-bound kernel hiding `fmodf`'s cost. The timing group measures
small-quotient and large-quotient inputs and adds a fast path only if a gap appears.

### 3. Rounding: CubeCL ops, one folding fix, `*_to_int` as compositions

- `floor`, `ceil`, `trunc` and `round` emit the CubeCL ops. Rounding a float16/bfloat16 value in
  float32 is exact, so one rounding at the store is enough.
- **Patch change.** `RoundOp`'s `const_eval` uses `round_ties_even()`, matching the executed
  `rint`, which rounds half to even. While editing, inspect the other backends' `RoundOp`
  lowering and record whether any rounds half away from zero. Only the C++ target is used in
  production.
- **Integer operands.** Inductor's lowering clones for `round` and `trunc`. `floor`/`ceil` of an
  integer reach `CubeCLOps` and return the operand unchanged, as eager.
- **`*_to_int` forms** are the rounding op followed by the existing `to_dtype` cast, as
  Inductor's generic handler composes them. A test-only op lowered to each `*_to_int` op
  exercises them, on the `int_truediv` pattern: no aten lowering of the pinned Inductor emits
  them.

### 4. Sign ops: Inductor's and Triton's forms, bit-level where exactness needs it

- **`sign`** is Triton's form: `int8(0 < x) − int8(x < 0)`, cast to `x`'s dtype. NaN and ±0.0
  give +0.0. For integers the same form gives −1, 0 or 1.
- **`signbit`**:
  - Float: `lt(bitcast(x, int_w), 0)`, with `int_w` the signed integer of the float's width
    (int32 for float32, int16 for float16/bfloat16). The bitcast rounds a half value to its dtype
    first; the sign bit survives that rounding.
  - Signed integers: `lt(x, 0)`.
- **`copysign`**: `bitcast((bits(x) & ~sign_mask) | (bits(y) & sign_mask))` at the float's
  width. This is exactly libdevice's `copysign`, NaN operands included. Integer operands arrive
  already promoted to float32 by Inductor's lowering.

### 5. Exponent ops: musl's algorithms over same-width integers

These follow musl, which eager's c10 `nextafter` for float16/bfloat16 also cites. Every branch is
written as a `where`.

- **`nextafter(x, y)`**:
  - Either operand NaN: `x + y` (NaN).
  - `x == y`: `y`.
  - `x == 0`: the smallest subnormal with `y`'s sign.
  - Otherwise: `bits(x) ± 1` in the float's own width. Step away from zero when
    `(x < y) == (x > 0)`, toward zero otherwise.

  float16/bfloat16 use int16 bits, so the step is in their own format. The kernels keep
  subnormals because NVRTC runs without FTZ, so float32 subnormal compares and bit steps are
  exact.
- **`ldexp(x, n)`**:
  - `n` is cast to int32 (the integer cast wraps, as eager and Triton).
  - musl's `scalbnf`: if `n > 127`, multiply by 2^127 and subtract 127 (twice, clamping the
    remainder); if `n < −126`, multiply by 2^−126·2^24 and add 102 (twice, clamping). Then
    multiply once by 2^n, built as `bitcast((n + 127) << 23)`.
  - musl's split keeps the result to one rounding.
  - float16/bfloat16 compute in float32. A half value scaled into float16 range is exact in
    float32, so only the store rounds. bfloat16 results in float32's subnormal range lie below
    half of bfloat16's smallest subnormal or are exact, so double rounding cannot change them.
    The tests cover both cases.
- **`frexp(x)`**: computed on float32 bits.
  - A subnormal `x` is first scaled by 2^64 and its exponent corrected by 64 (musl `frexpf`).
  - The exponent is `((bits >> 23) & 0xff) − 126`.
  - The mantissa is `bitcast((bits & 0x807fffff) | 0x3f000000)`.
  - ±0.0, ±inf and NaN return `(x, 0)` through a `where`.
  - `CubeCLOps.frexp` returns the `(mantissa, exponent)` pair. Inductor's two pointwise nodes
    each take one element. When both land in one kernel, each recomputes the shared bits, and
    NVRTC removes the duplicate work.
  - A float16/bfloat16 mantissa computed from the float32 value is exactly representable in
    the source dtype.

### 6. Float floor division follows Inductor's lowering (Triton's result)

Inductor lowers float floor division to `floor(div_rn(a, b))` plus corrections for infinities.
Triton runs that lowering, and this backend does too once `floor` and `isinf` exist. Results
equal Triton bit for bit.

Eager floors the exact quotient, through `fmod`, so it differs when the rounded quotient crosses
an integer.

Alternative: recognize `floor(div_rn(a, b))` in `CubeCLOps` and emit eager's algorithm. Rejected:
- it pattern-matches across Inductor's lowering;
- it interacts with the lowering's own infinity corrections;
- it costs an `fmod` and extra selects per element over Triton's two ops.

The repository's policy rules out trading speed against Triton for accuracy. The divergence is
an Inductor-level property, recorded on `x3c` and in the test.

### 7. Reference data and tests

- `testing/rounding_remainder_reference.py`, on the `float_arithmetic_reference.py` pattern and
  run with the Triton venv, records two things into
  `testing/data/rounding_remainder_triton.pt`:
  - seeded inputs (as SHA-256) with Inductor+Triton outputs for every scenario that names Triton;
  - eager outputs for the cases where Triton is wrong.

  Special values are constructed explicitly. Large-quotient and subnormal inputs are checked in
  float64 in the generator.
- One test file per capability: `test_float_remainder.py`, `test_float_rounding.py` and
  `test_float_sign_exponent.py`. `isinf` goes into `test_float_extrema.py`, and integer
  `sign`/`signbit`/rounding into `test_integer_arithmetic.py`.
- The existing rejection assertions for float `fmod`/`remainder` in `test_contracts_fp.py` and
  `test_integer_arithmetic.py` become acceptance checks.
- The FP audit route gains the `rem_*` and `loaded_rem` cases at contraction on. `loaded_floor`
  stays at no contraction only: it audits `FModFloorOp`, which the backend never emits.

### 8. Timing

`testing/rounding_remainder_timing.py`, on the `integer_timing.py` pattern, times the specs'
kernels under CubeCL and Inductor+Triton:
- the RTX 3080 first, confirmed on the RTX 3090;
- clock warm-up and alternating rounds;
- results to `results/rounding-remainder-timing-{3080,3090}.json` plus a markdown summary.

Remainder is timed with small and large quotients.

## Risks / Trade-offs

- [Bit-level decompositions are long (`ldexp` ≈ 15 descriptor ops) and may add registers in
  fused kernels] → The SASS inspection in the integration group checks one `ldexp` and one
  `nextafter` kernel. The timing covers them alone. They are rare in real graphs.
- [`fmodf` is a loop for large quotients, so compute-bound fused kernels with large quotients
  could be slower than Triton's inexact `frem`] → It is timed explicitly. A gap gets the
  fma-based small-quotient fast path (decision 2), or a bead if that takes much more effort.
- [Low-precision `nextafter` relies on the bitcast rounding a float32 compute value to its
  dtype] → That is the admitted `to_dtype_bitcast` contract. Tests cover a value computed in the
  kernel (`nextafter(x * 1, y)`), not only a loaded one.
- [The `RoundOp` folding fix changes the compiler binary and every artifact key] → The manifest
  check must show no changed case, as for the 4x9.2 patch change.
- [The census may show float floor division in workloads, where eager users would see the
  Inductor divergence] → This is the same as default Inductor+Triton; the census rerun
  records it.

## Migration Plan

No migration: the ops were rejected before, so no compiled program changes behaviour. Rollback
is reverting the change's commits and the patch hunk; the manifest check guards the rest.
