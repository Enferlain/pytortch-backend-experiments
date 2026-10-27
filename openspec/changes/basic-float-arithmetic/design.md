# Design

## Context

See proposal.md for the motivation and specs/ for the contracts.

The facts that shape the approach:
- **Front end.** Code generation (`inductor/codegen.py`) has one ops handler, `CubeCLOps`, derived
  from Inductor's `DefaultHandler`. Every op without its own method goes to `_default`, which
  accepts the names in `admission.OP_RULES` and routes them to `CubeCLKernel.operation`. That
  method types the operands by promotion and emits one descriptor op. Admission
  (`inductor/admission.py`) types the LoopBody graph from the same `OP_RULES`. The descriptor
  compiler mirrors the table as `VALUE_OPS` in `native/src/semantic.rs` and emits CubeCL IR ops by
  operand kind.
- **Inductor's generic overrides.** `square` → `mul(x, x)` and `relu` → `maximum(x, 0)` live on
  `OpOverrides`, which our handler does not inherit. Triton uses them unchanged.
- **Triton's emission** (`codegen/common.py` `pointwise_overrides_data`, `codegen/triton.py`):
  - `fma` → `tl.fma`, deliberately not `libdevice.fma`, to keep subnormals.
  - `mul_rn` → `libdevice.mul_rn`, which Triton compiles with flush-to-zero: SASS `FMUL.FTZ`, so a
    subnormal product becomes 0 (measured, group 5).
  - `div_rn` → `tl.div_rn` (IEEE) when both operands are float32, else a plain `x / y`. Loads are
    upcast to float32, so in practice always; the recorded float16/bfloat16 `addcdiv` results equal
    ours bit for bit.
  - `abs` → `tl_math.abs`.
- **Where Inductor emits them.** The CUDA lowerings emit `fma` for:
  - `add` with `alpha` (`_add_with_alpha_fma`; `sub` with `alpha` stays a multiply and a subtract,
    so it depends on contraction)
  - `addcmul`
  - `addcdiv`
  - `lerp`, through the decompositions in `decomposition.py`: scalar weight → `add(alpha)`,
    tensor weight → `addcmul` plus `abs`

  `mul_rn` and `div_rn` round the `addcmul`/`addcdiv` intermediate before the `fma`. `div_rn`
  also starts float floor division.

  `addcdiv` is CompositeImplicitAutograd, so Dynamo hands Inductor `add(s, mul(div(a, b), v))`.
  A post-grad pattern (`_fuse_addcdiv_to_fma`) rebuilds `aten.addcdiv` so the lowering above
  applies. With default settings and `value=1` it never fires: the joint-graph `remove_no_ops`
  has already deleted the `* 1`, so Triton's kernel is `s + a / b` with `div.full` (measured:
  ~1 ulp from eager in 355 of 1024 float32 rows). It looks unintended upstream; bead `4jf` tracks
  reporting it. This backend is unaffected: Stage 2 disabled `joint_graph_constant_folding`,
  because `remove_no_ops` also deletes `x + 0`, which turns −0.0 + 0 into −0.0 where eager gives
  +0.0. The `* 1` survives, the pattern fires, and `value=1` divides with `div_rn`, bitwise equal
  to eager (measured, group 4).
- **CubeCL.**
  - The IR has `FmaOp`. The C++ backend prints it as `fma(a, b, c)`, which is CUDA's
    single-rounding `fmaf` for float.
  - `FmaOp`'s constant folding is `a * b + c` in Rust, two roundings.
  - There is no rounded multiply or divide op. Plain `FMulOp`/`FDivOp` print as `*` and `/`,
    which NVRTC contracts under `--fmad=true` and approximates under `--prec-div=false` (our
    defaults, `compiler/nvrtc.py`).
  - CubeCL's `inst_combine` contraction matches `FMulOp` only.
- **CubeK** (`.stage0-sources/cubek` at `08659297`) has nothing to reuse for this family:
  - No elementwise op kernels: no `abs`/`relu`/`square`. `cubek-std`'s "unary" entry is a
    benchmark harness only.
  - No rounded multiply or divide, and no rounding controls.
  - It uses CubeCL's `fma` frontend, i.e. `FmaOp`, as a building block in `cubek-random`
    (sin/cos polynomials, Box–Muller normals) and `cubek-interpolate` (cubic and Lanczos filters).

  So emitting `FmaOp` follows CubeK's established pattern. The `FmaOp` constant-folding fix
  (decision 2) would also affect CubeK kernels with constant operands, which strengthens it as an
  upstream contribution.
- **Generated code** (SASS for sm_86, group 5: our `kernel.ptx` from NVRTC 13.0, assembled with
  `ptxas -arch=sm_86` 13.1 and read with `cuobjdump -sass`, as the driver JIT assembles it; Triton's
  cached cubins read the same way):
  - `addcmul(value=0.3)`: `FMUL p = a*b`, then `FFMA p*0.3 + s`, one `FMUL` per `FFMA` in every
    plan: the `mul_rn` product stays a separately rounded multiply, as in Triton.
  - `addcdiv(value=0.3)`: per element one IEEE division routine (`MUFU.RCP` with a Newton `FFMA`
    chain, `FCHK` sending hard operands to a called slow path), then the `FFMA` with the value: the
    same shape as Triton's `tl.div_rn`.
  - Triton's `mul_rn` is `FMUL.FTZ`: it flushes subnormal products to zero. Ours is a plain
    `FMUL`, which keeps them, as eager CUDA does: `addcmul(0, 1e-20, 1e-20, value=0.3)` gives
    3.0e-41 here and in eager, 0 in Triton (spec scenario "Subnormal product"). No recorded
    reference row has a subnormal product, so those agree.
- **Eager edge behaviour** (checked on the RTX 3080; NaN bits from the recorded reference data):
  - `abs` clears the NaN sign, but returns the canonical NaN (`0x7fffffff`) for every NaN operand.
    Triton's `abs` is `and.b32 x, 0x7fffffff`, which keeps the payload (`0xffc00000` →
    `0x7fc00000`) when nothing follows it. NVRTC compiles `fabsf` to PTX `abs.f32`, which returns
    the canonical NaN on sm_86, like eager (measured, group 3).
  - `abs(INT_MIN)` returns `INT_MIN` for every signed width.
  - `relu(-0.0)` returns −0.0, while Triton's `maximum(0, x)` returns +0.0.
  - `relu(NaN)` returns the operand unchanged, sign included. Triton's `maximum` with
    `PropagateNan.ALL` returns the canonical NaN.

## Goals / Non-Goals

**Goals:**
- Each op gets one descriptor op (or a front-end decomposition into existing ones), typed by the
  shared table on both sides, admitted on every path.
- Explicit rounding is honoured whatever the global numerics: contraction on or off,
  approximate or exact division.
- Constant folding agrees with execution for the folded ops.

**Non-Goals:**
- Changing the global division or contraction policy (4z8.17 stays).
- float64 arithmetic (dtypes family).
- `floor` and the rest of float floor division (`x3c`).
- Half-precision `fma` in storage precision. Values compute in float32, as in Triton.

## Decisions

### 1. `square` and `relu` decompose in `CubeCLOps`; the other four are descriptor ops

`square(x)` becomes `operation("mul", x, x)`. `relu(x)` becomes `operation("maximum", x,
constant(0, x.semantic_dtype))`. These are Inductor's own generic forms, which Triton uses, so the
descriptor and native layers need nothing new.

Admission still has to type `square`/`relu` in the LoopBody graph. They get `OP_RULES` rows
marked as decomposed, kept out of the native mirror check. The Python/Rust table comparison skips
names that never reach a descriptor.

The zero is created in `x`'s dtype rather than Triton's int32 constant. For int8/int16 operands
this avoids promoting the comparison to int32. Values are the same either way.

`abs`, `fma`, `mul_rn` and `div_rn` are new descriptor ops:
- `abs` because integer `abs` needs a defined form (decision 3).
- The other three because explicit rounding cannot be expressed with today's ops (decision 2).

*Alternative:* decompose `abs` as `where(x < 0, -x, x)` in Python. Rejected for floats: that
keeps −0.0's sign (−0 < 0 is false) and differs from `fabs`.

### 2. Explicit rounding through CubeCL ops, added to the patch stack

- **`fma`.** Emit CubeCL's `FmaOp`. Patch its `const_eval` for f32/f64 to `mul_add` (single
  rounding). Half types only appear in storage, never in compute.
- **`mul_rn` and `div_rn`.** Add two IR ops, `FMulRnOp` (`math.f_mul_rn`) and `FDivRnOp`
  (`math.f_div_rn`):
  - Constant folding: Rust's IEEE `*` and `/`, which are round-to-nearest.
  - Simplification: only the identities that hold bit for bit (`x / 1`).
  - C++ backend: printed as `__fmul_rn(a, b)` / `__fdiv_rn(a, b)`.
  - Scalarized over vectors (`unrolling!`), as `FmaOp` is, because vector plans may vectorize
    these kernels.
  - LLVM backend (the optional probe route): plain `fmul`/`fdiv` with no fast-math flags, which
    are IEEE there.
- `inst_combine` sees neither op as a multiply, so it never contracts them.
- The patch file, the README's list of patched files and the pinned SHA change together. The pin
  stays at `907676dc`.

*Alternatives considered:*
- **`mul_rn(a, b) = fma(a, b, -0.0)`, with no patch.** Exact in principle. But LLVM folds `fma(x,
  y, -0.0)` back into `fmul`, which the NVPTX backend may then contract with a following add under
  `--fmad=true`. Not reliable.
- **Per-kernel numerics flags.** For example `--prec-div=true` for any kernel containing `div_rn`,
  as the two-pass Welford kernel does, or `--fmad=false` for `mul_rn`. Rejected: this changes
  every other division or multiply in the kernel away from Triton's numerics. That is slower for
  division, and for contraction it changes results Triton contracts.

### 3. Integer and float `abs` emission

- **Float:** `FAbsOp` (`fabs`).
- **Signed integer:** `select(x < 0, 0 - x, x)`, with the wrapping `ISubOp` that `neg` already
  uses. C++ `abs` on `int`/`long long` is undefined at the minimum, which is why CubeCL's `SAbsOp`
  isn't used.
- **Unsigned:** the operand unchanged.
- **bool:** outside the domain, as in eager.

### 4. Type rules

`OP_RULES` and `VALUE_OPS` gain these rows:

| Op | Operands | Result |
|---|---|---|
| `abs` | numbers (floats and integers except bool) | same |
| `square` (decomposed) | numbers | same |
| `relu` (decomposed) | numbers | same |
| `fma` | floats, three operands | same |
| `mul_rn` | floats, two operands | same |
| `div_rn` | floats, two operands | same |

`fma`'s arity 3 is new in `VALUE_OPS`; the Rust validator checks arity per row already.
`operation` promotes all operands to one semantic dtype, as Inductor's type propagation does for
these ops.

### 5. Tests and reference data

- **Triton reference.** A generator, `testing/float_arithmetic_reference.py`, run in the Triton
  venv on the `integer_reference.py` pattern. It stores input hashes and Triton outputs in
  `testing/data/float_arithmetic_triton.pt` for every spec scenario that names Triton. The
  contract tests (`testing/test_float_arithmetic.py`) assert bitwise equality with that data and
  with eager where the spec says so.
- **Inputs that expose rounding.** The single-rounding and rounded-product scenarios need
  inputs where rounding is visible: `a * s + c` with `a * s` inexact, cancelling against `c`. The
  generator constructs them from seeded float32 values and checks each row in float64, where
  the sums are exact. Only float32 can have such rows: 16-bit products are exact in float32,
  and rounding the result to 16 bits hides a float32 rounding step.
- **Constant folding.** `fma`'s folding is tested in the FP-rewrite suite (`test_contracts_fp.py`
  on the native crate): a descriptor with three constant operands must give the same bits as one
  with runtime operands.

### 6. Timing

On the `integer_timing.py` pattern, in a new `testing/float_arithmetic_timing.py`: the Adam-style
update, an `addcdiv` kernel and an `abs`/`relu` kernel over 4M float32 elements. Alternating
CubeCL and Triton rounds, with clock warm-up, on the 3080 and confirmed on the 3090.

`div_rn` costs an IEEE division routine, like Triton's `tl.div_rn`. The SASS check confirms one
routine per element.

## Risks / Trade-offs

- **[Patch stack grows by two ops]** → They are small and self-contained, and good upstreaming
  candidates (with the `FmaOp` folding fix) to shrink later rebases. Note them for the next
  upgrade.
- **[Rebuilt compiler changes every artifact key]** → Expected. The manifest check must still pass
  with identical IR/CUDA/PTX for existing kernels (spec "Existing kernels keep their generated
  code"). Caches rebuild once.
- **[`relu(-0.0)` sign differs from eager]** → Accepted. Measured on both GPUs (RTX 3080 and
  3090, 2026-10-26):

  | Implementation | Ops | Result for −0.0 |
  |---|---|---|
  | Eager CUDA | `relu`, `clamp_min(0)`, `clamp(0, 6)`, `hardtanh`, `relu6` | −0.0 |
  | Eager CUDA | `torch.maximum(x, 0)`, `threshold(x, 0, 0)` | +0.0 |
  | Inductor+Triton | all seven | +0.0 |

  - Eager's −0.0 is an artifact of how `clamp` is written: it returns `x` unless `x < 0`. It is
    not a `relu` rule, and eager is not consistent across these ops.
  - Inductor lowers all of them to `maximum`/`minimum`. Following it keeps `relu` consistent with
    the clamp family and with the `float-extrema` ordered-zero contract, and bitwise equal to
    Triton.
  - The values compare equal (`==`, `allclose`, `torch.equal`). Only sign-sensitive consumers
    (`1/x`, `copysign`, `atan2`, `signbit`) can tell them apart. `relu`'s backward uses `x <= 0`
    and is unaffected.
  - Eager-bitwise signed zero would be a separate decision covering the whole `maximum`/`clamp`
    family.
- **[NaN payloads differ from Triton or eager]** → Accepted. `abs` and `relu` leave the payload
  open, as `float-extrema` does for `maximum`; the sign of an `abs` NaN is cleared. Our `abs` is
  `abs.f32` (canonical NaN, as eager), Triton's a bit mask (payload kept when stored directly).
  Matching either would cost an instruction per element in some contexts (`abs.f32` folds into a
  following FP instruction as a free operand modifier; a mask does not) for bits no PyTorch
  contract covers.
- **[Subnormal products differ from Triton]** → Ours keeps them, as eager; Triton's `mul_rn`
  flushes them. No cost: both are one `FMUL`.
- **[`addcdiv(value=1)` differs from Triton]** → Ours is eager-exact where Triton's is 1 ulp off in
  355 of 1024 recorded float32 rows (see Context). Measured at 4M elements (timing results): 96.94
  against 96.79 µs (+0.2%) on the RTX 3080, 80.42 against 79.86 µs (+0.7%) on the RTX 3090. About
  0.1 µs is the IEEE division itself (Triton's own `div_rn` kernel, value 0.3, takes 96.91 µs on the
  3080); the rest is a general difference also seen with value 0.3 (+0.1% on the 3080). Kept (decided with the user, 2026-10-27): Triton's result
  comes from a pass-ordering accident, not a numerical choice, and ours costs only the division.
- **[Adam step slower than Triton]** → Triton fuses lerp, addcmul and addcdiv into one kernel
  with three outputs; this backend emits one kernel per output (the single-output fusion rule),
  so the inputs are read up to three times: 248 µs against 197 µs at 4M elements on the RTX 3080.
  Each of our kernels matches Triton running the same update alone. Multi-output fusion is bead
  `4z8.21`, widened to pointwise kernels with this measurement (decided with the user).
- **[NVRTC treats `__fmul_rn`/`__fdiv_rn` differently than expected]** → The SASS check in the
  timing task verifies `FMUL.RN`-equivalent emission without fusion, and an IEEE division routine.
  The contraction-sensitive scenarios fail if it is wrong.
- **[Eager equality for `lerp`/`addcmul` depends on Inductor's lowering matching eager]** →
  Triton's bitwise equality is the floor, and eager equality is asserted where Inductor intends it.
  If eager differs on some input, record the case and compare with Triton. This would be an
  Inductor property, not ours.

## Migration Plan

- Rebuild the native pieces with the extended patch (`native/build.sh`).
- Run the full suite and manifest check.

Rollback is reverting the commit and the patch file together; caches keyed on the old binary
digest become valid again.
