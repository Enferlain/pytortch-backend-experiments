# cubecl audit condensed findings

## CubeCL Floating-Point IR Rewrite Audit

[chatgpt url](https://chatgpt.com/share/6acc89dd-ec90-83eb-8fe7-db25353395a0)

### 1. Rewrite classification

| Operation | Exact current rewrite | Classification | Shortest counterexample | Required assumption / condition |
|---|---|---|---|---|
| `FAddOp` | `0.0 + x -> x` | `requires assumptions` | `+0 + -0` | Signed zero insignificant (`nsz`-like) |
| `FAddOp` | `x + 0.0 -> x` | `requires assumptions` | `-0 + +0` | Signed zero insignificant |
| `FSubOp` | `x - x -> +0.0` | `requires assumptions` | `inf - inf -> NaN` | `x` finite/non-NaN |
| `FSubOp` | `x - +0.0 -> x` | `safe`* | — | *At ordinary value semantics; exact sNaN/payload observability unresolved |
| `FMulOp` | `0.0 * x -> +0.0` | `requires assumptions` | `0 * inf -> NaN` | No NaN/Inf; signed zero insignificant |
| `FMulOp` | `x * 0.0 -> +0.0` | `requires assumptions` | `inf * 0 -> NaN` | Same |
| `FMulOp` | `1.0 * x -> x` | `safe`* | — | *Ignoring sNaN/payload effects |
| `FMulOp` | `x * 1.0 -> x` | `safe`* | — | Same |
| `FDivOp` | `0.0 / x -> +0.0` | `requires assumptions` | `0 / 0 -> NaN` | Divisor nonzero/non-NaN; signed zero insignificant |
| `FDivOp` | `x / 1.0 -> x` | `safe`* | — | *Ignoring sNaN/payload effects |
| `FRemOp` | `0.0 % x -> +0.0` | `requires assumptions` | `0 % 0 -> NaN` | Divisor nonzero/non-NaN; signed zero insignificant |
| `FRemOp` | **`x % 1.0 -> x`** | **`plain bug`** | **`1.5 % 1.0 = 0.5`, rewrite gives `1.5`** | None can rescue it while preserving CubeCL's own current `%` constant evaluator |
| `FModFloorOp` | `0.0 mod_floor x -> +0.0` | `requires assumptions` | `0 mod_floor 0 -> NaN` under its evaluator | Finite, nonzero, non-NaN divisor |
| `FMinOp` | `min(x, x) -> x` | `unclear` | — | Depends on exact NaN/signed-zero/payload contract |
| `FMaxOp` | `max(x, x) -> x` | `unclear` | — | Same |
| `FMulOp` + `FAdd/FSub` | `a*b+c`, `a*b-c`, `c-a*b` → `FmaOp` | `requires assumptions` | Separate multiply/add can round differently from fused FMA | FP contraction explicitly permitted |

### 2. Confirmed plain bugs

- **`FRemOp: x % 1.0 -> x`**
  - Wrong for ordinary finite values.
  - Minimal example: `1.5 % 1.0` evaluates to `0.5`, but the rewrite returns `1.5`.
  - Also wrong for `2.5 % 1.0`, `-1.5 % 1.0`, etc.
  - It directly conflicts with `FRemOp`'s own ordinary constant evaluator, which computes `lhs % rhs`.

There are also analogous internally contradictory integer `SRemOp` / `URemOp` `simplify!` rules returning `lhs` for `% 1`, although their earlier custom constant folds produce `0`.

### 3. Reachability

**Yes. The suspicious rules are actually reachable.**

- `simplify!` generates `SimplifyInterface::check_fold`.
- `SimplifyOpsPass` calls that interface and replaces the operation with the returned SSA value.
- `const_eval!` generates `ConstFoldInterface`.
- SCCP invokes that interface and can supply `None` for unknown operands, so custom partial folds such as `0*x`, `0/x`, and `0%x` can fire even when the other operand is runtime-unknown.
- Current WGSL, SPIR-V, C++, and LLVM pipelines schedule both SCCP and `SimplifyOpsPass`.
- `InstCombinePass` is also scheduled and therefore the FMA contraction rewrite is live.

The separate `ExpressionCanonicalize`/value-numbering equivalences exist, but were **not shown to be routinely scheduled in the main backend pipelines**.

### 4. Additional suspicious FP rewrites

- `FModFloorOp: 0 mod_floor x -> +0`
- `FMinOp: min(x,x) -> x`
- `FMaxOp: max(x,x) -> x`
- Optimizer contraction:
  - `a*b+c -> FmaOp`
  - `a*b-c -> FmaOp(a,b,-c)`
  - `c-a*b -> FmaOp(-a,b,c)`
- FP expression canonicalization also treats these as equivalent:
  - `FAdd(a,b)` / `FAdd(b,a)`
  - `FMul(a,b)` / `FMul(b,a)`
  - `FMin(a,b)` / `FMin(b,a)`
  - `FMax(a,b)` / `FMax(b,a)`
  - `a > b` / `b < a`
  - `a >= b` / `b <= a`

Notably, there is **no** float `x == x -> true` or `x != x -> false` fold.

### 5. Most important source locations

- `crates/cubecl-ir/src/dialect/math.rs`
  - `FAddOp`, `FSubOp`, `FMulOp`, `FDivOp`: roughly lines 251–518.
  - `SRemOp`, `URemOp`, `FRemOp`, `FModFloorOp`: roughly lines 550–700.  
- `crates/cubecl-ir/src/dialect/cmp.rs`
  - `FMinOp`, `FMaxOp`, float comparisons, clamp and NaN-propagating extrema.  
- `crates/cubecl-macros-internal/src/generate/simplify.rs`
  - `simplify!` → `SimplifyInterface::check_fold`. 
- `crates/cubecl-macros-internal/src/generate/const_eval.rs`
  - `const_eval!` → `ConstFoldInterface`, including partial custom folds/materialization. 
- `crates/cubecl-ir/src/rewrite.rs` and `crates/cubecl-opt/src/passes/sccp.rs`
  - Live invocation of `SimplifyInterface` and `ConstFoldInterface`.

## CubeCL Floating-Point Semantics and FastMath Audit
[chatgpt url](https://chatgpt.com/share/6acc89ea-f324-83eb-8862-a6a9cda4db97)

```yaml
"Default CubeCL floating-point model": "Safe/no-FastMath by intent; IEEE-like for ordinary special-value constraints, but only partially specified and backend-dependent in precision and several edge details. It is not strict IEEE-754 and not implicitly FastMath."

"Are NaN/Inf/signed-zero semantics relaxed by default?": partially

"Can current IR simplification safely assume fast-math?": no

"Does the architecture support gating algebraic rewrites on FastMath today?": partially

"Most important semantic evidence":
  - "Current CubeCL book: FastMath explicitly loosens normal restrictions around Inf, NaN, signed zero and precision."
  - "Current FastMath API: NotNaN and NotInf are UB-producing assumptions; UnsignedZero and the algebraic flags are explicit permissions."
  - "PR #432 / commit 2cc42af: 'The default is the existing safe one.'"
  - "PR #1000 / commit 20fa77d: FastMath intentionally became per-function and historically propagated per instruction."
  - "PR #1402 / commit 12c4976: Pliron migration removed the old per-instruction mode and backend consumers."
  - "PR #1553: explicitly documents that current fp_math_mode is restored after expansion and not recorded on operations."
  - "PR #1732 / commit 08c2b16: adds NaN-propagating extrema and explicitly removes LLVM NSZ, showing special values remain semantically significant."
```

1. **CubeCL’s default floating-point semantic model**

CubeCL is **safe/no-FastMath by default**, with ordinary code retaining restrictions around NaN, infinities, signed zero, and operation ordering unless explicitly relaxed. It is best described as **IEEE-like in these respects, but not strict IEEE-754**, because CubeCL does not fully specify rounding, subnormal/FTZ behavior, signaling NaNs, FP exceptions, or NaN payload preservation. `FastMath` is an explicit opt-in mechanism for weakening those constraints. Backend behavior can differ, but that does not redefine CubeCL’s intended default contract.

2. **Direct answers**

- **Are NaN assumptions opt-in?** Yes. `NotNaN` explicitly assumes NaNs do not occur.
- **Are infinity assumptions opt-in?** Yes. `NotInf` explicitly assumes ±Inf do not occur.
- **Is ignoring signed zero opt-in?** Yes. `UnsignedZero` explicitly permits ignoring the sign of zero.
- **Is reassociation/unsafe algebra opt-in?** Yes. `AllowReassociation` and `AllowTransform` explicitly permit those transformations.
- **Is default CubeCL implicitly fast-math?** No.

3. **Relevant FastMath flags**

| Flag | What it permits |
|---|---|
| `NotNaN` | Assume operands/results are never NaN; violating that assumption is documented as undefined behavior. |
| `NotInf` | Assume operands/results are never ±Inf; violating that assumption is documented as undefined behavior. |
| `UnsignedZero` | Ignore the distinction between `+0` and `-0`. |
| `AllowReassociation` | Reassociate floating-point expressions even if evaluation order/precision changes. |
| `AllowTransform` | Permit broad mathematical/algebraic transformations, including reassociation/contraction, even if precision changes. |

`AllowTransform` does **not** document an implication that `NotNaN`, `NotInf`, or `UnsignedZero` are automatically true; those remain separate assumptions.

4. **Can current IR simplification use those assumptions?**

**No.**

Current frontend expansion temporarily sets `fp_math_mode`, but after the Pliron migration that mode is **not attached to individual IR operations**. The simplification pass therefore has no applicable per-operation FastMath state to inspect when deciding whether an algebraic rewrite is legal.

5. **Assumptions required for the questioned rewrites**

| Rewrite | Required assumptions |
|---|---|
| `x + 0 -> x` | `UnsignedZero` |
| `x - x -> 0` | `NotNaN` + `NotInf`; CubeCL’s globally undocumented rounding/zero-sign details prevent a stronger unconditional claim |
| `0 * x -> 0` | `NotNaN` + `NotInf` + `UnsignedZero` |
| `0 / x -> 0` | `NotNaN` + `UnsignedZero` **plus independent proof that `x != 0`** |
| `0 % x -> 0` | `NotNaN` **plus independent proof that `x != 0`**; if the rewrite fails to preserve the dividend’s zero sign, `UnsignedZero` is additionally needed |

The important distinction is that FastMath cannot express all necessary value facts. In particular, there is **no FastMath flag meaning “denominator is nonzero.”**

6. **Strongest CubeCL-native evidence**

1. **Current CubeCL book, `math_optimizations.md`:** describes FastMath as *loosening* normal floating-point restrictions, specifically mentioning `Inf`/`NaN`, signed zero, and precision, and describes unmarked code as full-precision relative to the marked section.
2. **Current `FastMath` API in `cubecl-ir/src/properties.rs`:** explicitly defines `NotNaN`, `NotInf`, `UnsignedZero`, reassociation, transform, reciprocal, contraction, and reduced-precision permissions; `NotNaN`/`NotInf` explicitly make violations undefined behavior.
3. **PR #432 / commit `2cc42af...`:** when FastMath was introduced, its description explicitly said **“The default is the existing safe one”** and characterized the added modes as unsafe optimizations.
4. **PR #1000 / commit `20fa77d...`:** deliberately changed FastMath from per-kernel to **per-function**, so relaxed assumptions apply only to selected performance-critical regions rather than implicitly to the whole kernel.
5. **PR #1732 / commit `08c2b16...`:** recent work added explicitly NaN-propagating extrema and specifically removed LLVM `NSZ`, strong evidence that NaN and signed-zero behavior remain semantically meaningful when not relaxed.

7. **Undocumented / ambiguous areas**

CubeCL does not currently give a complete global contract for **subnormals/FTZ, rounding mode, signaling NaNs, floating-point exception flags/traps, or NaN payload/sign preservation**. Some operations define these edge cases locally, but they cannot safely be generalized to all CubeCL floating-point operations.

There is also an implementation/contract mismatch around **contraction**: current LLVM GPU lowering enables contraction for some arithmetic even though CubeCL exposes `AllowContraction` as an explicit FastMath permission. That prevents describing default CubeCL as fully strict or fully specified, but it does **not** justify assuming the broader `NotNaN`/`NotInf`/`UnsignedZero`/reassociation permissions by default.

## CubeCL Floating-Point Rewrite History and Backend Audit
[chatgpt url](https://chatgpt.com/share/6acc8a0c-ac84-83eb-a824-0b3b5f9c42a7)

```text
Were the questionable float rewrites explicitly justified by maintainers?
no

Do they appear inherited from integer algebra?
strong evidence

Did the Pliron rewrite introduce the semantics or mostly preserve them?
Mostly preserve them. The zero/one algebraic folds largely predate Pliron in
a type-generic simplifier. Pliron did introduce FSub x-x -> 0 and introduced
the x % 1 -> x behavior; the latter is strongly indicated to be a refactor
regression because it simultaneously broke SRem and URem.

Does backend lowering independently assume away NaN/Inf/signed zero?
No, not for the affected operations in the lowering inspected. LLVM
FAdd/FSub/FMul may carry only `contract`; FDiv/FRem use no fast-math flags.
No per-op nnan/ninf/nsz/reassociation/reciprocal assumption was found in
the SPIR-V, C++/CUDA/HIP/Metal-shared, or WGSL paths.

Most important provenance evidence:
The pre-Pliron optimizer applied zero/one identities through one generic
Arithmetic simplifier with no float guard; pre-Pliron Rem was explicitly
Rust Rem::rem and separate from ModFloor; the old x % 1 rule returned zero;
and #1402 changed that rule to return lhs for signed integer, unsigned
integer, and floating remainder simultaneously. Current SRem/URem constant
evaluation still returns zero for % 1, making the Pliron `% 1 -> x` rule an
especially clear migration regression rather than evidence of intentional
relaxed floating-point semantics.
```

1. **Questionable rewrites**

| Rewrite | Introduced / historical origin | Explicit float-specific justification? | Inherited from shared integer/float algebra? |
|---|---|---|---|
| `x + 0 -> x` | Pre-Pliron optimizer, PR **#157**, merged as `c40c0464aab2707487143b2e8d3cf71917c2f841` (2024-10-08). Implemented in the generic arithmetic simplifier before integer and float ops were split. | **No.** No NaN/Inf/signed-zero or relaxed-FP rationale was found. | **Yes, strong evidence.** It originated in a type-generic arithmetic rule. |
| `0 + x -> x` | Same pre-Pliron generic optimizer in PR **#157**. | **No.** | **Yes, strong evidence.** |
| `x - x -> 0` | Added during the Pliron conversion, PR **#1402**, commit `12c497629d4b94baaae0feb172c4c24405c2c8f6` (2026-08-11), when subtraction became separate `ISubOp`/`FSubOp`. | **No.** | **Some evidence.** It appears to have been copied symmetrically into integer and float subtraction during the split, but unlike the older zero rules it did not predate Pliron. |
| `0 * x -> 0` | Pre-Pliron optimizer, PR **#157**, generic `Mul` simplification. | **No.** | **Yes, strong evidence.** |
| `x * 0 -> 0` | Same PR **#157**, generic `Mul`. | **No.** | **Yes, strong evidence.** |
| `0 / x -> 0` | Pre-Pliron optimizer, PR **#157**, generic `Div`. | **No.** | **Yes, strong evidence.** |
| `0 % x -> 0` | Added to the old generic modulo/remainder simplifier in PR **#169**, merge `6ed4657b6da956c75922e09622710c22cd1a0739` (2024-10-17). PR **#1322**, commit `cb87b0d2ba4a14234d751a536dabb0540646e190` (2026-05-15), later extended the shared simplifier to explicit `Rem` as well as floor-mod. | **No.** | **Yes, strong evidence.** |
| `x % 1 -> x` | Introduced specifically by the Pliron conversion, PR **#1402**, `12c497…` (2026-08-11). Before Pliron, the corresponding generic rule was **`x % 1 -> 0`**. The Pliron port changed it to `lhs` for **`SRemOp`, `URemOp`, and `FRemOp` simultaneously**. | **No.** | **Not merely inherited; stronger evidence of a refactor regression.** The old shared rule was correct-looking `-> 0`; the Pliron port changed all integer and float variants to `-> x`. |

2. **Material timeline**

- **2024-10-08 — PR #157, `c40c0464…`, “IR Optimizer and changes to SPIR-V compiler”.** Established the early type-generic arithmetic simplifier. The zero/one identities for add, multiply and divide were written against shared arithmetic ops, not float-specific ops.

- **2024-10-15 — `7677ba32aeee5c607c469ccd856751dc6e1a7644`, “Swap mod and remainder to make it consistent with other backends”.** Clarified the backend distinction between truncating remainder and modulo: signed/float remainder maps to `SRem`/`FRem`, while floor-style modulo maps to `SMod`/`FMod`.

- **2024-10-17 — PR #169, `6ed4657…`.** Added generic remainder/modulo identities including `0 % x -> 0` and **`x % 1 -> 0`**.

- **2026-05-15 — PR #1322, `cb87b0d2…`.** Arithmetic refactor made Rust `%` explicit as `Arithmetic::Rem` and floor modulo explicit as `ModFloor`; the old generic simplifier was extended to cover `Rem`. This preserved the old shared integer/float algebra.

- **2026-08-11 — PR #1402, `12c497629d4b94baaae0feb172c4c24405c2c8f6`, “Make everything Pliron”.** Split generic arithmetic into integer and float operations. Most old identities were copied into both families. It also added `FSub x,x -> 0` and changed the old remainder rule from `x % 1 -> 0` to **`x % 1 -> x`** for signed integer, unsigned integer, and float remainder.

- **2026-09-09 — PR #1626, `2f81491f…`.** Fixed structural problems with self-operand folds such as subtraction on vector results, but retained scalar `FSub x,x -> 0`.

- **September 2026 — PR #1701.** Zero-vector normalization behavior was explicitly discussed in terms of kernels opting into **FastMath**, showing later code/review treated relaxed special-value behavior as something associated with explicit fast-math assumptions.

- **September 2026 — PR #1732, `08c2b16e…`.** Fixed signed remainder/floor-modulo semantics and added NaN-propagating extrema, showing later maintainers explicitly cared about special-value and sign semantics.

3. **Strongest evidence for/against the theory**

> “These float rewrites were accidentally inherited from integer algebra during the Pliron rewrite.”

The theory is **substantially right but historically imprecise**.

The strongest evidence **for** accidental integer-style inheritance is that most questionable rules did **not originate in the Pliron rewrite at all**. They already existed in a **single pre-Pliron, type-generic arithmetic simplifier** that applied equally to integer and floating-point operations. When Pliron later split arithmetic into `I*Op` and `F*Op` classes, those rules were largely copied over unchanged.

The strongest evidence specifically implicating the **Pliron migration** is `x % 1 -> x`: immediately before Pliron the shared rule was `x % 1 -> 0`; #1402 changed it to `-> lhs` for `SRemOp`, `URemOp`, and `FRemOp` together. Current integer constant evaluation still produces zero for `% 1`. That is unusually strong evidence of a mechanical porting error rather than a floating-point policy decision.

`x - x -> 0` also first appears during the integer/float split and is installed in both `ISubOp` and `FSubOp` without FP-specific reasoning. This is suggestive of mechanical parity, although less conclusive than the remainder error.

Evidence **against** the idea that the entire problem was created by Pliron is therefore substantial: the zero folds were already present years earlier in generic arithmetic.

4. **Actual meaning/lowering of `FRemOp`**

- **Frontend meaning:** floating-point Rust `%`; historically documented as **Rust `Rem::rem`** semantics, distinct from CubeCL's floor modulo operation.
- **LLVM:** LLVM **`frem`**.
- **SPIR-V:** **`OpFRem`**, not `OpFMod`.
- **CUDA/HIP/C++ shared backend:** **`fmod(lhs, rhs)`**.
- **WGSL:** floating-point **`%`**.
- **Metal:** uses the shared C++-style path for this operation rather than a distinct floor-mod lowering.

`FModFloorOp` is separate:

- SPIR-V uses **`OpFMod`**.
- Other backends synthesize floor modulo with the equivalent of `x - y * floor(x / y)`.

The distinction between `Rem` and floor-modulo predates Pliron and was explicitly represented in CubeCL before the current IR split.

5. **Compatibility of `x % 1.0 -> x` with `FRemOp`**

**No.**

Given CubeCL's actual `FRemOp` meaning—Rust/C `fmod`-style truncating remainder, LLVM `frem`, SPIR-V `OpFRem`—`x % 1.0` is a remainder operation, not the identity function. The historical CubeCL simplifier itself previously used `x % 1 -> 0`.

The fact that #1402 simultaneously changed **integer** `SRemOp` and `URemOp` to the same erroneous `-> lhs` rule is particularly strong evidence that the current `FRemOp` rule is a migration bug, not an alternate definition of floating remainder.

6. **Backend fast-math assumptions**

No inspected lowering independently applies assumptions equivalent to **`nnan`**, **`ninf`**, or **`nsz`** for these operations.

- `FAdd`, `FSub`, `FMul` on LLVM may receive only **`contract`**, where the target permits FMA contraction.
- `FDiv` uses an empty/no-fast-math flag set.
- `FRem` uses an empty/no-fast-math flag set.
- No corresponding per-operation `nnan`, `ninf`, or `nsz` decoration was found in the SPIR-V, CUDA/HIP/C++, WGSL, or shared Metal lowering.

So the questionable IR simplifications are **stronger than assumptions independently introduced by these backend lowerings**.

7. **Strongest supporting sources**

- **PR #157 / merge `c40c0464aab2707487143b2e8d3cf71917c2f841`** — establishes that the original zero/one arithmetic identities lived in a **type-generic pre-Pliron simplifier**, with no float-specific justification.

- **PR #169 / merge `6ed4657b6da956c75922e09622710c22cd1a0739`** — establishes the historical remainder identities, especially the pre-Pliron **`x % 1 -> 0`** rule.

- **PR #1402 / `12c497629d4b94baaae0feb172c4c24405c2c8f6`, “Make everything Pliron”** — shows the integer/float op split, addition of `FSub x-x -> 0`, and the regression changing `% 1` to `-> lhs` in `SRemOp`, `URemOp`, and `FRemOp`.

- **`7677ba32aeee5c607c469ccd856751dc6e1a7644`, “Swap mod and remainder to make it consistent with other backends”**, plus the old arithmetic documentation — establishes that CubeCL intentionally distinguishes truncating remainder (`Rem`) from floor modulo (`ModFloor`).

- **Current backend lowering files for LLVM/SPIR-V/C++/WGSL** — establish `FRemOp -> frem / OpFRem / fmod / %` and show that `FDiv`/`FRem` do not receive `nnan`, `ninf`, or `nsz`-equivalent fast-math flags.

Provenance verdict: likely accidental

Most questionable float folds descend from an old type-generic integer/float simplifier without FP-specific justification, while the especially clear `x % 1 -> x` rule was introduced as a Pliron migration regression across integer and float remainder alike.
