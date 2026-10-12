## Executive findings

I audited the current `main` snapshot at commit **`6bba5fdf04206d9c3710dfbc46aa9850450ab405`**, committed October 11, 2026. 

I found **15 directional floating-point algebraic identities/special folds in `cubecl-ir`**, represented by 14 macro arms because the `FMulOp` zero-fold handles both operand positions in one arm. Separately, there are **11 floating expression-canonicalization equivalences** (`Commutative!`, swapped comparisons, and FMA multiplicand normalization), plus a live optimizer-level FMA-combine family in `cubecl-opt`.

The most important result is:

**`FRemOp: x % 1.0 -> x` is plainly incorrect even for ordinary finite values, and it contradicts `FRemOp`'s own constant evaluator.** `1.5 % 1.0` is folded by the regular `const_eval!` implementation as `0.5`, whereas the `simplify!` rule rewrites the same operation with a nonconstant `x` to `x`, i.e. `1.5`. This does not require assuming IEEE remainder, LLVM `frem`, C `fmod`, or any backend semantics: the inconsistency is visible entirely inside the current IR definition. `FRemOp` is at `math.rs:636–650`. 

Several other rules are not valid unless CubeCL grants floating-point assumptions that are **not encoded on the operations or rewrite**:

- `FSubOp: x - x -> +0` requires `x` not to be NaN or infinity.
- `FMulOp: 0 * x -> +0` requires at least no NaN/infinity and no significant signed zero.
- `FDivOp: 0 / x -> +0` requires a suitable nonzero/non-NaN divisor and insignificant signed zero.
- `FRemOp: 0 % x -> +0` requires a suitable nonzero/non-NaN divisor and insignificant signed zero.
- `FModFloorOp: 0 mod_floor x -> +0` requires a finite, nonzero, non-NaN divisor under its own constant-evaluation formula.
- `FAddOp`'s zero identities need a signed-zero assumption: its matcher treats `+0` and `-0` alike.

These are more than hypothetical IEEE objections. CubeCL explicitly represents `is_nan` and `is_inf`, and `FSubOp` itself deliberately distinguishes positive zero using `val.to_bits() == 0`, so the optimizer cannot obviously defend the other rules on the basis that signed zero / NaN / infinity are globally irrelevant.  

The rules that **appear safe at ordinary numeric-value semantics** are `x - +0 -> x`, multiplication by `+1`, and division by `+1`. I would still not label those bit-level unconditional until CubeCL specifies whether signaling-NaN quieting/payload changes are observable.

`FMinOp` / `FMaxOp` do have `min(x,x) -> x` / `max(x,x) -> x`. Float `x == x -> true` and `x != x -> false`, however, **do not exist**. The neighboring integer and boolean comparisons do have those self-comparison folds. That is evidence of selective NaN awareness rather than a completely mechanical integer-to-float copy.

### Complete rewrite table

The first block contains the **15 direct algebraic identities/special folds**. “New constant” means a `const_eval! custom:` arm returns a `FloatAttr`, which `ConstFoldInterface::fold_in_place` materializes; “existing SSA” means `simplify!` returns one of the operation's existing operands.   

| Operation | Current rewrite | Mechanism | Counterexample | Required assumption | Preliminary classification |
|---|---|---|---|---|---|
| `FAddOp` (`math.rs:251–265`) | `0 + x -> x` | `simplify!`; existing `rhs` SSA | `+0 + -0` normally produces `+0`; rewrite returns `-0` | Signed zero must be insignificant; arithmetic effects on sNaN must be irrelevant | **Valid only under specific floating-point assumptions** |
| `FAddOp` | `x + 0 -> x` | `simplify!`; existing `lhs` SSA | `-0 + +0` normally produces `+0`; rewrite can retain `-0` | Same | **Valid only under specific floating-point assumptions** |
| `FSubOp` (`math.rs:319–337`) | `x - x -> +0` | `const_eval! custom`; **new `+0` constant**; scalar only | `inf - inf -> NaN`; `NaN - NaN -> NaN` | `x` must be finite/non-NaN | **Valid only under specific floating-point assumptions** |
| `FSubOp` | `x - +0 -> x` | `simplify!`; existing `lhs`; specifically `to_bits() == 0` | No ordinary numeric counterexample | No observable sNaN quieting/payload/arithmetic status | **Valid only under specific floating-point assumptions**; otherwise appears safe |
| `FMulOp` (`math.rs:386–408`) | `0 * x -> +0` | `const_eval! custom`; **new `+0` constant** | `+0 * -1 -> -0`; `0 * inf -> NaN`; `0 * NaN -> NaN` | No significant signed zero; `x` finite and non-NaN | **Valid only under specific floating-point assumptions** |
| `FMulOp` | `x * 0 -> +0` | same custom arm; **new `+0`** | `-1 * +0 -> -0`; `inf * 0 -> NaN` | Same | **Valid only under specific floating-point assumptions** |
| `FMulOp` | `1 * x -> x` | `simplify!`; existing `rhs` | Only sNaN/payload-type concerns | Ordinary non-trapping value semantics | **Valid only under specific floating-point assumptions**; appears safe |
| `FMulOp` | `x * 1 -> x` | `simplify!`; existing `lhs` | Same | Same | **Valid only under specific floating-point assumptions**; appears safe |
| `FDivOp` (`math.rs:502–518`) | `0 / x -> +0` | `const_eval! custom`; **new `+0` constant** | `+0 / -1 -> -0`; `0 / 0 -> NaN`; `0 / NaN -> NaN` | Divisor nonzero/non-NaN and signed zero insignificant | **Valid only under specific floating-point assumptions** |
| `FDivOp` | `x / 1 -> x` | `simplify!`; existing `lhs` | Only sNaN/payload-type concerns | Ordinary non-trapping value semantics | **Valid only under specific floating-point assumptions**; appears safe |
| `FRemOp` (`math.rs:636–650`) | `0 % x -> +0` | `const_eval! custom`; **new `+0` constant** | `-0 % 2 -> -0`; `0 % 0 -> NaN`; `0 % NaN -> NaN` under its own ordinary `%` evaluator | Divisor nonzero/non-NaN; signed zero insignificant | **Valid only under specific floating-point assumptions** |
| `FRemOp` | **`x % 1 -> x`** | `simplify!`; existing `lhs` | **`1.5 % 1 = 0.5`, `2.5 % 1 = 0.5`, `-1.5 % 1 = -0.5`** | Would require `% 1` itself to be identity, contradicting `FRemOp`'s `lhs % rhs` constant evaluator | **Invalid even for ordinary finite values** |
| `FModFloorOp` (`math.rs:690–699`) | `0 mod_floor x -> +0` | `const_eval! custom`; **new `+0` constant** | Its all-constant formula gives NaN for `0/0`, `0/NaN`, and even `0/inf` because of `0 * inf` | Divisor finite, nonzero, non-NaN | **Valid only under specific floating-point assumptions** |
| `FMinOp` (`cmp.rs:91–100`) | `min(x,x) -> x` | `simplify!`; existing `lhs` | No ordinary numeric counterexample; possible sNaN quieting / exact NaN representation issue | Need exact ordinary-min NaN semantics to say bit-exactly | **Semantics-dependent / cannot classify yet** |
| `FMaxOp` (`cmp.rs:172–183`) | `max(x,x) -> x` | `simplify!`; existing `lhs` | Same | Same | **Semantics-dependent / cannot classify yet** |

The expression-canonicalization rules are also part of the current IR definition, although unlike `SimplifyOpsPass`, their current consumer is the value-numbering/PRE analysis stack rather than an obviously scheduled production backend pass. `Commutative!` sorts expression operands; `invert_cmp_expression!` changes the normalized operation ID as well as swapping operands.  

| Operation | Current rewrite | Mechanism | Counterexample | Required assumption | Preliminary classification |
|---|---|---|---|---|---|
| `FAddOp` | canonicalize `add(a,b)` and `add(b,a)` identically | `Commutative!` / `ExpressionCanonicalize` | Possible only for exact NaN payload/signaling behavior | NaN representation effects not operand-order-sensitive/observable | **Valid only under specific FP assumptions** |
| `FMulOp` | same, multiplication | `Commutative!` | Same caveat | Same | **Valid only under specific FP assumptions** |
| `HypotOp` | `hypot(a,b)` ≡ `hypot(b,a)` | `Commutative!` | No numeric counterexample found | Usual symmetric hypot semantics | **Unconditionally valid at numeric-value level** |
| `RhypotOp` | same | `Commutative!` | None found | Usual symmetric hypot semantics | **Unconditionally valid at numeric-value level** |
| `FMinOp` | `min(a,b)` ≡ `min(b,a)` | `Commutative!` | Depends on exact NaN/zero tie semantics | Ordinary `minnum` semantics must really be symmetric | **Semantics-dependent / cannot classify yet** |
| `FMaxOp` | analogous | `Commutative!` | Same | Same | **Semantics-dependent / cannot classify yet** |
| `FEqualOp` | equality operands sorted | `Commutative!` | None, including NaN: `a==b` and `b==a` agree | None at boolean-result level | **Unconditionally valid** |
| `FNotEqualOp` | inequality operands sorted | `Commutative!` | None | None at boolean-result level | **Unconditionally valid** |
| `FGreaterThanOp` | `a > b` canonicalizes as `b < a` | `invert_cmp_expression!` | None, including NaN for the current ordered comparisons | None | **Unconditionally valid** |
| `FGreaterThanOrEqualOp` | `a >= b` canonicalizes as `b <= a` | same | None | None | **Unconditionally valid** |
| `FmaOp` (`math.rs:764ff`) | sort only multiplicands `a,b` | custom `ExpressionCanonicalize` | Only exact NaN selection/payload issues | Multiplication operand order semantically irrelevant | **Valid at numeric-value level** |

`FmaOp`'s IR constant evaluator itself computes `a * b + c`; its expression normalizer swaps only `a` and `b`. 

There is one additional live optimizer rewrite family outside `cubecl-ir`:

| Operation | Current rewrite | Mechanism | Counterexample | Required assumption | Preliminary classification |
|---|---|---|---|---|---|
| `FMul` + `FAdd/FSub` | `a*b+c`, `a*b-c`, `c-a*b` → `FmaOp`, adding `FNegOp` where necessary | `cubecl-opt` `InstCombinePass`; creates a **new FMA op** | For f32, `(1+2^-23)*(1-2^-23)-1` can differ between separately rounded multiply/add and fused evaluation | Floating contraction must be permitted | **Semantics-dependent / cannot classify yet** |

That one genuinely needs semantic confirmation. `InstCombinePass` constructs `FmaOp` unconditionally, while the C++ backend emits `fma`/`__hfma`, and the LLVM backend lowers floating `FmaOp` to `llvm.fmuladd`. At the same time, the IR `const_eval!` spellings use ordinary `a * b + c`. GPU LLVM lowering explicitly gives add/sub/mul `CONTRACT`, but CPU does not.     

## `FAddOp`

Current source, `math.rs:251–265`:

- ordinary `const_eval!`: when both operands are constants, compute `lhs + rhs`;
- `simplify!`: if `lhs.float_as_f64() == 0.0`, return the existing `rhs`;
- `simplify!`: if `rhs.float_as_f64() == 0.0`, return the existing `lhs`;
- `Commutative!(FAddOp)` also makes operand order irrelevant to expression canonicalization. 

The matcher is numeric `Some(0.0)`, not a bit test, so it does not distinguish `+0` from `-0`. That is conspicuous beside `FSubOp`, which explicitly uses a bit test for its zero identity.

The problematic case is not NaN; it is signed zero. For example, with a constant `+0` and an unknown `x` that happens to be `-0` at runtime, `x + +0 -> x` preserves `-0`, whereas normal floating addition gives `+0` under the usual rounding mode.

So this rewrite needs at least an **NSZ-like assumption**, but no such assumption is attached to `FAddOp` or to `SimplifyOpsPass`.

## `FSubOp`

At `math.rs:319–337`, there are two very different transformations. 

`x - x -> +0` is a `const_eval! custom:` rule. It checks SSA identity, is explicitly restricted to `vector_size == 1`, and creates `float_attr(..., 0.0)`. Because it is a custom fold, **the operand need not itself be constant**.

That is incorrect unless `x` is constrained:

- `inf - inf` is NaN, not zero.
- `NaN - NaN` is NaN, not zero.

This is especially meaningful inside CubeCL because `math.is_nan` can observe the difference. 

By contrast, `x - +0 -> x` is a `simplify!` rule and uses:

`ConstantValue::Float(val) if val.to_bits() == 0`

That matches **positive zero only**. It therefore avoids the obvious signed-zero error that the neighboring `FAddOp` matcher does not.

## `FMulOp`

At `math.rs:386–408`: 

The ordinary all-constant arm computes `lhs * rhs`.

The special arm takes whichever operand is known constant with `lhs.or(rhs)`, tests it numerically for zero, and constructs a **fresh positive-zero `FloatAttr`**. This is reachable when only the zero operand is known.

Consequently:

- `+0 * -1` should have negative-zero sign but becomes `+0`;
- `-0 * +1` becomes `+0`;
- `0 * inf` becomes `+0` instead of NaN;
- `0 * NaN` becomes `+0` instead of NaN.

Importantly, **when both operands are compile-time constants**, the ordinary `lhs * rhs` arm is tried first and produces the ordinary result. The incorrect custom rule appears precisely when the other operand is unknown. That means CubeCL can assign different semantics to the same operation based purely on how much SCCP happens to know.

The two `* 1.0` rules are `simplify!` replacements and simply reuse the other SSA value.

## `FDivOp`

At `math.rs:502–518`: 

The special rule is exactly:

`known zero lhs / arbitrary rhs -> newly materialized +0`

It does **not** inspect `rhs`.

Thus the rule collapses:

- `+0 / -1` from `-0` to `+0`;
- `0 / 0` from NaN to `+0`;
- `0 / NaN` from NaN to `+0`.

Again the regular all-constant evaluator computes `lhs / rhs` first if both are known, so this is an inconsistency between complete and partial constant knowledge.

`x / 1.0 -> x` is a separate `simplify!` rule returning the original `lhs` and appears numerically sound.

## `FRemOp`

This is the clearest defect.

Current definition, `math.rs:636–650`: 

The regular evaluator establishes only this much semantically:

`constant lhs, constant rhs -> lhs % rhs`

I am **not** inferring from the name that this means LLVM `frem`, C `fmod`, IEEE remainder, etc. That is for the other semantic/backend investigation.

The first special rule is:

`known 0 lhs % arbitrary rhs -> newly materialized +0`

It does not preserve a negative-zero dividend, and it does not check a zero or NaN divisor. It therefore disagrees with the op's own ordinary constant evaluator for cases such as `-0 % 2`, `0 % 0`, and `0 % NaN`.

The second is:

`rhs == 1.0 -> return lhs`

and it is indeed a **`simplify!` rule**, not a constant-folding special case.

That produces:

```text
1.5 % 1.0  -> 1.5       // rewrite
1.5 % 1.0  -> 0.5       // FRemOp's regular const_eval
```

Likewise `2.5 -> 2.5` instead of `0.5`, and `-1.5 -> -1.5` instead of `-0.5`.

So this rule can be called incorrect **without first settling the eventual backend interpretation of `FRemOp`**. Either `const_eval!` or `simplify!` is wrong; they cannot both represent the same IR semantics.

There is also strong copy-pattern evidence immediately above it. `SRemOp` and `URemOp` have:

- normal `%` constant evaluation;
- a custom `0 % x -> 0`;
- a **correct custom `x % 1 -> 0`**;
- but then, inexplicably, the same `simplify! rhs == 1 -> lhs` rule. 

The integer mistake is normally masked in the standard pipeline because SCCP runs first and its custom `x % 1 -> 0` can fold an unknown lhs plus known-one rhs to zero. `FRemOp` lacks that corrective custom arm, so SCCP leaves it alone and the later `SimplifyOpsPass` reaches the bad `x % 1 -> x` rule.

That is unusually strong evidence of a division-style identity being carried into remainder code.

## Other arithmetic

`FModFloorOp` at `math.rs:690–699` has no `x % 1 -> ...` simplification. Its ordinary evaluator is explicitly:

`lhs - floor(lhs / rhs) * rhs`

and it has one special fold:

`known zero lhs -> freshly materialized +0`. 

That fold is consistent for ordinary finite nonzero divisors, but not universally. Using the operation's own constant-evaluation expression, zero with a zero or NaN divisor gives NaN; zero with infinity also encounters `0 * inf`, giving NaN. Therefore it needs a divisor-domain assumption.

The nearby signed integer `SModFloorOp` has an explicit `x % 1 -> 0` custom fold and **no** corresponding bad simplification. 

I found no other special FP algebraic arms in `cubecl-ir`. The other math operations use ordinary all-operands-constant `const_eval!` only. That inventory includes `FAbs`, `Exp`, `Log`, `Log1p`, `Expm1`, trigonometric/hyperbolic and inverse functions, degrees/radians, `Sqrt`, `Rsqrt`, round/floor/ceil/trunc, `Recip`, `FNeg`, `IsNan`, `IsInf`, `ArcTan2`, `Powf`, `Hypot`, `Rhypot`, and `Fma`. `ErfOp`'s evaluator is commented out; `PowiOp` says `TODO const_eval`. `Hypot` and `Rhypot` are additionally marked commutative.  

The separate `InstCombinePass` deserves attention even though it lives in `cubecl-opt`: it unconditionally converts multiply-plus/minus patterns into `FmaOp`. This is scheduled in WGSL, SPIR-V, C++, and LLVM compilation.    

## Comparisons / min / max / clamp

`FMinOp` and `FMaxOp` each have exactly one `simplify!` rule:

- `min(x,x) -> x`
- `max(x,x) -> x`

Both return the existing `lhs` SSA value.  

The backend source provides useful intended-semantic evidence: LLVM lowering deliberately maps ordinary `FMinOp` / `FMaxOp` to `llvm.minnum` / `llvm.maxnum`, with a comment saying these ignore one NaN operand like C++ `fminf` / `fmaxf` and Rust `f32::min`; the NaN-propagating operations are separate. 

I found **no** simplification for `FClampOp`; it only performs all-constant `inp.clamp(min,max)` evaluation. `FMinNanOp`, `FMaxNanOp`, and `FClampNanOp` also have constant evaluation only—no self-identity simplification and no `Commutative!`. Their IR comments explicitly specify NaN propagation and state that opposite-signed zero may yield either sign and NaN payload/sign are unspecified. 

For float comparisons:

- `FLessThanOp`: only constant evaluation.
- `FGreaterThanOp`: constant evaluation plus expression canonicalization as swapped `FLessThanOp`.
- `FLessThanOrEqualOp`: only constant evaluation.
- `FGreaterThanOrEqualOp`: canonicalizes as swapped `FLessThanOrEqualOp`.
- `FEqualOp`: ordinary constant evaluation + commutative expression canonicalization.
- `FNotEqualOp`: ordinary constant evaluation + commutative expression canonicalization.

There is **no `x == x -> true`**, **no `x != x -> false`**, no `x < x -> false`, and no `x <= x -> true` for floating values.   

The contrast with adjacent integer code is quite explicit: integer and boolean comparison ops do have those self-comparison custom folds. That omission on the float variants is a good sign: the comparison implementation was at least partially audited for NaN rather than copied verbatim.

## Macro and pass reachability

These macros are executable code generation, not declarative hints.

`simplify!` expands into an implementation of `crate::interfaces::SimplifyInterface::check_fold(...) -> Option<Value>`. The generated method gets optional constant attributes for operands and returns an SSA `Value`. 

`SimplifyInterface` itself is a real op interface. 

`SimplifyOps`:

1. matches every operation implementing that interface;
2. gathers directly available constant operands;
3. calls `check_fold`;
4. replaces the operation with the returned SSA value.

So `FRemOp x % 1 -> x`, for example, really removes the remainder operation. 

`const_eval!` expands into Pliron's `ConstFoldInterface`. Regular arms require their constant operands to exist; `custom:` arms deliberately receive `Option` attributes and can succeed with only part of the operands known. The generated `fold_in_place` calls `fold_with_materialization`, hence an returned `FloatAttr` becomes a constant SSA definition.  

This partial-operand behavior is important because SCCP explicitly maps an unknown lattice operand to `None` and still invokes `check_fold`. Therefore:

```text
known +0 * unknown x
known +0 / unknown x
known +0 % unknown x
unknown x - the same unknown x
```

can all reach their custom folds. 

After analysis, `SCCPPass` invokes `fold_in_place` to rewrite/materialize the results. 

These paths are unquestionably live: current WGSL, C++, SPIR-V, and LLVM compilation pipelines schedule both `SCCPPass` and `SimplifyOpsPass`.     

The **expression-canonicalization** machinery is different. `Commutative!` implements `ExpressionCanonicalize` by sorting expression operands. Value-numbering invokes that interface.  

At this commit, repository search finds the `ValueClassAnalysis` / `ValueNumberingAnalysis` pair instantiated in the PRE-analysis helper and tests, rather than in the normal backend pass pipelines. So those canonical equivalences are implemented and reachable through that analysis stack, but I would **not** describe them as routinely production-applied in the same sense as SCCP and `SimplifyOpsPass`.  

## Integer-to-float copy-pattern evidence

The arithmetic layout strongly suggests structural reuse:

`IAddOp` is immediately followed by `FAddOp`, with the same two zero identities. 

`ISubOp` and `FSubOp` both have a custom `x-x -> 0` and a rhs-zero identity. The float version at least refines the latter to **positive zero only**, which shows somebody was thinking about signed zero there.  

`IMulOp` and `FMulOp` share zero annihilation and one identities. The integer zero annihilation is algebraic; directly carrying it to floating point introduces the NaN/infinity/signed-zero problems. 

`SDivOp`/`UDivOp` and `FDivOp` share `0/x -> 0` and `/1 -> lhs`. The integer versions additionally have `x/x -> 1`; importantly, that one was **not** copied into floating division.  

The strongest correspondence is remainder:

```text
SRemOp:
    const_eval x % 1 -> 0
    simplify  x % 1 -> x      // internally contradictory

URemOp:
    const_eval x % 1 -> 0
    simplify  x % 1 -> x      // same

FRemOp:
    regular const_eval lhs % rhs
    simplify x % 1 -> x       // same bad simplify shape
```

 

That looks much more like a reused `/ 1 -> x` identity than a deliberately derived remainder identity.

At the same time, comparisons provide counterevidence to a blanket “all floats copied from integers” theory: the integer self-comparison folds were **not** copied onto floating comparisons. So the likely pattern is **selective structural reuse with uneven FP review**, not wholesale generation from the same template.

## Open semantic questions for the other investigations

The current source is sufficient to identify the `FRem x % 1 -> x` contradiction and the assumption-dependent partial folds, but a few questions should be deliberately left to the semantic/backend investigation:

1. **What exact runtime contract does `FRemOp` have?** The IR's constant evaluator currently says Rust-style `lhs % rhs`; this audit does not assume that proves the intended CUDA/SPIR-V/WGSL/LLVM contract. It does prove the current optimizer is internally inconsistent.

2. **Does CubeCL intend any implicit fast-math contract globally?** I found no per-operation `nnan`, `ninf`, `nsz`, reassociation, etc. attached to these `simplify!` / custom-fold rules. The LLVM backend explicitly uses `CONTRACT` only in a targeted place for GPU add/sub/mul, which argues against silently assuming every fast-math property everywhere.  

3. **Is signed zero semantically observable?** The source strongly suggests “yes, at least sometimes”: `FSubOp` tests exact positive-zero bits, and NaN-extrema documentation explicitly discusses opposite-signed zeros. A definitive semantics statement would determine whether the `FAdd`, zero-multiply/divide/remainder cases are outright bugs rather than merely missing assumptions.

4. **Are signaling NaNs / NaN payloads bit-observable?** This controls whether apparently benign `x*1`, `x/+1`, `min(x,x)`, etc. are truly unconditional or only value-equivalent.

5. **What is `FmaOp` supposed to mean: fused or algebraic multiply-add?** Its constant evaluator writes `a * b + c`, while multiple backends lower it to FMA facilities and `InstCombinePass` creates it automatically. That merits its own correctness check.

6. **What is the intended special-value contract for `FModFloorOp`?** Its constant-evaluation formula is explicit, but there is no corresponding float semantic comment like the one provided for signed integer floor modulo.

### Bottom line

The highest-confidence correctness bug is **`FRemOp: x % 1.0 -> x`**. It should be investigated first because it is wrong for normal finite inputs and disagrees directly with CubeCL's own `FRemOp` constant evaluator.

The next group worth treating as potentially unsound rather than harmless fast-math is **`FSub x-x`, `FMul 0*x`, `FDiv 0/x`, `FRem 0%x`, `FModFloor 0%x`, and the `FAdd` zero identities**. They are live, unconditional optimizer rules, but each requires assumptions that the IR does not currently encode. The internal signed-zero checks and `is_nan`/`is_inf` operations make it hard to justify those assumptions as globally implicit.