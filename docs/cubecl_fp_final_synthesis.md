# CubeCL floating-point simplifications: final synthesis

## Executive verdict

**Yes: the questioned generic simplifications are inconsistent with CubeCL’s intended default semantics.**
This conclusion follows from CubeCL’s explicit opt-in FastMath design, current documentation and operation definitions, rather than an assumed strict IEEE-754 contract.
The strongest ordinary-value bug is `FRemOp: x % 1.0 -> x`: CubeCL defines this operation as truncating remainder, so `1.5 % 1.0` is `0.5`, not `1.5`.
The other questioned folds discard special-value or signed-zero behavior without applicable assumptions.
Most zero identities predate Pliron; the migration added the self-subtraction fold and changed the old remainder-by-one rule incorrectly across integer and float operations.
**The old float `x % 1 -> 0` rule was also incorrect and must not be restored.**
Current frontend FastMath state is not recorded on these operations, so immediate assumption-based gating is unavailable.
A narrow correctness patch should remove the offending float rules and the contradictory integer remainder simplifiers while retaining ordinary constant evaluation and sign-safe existing identities.

**Verification scope:** source and exact-history checks at [main `6bba5fdf04206d9c3710dfbc46aa9850450ab405`][head], retrieved 2026-10-12; no repository source was modified and no CubeCL build/device execution was performed. Backend statements describe inspected lowering, not a complete driver-level guarantee.

## Final rewrite table

“Required assumptions” lists sufficient restrictions for a legitimate restricted-domain rule, **not assumptions that current IR actually carries**. `NotNaN`, `NotInf` and `UnsignedZero` are existing CubeCL flags; nonzero-divisor and sign/range facts are separate value proofs. Verdicts concern the current unconditional rule.

| Operation | Current rewrite | Final verdict | Why | Required assumptions | Recommended action |
|---|---|---|---|---|---|
| FAdd | `0 + x -> x` | invalid under CubeCL default semantics | Matching `+0` can turn `+0 + -0` into `-0`; ignoring zero sign is explicitly opt-in. | `UnsignedZero`, or a sign-safe domain and applicable rounding semantics. | Remove current zero simplifier. |
| FAdd | `x + 0 -> x` | invalid under CubeCL default semantics | Same problem for `-0 + +0`. | `UnsignedZero`, or separate sign/rounding proof. | Remove current zero simplifier. |
| FSub | `x - x -> +0` | invalid under CubeCL default semantics | Runtime `x = Inf` or NaN must not become zero. Current custom fold is scalar-only. | `NotNaN + NotInf`, or proof that x is finite; producing specifically +0 also needs an agreed cancellation-zero rule, otherwise `UnsignedZero`. | Remove custom self-operand fold. |
| FMul | `0 * x -> +0` | invalid under CubeCL default semantics | Discards NaN/Inf cases and the product’s zero sign. | `NotNaN + NotInf + UnsignedZero`, or equivalent operand/sign proofs. | Remove custom partial fold. |
| FMul | `x * 0 -> +0` | invalid under CubeCL default semantics | Same in reverse operand order. | Same. | Remove custom partial fold. |
| FDiv | `0 / x -> +0` | invalid under CubeCL default semantics | `0/0` and `0/NaN` are not zero; `+0/-1` has negative zero sign. | `NotNaN + UnsignedZero` **and proof `x != 0`**. Infinite divisor does not itself defeat ordinary zero division. | Remove custom partial fold. |
| FRem | `0 % x -> +0` | invalid under CubeCL default semantics | Discards zero-divisor/NaN cases and changes `-0 % 2` to +0. | `NotNaN + UnsignedZero` **and proof `x != 0`**; require a finite divisor as well for a conservative cross-backend domain. | Remove custom partial fold. |
| FRem | `x % 1 -> x` | plain correctness bug independent of special values | CubeCL’s own evaluator and operation meaning give `1.5 % 1 = 0.5`. | No FastMath flag licenses treating remainder as identity for unrestricted ordinary x. A narrow range proof could justify a different specialized rule. | Remove; do **not** replace with float `-> 0`. |
| FModFloor | `0 mod_floor x -> +0` | invalid under CubeCL default semantics | Its own formula produces NaN for a zero, NaN or infinite divisor, unlike the partial fold. | `NotNaN + NotInf` **and proof `x != 0`**; zero-sign permission/proof only if needed by the chosen operation-specific sign contract. | Remove custom partial fold. |
| SRem | `x % 1 -> x` | plain correctness bug independent of special values | Contradicts existing correct integer `const_eval!` fold to zero. | None needed for `x % 1 -> 0`. | Delete duplicate bad `simplify!`; retain correct fold. |
| URem | `x % 1 -> x` | plain correctness bug independent of special values | Same integer contradiction. | None needed for `x % 1 -> 0`. | Same. |
| FMul | `1*x -> x` | safe under CubeCL default semantics | Ordinary value identity preserving signed zero and NaN/Inf classification. | No additional permission under the supported ordinary value interpretation.* | Keep. |
| FMul | `x*1 -> x` | safe under CubeCL default semantics | Same. | Same.* | Keep. |
| FDiv | `x/1 -> x` | safe under CubeCL default semantics | Same ordinary value identity. | Same.* | Keep. |
| FSub | `x-+0 -> x` | safe under CubeCL default semantics | Existing implementation explicitly matches positive zero by bits. | Same.* | Keep its positive-zero check. |

\* “Safe” does not assert preservation of signaling-NaN quieting, exception flags, NaN payloads, or every backend’s denormal processing. CubeCL does not supply that global promise. These unresolved details do not warrant removing the retained identities in this targeted patch. The assumptions listed for remainder distinguish ordinary `fmod` behavior, where a zero dividend and infinite divisor can produce zero, from a conservative portable domain; the backend’s treatment of infinity needs its own guarantee.

**Current mechanism and reachability are verified in [math.rs][math]:**

- FAdd zero identities and FRem’s by-one identity are `simplify!` rules.
- FSub self-subtraction and FMul/FDiv/FRem/FModFloor zero folds are `const_eval!` **custom partial folds**, alongside ordinary both-constant evaluation.
- [Macro generation][ce] accepts optional operand constants; [SCCP analysis][sccpa] passes `None` for runtime-unknown operands, and [SCCP rewriting][sccpp] materializes folded values.
- [SimplifyOpsPass][rewrite] invokes `SimplifyInterface::check_fold` and substitutes the returned SSA value.
- LLVM, C++, SPIR-V and WGSL schedule SCCP and SimplifyOpsPass in their normal compilation pipelines: [LLVM][llvmbase], [C++][cppbase], [SPIR-V][spirvcompiler], [WGSL][wgslcompiler].

**Important reachability qualification:** ordinary both-constant evaluation precedes custom folds. Thus a constant `1.5 % 1` may fold correctly through SCCP, while `runtime_x % 1` reaches the wrong simplifier. Integer SCCP already folds `runtime_x % 1` correctly to zero and can mask the duplicate bad simplifier in normal pass order. Direct SimplifyOpsPass/interface tests are needed; the existence of a bad rule does not imply every compiled kernel exhibits it.

**Signed zero:** Rust float-literal matching in `Some(0.0)` matches both +0 and -0; it is not a bit/sign test. FAdd returns the other SSA value, preserving that value’s original zero sign even when arithmetic would differ. The custom FMul/FDiv/FRem/FModFloor folds instead explicitly call `float_attr(..., 0.0)`, materializing +0. FSub self-subtraction also materializes +0. By contrast, `FSub x-+0` checks `val.to_bits() == 0`, excludes -0, and preserves x. Float attributes retain a floating representation and expose it as f64; see [attribute implementation][attrs]. Do not impose a blanket “floor modulo preserves the dividend’s -0” rule: its formula and backend-specific sign details differ from truncating remainder.

## What actually happened historically

| Stage | Directly verified history | Interpretation |
|---|---|---|
| PR #157, 2024-10-08, [`c40c0464`][c157] | The [original generic simplifier][old157] contains add-zero, multiply-zero, divide-zero and zero/one identities without a float guard. Multiplication and zero-numerator division reuse a zero operand rather than explicitly constructing +0. | Most questioned algebra predates Pliron and was shared across integer/float types. |
| PR #169, 2024-10-17, [`6ed4657b`][c169] | Its exact diff adds generic modulo `0 % x -> 0` and `x % 1 -> 0`. | Integer-style identities already affected generic floating arithmetic; the float by-one fold was already wrong. |
| PR #1322, 2026-05-15, [`cb87b0d2`][c1322] | The refactor creates a shared [post-processing simplifier][old1322] covering `ModFloor | Rem`. Before Pliron, [arithmetic documentation][oldarith] calls Rem Rust `Rem::rem` and separately describes ModFloor. | The remainder/floor-mod distinction and the shared unsafe algebra predate Pliron. |
| PR #1402, 2026-08-11, [`12c49762`][c1402] | [Immediately before migration][oldpre] the shared by-one rule constructs zero. [Migrated math.rs][new1402] adds FSub self-subtraction and makes SRem/URem/FRem by-one simplifiers return lhs; the integer correct constant folds remain alongside them. Float zero custom folds explicitly construct +0. | New self-subtraction rule; new integer remainder regressions; change from one wrong float remainder rule to another; inherited zero folds with additional sign canonicalization for multiply/divide. |

**Migration/refactor bug is a strong inference, not a verified maintainer admission.** Simultaneously breaking integer remainder, while leaving its contradictory correct constant fold in the same file, is particularly strong evidence. The checked PR descriptions and diffs supply no float-specific justification for the questioned rules. However, neither “Pliron introduced all the bad float folds” nor “the previous float by-one rule was correct” survives direct verification.

**Reconciliation of the audits:**

- Their overlapping operational findings agree: these rules exist and are reachable; FRem’s by-one rule is a plain bug.
- The semantic-contract conclusion principally comes from the second audit’s CubeCL-native documentation/history; IEEE counterexamples from the first audit alone would not establish that contract.
- Provenance and backend mappings principally come from the third audit and are confirmed above.
- The second audit’s YAML “partially” on default relaxation and “partially” on gating is too ambiguous: implementation gaps exist, but they do not establish a default permission or a usable per-operation gate.
- **Correct the claimed PR #1732 NSZ removal:** its complete 14-file patch and merge diff contain no NSZ removal; affected LLVM arithmetic already had the current flag pattern before it. [PR #1732][p1732] / [`08c2b16e`][c1732] supports intentional NaN-propagating extrema and signed-remainder correctness, **not that specific NSZ historical claim**.
- [PR #1701][p1701] / [`e94b19c9`][c1701] adds a test comment attributing zero-normalization latitude to kernels opting into FastMath. This is supporting code-comment evidence, not a separately verified maintainer discussion or a complete semantic specification.

## CubeCL’s actual semantic contract

The strongest supported statement is: **CubeCL intends explicit, scoped permission for relaxing NaN, infinity, signed-zero and algebraic/precision constraints; unmarked code is not implicitly under every FastMath assumption.**

The current [CubeCL book][book] frames FastMath as loosening ordinary restrictions and applying only to selected functions. The [current API][properties] separately defines `NotNaN`, `NotInf`, `UnsignedZero`, `AllowReciprocal`, `AllowContraction`, `AllowReassociation`, `AllowTransform` and `ReducedPrecision`; NaN/infinity assumption violations are explicitly undefined. [Original PR #432][p432], merged as [`2cc42af0`][c432], states: **“The default is the existing safe one.”** [PR #1000][p1000] / [`20fa77da`][c1000] deliberately makes FastMath per-function rather than per-kernel. That directly contradicts a blanket implicit-fast-math interpretation.

This is **not a strict IEEE-754 promise**. The checked contract does not establish uniform subnormal/FTZ handling, all rounding modes, signaling-NaN behavior, FP exception flags/traps, or NaN payload/sign preservation. Operation-local guarantees can be narrower: the extrema introduced in #1732 explicitly allow either sign for opposite-signed zeros and leave NaN payloads/signs unspecified. Default contraction and backend precision also need separate clarification.

**FRem’s own meaning is especially well supported.** The [current frontend][frontend] expands Rust Rem/% to FRemOp and `mod_floor` to the distinct FModFloorOp; its constant evaluator computes `lhs % rhs`. Inspected lowering is:

| Backend path | FRemOp | FModFloorOp |
|---|---|---|
| [LLVM][llvmmath] | LLVM `frem` | [Polyfill][llvmpoly]: `lhs - rhs * floor(lhs/rhs)` |
| [SPIR-V][spirvmath] | `OpFRem` | `OpFMod` |
| [Shared C++ / CUDA / HIP][cppmath] | `fmod(lhs,rhs)` | Floor formula |
| [WGSL][wgslmath] | Float `%` | Floor-formula polyfill |

These source mappings corroborate the operation definition; they do not create CubeCL’s global contract. **`1.5 % 1.0 = 0.5` is independent of special values and relaxation policy within that definition**—not independent of what “remainder” means. It is exactly representable and exposes the erroneous identity even if all special values are excluded. Negative ordinary cases, e.g. `-1.5 % 1 = -0.5` versus floor modulo `0.5`, establish the distinction.

IEEE/ordinary arithmetic supplies illustrative counterexamples (A); PyTorch (B) is addressed separately below; no Burn behavior (C) is needed for the verdict. Documentation/API/history establish CubeCL intent (D), while lowering checks corroborate implementation (E).

## Why FastMath does or does not rescue the current rewrites

**It does not rescue the unconditional generic rules.**

1. The required permissions are separate. Neither contraction nor reciprocal permission implies `NotNaN`, `NotInf` or `UnsignedZero`. `AllowTransform` includes contraction/reassociation and tolerates precision changes; it does not document that all separate special-value assumptions automatically hold, nor license an arbitrary remainder identity.
2. Some transformations require value facts: there is no dedicated FastMath “divisor is nonzero” flag. Do not substitute an input no-NaN assumption for an actual denominator proof.
3. Current [fast_math_expand][options] saves, sets and restores `scope.state().modes.fp_math_mode`. [Scope state][scope] retains the field, defaulting to an empty set, but operation registration merely appends the op. A targeted repository-wide search finds this field read/write only in expansion plus its declaration; the arithmetic op definitions and simplification interfaces have no attached FastMath state.
4. The exact [Pliron scope diff][c1402] removes `inst.modes = self.state().modes` from registration. [PR #1553][p1553] documents that missing gate. Its statement that every LLVM instruction has default flags is historical, **not accurate for current GPU add/sub/multiply**.

Current [LLVM arithmetic lowering][llvmmath] gives FAdd/FSub/FMul **only contract on AMDGPU/NVPTX**, and empty flags on CPU; FDiv/FRem get empty flags. It supplies no nnan/ninf/nsz/reassoc/arcp for these operations. SPIR-V conversion emits ordinary opcodes with uniformity decoration; C++ and WGSL emit the expressions shown above without corresponding per-op assumptions. Capability booleans such as `supports_features.fast_math` are not operation permissions. Native Metal currently requests [Safe math mode and Precise functions][metal], additional evidence against a universal unsafe default.

Driver/compiler latitude and operation-specific limitations still exist, especially on shader backends. They are not a legitimate basis for discarding information in generic IR shared with other backends.

## Minimal upstream patch

### Correctness patch now

Confine implementation changes to the relevant rules in `crates/cubecl-ir/src/dialect/math.rs`, plus regression tests:

- **FAdd:** remove the entire current `simplify!` zero-identity block. Do not add new negative-zero-only variants in this patch; that avoids expanding the rounding/sign contract while correcting the existing matcher.
- **FSub:** remove the custom self-subtraction fold. Retain ordinary constant subtraction and the existing positive-zero-only `x-+0` simplifier.
- **FMul:** remove the custom zero-annihilation fold, retaining ordinary constant multiplication and both multiply-by-one simplifiers.
- **FDiv:** remove the custom zero-numerator fold, retaining ordinary constant division and divide-by-one.
- **FRem:** remove its custom zero-numerator fold and its entire by-one `simplify!` block. Retain actual constant remainder; do not restore `x%1 -> 0` for floats.
- **SRem / URem:** delete their erroneous duplicate by-one `simplify!` blocks; retain existing correct `const_eval!` by-one-to-zero folds.
- **FModFloor:** remove the custom zero-numerator fold; retain its both-constant formula.

**Exclude min/max:** normal versus NaN-propagating operations have distinct local contracts and need their own signed-zero/NaN audit. **Exclude FMA contraction:** the live [InstCombinePass][combine] and LLVM GPU contract flags form a separate precision/permission issue spanning IR and lowering; report and track it, but do not make this arithmetic patch depend on redesigning it. **Exclude other canonicalization, denormal/rounding/payload guarantees and integer zero-divisor rules:** they require distinct reachability or operation/backend contracts not established by this targeted verification.

No immediate FastMath gating is proposed. Removing information-destroying partial folds does not require restoring the architecture first.

## Follow-up architecture work

Carry effective FastMath assumptions as explicit per-operation IR attributes during expansion; default to empty and preserve function scope boundaries. Define propagation through cloning, CSE, vectorization/unrolling, lowering and new arithmetic emitted by polyfills, with conservative combination when modes differ. Expose those assumptions to constant folding/simplification and preserve them into backend flags or intrinsic selection.

Separately represent range/sign facts, including nonzero denominators. Restore aggressive folds only when their actual preconditions are proven. Clarify contraction and FmaOp evaluation, including the distinction between InstCombine and target compiler contraction; document backend/default limitations without claiming strict IEEE-754. Keep extrema and PyTorch compatibility as separate operation-level work.

## Regression tests

### IR simplification tests

Use existing textual-IR/SCCP infrastructure, exemplified by [cubecl-opt/tests/sccp.rs][tests], and construct a function that **returns each result**, preventing dead-code removal.

1. **SimplifyOpsPass alone:** runtime-unknown x with `+0+x`, `x++0`, `-0+x`, `x+-0`, `x%1` and integer `x%1`. Assert the removed rules do not return x or replace arithmetic. The integer cases must not produce lhs; SCCP may still correctly produce zero.
2. **SCCP alone:** runtime-unknown x/y with `x-x`, `±0*x`, `x*±0`, `±0/y`, `±0%y` and `±0 mod_floor y`. Assert no unjustified constant +0 is inferred/materialized. Check both operand orders for multiply.
3. **Pipeline order:** run SCCP→Simplify and Simplify→SCCP. Feed inputs through block arguments/control-flow merges as well as direct arguments. Float remainder must not depend on whether its numerator became known before simplification; integer remainder must end as zero rather than x.
4. **Positive controls:** fully constant `1.5%1=0.5`, `-1.5%1=-0.5`, `2%1=+0`, and floor modulo `-1.5 mod_floor 1=0.5`; retained `1*x`, `x*1`, `x/1`, `x-+0` simplify to x. Verify `x--0` does not use the positive-zero subtraction rule.
5. Use f32 and supported additional float types; include scalar and a representative vector/splat path after normal unrolling. Verify IR result types, not only printed text.

**Do not rely solely on fully constant special-value tests:** ordinary evaluation can return the correct result before the invalid partial fold or identity matters. Test unknown operands first, then evaluate the resulting runtime kernel on edge inputs.

### Runtime/backend tests

Use unmarked kernels, load x/y from buffers, and cover both a **literal-zero/literal-one kernel** and a **both-operands-loaded kernel**. Existing [runtime binary tests][runtime] provide the shared harness.

| Test | Required observation for an ordinary-special-value-capable backend |
|---|---|
| Loaded x `% 1`, x = 1.5, 2.5, -1.5 | 0.5, 0.5, -0.5; exact ordinary-value regression. |
| Loaded x `mod_floor 1`, x = -1.5 | 0.5, distinguishing it from FRem. |
| Loaded integer x `% 1`, signed ±7 and unsigned 7 | 0. |
| Loaded x `- x`, x = +Inf, -Inf, NaN | NaN classification. |
| Literal/loaded zero `* x` and `x * zero`, x = Inf/NaN | NaN classification. |
| Zero `/ y`, y = 0 or NaN | NaN classification. |
| +0 `/ -1`; -0 `/ +1` | Negative zero. |
| Zero `% y`, y = 0 or NaN; -0 `% 2` | NaN in first cases; negative zero in last. |
| +0 `+ x` / `x + +0`, x = -0 | Positive zero under ordinary round-to-nearest semantics; also exercise negative-zero literals. |
| Zero `mod_floor y`, y = 0, NaN, Inf | NaN for the inspected formula paths; validate the SPIR-V operation’s supported domain separately. |

For f32 zero sign use `to_bits()`: +0 = `0x00000000`, -0 = `0x80000000`. Equal-to-zero and epsilon comparisons cannot distinguish them. Use `is_nan()` for NaN, without asserting a particular payload/sign. Run supported float types/vector widths through representative backend CI.

These are **targeted backend expectations, not an invented global promise**: gate special-value runtime tests on established backend capability/contracts, especially WGSL/Vulkan edge behavior. The generic IR no-unjustified-fold tests are unconditional. No all-backend bit identity is required; ordinary 1.5 remainder tests are exact, and wider precision/subnormal policies belong to follow-up.

## PyTorch downstream consequence

**Removing these folds is necessary but not sufficient** for a general downstream implementation requiring PyTorch-compatible edge behavior.

Map operations correctly: [torch.fmod][torchfmod] is truncating remainder with dividend sign, whereas [torch.remainder][torchrem] is floor-style modulo with divisor sign. Do not map both to FRemOp. Differential-test the pinned PyTorch target against emitted kernels for sign combinations, fractional divisors, NaN/Inf/zero divisors, signed zero and large quotients; a bare floor formula alone is not a compatibility guarantee. Also establish extrema/NaN policy, contraction/reassociation, dtype promotion, conversion/accumulation precision, reduction order, subnormal/FTZ and approximate-math tolerance. Test eager and compiled modes separately where relevant.

[PyTorch itself does not promise universal cross-platform bit identity][torchnumeric]. Define compatibility per operation, dtype, backend and numerical policy, using exact classifications/sign tests where required and justified tolerances elsewhere. Burn behavior neither establishes the CubeCL verdict nor replaces these tests.

[head]: https://github.com/tracel-ai/cubecl/commit/6bba5fdf04206d9c3710dfbc46aa9850450ab405
[math]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-ir/src/dialect/math.rs#L251-L776
[ce]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-macros-internal/src/generate/const_eval.rs
[sccpa]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-opt/src/analyses/dataflow_solver/sccp.rs#L90-L111
[sccpp]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-opt/src/passes/sccp.rs#L65-L83
[rewrite]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-ir/src/rewrite.rs#L109-L130
[llvmbase]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-llvm/src/shared/base.rs#L438-L449
[cppbase]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-cpp/src/shared/base.rs#L289-L303
[spirvcompiler]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-spirv/src/compiler.rs#L242-L258
[wgslcompiler]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-wgpu/src/compiler/wgsl/compiler.rs#L159-L173
[attrs]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-ir/src/attributes/mod.rs#L302-L465
[properties]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-ir/src/properties.rs#L553-L583
[book]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/cubecl-book/src/advanced-usage/math_optimizations.md
[options]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-core/src/frontend/options.rs
[scope]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-ir/src/scope.rs#L408-L410
[frontend]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-core/src/frontend/operation/binary.rs#L430-L556
[llvmmath]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-llvm/src/shared/to_llvm/math.rs#L360-L405
[llvmpoly]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-llvm/src/shared/polyfill/math.rs#L145-L150
[spirvmath]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-spirv/src/ops/math.rs#L21-L34
[cppmath]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-cpp/src/shared/binary.rs#L111-L139
[wgslmath]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-wgpu/src/compiler/wgsl/ops/math.rs#L115-L132
[metal]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-metal/src/compute/artifact.rs#L208-L218
[combine]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-opt/src/passes/inst_combine.rs
[tests]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-opt/tests/sccp.rs
[runtime]: https://github.com/tracel-ai/cubecl/blob/6bba5fdf04206d9c3710dfbc46aa9850450ab405/crates/cubecl-core/src/runtime_tests/binary.rs
[c157]: https://github.com/tracel-ai/cubecl/commit/c40c0464aab2707487143b2e8d3cf71917c2f841
[old157]: https://github.com/tracel-ai/cubecl/blob/c40c0464aab2707487143b2e8d3cf71917c2f841/crates/cubecl-opt/src/passes/constant_prop.rs#L18-L66
[c169]: https://github.com/tracel-ai/cubecl/commit/6ed4657b6da956c75922e09622710c22cd1a0739
[c1322]: https://github.com/tracel-ai/cubecl/commit/cb87b0d2ba4a14234d751a536dabb0540646e190
[old1322]: https://github.com/tracel-ai/cubecl/blob/cb87b0d2ba4a14234d751a536dabb0540646e190/crates/cubecl-core/src/post_processing/constant_prop.rs#L25-L77
[c1402]: https://github.com/tracel-ai/cubecl/commit/12c497629d4b94baaae0feb172c4c24405c2c8f6
[oldpre]: https://github.com/tracel-ai/cubecl/blob/cb9ed88561d42bbd89e903c7a7a1eaf2c1d25c62/crates/cubecl-core/src/post_processing/constant_prop.rs#L25-L77
[oldarith]: https://github.com/tracel-ai/cubecl/blob/cb9ed88561d42bbd89e903c7a7a1eaf2c1d25c62/crates/cubecl-ir/src/arithmetic.rs#L60-L63
[new1402]: https://github.com/tracel-ai/cubecl/blob/12c497629d4b94baaae0feb172c4c24405c2c8f6/crates/cubecl-ir/src/dialect/math.rs#L288-L610
[p432]: https://github.com/tracel-ai/cubecl/pull/432
[c432]: https://github.com/tracel-ai/cubecl/commit/2cc42af02671d90255ab823e29a4a3ad2e564333
[p1000]: https://github.com/tracel-ai/cubecl/pull/1000
[c1000]: https://github.com/tracel-ai/cubecl/commit/20fa77dabd0be4db7ded05bdf033dbb30d455a6e
[p1553]: https://github.com/tracel-ai/cubecl/pull/1553
[p1701]: https://github.com/tracel-ai/cubecl/pull/1701
[c1701]: https://github.com/tracel-ai/cubecl/commit/e94b19c922ba73dc8c0bba95d3e24677008873bf
[p1732]: https://github.com/tracel-ai/cubecl/pull/1732
[c1732]: https://github.com/tracel-ai/cubecl/commit/08c2b16eaab030f64b71bd54bb3ebc3d1329eb12
[torchfmod]: https://docs.pytorch.org/docs/2.14/generated/torch.fmod.html
[torchrem]: https://docs.pytorch.org/docs/2.14/generated/torch.remainder.html
[torchnumeric]: https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html

## Final conclusion

```text
Upstream CubeCL:
likely bug

Confidence:
high

Reason:
Explicit opt-in FastMath intent conflicts with unconditional generic special-value/sign-discarding folds. FRem x%1->x additionally contradicts CubeCL’s own ordinary remainder definition. History supports inherited generic algebra plus specific Pliron regressions; the claimed #1732 NSZ removal is not supported.

Recommended immediate patch:
Remove FAdd zero simplification, FSub self-subtraction, FMul/FDiv/FRem/FModFloor partial zero folds, and FRem by-one simplification. Delete SRem/URem by-one identity simplifiers while retaining their correct constant folds. Keep ordinary constant evaluation, multiply/divide-by-one, and positive-zero-only subtraction.

Recommended follow-up:
Restore per-operation FastMath attributes and independent value proofs; resolve contraction/FMA, extrema and backend edge contracts separately.

For PyTorch compatibility downstream:
Necessary but not sufficient. Correct remainder-versus-floor-mod mapping and validate operation-specific edge behavior, precision and compilation policies against the selected PyTorch target.
```

