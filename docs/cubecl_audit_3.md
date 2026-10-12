### Provenance conclusion

The evidence points to **two different provenance stories**, not one coherent “CubeCL deliberately uses relaxed floating-point algebra” policy.

Most of the questionable zero/one simplifications—`x + 0`, `0 + x`, `0 * x`, `x * 0`, `0 / x`, and eventually `0 % x`—come from a **pre-Pliron, type-generic algebraic simplifier**. At that point CubeCL had shared arithmetic operations rather than distinct integer and floating-point IR ops, and these identities were applied without an integer/float guard or a fast-math condition. The introducing PRs contain no float-specific argument about NaN, infinity, or signed zero. That is strong evidence of **generic/integer-style algebra being inherited by floats**, although absence of discussion is not itself proof that maintainers considered it erroneous.  

`x - x -> 0` has a different origin: it appears in the Pliron conversion when arithmetic was split into integer and float operations. The same conversion deliberately excludes floats from a self-equality fold such as `x == x -> true`, but still gives `FSubOp` the self-subtraction fold. There is no recorded FP justification. That makes it look more like **mechanical parity with integer `Sub` during the split** than a documented relaxed-FP decision, though its provenance is less conclusive than the old generic zero folds.  

`x % 1 -> x` is qualitatively different again. Pre-Pliron CubeCL had `x % 1 -> 0`. During the Pliron conversion that became `-> lhs` for **`SRemOp`, `URemOp`, and `FRemOp` alike**. Current integer constant evaluation still computes `% 1` as zero, directly contradicting the simplifier. This is very strong evidence of a **Pliron-port regression/typo**, not an intentional floating-point semantic choice.  

The remainder *operation itself*, however, is quite intentional. Long before Pliron, CubeCL explicitly distinguished `Rem`—documented as Rust `Rem::rem`—from `ModFloor`, documented as PyTorch/SPIR-V-style modulo. Current backends preserve that distinction: `FRemOp` maps to LLVM `frem`, SPIR-V `OpFRem`, C/C++ `fmod`, and WGSL `%`; `FModFloorOp` maps to floor-based modulo or `OpFMod`. 

Finally, current backend lowering does **not** independently explain these questionable simplifications by assuming away special values. LLVM gives `FAdd`/`FSub`/`FMul` only the `contract` flag where FMA contraction is supported; `FDiv` and `FRem` receive an empty fast-math flag set. I found no corresponding per-operation `nnan`, `ninf`, `nsz`, reassociation, or reciprocal flags in SPIR-V, C++, or WGSL lowering.   

### Rewrite history table

| Rewrite | Introduced in | Historical context | Float-specific reasoning present? | Likely provenance |
|---|---|---|---|---|
| `x + 0 -> x` | optimizer work in PR **#157**, merged as `c40c0464aab2707487143b2e8d3cf71917c2f841`, 2024-10-08, Genna Wingert, “IR Optimizer and changes to SPIR-V compiler” | Generic `Operator::Add`, before integer/float op split | No | Strong generic/integer-algebra inheritance |
| `0 + x -> x` | PR **#157**, same commit family | Same generic simplification table | No | Strong generic/integer-algebra inheritance |
| `0 * x -> 0` | PR **#157** | Generic `Operator::Mul`; no numeric-type restriction | No | Strong generic/integer-algebra inheritance |
| `x * 0 -> 0` | PR **#157** | Same | No | Strong generic/integer-algebra inheritance |
| `0 / x -> 0` | PR **#157** | Generic `Operator::Div` | No | Strong generic/integer-algebra inheritance |
| `0 % x -> 0` | PR **#169**, merge `6ed4657b6da956c75922e09622710c22cd1a0739`, 2024-10-17, Genna Wingert, “Add value-based partial redundancy elimination” | Added for the then-generic `%`/`Modulo` path; frontend Rust `%` used it for floats too | No | Strong generic algebra inheritance |
| historical `x % 1 -> 0` | PR **#169** | Added alongside `0 % x`; mathematically correct for both integer and truncating float remainder where defined | No special FP discussion | Generic simplification |
| `x - x -> 0` | Pliron PR **#1402**, `12c497629d4b94baaae0feb172c4c24405c2c8f6`, 2026-08-11, Genna Wingert, “Make everything Pliron” | Added separately to `ISubOp` and `FSubOp` during semantic op split | No | Likely mechanical I/F parity during split; ambiguous intent |
| `x % 1 -> x` | Pliron PR **#1402**, `12c497…` | Old `-> 0` rule became `-> lhs` independently in S/U/F remainder ops | No | **Strong refactor-regression evidence** |

The first optimizer landed as a long-running PR with multiple internal commits—including `15a228e37d48440309648f8159faac7aed0d199c` (“More optimizations”, 2024-10-05) and `f442597fab9dfeb0b65f8ef646406a3e5fded380` (“Simplify constant simplification”, 2024-10-07)—before the October 8 merge.  The important provenance fact is that the landed simplification table was generic, not float-specific.

### Timeline

**September–October 2024 — optimizer foundation.** PR #157 builds CubeCL's early IR optimizer. The simplifier handles generic arithmetic operations and establishes the zero/one rules for add, multiply, subtract-by-zero, and division. Nothing in its review discussion that I found talks about NaN, infinity, signed zero, fast-math, or a deliberate relaxed floating-point model.  

**October 15, 2024 — remainder/modulo terminology is actively corrected.** Internal commit `7677ba32aeee5c607c469ccd856751dc6e1a7644`, **“Swap mod and remainder to make it consistent with other backends”**, changes SPIR-V mappings so “remainder” corresponds to signed/float remainder (`SRem`/`FRem`) and modulo corresponds to `SMod`/`FMod`. This is evidence that CubeCL did care about the distinction between truncating remainder and floor-style modulo. 

**October 17, 2024 — PR #169.** The generic constant simplifier gains `%` identities including `0 % x -> 0` and `x % 1 -> 0`. The frontend's Rust `%` operation at the time routes through that generic `Modulo` representation, including float values. Again, no FP-edge-value rationale appears in the review discussion I found.  

**May 15, 2026 — PR #1322 mega-refactor.** Commit `cb87b0d2ba4a14234d751a536dabb0540646e190` reorganizes the frontend and arithmetic naming. Rust `%` becomes `Arithmetic::Rem`, while the separate floor-semantic operation becomes `ModFloor`. The simplifier is updated from the old operation name to cover `ModFloor | Rem`; it does not revisit float edge semantics. Review discussion was essentially structural (“LGTM”), not a floating-point semantics review.  

Immediately before Pliron, the IR therefore has a single arithmetic enum. Its documentation explicitly says:

- `Rem`: Rust `Rem::rem`
- `ModFloor`: PyTorch `%`, / SPIR-V mod-style semantics.

That distinction is already established before the later split. 

**August 11, 2026 — PR #1402, “Make everything Pliron.”** Commit `12c497629d4b94baaae0feb172c4c24405c2c8f6` replaces the generic arithmetic representation with semantically separated `IAddOp`/`FAddOp`, `ISubOp`/`FSubOp`, etc. 

Most old simplifications are copied into both integer and floating versions. Two important things happen here:

1. `x - x -> 0` is added to both integer and floating subtraction.
2. The old remainder simplification `x % 1 -> 0` becomes `x % 1 -> lhs` in **all three** new remainder operations.

The latter is the strongest migration-regression evidence because it breaks integers too.

The same Pliron diff deliberately prevents the generic `x == x -> true` self-fold from applying to floats. So the migration code is not based on a blanket assumption that float values cannot be NaN; there is an explicit float exclusion in one place while float self-subtraction remains folded. 

**September 9, 2026 — PR #1626.** `2f81491f82b4c492b9d8ede5afe4557ef928e64f`, “decline the self-operand folds when the result is a vector,” fixes a structural problem with self-folds such as `x - x`: they were constructing scalar constants for vector results. The fix restricts them rather than reevaluating FP semantics, so scalar `FSub x-x` remains. 

**Late September 2026 — PR #1701.** Tests involving normalization of a zero vector are changed because `0/0` produces NaN and those kernels opt into FastMath; Metal can legally turn that into another result under those assumptions. This is particularly useful provenance evidence: later maintainers explicitly treat **FastMath opt-in** as the thing which changes special-value guarantees. 

**September 2026 — PR #1732.** `08c2b16eaab030f64b71bd54bb3ebc3d1329eb12`, “correct signed remainder semantics and add NaN-propagating extrema,” directly fixes backend discrepancies around signed remainder/floor modulo and adds explicitly NaN-sensitive extrema behavior. The patch discusses sign rules and NaN propagation rather than assuming them away.  

Taken together, the later history looks more FP-semantics-conscious than the original generic simplifier.

### Backend meaning table

| CubeCL op | Frontend meaning | LLVM | SPIR-V | CUDA/HIP/C++ | WGSL | FastMath flags |
|---|---|---|---|---|---|---|
| `FAddOp` | floating `+` | LLVM `fadd` | `OpFAdd` | `lhs + rhs` | `lhs + rhs` | LLVM: only `contract` when target supports FMA; no `nnan/ninf/nsz/reassoc/reciprocal` |
| `FSubOp` | floating `-` | LLVM `fsub` | `OpFSub` | `lhs - rhs` | `lhs - rhs` | Same: at most `contract` |
| `FMulOp` | floating `*` | LLVM `fmul` | `OpFMul` | `lhs * rhs` | `lhs * rhs` | Same: at most `contract` |
| `FDivOp` | floating `/` | LLVM `fdiv` | `OpFDiv` | `lhs / rhs` | `lhs / rhs` | LLVM explicitly uses empty/no fast-math flags |
| `FRemOp` | Rust `%`, truncating floating remainder | LLVM `frem` | `OpFRem` | `fmod(lhs, rhs)` | float `%` | LLVM explicitly empty/no FMF; no per-op FP-fast-math decoration found elsewhere |
| `FModFloorOp` | explicit floor modulo | polyfill `x - y * floor(x/y)` | `OpFMod` | `x - y * floor(x/y)` | same floor formula | No affected per-op FMF found |
| `SRemOp` | signed truncating integer remainder | `srem` | signed-rem lowering | `%` | `%` | N/A |
| `URemOp` | unsigned remainder | `urem` | `OpUMod` | `%` | `%` | N/A |
| `SModFloorOp` | signed floor modulo | dedicated floor-mod lowering | dedicated signed-floor-mod behavior | floor-mod implementation | floor-mod implementation | N/A |

LLVM's fast-math helper is explicit: `fma_contraction` starts with an empty/default set and enables only `contract` if the target advertises suitable FMA support; `no_fast_math` is simply the default empty attribute. `FDivOp` and `FRemOp` select the latter.   

SPIR-V lowering directly selects `FAdd`, `FSub`, `FMul`, `FDiv`, and `FRem`; `FModFloorOp` selects `FMod`. I found no code attaching an `FPFastMathMode` to these emitted arithmetic instructions.  

The CUDA and HIP generators share CubeCL's C++ arithmetic lowering here. Metal also uses the shared C++ path for these affected operations; I found no distinct Metal override changing their arithmetic meaning. `FRemOp` is explicitly printed as `fmod(lhs, rhs)`. 

WGSL maps `FRemOp` to WGSL `%`, while floor modulo is generated separately from `floor`.  

### `FRemOp` semantic mapping

This part is unusually consistent.

CubeCL's historical IR documentation describes `Rem` as **Rust `Rem::rem`** rather than PyTorch-style modulo. `ModFloor` is separately documented as the floor-mod/PyTorch-style operation. 

The current frontend preserves that distinction:

- ordinary Rust `%` on floating Cube values becomes `FRemOp`;
- explicit floor modulo becomes `FModFloorOp`;
- signed integer `%` becomes `SRemOp`;
- unsigned `%` becomes `URemOp`. 

For `FRemOp` specifically:

- **LLVM:** LLVM `frem`.
- **SPIR-V:** `OpFRem`, not `OpFMod`.
- **CUDA/HIP/C++:** `fmod(lhs, rhs)`.
- **WGSL:** floating `%`.
- **Metal:** through the shared C++ representation, the same `fmod`-style path rather than the floor-mod path.

So, yes: **`FRemOp` is intended to be truncating/C-`fmod`-style remainder.** It is not CubeCL's floor modulo operation.

That is also why `FModFloorOp` exists. On non-SPIR-V backends it is explicitly reconstructed as approximately:

```text
x - y * floor(x / y)
```

while SPIR-V can directly issue `OpFMod`.  

This distinction is older than Pliron and therefore should not be explained as an artifact of the new dialect.

The current `x % 1 -> x` simplification is particularly incompatible with this history. For truncating remainder, the old optimizer had `x % 1 -> 0`; during the Pliron split that return value changed to `lhs`. It also changed for integer `SRem`/`URem`, which removes any plausible “maybe floating remainder was intentionally redefined” explanation. 

### Integer vs float historical comparison

Before Pliron, there was not enough IR structure for the simplifier to cleanly express “integer identity, but not floating identity.” Add, subtract, multiply, divide, and remainder lived in shared `Arithmetic` variants and `ConstOperandSimplify` matched those variants without inspecting “is this an integer or a float?” first. 

That explains a lot of today's questionable FP folds:

```text
Add:
0 + x -> x
x + 0 -> x

Mul:
0 * x -> 0
x * 0 -> 0
1 * x -> x
x * 1 -> x

Div:
0 / x -> 0
x / 1 -> x
```

They were not originally written as `FAddOp` or `FMulOp` semantics. They were written as **generic arithmetic canonicalizations** and therefore automatically covered floats.

Remainder provides the clearest historical control case. The old generic simplifier had:

```text
0 % x -> 0
x % 1 -> 0
```

Once operations were split, the first one remained essentially unchanged but the second accidentally became `-> lhs`.

Current integer operations expose the error plainly. `SRemOp`/`URemOp` constant evaluation know `% 1 == 0`, while their simplification rule returns the dividend instead. `FRemOp` has the same erroneous simplifier. 

There are also telling current float-specific asymmetries:

- `FSubOp`'s `x - 0 -> x` test uses `to_bits() == 0`, which specifically recognizes the positive-zero bit pattern.
- `FAddOp`'s zero tests use numeric equality against `0.0`, so `-0.0` is also treated as zero.
- `FMulOp` similarly recognizes numeric zero rather than preserving zero sign/special-value effects. 

That does not look like one carefully articulated floating-point simplification policy.

The Pliron implementation also proves that developers were capable of distinguishing integer and floating algebra where they noticed a semantic difference. For self-equality, the generic `x == x -> true` optimization explicitly avoids floats. Yet `FSub x-x -> 0` exists beside it. 

### Pliron rewrite impact

The Pliron conversion did **not** invent most questionable simplifications.

It mostly did this:

```text
generic Add
    -> IAddOp
    -> FAddOp

generic Mul
    -> IMulOp
    -> FMulOp

...
```

and reproduced existing canonicalization rules in the new ops.

So for:

- `x + 0 -> x`
- `0 + x -> x`
- `0 * x -> 0`
- `x * 0 -> 0`
- `0 / x -> 0`
- `0 % x -> 0`

the semantic behavior substantially **predates Pliron**.

But Pliron is responsible for two important changes.

First, it introduces the `x - x -> 0` simplification into the new subtraction ops, including `FSubOp`. There is no evidence in the #1402 discussion that this was intended as a deliberate relaxed floating-point assumption. 

Second, it turns the old:

```text
x % 1 -> 0
```

into:

```text
x % 1 -> x
```

for `SRemOp`, `URemOp`, and `FRemOp`.

Because that same change is nonsensical for integer remainder, this is best classified as a **refactor regression**, not an intentional change in FP semantics.

So “Did Pliron introduce the semantics?” has a mixed answer:

- **most zero identities:** no, inherited;
- **`x - x -> 0`:** yes, appears with the semantic split;
- **`x % 1 -> x`:** yes, and almost certainly as a porting mistake;
- **`Rem` vs floor-mod meaning:** no, distinction is much older.

### Evidence from later FP correctness fixes

Later CubeCL history is notably more explicit about special-value semantics than the old optimizer work.

PR #1701's zero-vector normalization change is particularly instructive. Its rationale is not “CubeCL always assumes no NaN.” It says these particular kernels **opt into fast math**, and therefore the compiler may assume away the NaN from `0/0`; Metal had in fact optimized it unexpectedly. 

That is evidence for a conceptual distinction:

```text
ordinary operation
vs.
operation/kernel under explicitly relaxed FastMath assumptions
```

It does not by itself define CubeCL's whole FP contract, but it weakens the hypothesis that unconditional old algebraic folds were deliberately implementing an always-relaxed model.

PR #1732 is even more direct. It repairs signed remainder/floor-mod behavior and introduces NaN-propagating extrema behavior. Its implementation and tests explicitly care about sign conventions and NaN propagation rather than treating backend behavior as sufficient. 

The sequence is therefore interesting:

1. old optimizer: generic algebra, little/no recorded FP-edge discussion;
2. semantic op split: much more type information, but old rules largely copied;
3. later correctness work: explicit concern for NaN, signed remainder, floor modulo and fast-math scope.

I would not use that to retroactively declare every old fold a confirmed bug, but it is evidence that the **current codebase's semantic thinking is more discriminating than the provenance of those folds**.

### Burn relationship

I found no evidence that Burn motivated CubeCL to intentionally weaken these floating-point identities.

In fact, current Burn provides contrary practical evidence.

Burn has a distinct public `fmod` API rather than treating all remainder-like operations as equivalent. 

Its CubeCL tensor `remainder` implementation uses `Vector::mod_floor`, i.e. CubeCL's floor-modulo semantics rather than `FRemOp`. 

Burn also exposes special-value-sensitive operations such as `is_nan` on its CubeCL backend. 

So there is no historical trail saying anything like:

> Burn doesn't care about NaN/signed zero, therefore CubeCL may universally use integer-like FP identities.

What the current Burn/CubeCL integration instead shows is that consumers distinguish:

- truncating fmod-like remainder;
- floor remainder/modulo;
- NaN-sensitive operations.

That is not evidence for relaxed semantics.

### Open questions

The main remaining historical uncertainty is **intent**, not mechanism. The mechanism is fairly clear: several FP folds come from a generic optimizer and were later copied into float-specific Pliron operations. What the repository does not give us is a maintainer statement from the original optimizer era saying whether generic arithmetic was knowingly supposed to embody fast-math-like FP behavior.

`x - x -> 0` is the least historically settled rule. Unlike the zero folds, it does not simply descend from the oldest generic simplifier; unlike `% 1 -> x`, it is not disproven internally by integer semantics. It appears during #1402 without explanation, despite the same migration taking care to exclude floats from `x == x -> true`.

The current handling of signed zero is also internally uneven. The bitwise `+0` check in `FSubOp` versus numeric-zero checks in `FAddOp`/`FMulOp` deserves separate semantic review; provenance alone does not tell us whether that is an intentional distinction or accumulated patchwork.

Finally, backend compiler flags outside the per-operation CubeCL lowering layer can of course affect generated machine code. The finding here is narrower: **the CubeCL lowerings for these ops do not themselves attach `nnan`, `ninf`, `nsz`, reassociation, or reciprocal assumptions**, and LLVM division/remainder explicitly use an empty fast-math set. That should not be confused with a guarantee about every downstream vendor compiler mode.

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