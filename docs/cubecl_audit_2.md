### Semantic conclusion

CubeCL’s intended default floating-point model is best described as **safe/no-FastMath by default, IEEE-like with respect to the existence and observability of special values, but not strict IEEE-754 and not completely specified**.

The most important distinction is that CubeCL has an explicit semantic-relaxation API rather than silently making all GPU-style fast-math assumptions. The current book says floating-point operations normally have restrictions around `Inf`, `NaN`, and signed zero, and that functions can be marked with **“loosened restrictions”** through `FastMath`. Its example describes the marked inverse-square-root region as reduced precision / reduced special handling and says: **“Everything else is full precision.”**  Historical PR #432 was even more direct: **“The default is the existing safe one”**, with unsafe optimizations explicitly selected. 

That does **not** mean CubeCL promises strict IEEE-754. I found no CubeCL-wide contract for signaling NaNs, floating-point exceptions, NaN payload preservation, a universal rounding mode, or subnormal/FTZ behavior. Recent code also has backend- and operation-specific precision behavior, and the current LLVM GPU path permits FMA contraction independently of CubeCL's `FastMath::AllowContraction`. So “strict IEEE-754” would overstate the promise.

The intended model is therefore roughly:

> **Normal CubeCL code does not grant the compiler `NotNaN`, `NotInf`, `UnsignedZero`, reassociation, reciprocal, or reduced-precision permissions unless they are explicitly opted into. Beyond those guarantees, exact floating-point details can be operation- and backend-specific.**

There is, however, a significant **current implementation gap**: after the Pliron migration, the frontend still sets `fp_math_mode` while expanding a `#[cube(fast_math = ...)]` function, but that mode is not attached to the resulting operations. A merged 2026 PR explicitly observed that nothing reads the field after expansion, and that LLVM receives default fast-math flags instead.  Consequently, current `cubecl-ir` simplification has no applicable FastMath mode to query.

That distinction matters a lot for the algebraic-rewrite question: **the intended semantic architecture provides exactly the sorts of assumptions needed to gate these rewrites, but current Pliron IR does not carry those assumptions to the optimization pass that would need them.**

---

### Evidence hierarchy

I would rank the evidence this way:

1. **Current CubeCL book and current public `FastMath` API contract.** These explicitly distinguish ordinary behavior from opt-in relaxed behavior.  
2. **Current CubeCL operation contracts.** In particular, recent NaN-propagating extrema explicitly define NaN and signed-zero behavior. 
3. **Merged maintainer PRs describing the intended model.** #432 establishes safe-by-default; #1000 establishes per-function FastMath and historically wired it through each instruction; #1402 explains the Pliron transition; #1553 documents the present plumbing gap.   
4. **Current tests and optimizer/lowering source.** These reveal what the implementation actually preserves or relaxes today.
5. **Backend behavior.** LLVM, CUDA, HIP, SPIR-V, WGSL and Metal are evidence about implementation, but they are not CubeCL's semantic definition unless CubeCL explicitly adopts their behavior.
6. **IEEE-754, PyTorch and Burn.** Useful for comparison and counterexamples only. I found no evidence that either PyTorch or Burn defines CubeCL's floating-point contract. Burn compatibility checks in CubeCL PRs are downstream validation, not a normative semantic specification.

This ordering is important because there are places where **D, CubeCL's intended contract, and E, its present backend implementation, no longer line up cleanly**.

---

### FastMath flag table

The complete current `FastMath` definition is in `crates/cubecl-ir/src/properties.rs`. It describes FastMath as:

> “Unchecked optimizations for float operations” which may cause precision differences or undefined behavior if conditions are violated. 

| Flag | CubeCL meaning | Semantic assumption | Where consumed |
|---|---|---|---|
| `NotNaN` | Assume values are never `NaN`; if they are, result is explicitly **undefined behaviour** | NaN can be treated as impossible | **No current consumer found.** Historically mapped to SPIR-V `NOT_NAN` and used to select CUDA/HIP fast intrinsics |
| `NotInf` | Assume values are never `Inf` or `-Inf`; violation is explicitly **undefined behaviour** | ±∞ can be treated as impossible | **No current consumer found.** Historically SPIR-V `NOT_INF`; required by several C++ fast intrinsics |
| `UnsignedZero` | “Ignore sign on zero values” | `+0` and `-0` may be treated interchangeably | **No current consumer found.** Historically SPIR-V `NSZ`; required by fast div/reciprocal selection |
| `AllowReciprocal` | Division may be replaced by multiplication with reciprocal even if precision changes | Reciprocal substitution is permitted | **No current consumer found.** Historically gated `__fdividef` / reciprocal paths |
| `AllowContraction` | Multiple FP operations may be contracted into fewer operations even if precision changes | e.g. implicit multiply+add → FMA may be permitted | CubeCL flag itself: **no current consumer found.** Current LLVM GPU lowering independently sets LLVM `contract` |
| `AllowReassociation` | FP expressions may be reassociated even if precision changes | Parenthesization/order need not be preserved | **No current consumer found.** Historically mapped to SPIR-V `ALLOW_REASSOC` |
| `AllowTransform` | Allow “all mathematical transformations,” including contraction/reassociation, even with precision changes | Broad algebraic transformation permission | **No current consumer found.** Historically mapped to SPIR-V `ALLOW_TRANSFORM` plus contract/reassoc |
| `ReducedPrecision` | Allow lower-precision intrinsics | Approximate / reduced-precision evaluation is permitted | **No current consumer found.** Historically selected CUDA/HIP fast intrinsics |

The exact source wording separates two different kinds of relaxation. `NotNaN` and `NotInf` are **preconditions whose violation is UB**. The remaining flags are expressed as **permissions to change observable numerical behavior**, not as UB-producing input preconditions. 

One important omission: there is **no dedicated FTZ/subnormal flag, rounding-mode flag, signaling-NaN flag, exception flag, or NaN-payload flag**.

`FastMath::all()` simply selects all eight flags. 

---

### Default semantics

#### Safe by intent, not implicitly fast-math

The current CubeCL book presents FastMath as an explicit loosening of normal floating-point restrictions. It says special-value and signed-zero handling are among those restrictions and says the effect of opting in is backend-dependent. 

PR #432, which introduced the architecture, is particularly useful because it states the design intent rather than merely describing an implementation detail:

> “The default is the existing safe one.”

It also explains that some flags permit reordering/fusion despite precision changes while `NotNaN`/`NotInf` are assumptions that have to be true. 

PR #1000 later deliberately changed FastMath from **per-kernel to per-function** so that only performance-critical code would receive those assumptions, rather than forcing them on the whole kernel. It also removed CUDA's coarse global `--use-fast-math` approach in favor of selecting individual intrinsics only when their required CubeCL flags were enabled. 

So the architectural intent is not “GPU code is fast-math anyway.” It is almost the opposite.

#### NaN

NaN is observable by default. CubeCL has `is_nan`, and recent work deliberately added `min_nan`, `max_nan`, and `clamp_nan` operations whose contracts require propagating NaN from either operand. 

The existence of `NotNaN` as an explicit UB-producing opt-in assumption is even stronger evidence: without that flag, an optimizer cannot simply reason that NaNs are impossible. 

NaN **payload and sign**, on the other hand, are not generally promised. The new extrema explicitly say their NaN payloads and signs are unspecified.  A Metal bf16 conversion regression test likewise permits a GPU to canonicalize NaNs and compares only their classification. 

#### Infinity

Infinity is likewise an observable normal input unless `NotInf` is opted into. `NotInf` says supplying infinity makes the result undefined. That would be unnecessary if infinities were already outside the default domain. 

Current CPU transcendental work also treats things such as `log(0) = -inf` as part of its no-FastMath edge behavior rather than deleting those cases. The same merged PR specifically identifies removing NaN/infinity handling under `NotNaN`/`NotInf` as a future optimization that cannot yet be done because the FastMath mode does not reach lowering. 

#### Signed zero

Signed zero is not globally ignorable under default CubeCL. The dedicated `UnsignedZero` flag is the direct API evidence. 

Recent implementation history reinforces this strongly. Commit `08c2b16e...` specifically includes:

> `fix(llvm): remove NSZ from NaN-propagating min/max`

That is difficult to reconcile with any theory that CubeCL intends signed zero to be irrelevant by default. 

There are nevertheless **operation-specific relaxations**. The new NaN-propagating extrema explicitly permit either sign when the two inputs are opposite-signed zero.  That is a local contract, not a global `nsz`.

#### Subnormals / FTZ

This is **not globally specified**.

There is evidence that maintainers care about subnormal behavior in specific operations. The Metal f32→bf16 regression test includes subnormals, and recent CPU transcendental work tests them. 

Historically, the `ReducedPrecision` documentation in PR #1000 explicitly noted effects on “subnormals and rounding,” but that detail is absent from the current abbreviated enum comment. 

I would therefore not infer a global “gradual underflow required” rule, nor a global “FTZ allowed” rule. The current contract simply does not say enough.

#### Rounding

Also not globally specified.

Individual conversions do pin a rounding rule; for example recent bf16/fp8 work tests rounding boundaries. But I found no CubeCL-wide statement that all arithmetic is round-to-nearest-even, nor a user-visible rounding-mode model.

Consequently, a rewrite whose validity relies on a particular rounding mode should not be justified solely by “CubeCL is IEEE-like.”

#### Signaling NaNs and floating-point exceptions

I found **no CubeCL contract** for signaling NaNs, invalid/inexact/overflow flags, traps, or exception-state observability.

Those therefore cannot be used as reasons either to permit or forbid a transformation under CubeCL's documented model unless a future contract adds them.

---

### FastMath opt-in semantics

The original architecture was quite disciplined.

PR #432 added fast floating-point modes and mapped them to SPIR-V's per-operation `FPFastMathMode`. It explicitly used examples showing how an enabled reassociation flag can allow a transformation even when overflow behavior changes. 

PR #1000 then made the mode **lexically per-function**. A `#[cube(fast_math = ...)]` function temporarily changed the current expansion mode while its body was expanded, and every resulting old-IR `Instruction` stored a copy of that mode. 

The old C++ backend then had a helper conceptually equivalent to:

```text
if backend_supports_fast_math
   && type == f32
   && instruction_mode ⊇ required_flags
then fast intrinsic
else ordinary intrinsic
```

The requirements were operation-specific. Division, for example, required:

`AllowReciprocal | ReducedPrecision | UnsignedZero | NotInf`

while `exp`, `log`, `sin`, `cos`, `tanh`, `powf`, `sqrt`, and inverse sqrt required:

`ReducedPrecision | NotNaN | NotInf`. 

That same table remains in the current CubeCL book. 

This is valuable precedent for IR simplification: **CubeCL already designed FastMath as a set of precise semantic capabilities rather than one undifferentiated “unsafe math” switch.**

One caution is `AllowTransform`. Its wording is broad enough that it clearly authorizes more than simple reassociation. But CubeCL does **not** document that `AllowTransform` implies `NotNaN`, `NotInf`, or `UnsignedZero`. Those are separate flags. I would therefore not treat `AllowTransform` as silently satisfying special-value preconditions.

---

### FastMath plumbing through the IR

The intended path and current path are now different.

#### Frontend

The macro parser accepts:

```rust
#[cube(fast_math = ...)]
```

and generated expansion code wraps the function body in `fast_math_expand(...)`. 

Current `fast_math_expand`:

1. saves `scope.state().modes.fp_math_mode`;
2. sets it to the requested flag set;
3. expands the body;
4. restores the previous value. 

`GlobalState` still contains `modes: InstructionModes`. 

So lexical nesting still works **during frontend expansion**.

#### Before Pliron

Before PR #1402, each custom-IR `Instruction` itself had:

```rust
pub modes: InstructionModes
```

and therefore preserved the exact FastMath mode under which that operation had been expanded. 

Backend code received that mode and could make per-operation decisions. The old SPIR-V compiler converted it to `FPFastMathMode`; the old C++ compiler selected precise vs fast intrinsics from it.  

#### Pliron migration

PR #1402, merged as `12c497629d4b94baaae0feb172c4c24405c2c8f6`, replaced CubeCL's custom instruction IR with Pliron. 

In that migration:

- the old `Instruction.modes` field disappeared;
- the old C++ `modes: InstructionModes` lowering parameter disappeared;
- the old SPIR-V `FPFastMathMode` decoration path disappeared;
- but `GlobalState.modes` remained as expansion state.  

There is currently no equivalent per-op FastMath attribute that I could find.

#### Current simplifier

`SimplifyOpsPass` simply walks operations implementing `SimplifyInterface`, gathers constant operand attributes, and invokes `check_fold(ctx, &operand_attrs)`. 

There is no applicable FastMath argument and no per-op mode attribute.

In principle an IR operation could gain such an attribute and `check_fold` could inspect it through `ctx`; the architecture is not fundamentally incapable of that. But **today's operations do not carry it**.

The clearest confirmation is merged PR #1553. Its author explicitly investigated optimizations such as dropping NaN/Inf handling and reassociation, and wrote:

> **“There is no fast-math gate to reach for.”**

The PR explains that the frontend sets `GlobalState.modes.fp_math_mode`, expands the body, then restores it; nothing records that state on the operations, so a later lowering cannot recover it. 

So, specifically:

| Question | Current answer |
|---|---|
| Can `cubecl-ir` simplification inspect the applicable FastMath mode? | **No**, not from current operation state |
| Are modes attached to individual operations? | **No** |
| Are they retained after the Pliron conversion? | **No equivalent of the old `Instruction.modes` was retained** |
| Are they only used during backend lowering? | **Not currently; the mode is lost before lowering** |
| Is `fp_math_mode` stored but disconnected? | **Yes, according to current source and merged #1553** |
| Could the architecture be extended to gate simplification? | **Yes, by adding/preserving per-operation mode attributes or equivalent context** |

There are also backend capability fields that appear disconnected. `VulkanCompilationOptions.supports_fp_fast_math` is populated when Vulkan advertises float-controls support, but the only current reference I found is the assignment. 

Likewise CUDA/HIP/Metal populate a C++ `fast_math` support field, but current searches find no consumer of that field. Metal even has a comment saying its fast namespace supports per-op fast math, while no actual emitted `fast::` use appears in the current tree. 

I would therefore describe those as **“appearing unused/disconnected on current main,”** not definitively dead.

---

### Backend implementation

This section is deliberately about **E: implementation behavior**, not D: CubeCL's semantic definition.

#### LLVM / CPU / current CUDA LLVM

Current LLVM lowering does not translate CubeCL `FastMath`.

Floating comparisons get an LLVM `FCmpOp` with an empty/default `FastmathFlagsAttr`. 

Floating divide and remainder similarly receive no LLVM fast-math flags.

However, add/subtract/multiply use a target-dependent helper:

- CPU: no fast-math flags;
- NVPTX: LLVM `contract`;
- AMDGPU: LLVM `contract`.  

That means current CubeCL allows **implicit FMA contraction on LLVM GPU targets even with no CubeCL `FastMath::AllowContraction` available to the lowering**.

This is an important implementation fact, but I would not infer from it that the FastMath API contract intends `AllowContraction` to be enabled globally. It is better described as a current disconnect between the high-level mode architecture and LLVM lowering.

CUDA's default backend became LLVM with merged PR #1609. 

#### CUDA C++ / HIP C++

The current shared C++ emitter's previously active “fast intrinsic” forms are largely commented out. For example, the normal `exp` path is active while `__expf` is commented.  The analogous `__fdividef` implementation is also commented. 

Runtime code still advertises C++ fast-math capability, but I found no current read of that field.

This contrasts with the historical implementation from PR #1000, where exact CubeCL flag combinations actively selected these fast intrinsics. 

#### SPIR-V / Vulkan

Historically, this was the cleanest implementation of the architecture.

CubeCL converted each old instruction's mode into SPIR-V `FPFastMathMode`:

- `NotNaN` → `NOT_NAN`
- `NotInf` → `NOT_INF`
- `UnsignedZero` → `NSZ`
- `AllowReciprocal` → `ALLOW_RECIP`
- `AllowContraction` → `ALLOW_CONTRACT`
- `AllowReassociation` → `ALLOW_REASSOC`
- `AllowTransform` → transform + contract + reassociation. 

That path was removed in the Pliron migration.

Current Vulkan device setup still records whether `shader_float_controls2` is available, but I found no corresponding current consumption of that capability. 

So the book's claim that the Vulkan compiler “applies all enabled flags” accurately describes the intended/old architecture, but **does not match the current main plumbing I found**. 

#### WGSL

I found no CubeCL `FastMath` integration in the current WGSL backend.

The book says other backends do not currently support those optimizations. 

That should not be read as WGSL itself having strict IEEE semantics; it only means CubeCL is not explicitly lowering its FastMath flag set there.

#### Metal / MSL

Current Metal initialization states:

> “Metal honors per-op fast math via the `fast::` namespace (MSL 3+).”

and marks the backend as supporting fast math. 

But current source search finds no emitted `fast::` operations and no consumer of the capability field. So this currently **appears to be incomplete plumbing**.

A recent Metal-related test change is nevertheless semantically revealing: PR #1701 removed normalization of a zero vector from tests that opt into fast math because `0 / 0` yields NaN ordinarily, while under fast math the compiler may assume NaN does not arise; the description notes that Metal folded it to `1`. 

That is good evidence for the intended meaning of `NotNaN`, but not proof that current CubeCL mode propagation caused that particular Metal behavior.

---

### NaN / Inf / signed-zero evidence

Several very recent changes show that maintainers are treating special values as meaningful semantic cases, not accidental backend noise.

**PR #1732 / commit `08c2b16eaab030f64b71bd54bb3ebc3d1329eb12`**, merged September 29, 2026, added explicitly NaN-propagating `min`, `max` and `clamp`. Their CubeCL-level documentation says that either NaN operand is propagated, while NaN payload/sign are unspecified and opposite-signed zero may return either zero sign.  

The same commit specifically removed LLVM `NSZ` from those operations. 

**PR #1549**, merged September 2, 2026, added a Metal f32→bf16 regression test covering rounding boundaries, **signed zero, subnormals, infinities and NaN**. Non-NaN outputs are compared bit-for-bit, while NaN is compared by classification because GPU canonicalization is allowed.  

**PR #1553** added a CPU transcendental implementation and explicitly preserved default/no-FastMath edge behavior such as infinity, NaN, subnormals and zero sign where the implementation claims it. It separately identifies `NotNaN`, `NotInf`, and reassociation as the flags needed to remove some of that handling—then explains that those optimizations cannot currently be implemented because the mode is not carried into IR. 

**PR #1701** records the converse: a fast-math test should not expect `normalize(0)` to produce NaN because the fast-math contract permits the compiler to assume away that case. 

Finally, the still-open **issue #1219** provides useful evidence of the current plumbing failure: a kernel explicitly marked `FastMath::all()` nevertheless emits the precise `sqrt.rn.f32` CUDA path. 

Taken together, these are much more consistent with:

> “special values matter unless a specific relaxed assumption says they do not”

than with:

> “GPU behavior is inherently fast-math, so any algebraic identity is acceptable.”

---

### What assumptions common algebraic rewrites require

This table is about the **CubeCL FastMath assumptions needed to make the rewrite semantically defensible**. It is not claiming these rewrites currently exist.

I am also deliberately not assuming that `AllowTransform` magically implies the other flags. CubeCL does not document such an implication.

| Rewrite | Required CubeCL assumptions | Why / remaining caveat |
|---|---|---|
| `x + 0 -> x` | **`UnsignedZero`** is the key existing flag | With signed zero observable, e.g. `-0 + +0` need not have the same zero sign as simply returning `-0`. NaN class and infinity are not inherently a problem for this identity. sNaN/payload behavior is undocumented. |
| `x - x -> 0` | **`NotNaN` + `NotInf`** | NaN gives NaN; ±Inf − itself gives NaN. For ordinary finite values the identity holds. Exact zero sign can depend on rounding semantics, which CubeCL does not globally specify. |
| `0 * x -> 0` | **`NotNaN` + `NotInf` + `UnsignedZero`** | `0*NaN` and `0*Inf` are not zero; finite negative `x` can produce `-0`, so replacing with literal `+0` changes an observable sign. |
| `0 / x -> 0` | **`NotNaN` + `UnsignedZero` + proof that `x != 0`** | `0/0` is the fatal case and no FastMath flag means “nonzero.” Negative divisors affect zero sign. `NotInf` is not inherently needed for `0 / ±Inf` once zero sign is ignored. FTZ may require stronger range knowledge on a backend. |
| `0 % x -> 0` | **At least `NotNaN` + proof that `x != 0`** | A zero divisor is not excluded by any FastMath flag. General CubeCL floating-remainder edge semantics are not documented strongly enough to claim more. If the rewrite conflates signed-zero dividends, `UnsignedZero` may also be needed. |
| `x / x -> 1` | **`NotNaN` + `NotInf` + proof that `x != 0`** | NaN, infinities and zero all break the identity. `FastMath::all()` still does not establish nonzero. Subnormal/FTZ behavior can require further range knowledge. |
| `min(x, x) -> x` | For ordinary `min`: **`NotNaN` is the conservative sufficient flag**. For explicit `min_nan`, no FastMath assumption is needed at the documented value/class level. | Infinity and signed zero do not break idempotence when both operands are literally the same value. NaN behavior of ordinary `min` is backend/operation-specific; `min_nan` explicitly propagates it. |
| `x == x -> true` | **`NotNaN`** | Infinity is reflexive; `+0 == -0`; NaN is the relevant counterexample. |

A few details are worth highlighting.

#### `NotNaN` really is enough for `x == x`

CubeCL's current LLVM comparison lowering uses ordered equality for floating equality, so NaN is the exceptional case while infinities and zeros do not require `NotInf` or `UnsignedZero`. 

Architecturally, `NotNaN` is exactly the right assumption.

#### `x / x -> 1` cannot be justified by FastMath flags alone

Even:

```rust
FastMath::all()
```

does not assert `x != 0`.

That is a distinct value-range fact. Any IR optimizer wanting to make the rewrite generally would need analysis proving nonzero in addition to `NotNaN` and `NotInf`.

This same distinction applies to `0 / x` and `0 % x`.

#### `0 * x -> 0` is the clearest three-flag case

It needs all three special-value relaxations:

- `NotNaN`
- `NotInf`
- `UnsignedZero`

This is almost a textbook example of why CubeCL having independent flags is useful.

#### `AllowReassociation` / `AllowTransform` are a second axis

There are really two questions an optimizer can ask:

1. **Are special values / zero signs constrained enough that the two expressions agree?**
2. **Has the user granted permission for precision-changing algebraic restructuring?**

For a local exact identity, the first is often enough. For transformations changing evaluation order, overflow path, or rounding, `AllowReassociation` or `AllowTransform` may additionally be required.

Historical PR #432 explicitly demonstrates this distinction: `AllowReassociation` can authorize rewriting an expression even where reassociation changes whether an intermediate finite computation overflows to infinity. 

So I would not build an optimizer policy of “if `NotNaN`, do arbitrary algebra.” The flag set was designed to be finer-grained than that.

---

### Gaps or undocumented behavior

The largest semantic gap is not the definition of `FastMath`; that part is fairly good. It is the disconnect between that definition and current IR.

**First, current Pliron operations do not retain FastMath state.** This makes both IR simplification and backend lowering unable to distinguish ordinary code from `#[cube(fast_math = ...)]` code. The merged #1553 analysis confirms exactly this. 

**Second, current documentation is partly stale relative to implementation.** The book still describes Vulkan and CUDA/HIP FastMath lowering that existed before the Pliron migration, while current source no longer contains those active consumers. 

**Third, contraction currently bypasses the CubeCL flag model.** The public API says `AllowContraction` is the permission to contract operations even if precision changes, yet current LLVM lowering enables `contract` automatically for GPU add/sub/mul.   This needs either documentation saying contraction is part of ordinary GPU CubeCL semantics or wiring it back to the flag.

**Fourth, “full precision” is not a formal accuracy guarantee.** Recent no-FastMath CPU transcendental implementations can use polynomial approximations. That makes “full precision” in the book best understood as “not opting into the reduced-precision FastMath mode,” not “correctly rounded IEEE transcendentals.”

**Fifth, several IEEE topics remain unspecified at CubeCL level:** signaling NaNs, exception flags/traps, global rounding mode, global subnormal/FTZ policy, NaN payload/sign preservation, and probably the exact zero sign of many operations.

**Sixth, ordinary `min`/`max` semantics are less explicit than the newly added `min_nan`/`max_nan` family.** The latter now has a clear CubeCL contract; the former still relies more heavily on backend behavior. That matters for proving canonicalizations involving NaN.

**Seventh, PyTorch and Burn should not be used to fill these holes.** CubeCL frequently validates changes downstream against Burn, but that demonstrates compatibility rather than making Burn semantics normative. Likewise, nothing I found makes PyTorch's floating-point choices part of CubeCL's contract.

The practical architectural consequence is fairly straightforward: if CubeCL wants its algebraic simplifier to exploit FastMath correctly again, the cleanest continuation of its own historical design is to **materialize the current `InstructionModes::fp_math_mode` on each floating-point Pliron operation, or provide an equivalent immutable per-op/per-region semantic attribute**, and have simplification/lowering query that. That would restore the property the old IR already had, rather than inventing a new semantic model.

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