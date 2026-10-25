# Design

## Context

See proposal.md for motivation. The facts that shape the approach:

- **Three op vocabularies.** The descriptor compiler validates and emits ops in three places:
  - `native/src/semantic.rs` (`validate_ops`/`emit_with`) is the shared layer. Schema 2 uses it
    directly, schema 3 reductions through `reduction/analysis.rs`, and schema 4 as its fallback.
    It knows float math, comparisons on float32/int64, `where`, `masked`, `cast` and `index`. Its
    catch-all arm requires float32 for every other op.
  - `native/src/effects.rs` (schema 4) adds its own `iadd`, `imul`, `imin`, `imax`, `and`, `or`,
    plus int32 `constant`/`cast`/`index` for the RNG counter. Reductions don't see any of it,
    which is the `fmax(a, b).sum()` fusion gap noted on `4x9.1`.
  - `cubecl_torch/compiler/descriptor.py` checks the descriptor structure, and its `DTYPES` covers
    storage only.
- **Python front end.** The type rules live in `semantic_checks.node_reasons`:
  - The base `OPS` table, extended per path with `extra_ops`.
  - `operation_type` hooks (`effects.rng_type`, `predicate_type`).
  - A catch-all that raises "integer/bool arithmetic unsupported; indices only".

  Emission is `CubeCLKernel.operation` in `semantics.py` (promote, convert, emit; integer
  arithmetic rejected), overridden in `effects.py` for the internal integer ops. `reduction_checks`
  calls the same `node_reasons` without the effects path's extras.
- **Value types.** `VALUES = (float32, float16, bfloat16, bool, int64)`. int32 exists only as the
  effects path's extra value dtype. `get_computation_dtype` leaves integers and bool unchanged, so an
  integer value's compute dtype is its semantic dtype.
- **Eager semantics** (pinned v2.14.1 `BinaryShiftOpsKernels.cu`, `BinaryDivFloorKernel.cu`,
  `BinaryDivTruncKernel.cu`, `c10::div_floor_integer`):
  - Shifts guard the amount: negative or ≥ width gives 0 for a left shift, and `a >> (width − signed)`
    for a right shift.
  - Trunc division is C `a / b`.
  - Floor division subtracts 1 when the signs differ and the remainder is nonzero.
  - Division by zero is undefined on CUDA.
- **Triton codegen** (pinned Inductor `codegen/triton.py`):
  - Shifts are a plain `<<`/`>>`, which is LLVM poison for out-of-range amounts.
  - `truncdiv` is `//`, which is C division in Triton.
  - `floordiv` replaces a zero divisor by 1, then selects 0. It handles signs by negating and
    complementing around one truncating division.
  - `logical_not` is `a == 0`, and `to_dtype_bitcast` rounds to the source dtype first.
  - Inductor converts the operands of the aten logical ops to bool (`convert_input_to_bool`), and
    `bitwise_not` of bool becomes `logical_not`.
- **CubeCL IR** has every instruction needed: `IAdd`, `ISub`, `IMul`, `SNeg`, `SDiv`, `UDiv`,
  `SRem`, `URem`, `SMin`, `SMax`, `UMin`, `UMax`, the signed, unsigned and bool comparisons,
  `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `BitwiseNot`, `BoolNot`, `ShiftLeft`, `ShiftRight` and
  `ReinterpretCast`. The C++ backend emits `>>` (arithmetic on signed types), and `~` with integer
  promotion for `BitwiseNot`, so `~` must never be applied to a bool.

- **SASS check** (task 4.5, 2026-10-25). PTX assembled for sm_86 with ptxas 13.1; Triton's
  kernels come from the reference venv's cache.
  - int64 `a // b`: one 64-bit division routine `CALL` per element. NVRTC merges the quotient and
    remainder pair into one routine, so Triton's one-division complement form isn't needed. The
    scalar plan has 176 instructions; Triton's kernel (two elements per thread) has 168 instructions
    and 2 `CALL`s.
  - int32 `a << b` with the eager guard: 72 instructions and 6 `SEL`, the same as Triton's
    unguarded kernel.
- **CubeK/CubeCL reuse** (checked 2026-10-25, per the CubeK-first rule).
  - This change adds per-element op semantics, not a kernel strategy. CubeK has no
    integer-elementwise layer to reuse; it writes `a / b`, `a << b` and so on directly in `#[cube]`
    code, which lowers to the same IR ops this design emits.
  - `cubecl-std`'s `FastDivmod` (multiply-high and shift with a host-computed multiplier) is
    unsigned only (`u32`/`usize`) and needs a divisor fixed at launch. Value-level `truncdiv`/
    `floordiv` divide signed values by per-element divisors, so it doesn't apply. NVRTC already
    strength-reduces division by a compile-time constant. `FastDivmod` stays where it fits: layout
    and index division by positive shape symbols, as in the earlier `s1a`/`l1m` trials.
  - CubeK's reduce instructions (`cubek-reduce/src/components/instructions/{sum,max,min,any}.rs`)
    are generic over `Numeric`, with identities built by `from_int`. The integer-reduction
    follow-up bead should reuse that pattern in our combiners, which were modelled on CubeK's
    `ReduceInstruction`. Nothing here needs it.

## Goals / Non-Goals

**Goals:**
- One op table on each side (Python type rules, Rust validate/emit). An op is named once and
  dispatched on its operand dtype, and every kernel path uses that table.
- Width-carrying integer values that later families extend (int storage in `4x9.11`, integer
  combiners in reductions) by adding entries, not code paths.
- Eager-exact results wherever eager defines them, at Triton's instruction cost or within noise.

**Non-Goals:**
- A schema bump. Op names and dtypes are already strings in schemas 2–4. New dtypes and ops are
  additions, and old binaries reject them as unknown, which is the required behaviour.
- Removing the effects path's special validation of memory, indirect and RNG ops. Only its integer
  arithmetic moves.
- Vector-width changes. Integer values ride whatever physical plan their kernel's float/storage
  dtypes select. No new vectorization policy is added for integer data.

## Decisions

### 1. Op names are Inductor's, the dtype selects the instruction

The descriptor op is the Inductor op name (`add`, `bitwise_xor`, `floordiv`, ...). The validator
checks the operand and result dtypes against a per-op rule, and emission picks the CubeCL
instruction from the operand's element type. The comparisons already work this way for int64.

The internal spellings `iadd`, `imul`, `imin`, `imax`, `and` and `or` are replaced by
`add`, `mul`, `minimum`, `maximum`, `bitwise_and` and `bitwise_or` with int64/bool operands. Both the
Python emitters in `effects.py` and the Rust arms in `effects.rs` go.

*Alternative:* keep the internal names and add `isub`, `ixor`, ... beside them. That leaves two
vocabularies for one meaning and a validator per path, which is how the reduction gap arose.
*Cost:* corpus cases with indirect addressing or bool predicates change descriptor bytes, and so
artifact keys. The IR, CUDA and PTX must not change (spec requirement "Existing kernels keep their
generated code"). Group 2 does the rename alone so the manifest diff can be reviewed in isolation.

### 2. The dtype table

One table maps a dtype name to its element type, signedness, width and constant kind:

| dtype | CubeCL elem | signed | width | constant |
|---|---|---|---|---|
| `bool` | `Bool` | – | 1 | bool |
| `int8`, `int16`, `int32`, `int64` | `Int(I8..I64)` | yes | 8–64 | `i64` in range |
| `uint8` | `UInt(U8)` | no | 8 | `u64` in 0..=255 |

It lives in `ir.rs`, which grows `elem`/`constant` for the new widths and a `ConstantValue::UInt`
arm. Python mirrors it in a `dtypes` table in `semantic_checks.py`, which `semantics.DTYPES` and
`compiler/descriptor.py` (value dtypes, separate from storage `DTYPES`) import. The value dtype
set becomes `FLOATS + (bool,) + INTEGERS` on every path, replacing the effects path's
`(*VALUES, int32)`. Storage dtypes stay `FLOATS + (bool, int64)`.

### 3. Type rules, one table per side

Python gets `semantic_checks.OP_RULES`: op name → (operand class, result rule). The classes are
`float`, `int` (every integer width), `bool`, `intbool` (integers and bool) and `any`. The result
rules are `same`, `bool` and `float32`.

`node_reasons` types every op from the table:
- The `extra_ops`/`operation_type` hooks remain only for genuinely path-specific ops: `reduction`,
  `rand`, `load_seed`, `indirect`.
- `predicate_type` and the integer half of `rng_type` go away.
- `reduction_checks` uses the same table, which closes the fusion gap.

Rust gets `semantic.rs`'s `validate_ops`, which checks each op's arity and dtypes against the
same classes. Its float-only catch-all arm becomes the explicit `float` rule.

Operand promotion stays where it is. `CubeCLKernel.operation` promotes with
`torch.promote_types` and emits explicit `cast` ops, so every descriptor op sees same-dtype
operands, and Inductor's own int32 constants beside int64 operands are cast. The validator
requires equal operand dtypes (shifts included, since eager promotes both operands).

### 4. Emission per op

- **`add`, `sub`, `mul`, `neg` on integers:** `IAdd`, `ISub`, `IMul`, and `SNeg` (two's
  complement, so the same bits for uint8). Wrapping comes from computing at the value's own width.
  For int8/int16, the C++ promotion to `int` followed by assignment to an `int8_t`/`int16_t`
  variable truncates. The kernel-boundary test checks this per width rather than trusting it.
- **Comparisons:** the `S*` ops for signed, `U*` for uint8, and `BoolEqual`/`BoolNotEqual` for
  `eq`/`ne` on bool. `lt`, `le`, `gt` and `ge` on bool cast to uint8 and compare unsigned.
- **`maximum`/`minimum`:** `SMax`/`SMin` and `UMax`/`UMin`. On bool, `BitwiseOr`/`BitwiseAnd`.
- **Logical ops** (bool only): `logical_and`/`and_` and `logical_or`/`or_` are
  `BitwiseAnd`/`BitwiseOr` on bool. `logical_xor`/`xor` is `BoolNotEqual`. `logical_not` is
  `BoolNot`.
- **Bitwise ops:** `bitwise_and`, `or` and `xor` map to their CubeCL ops. `bitwise_not` is
  `BitwiseNot` on integers and `BoolNot` on bool, never `~` on a bool.
- **Shifts:** the eager guard. `in_range = (b as unsigned of the same width) < width`, then
  `select(in_range, a << b, 0)` for left shifts. Right shifts use
  `select(in_range, a >> b, a >> (width − 1))` for signed values and 0 for uint8. The unsigned
  compare folds eager's two conditions (negative, ≥ width) into one `setp`, so the cost is one
  compare and one select, which disappears next to the loads.
- **`truncdiv`, `mod`:** `SDiv`/`SRem` (`UDiv`/`URem` for uint8) on a divisor guarded with
  `select(b == 0, 1, b)`, as Triton does. The guard keeps NVRTC from exploiting division-by-zero UB
  when it can see a zero. The result for a zero divisor is then "a / 1", which is unspecified per
  the spec. The cost is one compare and one select before a 32/64-bit division routine.
- **`floordiv`:** eager's formula on the guarded divisor: `q = a / b`, `r = a % b`,
  `q − ((r != 0) & ((r < 0) != (b < 0)))`. One division pair compiles to a single divrem. Task 4.5
  confirms this in SASS against Triton's one-division sequence.
  - *Alternative:* Triton's complement trick (one division and no remainder). It is adopted only if
    the SASS or timing shows the remainder costs a second division routine.
  - The existing index `floor_div` keeps its positive-divisor form, because layout rules prove the
    divisor positive.
- **`int_truediv`:** emitted in Python as two `cast`s to float32 and a float `truediv`, the same
  ops Inductor's default (`ops.truediv`) produces for Triton. It needs no descriptor op, so the
  numerical policy for division (div.full, `division_rounding`) applies unchanged.
- **`to_dtype_bitcast`:** emitted in Python as Triton does. The value is converted to `src_dtype`
  when its compute dtype differs (rounding float32 compute to float16/bfloat16), then a descriptor
  `bitcast` op (`ReinterpretCast`, same size required), then a cast to the compute dtype when the
  target is float16/bfloat16. `bitcast` is the one new descriptor op that isn't an Inductor op name,
  because its semantics (same size, no rounding) differ from `to_dtype`.
- **Casts:** the existing `cast` op over the extended table. Float to integer and integer to integer
  use `CastOp`. The C++ backend emits a C cast. For int8/int16 that converts through int32 and
  truncates, as eager's `static_cast` does. Float to uint8 goes through int64, as eager's
  `static_cast_with_inter_type`. Measured 2026-10-25 against the recorded data: every NaN, ±inf and
  out-of-range conversion equals eager. Triton converts through int16 and saturates (its int8
  results can even exceed int8's range), so it differs there; the differences are listed in the test.
  - To bool is an explicit `ne(x, 0)`, not a cast, so NaN becomes True by definition and not by C++
    conversion rules. From bool is `CastOp`, which gives 0/1.
  - Float64 runtime scalars still cast only to floating compute.
- **`index_expr`/`value_expr` with an integer dtype:** the index is computed in int64 as today, then
  `cast` to the requested width. The effects path's int32 `index` arm goes.

### 5. Python handler surface

`CubeCLOps._default` stops listing ops by name. It routes every op in `OP_RULES` to
`V.kernel.operation`, so adding an op is one table row on each side, plus emission. Ops outside the
table keep the "semantic operation X unsupported" rejection, which `scoreboard/families.py` maps to
the op's family. The logical aliases (`and_`, `or_`, `xor`, `lshift`, `rshift`) normalize to
their `logical_*`/`bitwise_*` names before emission, so the descriptor has one name per meaning.

### 6. Tests and reference data

- **Contract module `test_integer_arithmetic.py`:** every op in the spec over every admitted dtype,
  with edge values (limits, 0, ±1, out-of-range shift amounts, all sign pairs, zero divisors), and
  bitwise equality with eager. Compiled in one graph per op family to keep the suite fast, with
  eager computed on the GPU.
- **Triton reference `integer_reference.py`** (Triton venv), the same pattern as
  `extrema_reference.py`: seeded inputs and recorded Triton outputs for shifts, float-to-integer
  casts of NaN and out-of-range values, and `int_truediv`. These are the cases where eager and
  Triton may differ, or where the spec defers to Triton. The test asserts the documented
  differences rather than skipping them.
- **Rust unit tests** in `semantic.rs`: rejection of mixed operand dtypes, `~` on bool, size-mismatched
  bitcast, out-of-range constants per width, and unknown dtypes.
- **Fusion tests:** `fmax(a, b).sum(-1)`, `fmax(a, b) * 2`, and a masked `where` inside a sum,
  asserting one kernel each, through the trace's kernel events.

### 7. Timing

The `perf` requirement uses the repository's CUDA-event method with clock warm-up, on the 3080
and confirmed on the 3090. It compares against the Triton venv on four graphs:
- a padding mask (`F.pad` constant, 2-D)
- an int64 floordiv
- a shift/xor mix
- an int8 wrap

Results go under `cubecl_torch/testing/results/integer-timing.json`, with a bead for any gap.

## Risks / Trade-offs

- **[Risk]** NVRTC folds UB in narrow-width or shift code that the eager guard was meant to define.
  → Every guard is an explicit select on the IR side, and the contract tests run the edge values on
  the GPU. Task 4.5 inspects one shift kernel's SASS.
- **[Risk]** The op rename changes more descriptors than expected, for example if the corpus uses
  int64 `add` in ways that now type differently. → The rename is its own group (2) with a manifest
  check that must show identical IR/CUDA/PTX for every changed case before anything else lands.
- **[Risk]** A promotion path in `operation` disagrees with Inductor's dtype propagation, for
  example uint8 with int8 (both promote to int16). → `operation` already uses `torch.promote_types`,
  the function Inductor's propagation uses. The mixed-dtype scenarios in the contract tests compare
  against eager.
- **[Trade-off]** `lt`/`gt` on bool cost a cast each. Inductor rarely emits them, so clarity wins
  over a special case.
- **[Trade-off]** The zero-divisor guard adds two instructions to every integer division where
  Inductor can't prove the divisor nonzero. Triton pays the same for floordiv and nothing for
  truncdiv. Kept for safety; the timing group shows whether it is measurable.

## Migration Plan

Not applicable at runtime: artifact keys include the compiler digest, so new binaries compile new
artifacts and old cache entries simply go unused. The structural manifest is re-recorded once,
after group 2's review, and again only if a later group changes an existing case, which it must not.
