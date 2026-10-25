# Proposal

## Why

The backend treats integers as index values and bool as predicates, and nothing more. Integer
values exist only as int64 index arithmetic (`index_expr`, and the internal `iadd`/`imul`/`imin`/
`imax` of indirect addressing), plus int32 as the RNG counter. Bool has only the `and`/`or` admitted
for Inductor's `fmax`/`fmin` decomposition, and only in pointwise kernels: the reduction path's
checks and its Rust validator lack them, so `fmax(a, b).sum()` splits into separate kernels.

Every other integer or bool operation is rejected. The 2026-10-25 census (`4z8.19`) finds the
`int` family in six of the nine model workloads:
- `and_` in cosyvoice3, minimax_h3 and wan22. This is the padding and pooling mask
  `ops.and_(start_cond, end_cond)` that Inductor's lowerings build.
- `logical_and` and `logical_not` in cosyvoice3, krea2, qwen_image21 and sdxl.
- `truncdiv` in cosyvoice3.
- Comparisons of bool or int32 values in krea2.
- int32 `index_expr` values in kohaku_moss_ops.

The `int` family is also the prerequisite of five others: rng (`4x9.5`), storage dtypes
(`4x9.11`), scan/sort (`4x9.13`), argmax/argmin (`4x9.8`) and any/prod/xor_sum (`4x9.9`). Building
integer values as a first-class value type now, with their width, lets those families add storage
and reductions without reworking the value layer. Integer and bool storage beyond today's
`bool`/`int64` is the dtypes family's job.

## What Changes

- **Integer value types with their width.** Values in a kernel may be bool, int8, int16, int32,
  int64 or uint8. Each keeps its width and signedness through arithmetic, as in eager and Triton,
  rather than being widened to int64.
  - These are the integer dtypes eager CUDA supports for arithmetic.
  - Values of these widths arise from constants, casts (`to_dtype`), `index_expr`/`value_expr` and
    loads of the storage dtypes already admitted (`bool`, `int64`).
  - Storage of int8/int16/int32/uint8 stays out of scope (`4x9.11`).
- **One op vocabulary, dispatched on dtype.** The semantic op layer the descriptor compiler shares
  across schemas 2–4 gains integer and bool forms of the existing ops, so pointwise, fused and
  reduction kernels all admit them. The op name stays Inductor's, and the operand dtype selects the
  instruction, as comparisons already do for int64.
  - The internal spellings `iadd`, `imul`, `imin`, `imax`, `and` and `or` fold into `add`, `mul`,
    `minimum`, `maximum`, `bitwise_and` and `bitwise_or`. The descriptors of affected corpus cases
    change; their IR, CUDA and PTX must not.
- **Integer forms of existing ops.** For integer operands: `add`, `sub`, `mul`, `neg`, `eq`, `ne`,
  `lt`, `le`, `gt`, `ge`, `where`, `maximum`, `minimum` and `masked`. Arithmetic wraps modulo 2^width,
  as eager CUDA and Triton do in practice. Comparisons accept bool and every integer width.
- **The int family's ops:**
  - logical: `logical_and`, `logical_or`, `logical_not`, `logical_xor`, `and_`, `or_`, `xor`
  - bitwise: `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`
  - shifts: `bitwise_left_shift`, `bitwise_right_shift`, `lshift`, `rshift`
  - division: `truncdiv`, `floordiv`, `int_truediv`, and integer `mod` (C remainder, emitted for
    integer `fmod`)
  - `to_dtype_bitcast`
- **Stated contract per op.** Each op is defined against eager CUDA PyTorch. Where eager defines
  more than Triton guarantees, the backend follows eager when it costs nothing measurable. The case
  in point is out-of-range shift amounts: eager returns 0 or the sign fill, while Triton emits a
  plain shift that LLVM treats as poison. Division by zero is unspecified in eager CUDA and stays
  unspecified here, but it must not trap or crash.
- **Casts.** Float↔integer, integer↔integer and bool conversions follow eager's rules (float to
  int truncates toward zero; narrowing wraps; to bool tests `!= 0`). Out-of-range float to int
  wraps through a wider integer as eager does (Triton saturates; the differences are recorded).
- **Coverage inventory.**
  - The family's 17 ops become implemented, each with a domain note.
  - The notes for `add`, `sub`, `mul`, `neg`, comparisons, `where`, `maximum`/`minimum` and
    `to_dtype` gain their integer domains.
  - Integer `mod` is split from float `mod`, which stays with `x3c`.
- **Scoreboard.** After the change, rerun the census inventory and record which `int` reasons are
  gone and what each region's next wall is.

Out of scope:
- int8/int16/int32/uint8 tensor storage, and float64 (`4x9.11`).
- Integer reductions (`sum`/`max`/`min` over integer inputs, found in cosyvoice3). They need the
  reduction framework's integer combiners and are filed as a follow-up bead.
- `abs` and `sign` for every dtype (`4x9.2`, `x3c`), and float `mod`/`fmod`/`remainder` (`x3c`).
- uint16/uint32/uint64, which eager CUDA supports for few ops.
- Op-level bool `add`/`sub`/`mul`/`neg`. Inductor never emits them: its lowering turns aten bool
  `add` into `logical_or` and bool `mul` into `logical_and` (both supported here), and bool `sub` is
  an eager error. The op table rejects them by name if they ever arrive.

## Capabilities

### New Capabilities
- `integer-arithmetic`: The value-level contract of integer and bool arithmetic in compiled
  kernels. It covers which value dtypes exist and how they arise, the integer forms of the shared
  ops, the logical/bitwise/shift/division ops and their edge-case semantics against eager and
  Triton, casts, admission in pointwise, fused and reduction kernels, and the requirement that the
  generated code of existing kernels stays unchanged.

### Modified Capabilities
None. `descriptor-compiler`'s schema rejection and determinism requirements, `inductor-backend`'s
observable rejection and manifest reproducibility, and `reduction-kernels`' admitted reduction set
are unchanged and must keep holding.

## Impact

- **Python front end:**
  - `cubecl_torch/inductor/semantics.py`, `effects.py`, `reductions.py`: the ops handler and
    `operation` typing for integer and bool values.
  - `semantic_checks.py`, `reduction_checks.py`: the admitted op set and type rules, one table shared
    by the pointwise, effects and reduction paths.
- **Compiler:**
  - `cubecl_torch/compiler/descriptor.py`: value dtypes.
  - `cubecl_torch/native/src/semantic.rs`: validation and emission of every integer/bool op.
  - `ir.rs`: element types and constants for every width.
  - `effects.rs`: drops its private integer ops.
  - `reduction/analysis.rs`: inherits the shared ops.
  - No CubeCL patch is expected. The IR already has the signed and unsigned arithmetic, shift,
    bitwise, division and remainder ops.
- **Tests:**
  - A new contract test module covering every op × dtype against eager, with edge values.
  - A Triton reference data file for the cases where eager and Triton can differ.
  - Rust validator tests.
  - Rejection tests for what stays out of scope.
- **Structural gate:** the manifest changes only in descriptor bytes of corpus cases that use the
  renamed internal ops. Their IR, CUDA and PTX stay identical, which is reviewed before re-recording.
- **Inventory and beads:** `coverage_inventory.py`/`.md`; close `4x9.1`; file the integer
  reduction follow-up; record the census rerun on `4z8.19`.
