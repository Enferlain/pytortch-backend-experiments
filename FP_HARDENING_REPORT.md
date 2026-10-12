# CubeCL floating-point hardening before Stage 3

12 October 2026, Europe/Budapest. Task `pytortch-backend-experiments-e30`.

## Result and scope

**The current Stage 0–2 CUDA path is hardened against the confirmed audit defects.**
The patched compiler passes seven IR test groups, 100 generated runtime cases,
and real Dynamo/AOTAutograd contraction-policy tests for FP32/FP16/BF16.
The established Stage 2 semantic surface remains unchanged. No reduction work,
new Inductor remainder/extrema operators, vectorization, or relaxed-math mode
was implemented.

All five supplied reports were read in full: the three individual audits,
`cubecl_audit_condense.md`, and `cubecl_fp_final_synthesis.md`. They describe the
same CubeCL pin used here: `6bba5fdf04206d9c3710dfbc46aa9850450ab405`.
The original four-fold Stage 2 patch was a partial fix. The complete local stack
now also removes the remaining remainder defects and replaces the negative-zero
addition identity with no partial add-zero simplification.

This is a tested operation/target/dtype policy, not a strict IEEE-754 or universal
cross-backend bit-identity claim. NaN payloads/signs, signaling NaNs, exception
flags/traps, all rounding modes and universal FTZ/subnormal behavior remain
outside the contract. NaN classification and required zero signs are checked;
ordinary division uses the documented `rtol=2e-6, atol=0` comparison.

## Findings and local fixes

| Audit concern | Current relevance | Disposition |
| --- | --- | --- |
| FAdd zero identities | Stage 2 arithmetic and signed zero | Remove the entire simplifier, including Stage 2's negative-zero variant; retain both-constant evaluation |
| FSub self-subtraction | Stage 2 `sub`, NaN/infinity | Remove runtime `x-x` constant inference; retain actual constant subtraction and positive-zero-only `x-+0` |
| FMul zero annihilation | Stage 2 `mul`, backward | Remove partial zero inference; retain constant multiplication and multiply-by-one |
| FDiv zero numerator | Stage 2 division | Remove partial zero inference; retain constant division and divide-by-one |
| FRem zero numerator and `%1 -> x` | Confirmed compiler defect; operation not accepted by Inductor backend | Remove both rules, keeping true constant truncating remainder |
| SRem/URem `%1 -> x` | SRem is used by signed-index floor division | Remove duplicate identity simplifiers; retain SCCP's correct by-one-to-zero folds |
| FModFloor zero numerator | Compiler defect; floating operation not accepted by Inductor backend | Remove partial inference; retain its actual constant formula |
| Implicit FMA combine | Reached by ordinary Stage 2 multiply/add/subtract | Default to no permission; an explicit pass option is required to contract |
| LLVM GPU `contract` flags | Verified Stage 0 LLVM alternative | Ordinary add/sub/multiply now use empty fast-math flags |
| C++ float constant `-0` emitted as integer `-0` | Newly exposed by runtime tests after removing add-zero folding | Integral float spellings get `.0`, preserving negative zero and avoiding oversized integer literals |
| LLVM NVPTX `frem` expansion loses negative zero | Newly exposed by runtime remainder probe | Replace NVPTX remainder instructions with existing libdevice `fmod` calls before LLVM optimization/code emission |
| Per-operation FastMath provenance missing | No relaxed mode is exposed by this frontend | Keep conservative behavior; track restoration before enabling relaxed assumptions |
| Extrema and explicit FmaOp contracts | Unsupported semantic operations in current backend | Keep capability gates; track operation-specific work before accepting them |

The synthesis correctly warns against restoring float `%1 -> 0`; it is also
wrong for ordinary fractional dividends. No such rule is added. The older audits'
specific historical NSZ-removal claim is corrected by the final synthesis and
is not used as evidence for this patch. No new history/maintainer-intent claim
is inferred from our execution tests.

## Actual compiler paths

Production capture remains:

```text
torch.compile → Dynamo/AOTAutograd → Inductor scheduler/replay
→ typed Stage 2 semantic description + PhysicalKernelPlan
→ existing Rust builder → CppCompiler<Cuda> → NVRTC (--fmad=false)
→ PTX → existing native launcher → PyTorch-owned tensors/current stream
```

Source verification confirms the live SCCP/Simplify pipelines in
`cubecl-cpp/src/shared/base.rs` and `cubecl-llvm/src/shared/base.rs`.
`SimplifyOpsPass` invokes `SimplifyInterface::check_fold`; SCCP invokes partial
`ConstFoldInterface` folds even when operands are unknown. Tests therefore return
runtime-unknown results rather than relying on fully constant special-value cases.

The new audit binary builds fixed KernelBuilder/Scope fixtures directly, so
unsupported frontend operations can be tested without weakening capability checks.
The execution seam is the same Stage 0 native enqueue after explicit probe ABI
validation. The probe contains CubeCL IR operations, not handwritten CUDA.

The LLVM remainder correction sits at its existing target-library boundary:

```text
FRemOp → LLVM frem → NVPTX lower_to_device_libs
→ redirect_frem using existing MathLibrary replacement/wrapper machinery
→ __nv_fmodf / __nv_fmod → existing link_libdevice → LLVM O3 → PTX
```

The emitted pre-codegen `kernel.ll` still shows `frem`; the runtime PTX is built
after redirection/linking. The C++ target emits its existing `fmod` call.
The new LLVM helper reuses existing widening/vector library wrappers, but only
NVPTX scalar FP32 runtime remainder was exercised here. LLVM FP16/BF16 remainder,
vector remainder, CPU/AMDGPU/shader execution are not claimed.

## Source changes and upstream suitability

The full pinned [patch](fp_hardening/cubecl-fp.patch) changes six CubeCL files.
The [upgrade patch](fp_hardening/cubecl-fp-upgrade.patch) composes with the
original Stage 2 patch to yield byte-identical source to the full patch.
Both application paths were checked in temporary directories.

| File / owner | Old responsibility | New responsibility / suitability |
| --- | --- | --- |
| `cubecl-ir/src/dialect/math.rs` | Unconditionally apply assumption-dependent arithmetic identities | Remove unsound partial rules; keep valid constant evaluation and ordinary safe identities. Generic upstream correctness fix. Remove now-unused private float materialization helper |
| `cubecl-opt/src/passes/inst_combine.rs` | Always contract multiply-plus/minus | Default permission false; explicit `allow_fp_contraction` enables this pass. Generic conservative fix; the pass-wide option is a local bridge until per-operation provenance exists |
| `cubecl-llvm/src/shared/to_llvm/math.rs` | Implicitly permit contraction on GPU ordinary arithmetic | Empty default flags on ordinary add/sub/multiply. Generic safe default; upstream should reconcile explicit permissions rather than infer them from target |
| `cubecl-cpp/src/shared/value.rs` | Print integral float text as a C++ integer token | Emit floating tokens with `.0` where necessary. Generic emitter correctness fix |
| `cubecl-llvm/src/shared/math_library.rs` | Redirect library-backed intrinsic calls and handle promotion/vector wrappers | Also replace floating remainder instructions through that existing machinery. Shared implementation, invoked only by NVPTX here |
| `cubecl-llvm/src/nvptx/codegen.rs` | Link libdevice only when intrinsic redirection needs it | Also link when remainder redirection needs it. Target-specific correctness fix, preserving existing compiler/library design |

No PyTorch source changes, launcher changes, tensor-copy adaptation or new runtime
ABI were needed. The original `stage2/cubecl-ieee.patch` remains historical
evidence; `stage2/setup.sh` now accepts/installs the exact full audited stack.
Unexpected local source, staged or untracked changes reject instead of being
overwritten. `stage0/Cargo.toml` adds the existing optimizer as a test dependency
and keeps `default-run = "cubecl-stage0"` after adding the probe binary.

## Evidence

### IR

[IR results](fp_hardening/artifacts/ir-tests.json): seven test groups pass.
They include 272 unknown-operand combinations across f16/BF16/f32/f64 and
Simplify-only, SCCP-only, and both pass orders; returned SSA identity/type checks;
retained safe identities; correct signed/unsigned integer `%1`; fully constant
fractional remainder/floor-mod controls; a control-flow merge; explicit contraction
permission; and floating literal sign/type checks.
No vector/unroll regression claim is made: our physical plan remains scalar.

### Actual CUDA execution

[Runtime results](fp_hardening/artifacts/results.json): 100 compiler invocations,
100 case/dtype/route checks, 114 matching Python/native kernel launches, zero
failures. C++ executes 24 floating fixtures for each of FP32/FP16/BF16 plus two
integer fixtures. LLVM executes the 24 FP32 fixtures plus two integer fixtures.
Additional calls test 0/1/255/256/257 tails, non-default streams and profiled launch
count. Profilers see exactly one generated `cubecl_fp_*` kernel per warmed call.

Literal and loaded operands cover NaN, ±infinity, ±zero, positive/negative ordinary
values, fractional divisors, large quotients and sampled tiny values. Tests use
NaN classification and exact zero-sign checks, without asserting payloads. The
counterexamples `(1+2^-23)*(1-2^-23)-1` and its add/reverse-subtract forms produce
exact separate-rounding zero, rather than a fused residual. Their source/PTX
contains no implicit FMA; LLVM ordinary arithmetic carries no contract flag.

[Inductor evidence](fp_hardening/artifacts/inductor.json): all three storage dtypes
produce one forward CubeCL kernel for multiply/subtract, match eager exact-zero
rounding and arbitrary-upstream-gradient backward, and retain lowp precision-cast
barriers. The backward artifact is reused for the two gradient outputs. Six
compiler invocations and twelve native launches are observed. Actual fmod and
remainder graphs reject at pre-fusion with zero compiler invocations.

### Important compatibility limit

Both compiler routes execute the CubeCL floor formula for float32 `1e10, 3` and
produce **0**, while eager `torch.remainder` produces **1**. The audit fix preserves
the IR's existing formula; it does not make that formula a PyTorch implementation.
This does not enter the supported graph surface: floating `fmod` and `remainder`
remain capability failures. Integer floor/mod indexing is a different operation
family and its existing wide-index tests still pass.

Nine [representative artifacts](fp_hardening/artifacts/representatives) retain
fixture/policy descriptions, CubeCL IR, CUDA or LLVM source, PTX, manifests and
NVRTC options where applicable. Full per-case outputs, raw profiler/log files and
upstream source snapshots are local and ignored. No unmodified upstream checkout
files are eligible for commit.

## Baseline regressions and independence

Stage 0 C++ and LLVM acceptance suites, Stage 1 acceptance, Stage 1.5 boundary
checks, the complete Stage 2 suite and descriptor components pass. The Stage 2
matrix remains 34 checks / 94 compiler invocations / 165 native launches.
They preserve required tails, forward/backward, dynamic/strided/broadcast/masked
fixtures, normal multi-node fusion, stream/lifetime and allocator-pressure tests.
The existing non-CUDA HIP source probe is rerun as compiler-structure evidence,
not GPU or launcher portability.

Triton package availability remains absent, optional import attempts are blocked,
no Triton Python/native module is loaded, and fresh CPython 3.14.5t processes
remain GIL-disabled with no `PYTHON_GIL` forcing. The audit runtime process has
three availability probes, one blocked optional import and three false GIL
checkpoints. The Inductor process records its actual additional optional imports.
This is not a concurrent compiler or universal free-threading safety claim.

Clippy (all targets, warnings-as-errors), rustfmt, Ruff and Python syntax checks
pass. The required review confirmed the tested behavior and found no wrong-result
bug. Small function-ownership/signature/type guards and manifest/fixture clarifications
were applied and affected checks rerun. [Review disposition](fp_hardening/artifacts/review-disposition.md)
records each finding. Active manifest patch hashes identify the full live patch;
the original Stage 2 patch hash is explicitly labeled historical.

Graph verification began at Tier 2. The upstream clone is deliberately excluded,
so affected compiler files and relevant pass/flag/library context were read
directly. The final graph-coverage call failed with `Transport closed`; new local
files were checked directly instead. No graph-completeness claim is made.

## Remaining boundaries

`pytortch-backend-experiments-1tc` tracks per-operation FastMath propagation before
relaxed modes are enabled. `pytortch-backend-experiments-x3c` tracks PyTorch
remainder/extrema contracts before those operators are accepted, including the
explicit FmaOp constant/lowering consistency question. These issues do not change
the current tested Stage 2 contract, which does not expose those semantics.

Stage 3 has not started. A later reduction investigation should begin with the
tested conservative numerical policy and explicitly choose any different policy;
it must not infer fast-math permission from GPU capability or revive these folds.
