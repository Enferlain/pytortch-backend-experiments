# Stage 5 follow-up: normal-overlay precision correction

## Result

**The constructor precision correction is installed in the normal experimental overlay.** It preserves FP16/BF16 tensor-storage rounding without rounding CUDA Python scalar operands as tensors. The installed PyTorch wheel, CubeCL compiler, physical plans and native launcher are unchanged. This follows the completed investigations in [STAGE5_ATTRIBUTION_REPORT.md](STAGE5_ATTRIBUTION_REPORT.md); no Stage 6 work was started.

The separate reused long-row sum discrepancy remains a numerical qualification gap. Its GPU results match a CPU model of the actual FP32 accumulation order exactly. No compiler wrong-result discrepancy is demonstrated for that input, but this does not make the unchanged eager-comparison tolerance pass. No tolerance, test input, reduction plan or capability was relaxed to hide it.

## Production change and ownership

`stage1/overlay.py:install` now installs two additional generic source substitutions from [stage1/precision.py](stage1/precision.py), alongside the existing selected-backend availability hook. The pinned wheel commit is still `5c4886908584029761b579af026dcfb627c84070`. Exact source anchors reject drift. Precision patch hunks use zero context; the serialized patch is independently applied with `git apply --unidiff-zero` to a temporary pinned-wheel source copy and must reproduce every overlay source hash. Only the regenerated local overlay is modified.

| Live Inductor owner | Correction | Reason |
|---|---|---|
| `index_propagation.py:SymPyOps.to_dtype` | Return `NotImplemented` for explicitly requested FP16/BF16 storage casts; align the default `use_compute_types=True` with the ops protocol | A `TypedExpr` dtype label does not perform low-precision rounding. Ordinary operation replay must see the requested conversion |
| `lowering.py:make_pointwise.inner_fn` | Exempt a scalar `BaseConstant`, including its existing view wrappers, from tensor-input precision emulation | Scalar promotion already owns operation/device-specific scalar policy. Tensor constructors remain pointwise producers and retain their storage barrier |

These are generic Inductor corrections, without backend-name conditions. They are plausible upstream candidates, not a claim of complete upstream-device conformance. CPU and configured Triton are regression neighbors; MPS and other targets are untested. No new dtype framework, CSE patch or backend constant workaround was added. The exemption relies on the current `promote_constants` representation; future constructor folding into `BaseConstant`, alternative scalar wrappers and explicit FP64→FP32 constant-rounding policy need separate validation. This fixes the configured FP16/BF16 barrier surface, not every floating conversion.

The inspectable [overlay patch](stage5/artifacts/precision/pytorch.patch) includes the four unchanged availability files and the two precision files. Hashes identify original/patched source. The semantic description still records explicit casts and compute/storage types; the existing bridge constructs CubeCL IR and the C++/NVRTC route compiles it with conservative FP settings.

`install(precision_casts=False)` is an explicitly experimental **legacy diagnostic mode**, used by `stage5.attribution` to reproduce its pre-fix baseline. Its optional precision probe reuses the normal correction definitions. Ordinary Stage 0–5 acceptance entry points use the default enabled correction. Legacy patch/hash evidence now goes to the ignored `stage1/artifacts/legacy-overlay` directory, preserving normal review artifacts. Historical attribution captures are retained; the new precision packet does not retroactively change their results.

## Regression matrix

[precision_checks.py](stage5/precision_checks.py) tests FP32, FP16 and BF16 with scalar operands in both orders; addition/subtraction; scalar and tensor comparisons; constructed, returned and pre-materialized tensors; explicit FP32→storage conversion; representable/nonrepresentable/subnormal-boundary/large fills; NaN/Inf/negative-zero fills; alpha neighbors; and arbitrary-gradient backward.

CubeCL passes **78 forward checks and nine backward checks**. Configured Triton passes **81 forward checks and nine backward checks**. Constructor, scalar and cast assertions are exact numeric equality with equal-NaN handling, dtype/classification and zero-sign checks. These do not require NaN payload identity. Three subtraction-with-alpha cases use the existing pointwise tolerances: eager CUDA uses fused alpha arithmetic, while this backend retains separate arithmetic. This existing numerical-policy limit is recorded rather than labeled exact. No ordinary FP contraction permission was enabled.

Explicit CUDA add-with-alpha reaches `ops.fma`; CubeCL deliberately rejects it at pre-fusion for all three dtypes. Tests assert zero additional CubeCL compiler invocations and native launches. Configured Triton executes that explicit operation and its results are checked separately. FMA coverage was not added to CubeCL.

CPU passes **81 forward checks and nine backward checks**; the tightened dynamic-reuse and negative compiler/launch-counter assertions also pass in the final CubeCL rerun. CPU add/subtraction-with-alpha cases use the same existing pointwise tolerances, while constructor/scalar/cast and backward checks remain exact. CPU alpha arithmetic needs its existing numerical qualification; this is separate from storage-barrier correctness. The initial CPU attempt lacked `openssl` on PATH. The installed `/nix/store/cgklhjl1llvv666r1rjycid4sl8xf92l-openssl-3.6.5-bin/bin/openssl` reports OpenSSL 3.6.5, and is supplied through a test-only PATH. No package or environment profile was installed/modified.

Dynamic constructor multiplication tests sizes `255,256,257,1048583,0,1,257`; nontrivial sizes reuse their dynamic graph, while 0/1 remain normally specialized. Nonempty calls execute one generated kernel and empty calls none. Three BF16 non-default-stream pressure trials reuse the existing current-stream/readiness/allocator lifetime path.

Six additional checks force fresh CubeCL IR/C++/NVRTC compilation through an isolated diagnostic cache partition, without deleting production cache entries or changing compiler options. They require six actual compiler invocations, six native launches, exact eager results, and PTX byte equality with the cached representative. Runtime symbolic extent slots are validated against these actual 1D descriptors and supplied separately; no static-ABI assumption is made. The retained representative files come from those fresh compiler directories.

Six warmed CubeCL profiler regions, covering scalar and constructed coefficients for all three dtypes, contain exactly one generated kernel each and match native counters. There is no pointwise ATen/Triton kernel in those regions. [Representative artifacts](stage5/artifacts/precision/representatives) preserve semantic casts, physical plans, CubeCL IR, CUDA C++, PTX and compiler manifests.

The original full **Stage 0–4 and FP hardening suites pass freshly**, including both Stage 0 C++/LLVM routes, Stage 1/1.5 boundaries, complete Stage 2 semantics/indexing/masks, Stage 3 reductions and Stage 4 effects/runtime boundaries. All Stage 5 controls and the **168-case CubeCL corpus pass freshly**. [Gate records](stage5/artifacts/precision/gates.json) and [new regression captures](stage5/artifacts/precision/regressions) retain these results independently of historical stage evidence. The corpus is correctness/census evidence here, not a new performance campaign. This run reused validated persistent artifacts: zero new Rust compiler invocations and 3,164 actual native launches. Freshly replayed descriptors and current runtime checks still determine which existing artifacts are eligible.

`precision_gates.py` saves preexisting eligible captures, runs fresh subprocesses sequentially, retains changed results under the new packet, and restores historical bytes. Its failure control proves nonzero exit propagation, retention of failed evidence, restoration of tracked/untracked captures and no source rewrite. The first complete gate run predated the untracked-capture preservation improvement: `launch/controls.json` was restored from its saved attribution copy; `launch/hook-checks.json` and the Stage 5 regression command summary were refreshed. No previous benchmark/Systems timing capture was overwritten.

CubeCL execution remains CPython **3.14.5t**, GIL-disabled with no `PYTHON_GIL` forcing, Triton unavailable/import-blocked and no loaded Triton Python/native module. The reference environment enables the GIL when loading Triton; no cross-environment host timing claim is made. Production remains PyTorch cu130/CUDA 13.0/NVRTC; no CUDA 13.1 offline diagnostic is substituted.

Conservative CubeCL FP patch hash stays `9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2`. No unsafe arithmetic identity, target-implied relaxed math, implicit FMA or LLVM `contract` flag was restored.

## Long-row reduction qualification

[reduction_order_checks.py](stage5/reduction_order_checks.py) models the actual existing scalar reduction:

```text
FP32 private accumulator t:
  a[t] = sequential FP32 additions of x[t + 256*k]
FP32 shared reduction:
  strides 128,64,32,16,8,4,2,1
```

The deterministic original history is retained: seed 6142, CUDA random tensors `64×31`, `64×257`, `8×257`, then `8×65539`, through one reused dynamic graph. Row 7 remains CubeCL `2.3123092651367188`, ATen FP32 `2.3122177124023438`, FP64 `2.312230602051727`. CPU emulation matches all eight GPU results exactly. The private chain has up to 257 additions before the shared tree. FP64 comparison locates a cancellation-sensitive accuracy limitation; the unchanged `rtol=atol=2e-5` eager gate still fails.

[PyTorch's numerical-accuracy guidance](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html) explains that floating-point operation order affects results and that mathematically equivalent executions need not be bit-identical. That is compatible with this observed order difference; it **does not waive this experiment's stated tolerance**. The current evidence establishes an accuracy/coverage gap, not a CubeCL rewrite or synchronization bug.

CPU-only component comparisons with two/four/eight/sixteen independent private chains shorten sequential accumulation and pass this particular recorded input. None is a GPU implementation or a validated replacement numerical policy. One input cannot establish general cancellation accuracy; picking a chain count because it passes would overfit. The numerical qualification packet keeps neighboring seeds, FP64 errors and special-value classification checks distinct from those proposed components.

`pytortch-backend-experiments-cwk` remains open for an evidence-backed numerical policy/correction before broader long-row tolerance claims. That work needs a stable summation decision tested against FP64, cancellation/large/tiny/special values and fused consumers; it must preserve explicit physical ownership and compiler FP constraints. No general reduction planner is justified by the earlier compile-history timing comparison.

## Review, provenance and remaining limits

Initial graph requests (`index_status`, `search_graph`, `check_index_coverage`) failed with `Transport closed`; exact affected source, callers and overlay/wheel bodies were read directly. The final coverage check reconnected (generation `2026-10-14T04:25:26Z`) and reported matching metadata with no recorded gap for the nine local implementation files; this remains a best-effort signal. No graph-completeness claim is made. Required review completed with no semantic correctness blocker. Its mutable-state patch-check, legacy evidence overwrite, diagnostic failure-retention and duplicated file-list findings were addressed. Post-review checks require all six normal overlay files, preserve normal evidence across a full legacy probe, test failure retention, and reapply the serialized patch exactly. The CubeCL matrix/cold compiler checks and reduction qualification were rerun. [Review disposition](stage5/artifacts/precision/review-disposition.md) and final validation hashes record the checks.

Early matrix attempts are explicitly failed: an unjustified exact alpha expectation; default-argument floating fixtures lifted by Dynamo into out-of-contract CPU scalar tensors; and a fixture-evaluation namespace error. They were corrected in the harness, not made accepted by weakening constructor/scalar assertions. CPU infrastructure/alpha attempts are separately retained. A fresh-compiler probe initially assumed a static representative ABI and failed on the actual symbolic extent; the corrected probe validates and supplies the declared extent slot. This failed diagnostic remains in the attempt ledger. The final fixtures use actual authored scalar literals and continue to reject unsupported CUDA scalar-tensor inputs.

No upstream checkout replacement, source publication, commit, push or Dolt remote sync was performed. The working tree also contains prior warm-dispatch work and user-added skills/instruction edits; those were preserved.

**Next direction: qualify and improve the remaining long-row FP32 accumulation accuracy.** The constructor precision problem is resolved on the executed neighboring surface. Stage 6 remains deferred while `cwk` is unresolved. No next stage is started automatically.

Reproduction (fresh processes, sequential overlay installation):

```bash
export CUDA_PATH="$PWD/stage0/cuda13-toolkit/nvidia/cu13"
export STAGE0_CARGO_PROFILE=release
stage0/.venv/bin/python -m stage5.precision_checks --output stage5/artifacts/precision/cubecl.json
stage0/.venv/bin/python -m stage5.precision_gates
stage0/.venv/bin/python -m stage5.precision_harness_checks
stage0/.venv/bin/python -m stage5.reduction_order_checks
# Run configured reference probes separately with the reference interpreter,
# TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib and the recorded libdevice path.
# CPU also needs the verified OpenSSL directory on its process PATH.
stage0/.venv/bin/python -m stage5.precision_curate
```

Use a new output/capture directory or preserve this packet before rerunning the matrix. Reference commands use `--impl triton` or `--impl cpu` and their own `--output`; curation expects the three canonical result files.
