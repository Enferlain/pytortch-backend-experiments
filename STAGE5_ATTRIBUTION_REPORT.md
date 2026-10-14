# Stage 5 follow-up — history, host attribution and constructor precision

The subsequent [precision correction report](STAGE5_PRECISION_REPORT.md) records installation into the normal overlay and fresh neighboring/full regression checks. The candidate-only status below describes this earlier capture; historical attribution now explicitly requests the legacy overlay mode.

## Result

**The three bounded investigations are complete.** Matched compile histories explain the saved split/unsplit reduction comparison. Nsight Systems identifies substantial framework/wrapper and stream/dispatch costs without establishing a new removable mechanism or a WSL cause. Tensor-constant triage identifies two generic Inductor precision problems; an isolated two-file correction passes the selected variants on CubeCL and configured Triton. **That correction is a counterfactual candidate, not an installed production fix.** No Stage 6 work, new reduction planner, host-runtime rewrite, compiler replacement or expanded operation coverage was implemented.

Evidence is under [stage5/artifacts/attribution](stage5/artifacts/attribution). Raw Chrome traces, `.nsys-rep`, SQLite databases and logs stay in the ignored `stage5/cache/attribution` directory. [The manifest](stage5/artifacts/attribution/evidence-manifest.json) records tool/source identities and hashes of curated descriptor/IR/CUDA/PTX packets. Observations below are fresh execution unless explicitly qualified as prior evidence or inference.

## Reduction compile history

Each experiment runs in a fresh process with **one stable backend callable and one `torch.compile(..., dynamic=True, fullgraph=True)` function**. Histories are symmetric for both backends:

- Large first: compile and execute `8×65539` directly.
- Short first: execute `64×31`, `64×257`, `8×257`, then `8×65539` through that same compiled function.

Input RNG is reset before each shape so both histories use identical target values. Every controlled target satisfies the unchanged FP32 `rtol=atol=2e-5` comparison. The first exploratory short-history attempt consumed different random inputs and failed that tolerance on one cancellation-sensitive output. Matching inputs fixes the history experiment, but does **not** erase that numerical counterexample; no tolerance was increased.

The diagnostic wraps the live `InductorChoices.reduction_split_factor` without changing its result and records backend graph invocations, scheduling/replay, selected artifacts and five separate CUPTI censuses. Unprofiled host/event samples precede those profiler regions.

| Backend/history | New graph at target | Split choice | Target GPU kernels | Median summed kernel duration, us | Host enqueue median, us |
|---|---:|---|---:|---:|---:|
| CubeCL large first |1|9 partials per row|2|6.308|347.031|
| CubeCL short first |0|Original unsplit graph reused|1|11.571|223.956|
| Configured Triton large first |1|9 partials per row|2|6.279|193.543|
| Configured Triton short first |0|Original unsplit graph reused|1|209.119|197.254|

Sources: the four `history-*-*_first.json` files; target kernel medians are recalculated from their five `profiles` arrays. These profiler durations are **not** unprofiled CUDA-event intervals. Host comparisons with Triton are environment-qualified: its native module enables the GIL, while CubeCL remains GIL-disabled. No Triton parity claim follows from two nearly equal split-kernel medians.

The live owner is `torch/_inductor/ir.py:Reduction.num_splits` and `choices.py:InductorChoices.reduction_split_factor` at PyTorch commit `5c4886908584029761b579af026dcfb627c84070`. The short initial reduction hint, 31, hits the early small-reduction case; a split-factor call is unnecessary. Dynamic graph reuse later runs the same unsplit topology with a larger runtime extent. Large-first lowering calls the existing heuristic with outer hint 8/reduction hint 65539, obtains split 9, then lowers the partial reduction with outer hint 72/reduction hint 7283.

The existing Inductor multilayer reduction and CubeCL bridge execute this correctly, using PyTorch-owned intermediate buffers. [Launch records](stage5/artifacts/attribution/reduction-launches.json) retain actual grids/blocks from Chrome kernel events; [representatives](stage5/artifacts/attribution/representatives) retain both CubeCL paths and Triton generated wrappers. No scheduler threshold or fusion rule was forced.

In these fresh dynamic captures, CubeCL's split passes launch 72 workgroups of 256 units, then two workgroups with dimensions `[32,4,1]`; Triton's split passes launch 72 blocks of 512 threads, then eight blocks of 64. Unsplit CubeCL launches eight 256-unit workgroups; reused Triton launches four 64-thread blocks with an iterative reduction loop. Those are the measured topologies, not assumptions copied from the earlier reference artifacts.

**Decision:** close this reduction investigation without adding a planner. The previous saved timings describe different compile histories, not an isolated physical-backend limitation. The finite topology's adaptation to a reused dynamic graph remains a performance qualification. A future benchmark must specify shared reuse history or intentionally specialize per shape; it must not create a fresh Triton backend closure while silently reusing a CubeCL graph.

This closes **compile-history attribution**, not the separate numerical qualification. A focused replay of the original sequential RNG history reproduces the failure: row 7 is CubeCL `2.3123092651`, ATen FP32 `2.3122177124`, and FP64 `2.3122306021`. CubeCL/ATen differ by `9.1552734375e-5`; their errors against FP64 are about `7.87e-5` and `1.29e-5`. [The counterexample](stage5/artifacts/attribution/cancellation-counterexample.json) retains all rows, exact seed/history, artifacts and the explicit failing gate. It is consistent with different accumulation order, but that is not a proof that the stated tolerance/coverage is adequate. `pytortch-backend-experiments-cwk` tracks the unresolved numerical qualification. No new planner, tolerance widening or compiler workaround is inferred from it.

## Host attribution

Linux Nsight Systems **2026.5.1.161-265138896106v0** was resolved from PATH and its entrypoint hashed. Production remains PyTorch cu130/CUDA 13.0 NVRTC and the existing driver JIT. No Windows offline binary utility or Nsight Compute measurement is substituted for this experiment.

Capture options, checked against installed help and NVIDIA's [focused profiling guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), are:

```text
--trace=cuda,nvtx,cublas --sample=none --cpuctxsw=none
--capture-range=nvtx --nvtx-capture=stage5_capture
--capture-range-end=stop
NSYS_NVTX_PROFILER_REGISTER_ONLY=0
```

PyTorch's ordinary NVTX strings require that last setting. The initial omitted-setting attempt generated no report despite a successful application exit; it is excluded. The runner now requires an actual `.nsys-rep`. Direct NVTX scopes distinguish indexed steps, forward/backward/optimizer, generated dispatch, reset and completion. Reset/completion are outside each measured step. Compiler/physical selection, all metadata keys, backward and optimizer are warm first, and CubeCL asserts zero warm compilation/tuning.

There are **19 accepted captures**: three independent 50-step captures per current/frozen CubeCL and tiny/larger workload; one capture per eager/configured Triton and workload; two 20-step deep captures; one targeted all-API deep capture. No perf, cProfile or torch.profiler runs inside these Systems captures. CUDA/API/kernel and NVTX statistics come from `nsys stats`; explicit SQLite queries assign events to steps by API correlation and NVTX timestamps. Raw commands and report/database hashes are preserved in [systems-analysis.json](stage5/artifacts/attribution/systems-analysis.json), with readable `systems-*-stats.txt` outputs.

### Workloads and comparison controls

The tiny graph is the existing Stage 4 embedding → linear → RMS → dropout → squared-mean loss with SGD outside compilation. Larger keeps that authored family but uses embedding weights `[4096,128]`, a bias-free linear `[256,128]`, and 256 indices. It exercises more vendor/generated work without adding an operator.

Current/frozen CubeCL dispatch uses the same locally captured graph, and independently generated PTX/selected plans are asserted equal within each comparison process. The frozen source remains Git `df4320d093a76a49b7ab33d52cf10d22a9498bd0`. RNG replay, outputs, gradients and updated parameters are checked after resets. The larger diagnostic uses the existing FP32 reduction tolerance, not a new mathematical coverage claim. Eager/Triton repeatability is checked within each backend; stochastic outputs are not asserted equal between different RNG implementations. Their reference environment enables the GIL.

Separate unprofiled measurements have three rounds of 100 samples per implementation/workload. They are independent-process observations, **not** another rotated, same-process causal ablation:

| Implementation | Tiny host enqueue, us | Larger host enqueue, us | GIL |
|---|---:|---:|---|
| Current CubeCL |3470.336|3627.770|disabled|
| Frozen CubeCL |3289.865|4626.827|disabled|
| Eager reference environment |2597.235|2123.980|enabled|
| Configured Triton |1684.160|1801.467|enabled|

The tiny comparison does not reproduce a stable improvement; the larger observation is favorable. These do not replace or erase the qualified prior paired launch capture. Samples/rounds are in `host-*.json`. A direct replay of the same generated dispatches with their live buffers, validation and current-stream/lifetime handling costs roughly 0.95–0.96 ms for current CubeCL. It omits GEMMs, optimizer and graph/autograd plumbing, and reuses intermediate buffers: **it is a dispatch diagnostic, not equivalent complete training, an eager fallback, or an additive savings estimate.** Scoped references are cleared on exit.

### Timeline evidence

| Current CubeCL | Profiled step medians across three runs, us | Generated dispatch union medians, us | Kernels per step |
|---|---|---|---|
| Tiny |4266.8 / 4780.8 / 4864.4|1397.8 / 1508.5 / 1575.5|20 total / 8 generated|
| Larger |4818.2 / 4644.3 / 4870.9|1575.2 / 1560.1 / 1706.6|22 total / 9 generated|

All steps have the exact expected census; generated counts match the shared native counter. Remaining wall time lies outside generated-dispatch ranges, in captured graph/framework/autograd/vendor/optimizer work or unlabelled execution/waiting. With sampling/context-switch collection disabled, the trace does **not** distinguish CPU execution from descheduling in that remainder.

No device/stream synchronization API is observed inside measured steps. Harness synchronization and reset copies appear outside them. Tiny current generated driver launches have a median around 27–29 us per API call in the default captures. Kernel queue intervals are separately recorded and nonzero; they are not assigned to Python, ctypes or WSL. Device durations/clock-state variation and profiling perturbation remain visible in the retained samples.

The all-API probe exposes many short device/context queries: 4460 `cudaGetDevice`, 1520 `cuDevicePrimaryCtxGetState`, 220 `cuCtxGetCurrent`, 320 capture queries and 160 generated driver launches over 20 steps. Their summed times per step are respectively approximately 141.1, 27.6, 4.9, 18.5 and 295.7 us in this separately instrumented capture. Many queries also belong to framework/vendor calls. Counts alone do not authorize caching or removing them.

Deep NVTX scopes separate lock acquisition, selection, proof signature, validation, metadata, current stream/capture queries, `record_stream`, binding and native enqueue. Exclusive range accounting subtracts descendant interval unions on the same thread. In the all-API deep run, illustrative exclusive totals per step include validation 189.0 us, signature 102.9 us, metadata 19.9 us, current-stream queries 128.7 us, `record_stream` 145.5 us and native enqueue 410.7 us. The native range includes ctypes/native work plus driver API and instrumentation; it is not pure CUDA time. Deep capture has material overhead and these numbers cannot be combined with unprofiled medians.

cuBLAS library calls are exported as NVTX ranges in this installation, rather than a separate `cublas_api_sum` report. Tiny default captures show about 31–39 us/step in heuristic queries and 139–166 us/step in matmul ranges. This is useful vendor attribution, not proof that those costs are removable by the CubeCL adapter.

**Decision:** close host attribution after the initial captures and the one all-API follow-up. There is no demonstrated new removable mechanism meeting the larger-workload criterion. Preserve validation, readiness, allocator protection and locking. The evidence does not require another launcher/binding rewrite, and cannot establish a WSL-specific cause. Framework/wrapper/optimizer work, necessary stream handling and launch count are now more concrete future investigation boundaries.

## Tensor-constant semantic decision

The governing distinction is storage materialization versus scalar opmath. [PyTorch's `full_like` contract](https://docs.pytorch.org/docs/2.14/generated/torch.full_like.html) returns the input dtype by default. The pinned `config.emulate_precision_casts` comment promises retained low-precision rounding boundaries; ordinary higher-precision compiled fusion does not resolve that enabled-policy obligation. Triton agreement is a comparison, not a semantic oracle.

Six variants are executed for FP32/FP16/BF16, with finite/random values, ±0, NaN and ±Inf:

| Variant | Baseline FP32 | Baseline FP16/BF16 | Isolated generic correction |
|---|---|---|---|
| Python scalar literal × x |exact|exact|exact|
| Fused `full_like` × x |exact|differs|exact|
| Return constructor and its product |exact|constructor exact; product differs|both exact|
| Already materialized constructor input × x |exact|exact|exact|
| FP32 constructor → explicit storage cast → product |exact|differs|exact|
| Constructor returned alone |exact|exact|exact|

“Exact” here means numeric equality with equal-NaN comparison, dtype, classification and zero sign checks; no NaN payload identity claim. All **18 variants per backend** pass those exact checks in the corrected probe. Baseline/probe results, post-AOT pre/post-pass graphs, metadata, lowering calls and symbolic cast logs are in `constants-{cubecl,triton}{,-probe}.json`.

For FP16 `x=65504`, eager constructor multiplication yields 21824; the baseline compiled constructor yields 21840. A Python scalar literal correctly yields 21840. Returning the constructor does not force its fused consumer to reload rounded storage. BF16 also differs over the recorded finite inputs even though that particular large-value example rounds to the same final result.

### Earliest concrete boundaries

1. `torch/fx/experimental/proxy_tensor.py:_maybe_record_pointwise_barrier` annotates the constructor, mul and explicit cast appropriately. Those annotations and the explicit cast node survive AOT and post-grad passes in the captured graphs.
2. `lowering.py:_full` produces a constant with the tensor's semantic dtype. It does not itself insert a storage cast, but `make_pointwise` **already requests** a downcast/upcast for the low-precision tensor input of mul. Therefore the initial theory that adding a constructor-specific barrier is required is incomplete.
3. `index_propagation.py:SymPyOps.to_dtype` returns `TypedExpr(value.expr, dtype)`. Float constants are converted to a Python `float`, not rounded to FP16/BF16. `LoopBodyBlock.__init__` installs `IndexPropagation`, which folds those requests before backend replay. The descriptor therefore receives an unrounded FP32 compute constant with low-precision semantic dtype. No independent CubeCL IR/device-code defect is demonstrated here.
4. Simply vetoing that fold exposes another shared problem: `make_pointwise` also requests tensor-input truncation for a Python scalar represented by an expanded `ir.Constant`. The existing fold had masked this wrong scalar rounding request. The naive veto fails 77/266 FP16 scalar-literal elements, including 21824 versus correct 21840. [The failed component evidence](stage5/artifacts/attribution/naive-cast-veto.failed.json) is explicitly unaccepted.

Provenance is thus still available at the lowering boundary: a scalar's view unwraps to `ir.BaseConstant`, while a constructed tensor is a `TensorBox`/pointwise producer. It is lost when propagation reduces both to the same typed symbolic constant. Backend constant handlers alone cannot recover it.

### Small tested candidate

[precision_probe.py](stage5/precision_probe.py) applies two exact source substitutions **only in the regenerated Stage 1 overlay, before importing Inductor**:

- `make_pointwise`: skip tensor-input emulation for `BaseConstant` after unwrapping existing views; preserve tensor-constructor input rounding. Existing comparison/scalar promotion remains responsible for callers that intentionally round scalar values.
- `SymPyOps.to_dtype`: decline symbolic folding of an explicitly requested FP16/BF16 storage conversion; let normal backend replay emit the cast. Align the default `use_compute_types=True` with the existing ops protocol.

No backend-name rule is introduced, scalar literals retain opmath precision, and signed integer symbolic propagation is unchanged. The selected constructor remains fused in one generated CubeCL kernel with explicit cast nodes; [baseline/probe packets](stage5/artifacts/attribution/representatives) preserve descriptor → CubeCL IR → CUDA C++ → PTX evidence. Normal overlay installation removes the experiment. The candidate `.patch` files are inspectable; the live wheel/local PyTorch checkout is untouched.

**Decision:** classify the constructor discrepancy as a generic Inductor precision-emulation defect with an executed, bounded candidate correction. Do not waive it through Triton agreement or round every scalar. The probe is not yet production coverage for every `BaseConstant` consumer, alpha/comparison variant, fill value, CPU/MPS backend or backward graph. Land a focused source fix with those neighboring regression checks before broader low-precision fusion claims.

The Triton baseline uses the live wheel, while its precision probe also installs the Stage 1 generic scheduler-availability overlay to load the two-file candidate. Those extra scheduling edits are not a numerical policy change, but this is not a pure single-variable Triton comparison. CubeCL baseline and probe both use the same Stage 1 overlay. The failed probe ledger records two naive-veto executions (the second added retained failure evidence) and one candidate that checked only bare `BaseConstant`, missing its `ExpandView`. All three fail before the accepted view-unwrapping candidate. One reference-harness attempt also failed because the frozen runtime lacked a Python launch counter; the shared native counter fixes that diagnostic compatibility issue.

## Validation, source changes and limits

Only new diagnostic/analysis/curation scripts, this report and the Stage 5 evidence ignore exception are permanent changes for this task. The two generic Inductor edits are isolated probe source, not production overlay policy. No production Python dispatch, PyTorch wheel, Rust, CubeCL, LLVM, NVRTC options or native launcher changed. The four production adapter hashes match the preceding launch capture, and conservative FP patch hash remains `9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2`.

Fresh validation includes the symmetric histories, baseline and corrected semantic variants, all 19 Systems captures and exact native censuses, evidence consistency checks, and the existing warm launch controls with the normal overlay restored. Warm controls revalidate tails/zero grid, bounded layout LRU, live capacity/dtype/layout/alias/alignment rejection, non-default streams, allocator pressure, scoped tracing, two free-threaded workers and prepared-scope release. Their fixed-path `launch/controls.json` was refreshed by that existing runner and copied into this packet; earlier benchmark/timing evidence was not overwritten.

CubeCL paths remain CPython 3.14.5t, GIL-disabled with `PYTHON_GIL` unset, Triton unavailable and no loaded Triton native module. PyTorch owns tensor/scratch storage; the existing current-stream/readiness/`record_stream` path is retained. Nsight observes no warm module loads/metadata allocation or synchronization attributable to generated dispatch. The full Stage 0–4/FP and 168-case corpus results are **prior executed evidence**, not fresh reruns in this diagnostic-only task.

[Analysis controls](stage5/artifacts/attribution/analysis-controls.json) separately test measured-step exclusion of reset synchronization/kernels, descendant-union subtraction, queue clamping when a kernel begins before API return, missing-step/native-census rejection and stale-report rejection. The stale-report test uses a temporary fake output directory and a mocked process; it is a harness failure-path test, not GPU acceptance evidence. The orchestrator now records its explicit environment and requires a freshly changed report, not merely an existing file. Earlier process-ledger rows contain application exit status only; accepted capture evidence is the analyzed report and matching step/native census.

Source discovery used the graph and coverage checks; ignored wheel/overlay sources were read directly. Zero graph edges for the nested backend factory did not establish absence of calls. Capture/patch/hash provenance and preliminary harness differences are explicit: instrumentation counters changed from current-only Python counters to the shared native counter for the older reference; missing-report attempts were rejected; scalar view unwrapping was required by the precision candidate. No failed sample became a pass and no numerical/census check was weakened.

The required review completed and cross-checked the artifacts. [Disposition](stage5/artifacts/attribution/review-disposition.md) records the addressed findings. Current report links, packet files and hashes are verified; preliminary snapshots were moved to ignored cache while the review's directory inventory still listed their old paths. Generated driver-API statistics now require matching CubeCL kernel correlation, with a vendor-driver negative control. Every CubeCL process now also checks the warmed frozen-reference output/gradients/parameters/RNG; targeted tiny/larger checks pass. These setup assertions and analysis guards were added after the primary measurements; their short validation rounds do not replace the 100-sample data.

Reproduce the diagnostic matrix with `stage0/.venv/bin/python -m stage5.attribution_run`, then the corrected precision component with `--only constants --precision-probe`. Run `stage5.system_analyze`, `stage5.attribution_curate` and `stage5.attribution_checks` for analysis/consistency controls. The one all-API follow-up is `stage5.attribution_run --only systems --all-api-probe`. Use the established `CUDA_PATH` and release profile. These commands write this task's evidence directory; preserve it before a new measurement campaign. Run the normal CubeCL overlay/launch controls afterward to restore the probe overlay. No commit, push or Dolt remote sync was performed.

## Recommendation

**Next: land the focused generic precision correction with neighboring regression coverage.** Keep Stage 6 deferred until that conservative-policy boundary is resolved or explicitly restricted. Preserve `cwk` as a separate numerical qualification before claiming the existing eager tolerance for all reused long-row inputs. Host and compile-history attribution are closed for this task; do not require LLVM vectors, general split planning, expanded effectful fusion, metadata eviction, concurrent dispatch redesign or tiny-fixture parity before the subsequent Stage 6 decision. No next stage was started.
