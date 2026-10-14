# Stage 5 follow-up — warm CPU dispatch

## Result

**Works: warm CPU dispatch was attributed and simplified; all correctness gates pass.** The final paired training capture is **3.653 ms → 3.192 ms host enqueue (12.6% lower)**, with the same selected physical plans and GPU kernel counts. This is a qualified capture, not a universal speedup: earlier captures were less favorable and the tiny FP32 case is inconclusive. No Stage 6 work, compiler/launcher replacement, physical-plan change or relaxed numerical policy is included.

## Measurement boundary

The benchmark replays the same captured graph through artifact/runtime/legality source loaded from Git `df4320d093a76a49b7ab33d52cf10d22a9498bd0` and through the current implementation. Both use the unchanged Stage 0–4 bridge. Reference selection is fixed to the optimized path's selected physical candidate. Frozen files provide honest old adapter provenance and distinct artifact cache identity; separate directories and exact independently generated PTX equality are asserted. Outputs, arbitrary training gradients, updated parameters and generator advancement match exactly before timing. This is a runtime comparison, not a second GPU tuning contest.

Three rotated rounds measure at least 100 warmed calls per mode/case. Reset and CUDA completion are outside host intervals. All initial compilation, metadata setup and physical selection finish before those rounds; zero warm compiler invocations and empirical tuning trials are asserted. CPU phase wrappers and cProfile are installed only in separate diagnostic regions. Nested phase intervals subtract child intervals once. Instrumented whole-step forward/backward/optimizer timings and exclusive dispatch timings are not combined with uninstrumented latency. CUPTI GPU durations are separately instrumented; queue-inclusive CUDA events are not treated as intrinsic GPU execution.

The normal kernel launch still has no device-wide synchronization. Synchronizations in the harness are measurement boundaries. Prepared native dispatch holds fixed tensors/stream only within its existing scoped diagnostic interface; its timings are a lower-bound comparison and do not represent an alternative production launch API.

## Changes

- Detailed per-launch pointer/stream/geometry records and warm selection-hit records become opt-in via experimental `CUBECL_STAGE5_TRACE=1` or `stage5.diagnostics.detailed_trace()`. The latter uses ContextVar, supports nesting/exception restoration and isolates worker preferences. Cold compile/tune/cache-decision evidence remains recorded. Native counters, per-kernel successful ordinary launch counts and per-dispatch selection-hit counts remain enabled.
- Stage 5 caches descriptor-derived size/stride/offset/span/dtype tuples for at most 64 scalar tuples per kernel, under the existing kernel lock. No tensor, storage, pointer, capacity, alignment or alias fact is cached in that LRU. Incoming descriptors are copied at construction to isolate caller mutation.
- Each invocation still checks argument count, bounded scalar/positive domain, physical/reduction extent, output domain, actual device/dtype/shape/stride, current original-storage capacity and disjoint output storage. Vector plans retain their direct-address/alignment validation. Checked integer expression evaluation and FP policy are unchanged.
- Storage wrappers and current data pointers are read once per validated tensor. Pointer values survive only for that invocation and are cleared in `finally`; the native pointer array is invocation-local. Its ctypes type and the immutable workgroup dimensions are prepared at construction. No persistent user-tensor references are added.
- Frozen-plan construction and immutable effect classification move out of warm dispatch. The local selection lookup no longer rehashes every physical candidate after the finite proof class has already determined the catalog. Current pointer alignment classes query each pointer once.

The experimental counters are `Kernel.launch_count` and `AutotunedKernel.selection_hits`; prepared batches use the authoritative native counter for repeated launches. Ordinary counting uses validated nonzero extent instead of reevaluating geometry. Pointwise output extent and schema-3 output/RNG extent checks match the old validators, preserving Stage4's mutation exemption. Native enqueue explicitly rejects missing validated invocation state.

**Only the added descriptor LRU is bounded.** The older immutable CUDA metadata/event cache still retains every prepared key for process lifetime. That pre-existing limitation is tracked as `pytortch-backend-experiments-dm4`; eviction/readiness redesign is outside this follow-up.

The cached validator is Stage 5-specific and mirrors the executed Stage 2/3/4 runtime contract. Those earlier sources remain unchanged and independently tested. This is an experimental runtime refinement rather than an upstream PyTorch API. Common metadata readiness, stream handling, allocator `record_stream`, native launch validation, compiler and semantic/physical boundaries are reused unchanged. There are no new PyTorch, CubeCL, Rust or C source changes.

## Evidence and remaining costs

The initial diagnostic confirmed multiple contributors rather than one isolated bottleneck: validation, current-stream/readiness/lifetime handling, selection, tracing, argument binding and ctypes/native/driver enqueue all consume CPU time. `driver_enqueue` includes the Python ctypes crossing and native validation/driver call; it does not attribute that total solely to CUDA itself. Framework/autograd/vendor/optimizer host work also contributes to the complete training step.

### Warm measurements and ablations

[Samples, rounds, phases and source identities](stage5/artifacts/launch/performance.json) retain every measurement. These are uninstrumented host medians in microseconds, excluding completion/reset:

| Fixture | Reference | Optimized + full trace | Counters only | Legacy selection ablation | Legacy validation/binding ablation |
|---|---:|---:|---:|---:|---:|
| FP32 add, 257 |240.64|215.62|244.93|228.46|250.77|
| FP16 add, 1,048,583 |310.65|279.34|244.84|259.56|261.48|
| FP32 row sum, 64×257 |277.45|259.69|232.47|254.41|250.26|
| Training, 5×9 |3652.97|3358.64|3192.48|3193.33|3319.99|

FP16 add and row sum counters-only medians are 21.2% and 16.2% lower; both are lower than reference in all three rotated rounds. Training round medians are reference **3815.6/3539.4/3680.3 us** versus counters-only **3324.7/3187.2/3128.4 us**. Aggregate MAD is 301.1 versus 250.2 us. These spreads are substantial, not confidence intervals. The tiny FP32 aggregate is 1.8% slower with strong order/process variation; no stable tiny-kernel win is claimed.

Earlier preliminary captures remain in `before.json` and `paired-first.json`. A preceding capture showed only about 2.3% training improvement with overlapping distributions. Earlier runs also used shared artifact identity/current legality, repaired after review; they are not the primary independent-artifact comparison. No positive run is substituted for the less favorable evidence. Ablation deltas are noisy and not additive causal percentages. The legacy-selection ablation uses reference globals/lock in this serial diagnostic, not a concurrent production mode.

Separate 20-step diagnostic regions yield these average exclusive wall costs (us/step), **not** a decomposition of the uninstrumented median:

| CPU phase | Reference | Optimized |
|---|---:|---:|
| Tensor/scalar validation |373.1|198.8|
| Compiled dispatch and tracing |177.8|28.6|
| Selection excluding signature |322.3|282.7|
| Proof signature |111.4|182.1|
| Argument binding |133.0|109.9|
| Metadata lookup/setup |16.9|17.1|
| Current stream/readiness/lifetime |395.1|437.7|
| ctypes/native/driver enqueue |305.0|343.6|

Validation and record construction are clear removable costs. Unchanged stream/driver phases and signature timing fluctuate; they do not imply a device-code regression. Separately instrumented forward/backward/optimizer medians are reference 1871.9/1587.9/447.3 us and optimized 1747.1/1604.4/502.6 us; these must not be combined with the uninstrumented table. Prepared fixed-buffer/stream native enqueue medians are approximately 24–25 us, a diagnostic lower bound only.

### GPU and CPU sampling

Every final mode records one generated GPU kernel for each pointwise/reduction fixture and **20 total / eight generated** for training. Generated counts match the native counter. Physical plans and independently emitted PTX match. No GPU speedup is attributed to runtime edits. Training GPU sums include a reference first-sample outlier: 199.8/39.7/39.6 us versus optimized 36.6/36.8/36.5 us; all samples are retained, not used to claim a device win.

`perf version 7.2.9` recorded CPU-clock/DWARF samples of 300 warmed training calls: 191 samples, zero lost. [Library attribution](stage5/artifacts/launch/perf-libraries.txt) places 50.3% in CPython, 8.9% in libtorch_python and 8.4% in libtorch_cpu; libcuda.so.1.1 accounts for 4.2%. This is coarse CPU execution sampling, not wall latency or exact Python function attribution. It includes reset and follows CUPTI profiling (libcupti appears in samples). cProfile supplies a separate function view. The evidence does not establish that WSL, ctypes or the GIL alone explains the remaining milliseconds.

### Correctness, reviews and reproduction

- All **16 Stage0–4/FP gates** pass, including Stage0 C++/LLVM, Stage1/1.5 boundaries, complete Stage2, Stage3 and Stage4 controls. [Fresh results](stage5/artifacts/launch/regressions.json), [details](stage5/artifacts/launch/regression-details.json).
- Stage5's **76 physical combinations / 754 extent checks**, capability/descriptor/proof controls, cache reload, compiler comparison and CubeCL math probes pass. The complete **168-case CubeCL corpus** passes correctness and exact profiler/native census checks. Unchanged ATen/Triton comparison data remain prior Stage5 evidence, not fresh comparisons. [Gates](stage5/artifacts/launch/gates.json), [controls](stage5/artifacts/launch/stage5-controls.json), [corpus](stage5/artifacts/launch/corpus.json).
- New [warm controls](stage5/artifacts/launch/controls.json) cover changing sizes/eviction, live dtype/device/layout/alias/alignment/storage-capacity rejection, descriptor snapshots, pointwise extent parity with Stage2, real rank-zero capture, counters and zero-grid no-op, scoped tracing/exception restoration, two concurrent workers with independent streams, allocator pressure and no tensor retention. They are integrated in `stage5.final_gates`.
- Triton package/native modules remain absent and CPython 3.14.5t remains GIL-disabled without `PYTHON_GIL` forcing. Current-stream/readiness/`record_stream` contracts were retained.
- Two review-mcp reviews completed. The first caught a missing output-extent guard and shared-cache PTX comparison; both were fixed and affected full Stage5 gates rerun. The follow-up found no dispatch correctness regression. Import routing now fails loudly, monkeypatch cleanup is exception-safe, unwarmed reference selection fails explicitly, and zero-grid counter equivalence is tested. [Disposition](stage5/artifacts/launch/review-disposition.md), [harness checks](stage5/artifacts/launch/hook-checks.json).

Two setup mistakes were repaired (legacy method's implicit `__class__`, overlay installation order). One strict cross-mode total GPU census stopped a post-review benchmark while generated/native counts matched. Names were not retained for that initial stopped sample, so its exact cause is not established. Failure capture now preserves names, and subsequent diagnostic/final captures satisfy the unchanged strict census. [Attempt history](stage5/artifacts/launch/gate-attempts.json) retains stopped gates. None became a numerical pass or an incomplete accepted GPU timing sample.

Final minor harness guards/cleanup were checked after timing and recorded as post-measurement provenance; production artifact/runtime/legality hashes match the primary capture. Old stage reports/generated evidence were restored after fresh results were preserved here. Raw logs, caches, perf data, Chrome traces and duplicate descriptors remain ignored.

Reproduce with `STAGE0_CARGO_PROFILE=release CUDA_PATH=$PWD/stage0/cuda13-toolkit/nvidia/cu13 stage0/.venv/bin/python -m stage5.launch_gates`. `--start <gate>` requires matching successful prerequisite rows. Old `stage5.final_gates` resumes must include the newly inserted `launch_controls` gate. No commit, push or Dolt remote sync was performed.

The conservative FP patch remains `9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2`. Production remains PyTorch cu130 / CUDA 13.0 NVRTC. Existing Windows 13.1 offline diagnostics and Linux Nsight hardware-counter evidence are unaffected; this follow-up changes CPU dispatch rather than generated device code.

## Recommendation

Continue **bounded Stage5 host/launch diagnosis before Stage6**. Safe redundant work is reduced, but stream/allocator handling, native enqueue and framework/autograd/vendor/optimizer dispatch still dominate this small graph. Attribute remaining costs before considering another binding/execution interface; the evidence does not require replacing the established compiler/launcher. Metadata retention is a separate deferred issue. No next implementation direction was started.
