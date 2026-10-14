---
name: compiler-regression-investigation
description: Isolate reproducible compiler correctness, numerical-accuracy, crash, compilation-time, runtime-performance, memory-use, code-size and backend-specific regressions between commits/versions. Use for Inductor/CubeCL/Pliron/Triton artifact changes, lost fusion or register growth. Coordinate lowering trace, semantic audit and profiling to distinguish introducing change, trigger and root cause.
---
# Compiler regression investigation
Read [suite protocol](references/suite-protocol.md). Use [reproducer/isolation guide](references/regression-method.md) and [verified history interfaces](references/history-tools.md) when needed.

## Workflow
1. **Classify and define a predicate.** Distinguish semantics, accuracy, crash, compile/runtime time, memory, code size and backend-specific behavior. Specify error signature, tolerance/bitwise obligations or statistical threshold. Changed PTX/register count/fusion is an artifact change until connected to required behavior/cost.
2. **Validate a stable reproducer.** Preserve input/seed/data/layout/aliasing/flags/mode. Repeat with controlled caches/processes/autotuning/asynchronous completion. Separate stochastic algorithm variation from compiler defects. Request **gpu-codegen-profiler** for timing and **compiler-semantic-audit** for oracle/tolerances.
3. **Establish comparable endpoints.** Retest good/bad revisions; account for transitive dependencies/lockfiles, LLVM/vendor compiler/driver/hardware, Python GIL mode and build features. Record dirty changes. Build/setup failure is inconclusive, not bad.
4. **Locate material divergence.** Re-establish each endpoint's path; architecture can change. Compare captured/transformed graphs, scheduling/fusion, CubeCL dialect passes, target IR/source, diagnostics/binaries/outputs. Keep raw artifacts and documented normalization. Earliest different text need not be earliest semantic/cost change. Request **compiler-lowering-trace**, including flags/order/cache/ABI changes.
5. **Reduce without changing the predicate.** Simplify operations/data/shape/layout/dtype or replay a boundary artifact while preserving endpoint classification. Retain essential vector lanes, tail masks, aliasing, streams and launches. Crossing into undefined inputs or different timing scope creates a different case.
6. **Bisect only reliable classifications.** Use an isolated checkout and tested harness. Set repeat/statistical policy for noisy performance; return inconclusive for overlap, drift and setup failure. Do not bisect flakiness. Report skipped intervals or introducing ranges when uniqueness is unavailable.
7. **Read introducing history.** Inspect diff/parents/merges/tests/discussion and resolved dependency changes. Separate new defect, newly triggered latent defect, contract change, stale cache, backend compiler/driver drift and benchmark error.
8. **Validate the explanation.** When authorized apply/revert suspected change under controlled conditions or instrument equivalently. Workarounds/pass disabling can suppress symptoms without proving cause. Test nearby values/shapes/dtypes/lanes/targets and at least one predicted unaffected case.
9. **Close at justified confidence.** Report predicate reliability, endpoints, minimized repro, first material boundary, introducing commit/range, root cause versus trigger, correction and validation. For development add meaningful tests at the responsible layer; for read-only work provide patch/test design.

## Handoffs and output
Keep version attribution here; other skills return mechanism evidence via the shared packet. Use installed PyTorch/Triton minifier skills only after verifying exact current interfaces and rechecking the reduced predicate.

Return an upstream-ready finding: reproduction conditions, contract-justified expected/actual behavior, comparable artifact excerpts, history evidence, uncertainty and falsifiable explanation. Backend mismatch without prior baseline does not prove regression.
