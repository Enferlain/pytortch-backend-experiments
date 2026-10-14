# Reproducer and isolation method
## Categories and predicates
| Category | Stable predicate | Important controls |
| --- | --- | --- |
| Semantic correctness | Established contract violation on exact data | UB/domain, aliasing, mask tails, output classes |
| Numerical accuracy | Agreed error statistic exceeds justified bound | Reference precision, accumulation, stochastic algorithm |
| Crash | Same failing stage and diagnostic/signature | Input validity, build/runtime dependencies |
| Compilation time | Per-stage or cold/warm distribution shift | Cache state, processes, instrumentation, work normalization |
| Runtime performance | Comparable repeated unprofiled timings | Async completion, clocks/load, autotune, selected variant |
| Memory use | Comparable peak/retained allocation or resource change | Scope: host/device/compile, allocator/caches/lifetimes |
| Code size | Comparable binary/section size or instruction count | Target/debug/link flags; text length is not binary size |
| Backend-specific | Established contract or prior baseline diverges on target | Actual compiler choice, supported feature/dtype/domain |

Start with minimal useful evidence, not maximal dumps. Preserve original and reduced reproducer. During reduction rerun good and bad; document retained triggering conditions.

## Stage ablation and artifacts
For PyTorch, current debugging backend registrations include `eager` and `aot_eager`; use verified `torch.compile(fn, backend="eager")` or `torch.compile(fn, backend="aot_eager")` as coarse isolation experiments, not equivalence proofs. Source: [debugging backend definitions](https://github.com/pytorch/pytorch/blob/60de3b7f3316d28ab80fb8029742e201d5c6ab7b/torch/_dynamo/backends/debugging.py); public API: [torch.compile](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html). Their decompositions/partitioners can differ from Inductor. A passing aot_eager run does not certify every Inductor AOT path. Keep backward invocation and training mode consistent.

Reusing a captured CubeCL/LLVM/target artifact isolates downstream work only if its metadata/ABI/options/runtime state match. Pass replay must preserve prerequisites, verifier constraints and invalidated analyses. In read-only work design instrumentation rather than modifying source. Do not assume a universal pass-disable flag.

Normalize only incidental naming/paths; keep dtype, fast-math, rounding, address/memory qualifiers, masks, layouts, constants and ABI intact. A changed optimization order can expose a latent illegal rewrite; the introducing commit and semantic defect can have different owners.

## Controlled history testing
Use independent build/output/cache directories per revision and retain logs. Resolve dependency changes deliberately: fixed dependency experiment answers a different question from historical lockfile reconstruction. Build failure can result from time-varying toolchains; classify separately. Retry endpoint tests after a suspected bisect result.

For performance, choose threshold and confidence policy from baseline variance before testing revisions. If repeated samples overlap or environment changes, return inconclusive and narrow manually or improve the harness; never force noisy samples into good/bad.

Use multiple-case validation proportionate to scope: nearest shape/dtype/lanes and another target if portability is claimed. Include one predicted unaffected case. Explain if a device/compiler is unavailable; offline lowering checks do not establish runtime correctness or performance.
