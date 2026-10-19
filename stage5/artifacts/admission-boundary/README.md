# Stage 5: correct the reduction admission boundary

This is an investigation of `7yq`, after the guarded column adoption. The
reference512 arithmetic remains qualified. Comparison with the old tree is an
integration defect to correct, not a numerical reason to discard it. No reduction
body, eligibility, precision, split rule, runtime warmup or tolerance changes here.
No performance conclusion, new topology, commit, push or Stage6 work.

## Executed counterexample to the current validation boundary

[results.json](results.json) executes the frozen uniform-loop reference512,
the normal scalar baseline and the existing compensated producer through the
**same retained final**, on the original RTX3090. All148 original input hashes
match the retained qualification and corresponding efficient-reference packets.
Every reference512 final output hash also matches the previous adoption check.

| Check, unchanged tolerances/policy | Result |
| --- | --- |
| reference512 complete reduction, `reference-or-better-v1` |148/148 pass |
| compensated complete reduction, same policy |148/148 pass |
| scalar baseline complete reduction, same policy |146/148 pass |
| reference512 vs scalar partial-buffer `compare` |143/148 pass |
| reference512 vs scalar **complete-output** `assert_close` |145/148 pass |

The five partial-comparison failures reproduce the earlier packet exactly.
The compensated path is checked under the complete qualification policy here;
the hypothetical strict old-tree final gate is tested only for reference512.
Normal compensation uses its existing separate corruption guard, unchanged.
Simply postponing that comparison until after the final is insufficient: it
fails `heldout_10007`, `heldout_30003`, and
`new_existing_finite_adversaries`. Their full values and assertion messages are
retained. In the adversary, reference512 and compensation return1,-1,0.25 where
the scalar baseline returns0. At `heldout_10007` element7, the scalar result
-1.54632568359375 fails against reference -1.5464019775390625 and FP64 truth
-1.5463906225886603; reference512 matches the reference. At `heldout_30003`, the
old result itself qualifies through no-greater-FP64-error, but its difference
from reference512 still exceeds the old-tree closeness gate. Qualification does
not require every independently qualified algorithm to be close to every other.

The original rejected adjacent-merge256 outputs are rechecked without another GPU
candidate. Their original finite-adversary failure is preserved. All148 synthetic
large output corruptions fail the existing policy. These controls do not replace
historical stream/special-value checks or claim universal error bounds.

## Relevant installed Inductor machinery

[mechanisms.json](mechanisms.json) binds exact installed source paths, SHA256s and
line excerpts. This is a local-version inventory, not an exhaustive upstream audit.
The installed PyTorch source is outside the project graph's indexed scope, so it
was read directly after graph lookup/coverage checks.

- `AlgorithmSelectorCache` already accepts arbitrary `ChoiceCaller` alternatives,
  uses a persistent cache and generates benchmark inputs. Its optional `VERIFY`
  defaults off; when enabled it takes `choices[0]` as expected and calls
  `assert_close`. Adopting it directly would retain the same wrong oracle if the
  old tree were the first choice. Its public call has custom input generation,
  but no caller-supplied complete-reduction qualification callback.
- `MultiKernelCall` benchmarks alternatives through `kernel.run` and `clone_args`,
  with a `CachingAutotuner` interface assumption. It does not supply this
  per-element numerical oracle or automatically group a partial and final.
- `SubgraphChoiceCaller` can compile and benchmark an actual multi-kernel graph;
  therefore Inductor is not fundamentally limited to single-kernel choices.
  It is a real reuse option if graph-level algorithm selection becomes necessary.
  Its `benchmark(*args, out=...)` executes the compiled graph but does not copy the
  graph return into `out`; an adapter must not assume `VERIFY` observes that return.
  Re-lowering a new subgraph is unnecessary for the immediate missing ownership
  fact in our existing scheduler.
- Dynamo's `same_two_models` can generate FP64 graphs, but `utils.same` falls
  back to aggregate RMSE/multipliers. That is not our existing **per-element**
  corresponding-efficient-reference policy. Reuse of its graph execution/input
  cloning is possible; reuse of its acceptance predicate would change policy.

Retain the existing Stage5 `KernelBenchmarker`, persistent cache, isolated
destinations and current-stream/storage ABI. None of the inspected mechanisms
requires another reduction algorithm or another runtime owner.

## Concrete existing boundary and scope

[graphs.json](graphs.json) observes five real, unmodified generated graphs through
`CubeCLScheduling.define_kernel`. The observer records descriptors, actual buffer
names and `SchedulerBuffer.users`, then delegates unchanged. It uses the existing
explicit baseline execution policy to avoid tuning during this structural probe.
It does not substitute kernels or override split counts.
This structural capture does not record the invoked wrapper filename, effective
cache directory or compiled-kernel hashes. The environment requests an isolated
cache after imports, but this packet does not prove effective cache isolation.
There is no timing or kernel-identity conclusion from these five graph records.

| Shape | Split policy, cold graph | Actual final extent |
| --- | --- | --- |
|8×65539 | guarded |9 |
|8×1048576 | guarded |32 |
|64×65539 | guarded |4 |
|8×1048576 | inherited |32 |
|64×65539 | inherited |4 |

All five have an intermediate `buf0` with one strong scheduler user, the final;
the final input is that exact producer output, and its result has an `OutputNode`
user. This is enough existing ownership information to begin a bounded pair
recognizer. It is not proof that every graph has this pattern. The ordinary
8×65539 graph supplies a naturally occurring split9 adoption target now.

Earlier retained inherited-policy tests first compiled8×65539 and reused that
dynamic split9 graph at the other shapes (`final_plan_checks.ordinary`). This
cold capture does not reproduce that history. Inherited policy is not synonymous
with split9. Admission must inspect the **actual** descriptor pair and scalar
scope, and its identity must distinguish every other split/consumer.

`backend.define_kernel` currently emits only the individual semantic descriptor;
`runtime.AutotunedKernel.select` cannot see this producer/consumer relation. The
missing adaptation is to carry a bounded admission context from that existing
scheduler information into the factory and tuning/cache key. It does not require
a generic graph scheduler, a new split planner, or a graph-wide autotuner.

## Existing implementation oracle and proposed correction

[oracle.json](oracle.json) checks the **existing** CPU FP32 topology model against
all148 frozen GPU partial buffers and complete outputs: all are bitwise identical.
All148 changed-partial controls are distinguishable. It uses the historical
field name `corruption_control_rejected` for those hash inequalities; that is
**bitwise distinguishability**, not a call to the numerical qualification gate.
The separate `results.json` output-corruption controls call `qualify_case`.
The model check uses the historical
test environment, one CPU thread and `set_flush_denormal(False)`. No new arithmetic
model or GPU candidate is created. Observed single host-wall model/copy times
were99.5/19.1ms for random_9241_1048576 and69.4/5.9ms for FP32 storage64×65539.
These are setup-cost observations, not a warm benchmark or intrinsic GPU cost.

The smallest justified integration has two separate responsibilities:

1. **Numerical admission:** recognize only the qualified pure-sum split9 producer
   and a qualified final/cast consumer, with the intermediate unobservable outside
   that pair. Bind both descriptors, scalar/dtype scope, compiler/precision and
   model versions into admission/cache identity. Use the existing full-corpus
   `qualify_case` policy in qualification gates. Admit a bounded tested shape
   scope first; no split32/4 inference or arbitrary prologue/epilogue admission.
2. **Implementation validation:** for that admitted reference512 candidate, use
   the existing independently implemented topology model as its first-use
   execution oracle instead of the old partial tree. Keep the other candidates'
   current checks, compensation guard, ABI checks and RNG/output isolation.
   An independently qualified algorithm is not the reference512 execution oracle.
   Timing then uses the existing selector; with the same fixed final its cost is
   common to the alternatives. Qualification does not force a winner.

This is a **proposed integration**, not implemented promotion. The exact model
check verifies implementation of the already-qualified arithmetic; it is not a
new numerical policy and cannot establish that arithmetic's accuracy by comparing
it to itself. Complete-reduction qualification remains independently referenced.

Before installing that CPU oracle in normal first-use tuning, its implementation
must handle the caller's CPU floating-point environment without changing it.
This probe explicitly establishes the historical environment and does not prove
ambient-environment independence. Unsupported CPU FP state should retain the
existing fallback unless a read-only environment check or preserving scoped
mechanism is established. Special-value class/sign requirements, supported input
layouts, candidate cache reload and one-time-only oracle execution also need
direct integration controls. There is no proposal to do CPU copies or model work
on warm launches or to ship dependencies on diagnostic artifact directories.

The original reason for withholding reference512 was therefore misframed. It is
qualified; the remaining work is this bounded selector/admission integration.
Record that work as `7yq`, rather than another arithmetic experiment or a stop
recommendation. Production eligibility remains unchanged during this investigation.

## Reproduction and evidence retention

Use the isolated Stage0 Python and original RTX3090:

```text
CUDA_VISIBLE_DEVICES=GPU-4cec666b-991c-fe31-e63f-6455bc828119 STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.reduction_admission_probe boundary
CUDA_VISIBLE_DEVICES=GPU-4cec666b-991c-fe31-e63f-6455bc828119 STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.reduction_admission_probe oracle
CUDA_VISIBLE_DEVICES=GPU-4cec666b-991c-fe31-e63f-6455bc828119 STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.reduction_admission_probe graphs
```

Captures refuse to overwrite existing result files. Move a previous packet aside
before intentional reproduction; do not silently replace it. The source evolved
by adding the independent oracle and graph probes. Exact source snapshots for
the first two captures are retained under `sources/` and match their recorded
hashes; the final reproduction source matches `graphs.json`. No failed GPU
capture or numerical case was discarded. No device timing was collected.

[audit.json](audit.json) independently recomputes all148 complete-output policy
results and old-tree closeness decisions from retained values, verifies source
snapshots, and checks the five ownership edges. `kernel-bindings.json` binds36
retained descriptor/manifest/PTX files so the execution identities survive cache
cleanup. Ruff F, formatting, Python compilation and `git diff --check` pass;
the production18-group suite was not rerun because this pass changes no production
logic. The earlier column-adoption suite and its preserved failures are unchanged.

The reproducible offline producers added after review are:

```text
stage0/.venv/bin/python stage5/artifacts/admission-boundary/sources/retain_kernels.py
stage0/.venv/bin/python stage5/artifacts/admission-boundary/sources/audit.py
```

Both verify existing output files instead of replacing them. The audit additionally
asserts element-wise equality with the original topology qualification outcomes.
`dependency-bindings.json` retains the six exact current Python dependencies,
including the previously uncommitted column-adoption runtime changes. Future
dependency drift requires an explicitly new capture or restoration of that source
state; it is not a license to reinterpret the original comparison. These sources
are provenance copies, not replacement normal-runtime modules.

Required [review-mcp](review.json) verified the central findings and identified
the missing offline producers; they are now retained and pass locally. Its
remaining source-only review limits and minor dispositions are recorded in
[review-disposition.md](review-disposition.md).
