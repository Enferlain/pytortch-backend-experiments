# Stage 5 reference512 normal admission

Implementation of `7yq`: the qualified FP32 uniform-loop partial becomes an
ordinary first-use choice only for an exact, private split9 producer/final pair.
The scalar old tree is no longer its implementation oracle. No reduction body,
work assignment, arithmetic, loop, protected load, split rule, precision barrier,
PyTorch storage/stream ownership or qualification tolerance is changed.

The complete descriptor templates are source in `stage5/reference512_templates.json`.
Initial producer scope is FP32 at5/8/64×65539 and8×131071/1048576, and only if
the actual generated partial/final descriptors already have split9. This does
not force split9: ordinary cold8×1M and64×65539 still choose32/4. FP16/BF16 and
other descriptors retain their existing eligibility.

## Existing mechanisms reused

Inductor's `DelayReplaceLine` resolves module-level factories after both kernels
have been generated. Existing `SchedulerBuffer.users` proves that the partial
has one strong final consumer and that the final is the only observable output.
No wrapper is added to warm calls. Existing ShapeEnv equality guards bind this
pair admission to the tested finite shapes. These additional guards specialize
the admitted graph shapes; they do not change the split planner. They are added
after descriptor construction so the final's existing scalar ABI is preserved.
Unadmitted neighboring graphs keep their original final choices.
The equality scans also classify rejected shapes, so neighbors may incur extra
guards or recompiles. No unchanged compile-cost or universal host-cost claim.

The pair's final chooses only between the existing byte-equivalent scalar final
and qualified descending-plane final9. The producer competes with the existing
compensated plan through the unchanged benchmark/MAD selector. Compensation is
the performance fallback; a scalar baseline outside long-row numerical eligibility
is not reintroduced as an admissible alternative. Deterministic mode retains that
qualified fallback when no validated timing cache exists.

Admission context, finite runtime shape fact, model/source/precision identities
enter tuning cache identity. Factory arguments are copied and complete templates
are cached as immutable strings. Context-free and paired kernels cannot share
the same factory/tuning entry accidentally.

## First-use implementation validation

The existing independently qualified CPU topology model produces expected FP32
partials. The fresh tuner compares the candidate's isolated scratch destination
bitwise to that model, ignoring only NaN payload encoding while preserving value
classes, infinities and finite/zero bits. The old checks for every other candidate
and compensation's separate corruption guard remain unchanged. Warm launches and
validated persistent-cache reloads perform no CPU model or input copy.

The CPU oracle is available on Linux x86_64 glibc. Standard `feholdexcept`,
`fesetround`, `fesetenv` save/restore the calling thread's complete FP environment,
including MXCSR, through success and exceptions. The local C ABI probe records
`sizeof(fenv_t)=32`, alignment4 and `FE_TONEAREST=0`; the ctypes buffer is larger
and long-double aligned. No process-wide thread setting is changed. The existing
model executes row by row: participating FP32 additions have at most18432 elements,
below the installed ATen TensorIterator32768 grain. Unused FP64 truth diagnostics
do not enter the expected partial. Unsupported platforms retain existing choices.
The complete-reduction `reference-or-better-v1` accuracy gate remains independent
of this implementation check; matching one's arithmetic model is not its accuracy
qualification.
On an unsupported platform the new candidate is excluded. An unexpected failure
to save, establish or restore the FP environment on a declared-supported target
is a hard validation failure; it is never swallowed to accept an unchecked
candidate. Compensation is the performance/deterministic fallback, not an error
suppression path.

## Verification packet

`corpus.json` preserves all148 original exact input hashes, normal-compiler
reference512 partial/final output hashes, row-wise CPU model identity with four
CPU threads, and the complete unchanged corresponding-reference policy. The low
precision cases are preservation controls, not new normal admission.

`controls.json` checks16 combinations of CPU thread count, rounding mode and
denormal setting, with exact saved-environment restoration on success and on an
exception. It also checks context/caller isolation, unsupported shape/context
rejections, and an actual fresh selection with deliberately corrupted model
expectations: rejection precedes timing and leaves the caller output untouched.

`cache-init.json` and `cache-reload.json` exercise the real generated ordinary
8×65539 graph. Initialization runs exactly one model check. Repeated warm calls
do not run it again. Fresh-process reload runs no model and no compiler.

`graphs.json` records seven actual generated graph calls, including the finite
adversary, supported→neighbor→supported transitions and unchanged cold split32/4
fallbacks. Actual invoked wrapper files/hashes, selected kernels and a separate
nondefault-stream/offset/canary check are retained. `identity.json` verifies all
six partial/final IR/CUDA/PTX bodies byte-for-byte against the frozen packets.
`specials.json` verifies NaNs/infinities/zero bits and caller default dtype/device
preservation.

All18 established regression groups have passing results across the two recorded
attempts, including a complete168-case/zero-failure run. The first9 successful
groups are retained in the first-attempt directory; the unchanged remaining groups
pass with the required CUDA_PATH. Both attempts restore17137 tracked artifacts
and reports; this does not undo the separately disclosed untracked JSON loss.

Required [review-mcp](review.json) found no correctness bug in the admitted path.
Small graph-lifetime/fallback hardening is applied; fresh init/reload,16 FP-state
controls, deterministic/missing-fallback/corruption controls and all seven graph
calls pass again afterwards. The original captures remain under `before-review`.
See [review-disposition.md](review-disposition.md) for every finding and limitation.
The independent `sources/audit.py` recomputes the148 policy cases, checks168 full
regression cases and18 log digests, checks six code identities and verifies cache,
state and actual-wrapper identities. `sources/retain.py` preserves exact current
source plus seven small executed-kernel identity packets without requiring the
artifact cache to survive future cleanup. Both producers are retained and pass.

## Preserved failures and limitations

The first regression recipe reused the earlier uncommitted adoption packet's
`regressions/corpus.json` output path. That old raw JSON was overwritten and could
not be recovered; the prior complete-run `corpus-final.log`, initial failed JSON
and all column traces/review evidence remain. The current168-case/zero-failure
result was moved to this packet's `regressions/corpus.json`. It must not be
presented as the original capture. `reference512_gates.py` now redirects every
explicit `--output` into the new packet. Its snapshot restoration protects tracked
history but did not protect that untracked old file; this limitation is disclosed
in the prior packet too. No old measurement was synthesized or relabeled.

The first gate attempt also omitted the required CUDA_PATH for LLVM libdevice.
The first nine passing groups and failed compiler-comparison log are retained
under `regressions-first-attempt`. The unchanged remaining groups resume with the
established toolkit path. Snapshot restoration runs after both attempts.

The first controls attempt mutated a returned inference tensor in the harness;
it now clones the tensor before deliberate corruption. The failed source/log are
retained. A compound ShapeEnv OR caused sustained SymPy sorting work before kernel
timing; sources, termination and a faulthandler stack are retained. Sequential
equality guards avoid that solver path. A subsequent early-equality attempt
specialized the final descriptor before it was constructed; its source/log are
retained. Deferring guards with the factory preserves both descriptors.

The initial smoke demonstrates selection, not a new performance benchmark. No
new paired performance conclusion, universal speedup, synthetic GPU primer,
production warmup, hardware-state investigation, Stage6, commit or push is claimed.
The first-use CPU model has host/setup cost and is deliberately absent from warm
execution. No default eligibility is inferred for split32/4 or other arithmetic.

Reproduce with the isolated Stage0 Python, release compiler, original RTX3090:

```text
CUDA_VISIBLE_DEVICES=GPU-4cec666b-991c-fe31-e63f-6455bc828119 STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.reference512_checks init
CUDA_VISIBLE_DEVICES=GPU-4cec666b-991c-fe31-e63f-6455bc828119 STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.reference512_checks reload
CUDA_VISIBLE_DEVICES=GPU-4cec666b-991c-fe31-e63f-6455bc828119 STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.reference512_checks controls
CUDA_VISIBLE_DEVICES=GPU-4cec666b-991c-fe31-e63f-6455bc828119 STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.reference512_checks corpus
CUDA_VISIBLE_DEVICES=GPU-4cec666b-991c-fe31-e63f-6455bc828119 STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.reference512_checks graphs
CUDA_VISIBLE_DEVICES=GPU-4cec666b-991c-fe31-e63f-6455bc828119 STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.reference512_checks identity
CUDA_VISIBLE_DEVICES=GPU-4cec666b-991c-fe31-e63f-6455bc828119 STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.reference512_checks specials
CUDA_VISIBLE_DEVICES=GPU-4cec666b-991c-fe31-e63f-6455bc828119 STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.reference512_gates
stage0/.venv/bin/python stage5/artifacts/reference512-admission/sources/retain.py
stage0/.venv/bin/python stage5/artifacts/reference512-admission/sources/audit.py
```

Captures refuse to overwrite result JSON. First-use initialization needs a fresh
tuning key; preserve an existing selection/capture before a deliberate repeat.
