# Required review disposition — 68x

Required review-mcp completed: **PASS, no correctness or legality blockers**.
Full returned review and trace: [review.json](../review.json). The reviewer read
all six original modules, supporting runtime classes and packet files, and
independently reproduced the ablation win counts. Five minor observations follow.

1. **Ordinary-artifact retention omitted from reproduction — fixed.** The initial
   copy was an orchestration command, and its absence from the README was a real
   reproduction gap. `stage5/composition_retain.py` now implements that same step,
   checks existing copies without overwrite and rejects mismatches. It verified
   all10 selected ordinary directories in this packet. A temporary fixture also
   passes fresh-copy, idempotency and mismatched-source refusal controls. The
   README now places this command before the offline analyzer. No timed code or
   captured artifact changed.
2. **Per-call compile/admission file I/O — not present on the warm call path.**
   The review misread the generated wrappers here: all `composition_compile`
   calls are module-level assignments. Warm `call()` methods invoke the resulting
   Dispatch/AutotunedKernel objects. The analyzer now proves, for every actual
   wrapper, that exactly both factory calls are top-level assignments and none
   occur inside the warm call method. Admission/proof/template file reads remain
   cold construction work; no caching optimization is needed for the measured
   path. This still provides no host-wall-time performance claim.
3. **First matching column source by dtype — scoped and deferred.** The pinned
   packet has exactly one column artifact per dtype, independently verified by
   the reviewer. Canonical full-semantic equality and mechanism/rank/ABI checks
   reject incompatible sources. A future packet expansion should use an explicit
   unique semantic/mechanism match. Current frozen-source lookup and tested code
   remain unchanged.
4. **Literal rank/ABI checks — intentionally bounded.** All admitted column forms
   have rank2 and metadata7/80. These checks also match the original ColumnKernel's
   literal guards. The column constructor bypass is needed for its original exact
   admission check; it already accepts `compile_ptx=False`. The separate packed
   RMS constructor always compiles, which motivates its frozen-loader override.
   No generalized rank/ABI API is introduced in this diagnostic.
5. **Extent-spelling origin — unresolved, disclosed.** The primary fit and first
   mixed test use different shape/construction histories. The differing order is
   already visible at the semantic descriptor boundary; the exact upstream
   symbol-order/emission origin was not isolated. No compiler defect is claimed.
   Complete-field equality plus the bounded extent proof and GPU controls make
   the numerical result independent of that unresolved origin. Future fused
   adapter feasibility remains tracked as `pytortch-backend-experiments-axk`.

The only post-review executable changes are the standalone retention helper and
the stronger offline wrapper-scope assertion. Original runtime, backend, CPU and
GPU qualification producers, timed wrapper bodies, traces and shader artifacts
remain unchanged. Pre-review source/analysis/integrity snapshots are retained
here; current source bindings include the new helper. Parent reruns the affected
offline audit, source checks and Ruff/diff gates. No GPU/performance recollection,
production eligibility change, commit, push or remote sync is needed.
