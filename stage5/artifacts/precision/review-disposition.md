# Required precision integration review

`review-mcp.review_with_context`, target `HEAD`, completed after 39 model calls.
Raw review/trace: ignored `stage5/cache/precision/review.md`.
The review verified both generic source edits against the live pinned overlay,
actual propagation/fallback protocol, scalar representations and curated evidence.
It found no semantic correctness blocker on the executed surface.

| Finding | Disposition / affected verification |
|---|---|
| Patch application control could silently narrow to a legacy four-file manifest | Require exactly the six current normal files, including index propagation/lowering, before temporary patch application; assert byte hashes. Harness passes |
| Legacy installation overwrote normal tracked patch/hash artifacts | Emit legacy evidence under ignored `stage1/artifacts/legacy-overlay`; normal evidence stays untouched. Installation transition control and full 18-variant corrected legacy probe pass; restore normal active overlay afterwards |
| Numerical diagnostic lacked retained failure evidence | Wrapper writes `accepted=false`, exception/traceback/source hash to a separate `.failed.json`, preserving the last completed qualification. Injected failure control and actual rerun pass; the actual unchanged eager gate remains false |
| Zero-context list duplicated precision module ownership | Derive context policy from `source_edits()` keys. Serialized patch still applies exactly to the pinned wheel with `--unidiff-zero` |
| Residual scalar representation and other cast-policy boundaries | Source comment identifies `promote_constants`' current Constant/BaseView(Constant) representation. Report explicitly limits future constructor folding/wrappers and FP64→FP32 casts. No speculative generalization |
| Passing superseded attempts had ambiguous qualification | Ledger distinguishes failed, exploratory and superseded passing captures; canonical final matrix remains separate |

Post-review: CubeCL78 forward/9 backward, six forced-cold compiler/native checks,
dynamic-reuse/FMA-negative counters/tails/side-stream pressure pass again.
Reduction ordering qualification passes its explicit diagnostic assertions while
retaining the original failing eager-comparison gate. Harness failure controls,
legacy probe, Ruff and diff whitespace checks pass. Source behavior of the two
precision edits is unchanged by review fixes (provenance comment only); baseline
Stage0-4/FP and all168Stage5 gates passed before these small evidence/control fixes.
CPU81/9 and configured Triton81/9 are the pre-review expanded matrices; their
numerical logic did not change. Their reference environment is GIL-enabled.
No second review was needed for these small corrections.
