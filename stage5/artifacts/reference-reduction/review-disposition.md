# Review disposition

Required review-mcp review of HEAD and the four diagnostic modules/report returned
after 51 model calls. Verdict: final comparison is methodologically sound; no
diagnostic correctness blocker. Production source and selection remain unchanged.
Raw review and trace are retained locally at
`stage5/cache/reference-reduction/review.md`.

Addressed findings:

- Report-generation drift: analyzer now mechanically renders or verifies all four
  measured tables and both quoted cost paragraphs, binding report and numerical
  source hashes. A stale-table negative control fails deliberately.
- Physical failure evidence: compiler/runtime exceptions now preserve reason and
  traceback, re-raise, and write a failed packet without changing canonical bytes.
  The original vector-veto attempt is retained; a fresh real veto is recorded in
  `long-vector-veto.json` with zero native launches.
- Nsight component failure evidence: same preservation convention, plus an eager
  correctness check on the exact profiled component fixture.
- Launcher ambiguity: observer requires exactly one post-dispatch launcher per
  invoked Triton autotuner; it cannot label benchmark alternatives as selected.
- Report claims: analyzer asserts the three captured dtype graphs and the explicit
  baseline policy label. Negative controls reject incorrect labels/counts.
- Hashing: diagnostic hash helper supports detached contiguous CPU bytes without
  imposing a contiguous requirement on future diagnostic input views.

Verification after these small evidence-handling follow-ups:

- Five fresh backend/history/policy captures passed and retain 115 matched inputs
  each, 11 timing fixtures and seven profiler captures per timing fixture.
- Existing four-plan physical corpus reran on 109 matched inputs.
- Both actual Nsight counter captures reran successfully; component correctness
  checked and raw CSV hashes validated.
- Nine exception/data/report integrity controls passed.
- Analyzer regeneration followed by strict verification passed; Ruff checks,
  formatting and git diff checks passed.

No production algorithm, physical policy, numerical tolerance or runtime contract
changed. These are the review's bounded harness/documentation follow-ups, not a
new reduction implementation requiring a second architecture review.
