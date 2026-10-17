# 8x0 review disposition

Required review-mcp returned **no blockers** and verified the pinned component,
ownership, launch, ordering, capture and trial-boundary claims against source
and artifacts. The complete response is retained in `review.md`.

1. **Reproducible evidence verification:** retained `verify.py` checks every
   source hash/anchor, baseline artifact/hash/integrity binding, protected
   source versus HEAD and upstream diff/revision identity. The README labels
   `source-evidence.json` as the manually assembled fixed source snapshot.
2. **Exact corpus tolerance policy:** capture now passes and records
   `equal_nan=True`; verifier executes only the pure `bench.py::tolerances`
   function (avoiding that module's CLI) and checks every recorded policy field.
3. **One FP32 sum and reduction-scope store:** verifier checks the actual copied
   semantic descriptors, including sum type/count, accumulator/storage dtype,
   store count/domain and entrypoint-to-launch identity. A descriptor assertion
   is used instead of a brittle generated-CUDA text pattern.
4. **Device/cwd/integrity:** capture uses the same explicit current CUDA device
   for tensors and compiler identity, resolves git through PROJECT, and verifier
   cross-checks copied integrity manifests and their canonical descriptor hash.

The final capture repeats all nine baseline cases after these changes; the
offline verifier, Ruff F checks, Ruff formatting and `git diff --check` pass.
These are small evidence-harness corrections; no candidate/backend arithmetic
was added, so the existing review covers the unchanged selected mechanism and
no second implementation review is needed. Candidate qualification remains the
separate `5nb` task.
