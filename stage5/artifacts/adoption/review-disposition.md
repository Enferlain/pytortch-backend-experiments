# Required review disposition

`review.json` is the complete review-mcp response for HEAD plus the untracked
admission/compiler/test sources. Verdict: no correctness or safety bugs.
The reviewer independently traced Python/native admission, FastDivmod parameters,
metadata ordering, live bounds and proof/cache isolation. It explicitly treated
the still-running regression suite as provisional, not accepted evidence.

1. **Example metadata blob disagreed with the manifest — fixed.** The column
   example now adds the four FastDivmod scalars, matching the real seven-scalar,
   80-byte ABI. The post-review control checks all four native example lengths
   and repeats IR/CUDA/PTX identity plus the full78/28-live/4-stream controls.
   This changes a non-executed example file, not GPU code or runtime metadata.
2. **Cross-tree template include — documented.** The native module now says to
   rebuild stage0 when templates change. Both sides deliberately share one source
   contract. A stale binary rejects rather than silently accepting a changed
   Python contract. This repository is currently a multi-stage experiment, not
   a standalone distribution of the stage0 crate.
3. **Unused generic layout path — retained intentionally.** The production
   module translates the qualified body rather than pruning another mechanism
   in this pass. Four exact templates are the entire admission boundary; any
   extension must qualify newly exercised paths. Presence of generic source is
   not evidence of generic capability. The inherited clone-on-copy Clippy
   warning is similarly retained rather than editing the qualified expansion.
4. **Non-numeric scalar conversion exception — scoped.** Generated wrappers
   supply integer symbolic scalars. Invalid Python objects can raise TypeError
   before launch, consistently with existing coercion conventions. No wider
   direct-call Python scalar API is claimed.
5. **Extra alignment classes in cache keys — retained.** This conservatively
   splits proofs and preserves the existing selector mechanism; it cannot admit
   unsupported extents. No cache-key optimization is part of this pass.

The component-census follow-up was also observed by the reviewer: generic `add`
fixtures must record rejection of a column-only plan, just as they record final9
rejection. Admitted column forms are tested by the dedicated controls. The
original census failure is retained under `failures/`.

No second review is required for the small example-file correction and comment;
affected checks are rerun. Final regression status is in `regressions/gate-status.json`.
The initial corpus SystemError and its subsequent repeats remain separate raw
evidence; they are not erased by a later successful run.
