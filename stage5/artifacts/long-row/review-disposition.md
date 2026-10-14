# Review disposition

Required `review-mcp` review completed after 36 model calls against HEAD plus the
new source files and stage/FP context. It found **no correctness blocker**. The
raw review and tool trace are retained in the ignored cache at
`stage5/cache/long-row/review.md`.

All actionable findings were addressed:

1. Qualified finite-difference diagnostics now retain dtype/shape and special
   checks plus a versioned coarse corruption guard. This is not a numerical
   acceptance tolerance or universal reassociation bound; controls preserve the
   known accurate-1/old-tree-0 case and reject gross finite corruption. The
   compensated plan remains the sole eligible long-row plan, so mixed qualified
   tuning is not an executed default path.
2. Explicit diagnostic policies reuse `LONG_REDUCTION_THRESHOLD` rather than
   duplicating its literal value.
3. The tuning error lists all four experimental options.
4. Guard traces now record planning anchors. Focused checks assert stability of
   the tested SM86 short anchor across histories/first shapes, in addition to
   exact bounded graph-reuse assertions.
5. Mask-widening controls include a valid negative int32 lower bound, an invalid
   lower bound and a lower-unbounded interval, as well as upper-bound/arithmetic
   negatives.
6. The gate runner restores historical artifacts in `finally` and retains fresh
   results under this packet. Final worktree verification must leave only the
   legitimate new overlay patch outside the packet.

An additional parent check initializes the caller's lazy choices handler before
temporarily patching its factory, preventing the temporary factory from becoming
the cached default. Fresh-handler and post-backward identity checks cover this
restoration boundary.

The matched reference investigation was also extended with 8K/131K/million-first
Triton captures, so per-case qualification follows the efficient corresponding
regime rather than assuming the 65K-first topology is best for every size.
Additional harness controls reject wrong roles, input hashes, missing timing
rounds, forbidden native Triton mappings and incomplete captures.

These are small review/verification corrections rather than a new physical
algorithm or coverage extension. The affected checks and complete gate runner
are rerun before final handoff. No native-agent review substitutes for the
required review.
