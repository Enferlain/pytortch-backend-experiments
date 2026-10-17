# wqc review disposition

Required `review-mcp` review of HEAD plus the explicitly named untracked
implementation/evidence returned **no blockers**. It independently traced the
actual CubeK/CubeCL component order and adapter indexing, checked guards,
artifact bindings and numerical records, and confirmed the timing caveats and
performance stop. Full response/trace: [review.md](review.md).

1. **Filter precedence:** added explicit inner parentheses to the graph-case
   filter. Its meaning is unchanged; no capture was repeated or replaced.
2. **Review record:** retained the full review and this disposition in the
   previously forward-referenced directory.
3. **Evidence assertions:** the analyzer now preserves a static count of
   128-bit LDG opcode suffixes and asserts zero for the vector4 primary. It
   also asserts that every retained process-query result has no compute rows.
   These checks pass against the original captures. They do not establish
   GPU exclusivity or per-kernel clock frequency.
4. **Chronology:** the initial gate is producer-attested, with the original
   source snapshots and CPU JSON hash bound into every candidate identity.
   The actual pre-source CPU run is preserved; later reproduction uses a
   separate `cpu-recheck` record. No original gate was rewritten.
5. **Unused scalar read expansion:** retained the harmless inherited method.
   The actual operand is typed `VirtualTensor<f32, Four>` / `Operand`; the
   scalar `Replay` instances are metadata and the output sink. No selectable
   scalar fallback or extra mechanism is introduced. No Rust source/binary
   change was needed for this note.
6. **Combined worktree:** hs6 and wqc are separate Beads tasks/evidence packets
   and currently share the uncommitted worktree. The 13,266-file preservation
   check includes the completed hs6 packet and the previously committed 5nb
   files. No commit/push was made.

Affected full offline checks pass after the small follow-ups: 36 CPU/GPU cases,
868,800 output elements, 27 live negatives/three stream-offset controls,
12 static negatives, 36 scalar-control qualifications, 180 exact raw timing
traces and matching executed-SASS/raw-counter totals. Scalar-control
IR/CUDA/PTX equals the frozen scalar code modulo entrypoint names. All protected
earlier file hashes pass; pinned build/provenance checks pass. Ruff F/format,
Cargo format and `git diff --check` pass. GPU code, binary, numerical outputs
and timing records were unchanged by review fixes; no retiming or second
mechanism was warranted. The historical 18-group production suite was not
rerun or claimed for this isolated diagnostic addition.

**Decision:** close wqc as a numerically qualified trial with no observed graph
win against ordinary, retaining the isolated clock-transition limitation.
Keep diagnostic-only, with no tuning, fallback or automatic eligibility.
The remaining packed-load capability question is tracked separately in
`pytortch-backend-experiments-9e8`, unstarted and source-first. It is not an
extension of this stopped trial. Stage 5 stays active; cwk.5 stays parked;
Stage 6 stays deferred.
