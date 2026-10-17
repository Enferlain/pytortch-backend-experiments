# hs6 review disposition

Required `review-mcp` review of HEAD plus the explicitly named untracked host
probe and evidence harness returned **no blockers**. The full response/trace is
retained in [review.md](review.md). It confirmed the mandatory-strategy API,
conditional plane geometry/vector policy, source/artifact attribution, exact
SASS region and changed-order requirements for a future vector4 trial.

1. **Stale pass marker:** the verifier now writes `complete=false` before any
   probe or assertion. The focused `check_failure_marker.py` starts from a valid
   result, injects a corrupt host-probe response, observes rejection and checks
   that the preceding pass is cleared. The full valid verifier was then rerun
   and restored `complete=true`. This test modifies no frozen input/evidence.
2. **Retained failure:** `setup-order-failure.txt` is now hash-bound in
   `evidence.json`, alongside the accepted build record.
3. **Review directory:** this disposition and the complete review now occupy the
   previously forward-referenced directory.
4. **Cap wording:** clarified that the binding cap is CubeK's constant eight
   planes in its GPU branch; the modeled hardware cap is 32 and does not bind.
5. **Frozen SASS/binary anchors:** intentionally retained. This verifies the
   exact original artifacts, including the actual host executable digest.
   Different artifacts/toolchains require a new evidence packet, not silent
   normalization or a verification-time identity refresh.
6. **Path dependencies:** the registry-package check is one part of identity.
   Source/copy digests, upstream heads/diffs, the locked/offline build and the
   historical-file manifest enforce the relied-on path-source provenance.

Affected checks pass after the small fixes: host helper nine rows/four controls,
source/copy and lock checks, exact retained IR/CUDA/PTX/SASS assertions and
13,249 historical file hashes/upstream states. Ruff F/format, Cargo format and
`git diff --check` pass. No Rust probe or GPU source changed after review; the
accepted host build remains `--release --locked --offline`. No production GPU
regression suite or benchmark was rerun or claimed.

Close hs6 as completed attribution. The single proposed vector4 implementation
is tracked separately as `pytortch-backend-experiments-wqc`, unstarted. It must
qualify the changed arithmetic on CPU before any GPU implementation. The
original 5nb result stays closed and byte-identical. Stage 5 stays active,
cwk.5 stays parked, Stage 6 stays deferred. No commit, push or remote sync.
