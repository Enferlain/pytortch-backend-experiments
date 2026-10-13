# Stage 4 review disposition

Required `review-mcp.review_with_context` completed against `HEAD`, including the new untracked implementation. The full review and diagnostic trace remain in the ignored local `review-core.txt`. The reviewer found no wrong-result or memory-safety bug in the implemented surface. This review covered the core while the acceptance run was still in progress; final test results below are independent execution evidence, not claims made by the reviewer.

| Finding | Disposition |
|---|---|
| Missing Stage4 ABI availability guard | Added `stage4_metadata` / `stage4_launch` export checks at backend selection. Two mocked stale-library component tests reject with `CubeCLBackendUnavailable`. |
| Atomic write dependency access could crash | Check one named write dependency before querying its dtype; unknown dependencies become capability rejections. |
| Raw literal first operand assumed typed | Select a typed operand explicitly; absent typed operands reject. Scalar constants retain the common replay's already-promoted semantic dtype. |
| Diagnostic drift | Corrected effect/fusion and internal extrema messages, supported input-count diagnostic and unreachable schema4 manifest condition. |
| Counter-width invariant implicit | Source reread showed random lowering explicitly requests i32, without a global fits-in-i32 guarantee. Added a pre-fusion ShapeEnv extent guard, runtime check and constant-descriptor check. A live FakeTensor CUDA graph of length 2^31 rejects before compilation/RNG consumption; signed counter bit patterns are separately tested. |
| Direct i32 index descriptor arm | Retained as a validated component representation; normal replay is i64 index plus explicit cast. No additional tensor-integer coverage claimed. |
| Process-lifetime caches/modules, no capture | Explicit limitations retained. Stage4 rejects CUDA graph capture before enqueue. No concurrent compilation claim. |
| Acceptance/regressions not yet complete at review time | Final 15-category acceptance, 11 runtime checks, descriptor/component checks and all 11 baseline gates pass. Machine evidence is preserved beside this file. |

Additional final validation hardening: memory `value` leaves require a checked-indirect SSA identity, and checked provenance is scoped to lazy regions. Two neighboring malformed-descriptor tests reject before IR construction. This changes validation only; the accepted IR/compiler route remains unchanged.

The follow-up changes were bounded validation/availability/diagnostic corrections, so a second broad review was not requested. Rust clippy/fmt, Python Ruff checks, native C warning checks and affected acceptance/component gates were rerun. No native review fallback was used.
