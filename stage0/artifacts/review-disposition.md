# Review dispositions

The initial review and a focused follow-up both returned. Raw outputs are
retained in review.md and review-followup.md.

Initial findings addressed:

- Removed stream-handle metadata-readiness memoization entirely. Metadata is
  immutable; completed upload state is cached per N. prepare_capture(N) completes
  the upload event outside capture. Capturing launches query no CUDA event.
- Added fallback for a null CUDA error string.
- Added STAGE0_CARGO_PROFILE and clear missing-binary errors.
- Made NVRTC selection deterministic; ambiguous libraries require an override.
- Removed the dead output allocation; documented bounded per-N cache retention.
- Used the actual kernel device for timing labels.
- Qualified metadata field-order evidence: equal lengths cannot distinguish it
  numerically. Source/emitted IR establish that part of the contract.
- Corrected lifetime telemetry to separate recycled addresses from reuse
  observed before the completion event.

Follow-up low-severity findings addressed:

- prepare_capture's event-only synchronization now occurs outside the cache lock.
- Public prepare guards the kernel's actual CUDA device before capture checks.
- Public prepare returns no mutable metadata reference; metadata cache is private.
- CPU-only torch produces an explicit CUDA requirement error in NVRTC discovery.

Metadata cache misses remain serialized setup operations under the lock to
avoid publishing duplicate/partly initialized entries. The bounded concurrent
launch contract uses prepared metadata; this is not a claim of unrestricted
concurrent setup performance.

Both GPU suites were rerun after the final corrections. No third review was
needed for these small follow-up changes; the focus review had already verified
the corrected completion/capture/native-error paths.
