# Review disposition

Required `review-mcp` review against HEAD returned without a blocking
correctness, benchmark-validity or eligibility finding. `review.md` preserves
the full response and trace. The reviewer inspected the initial 148-case capture;
the later existing finite adversaries add one rejected case with three failed
elements. No review endorsement of candidate admission is implied.

Actionable low findings:

1. **Local invariant:** `build_physical` now returns an error for adjacent merge
   with any non-four chain count before IR construction. A Rust unit test covers
   0,1,2,3,5. Both valid caller paths were already safe. Generated valid-kernel
   CUDA/PTX identity is checked after this host validation change.
2. **Historical attribution:** the current summary explicitly attributes the
   preceding unseeded-failure/retry discussion to the load-batch investigation.
   This pass's own completed 18 gates and the candidate's adversarial rejection
   are reported separately.
3. **Gate completion/preservation:** the runner initializes an incomplete
   `gate-status.json` and records expected/finished counts, then marks completion
   only after all commands succeed. It snapshots the untracked prior packet too,
   redirects both corpus and precision outputs into this packet, and restores
   previous bytes on failure. CPU stand-ins execute success and intentional
   failure paths, verifying output redirection, untracked history restoration
   and completion status. The initial actual run's precision-output correction
   is documented in `precision-output-preservation.json`; all 412 parent-packet
   hashes remain unchanged.

Informational findings:

- Timing-harness outputs are diagnostic records, not a numerical admission gate.
  Separate direct and real-graph captures qualify their stated inputs; adding
  the existing finite adversaries specifically prevents default admission.
- An explicit default `private_merge="halving"` is accepted by Rust and is
  normalized away by Python `to_dict`; old identities remain unchanged. This is
  harmless default normalization, not extra adjacent capability.
- Fixed reference lookup failures remain explicit `StopIteration`; no fallback
  is introduced for a missing corresponding-regime reference.

All 18 full regression gates precede the small local guard and harness/status
corrections above. `post-review-gates.json` records affected controls, numerical
capture, real graph, normalization and analyzer reruns plus release Clippy.
`post-review-code-identity.json` verifies valid partial/final CUDA and PTX remain
byte-identical after the local guard. These are small review corrections; no
second kernel candidate or additional review workflow is introduced.
