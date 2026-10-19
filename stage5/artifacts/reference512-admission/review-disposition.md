# Reference512 admission review disposition

Required review-mcp returned after26 model calls: implementation matches intent,
no correctness bug found in the admitted path. Full output: `review.json`.

1. **Scheduler lifetime: hardened.** Records are reset when the owning GraphLowering
   changes; each deferred factory captures its own graph and record list. A later
   scheduling instance reuse cannot redirect the earlier closure to another graph.
   The seven actual graph/neighbor/adversary transitions pass again afterwards.
2. **Missing fallback: hardened.** Selection now rejects a missing qualified fallback
   explicitly, before timing, rather than reaching StopIteration. The negative
   control verifies this. A fresh deterministic selection retains compensation.
3. **Unexpected environment failure: explicit fail-closed decision.** Unsupported
   platform/API excludes reference512; an unexpected failure of save/setup/restore
   on supported glibc is a validation error, not a swallowed performance fallback.
   Restore remains in finally. The16 rounding/FTZ/thread controls pass again.
4. **Additional shape guards: disclosed.** Neighbor choices remain unchanged, but
   equality classification may increase guards/recompilation. No compile-time or
   universal host-overhead improvement is claimed.
5. **Symbol-slot convention: asserted/documented.** Two scalar slots/symbols are
   required, with slot0=N and slot1=B following the exact template and existing
   Stage2 symbol plumbing. No descriptor or scalar ABI change.
6. **Minor items.** Model-check events now include plan, policy and oracle identity.
   The final context's constant false admission fact is harmless and retained.
   Bit-view checks remain bounded to the exact contiguous FP32 template. Empty
   negative-control lock directories are disposable cache data, not evidence.
7. **Code identity pending in the review's initial tree: completed.** The main
   agent ran all6 normal-compiler IR/CUDA/PTX comparisons during review. Each is
   byte-identical; `identity/` and `identity.json` are retained and independently
   rehashed by the offline audit. The initial listing was stale relative to that
   later completed capture, not evidence that a comparison failed.

Post-review affected checks pass: fresh tuning1model check, fresh-process reload
0model/0compiler, context and corruption controls,16 exact FP environment restores,
deterministic fallback, missing-fallback rejection, seven generated graph calls
and nondefault stream/offset canaries. No GPU body changed. The original148 corpus,
six identity controls and18-group results remain applicable. The independent audit,
Ruff F/formatting, Python compilation and diff checks pass. These small hardening
and diagnostic corrections did not require another review.

The regression runner's untracked-artifact loss was discovered by the main agent
while the review ran; its original reused --output path overwrote the prior
adoption corpus JSON. The current README and prior adoption packet explicitly
disclose this; the original complete-run log still hashes correctly. The new
168-case result is retained separately. The runner now redirects outputs into its
own packet and resolves relocated successful logs. The review's tracked-snapshot
assessment must not be read as a claim that all untracked history was protected.
