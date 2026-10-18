# Review disposition

The required review-mcp review is retained in [review.json](review.json), including
its trace. It found no numerical/ownership blocker and two medium provenance
issues. No native agent review was invoked.

| Finding | Disposition and verification |
| --- | --- |
| Medium1: shared adapter hash shadowed | Fixed: preserve `compiler_identity.adapter`, add `layout_adapter`. Original identities/phase sources remain immutable. Separate post-review78-case numerical and90 live/12 stream gates pass; all12 corrected-identity PTX bodies equal the originals after entrypoint-name normalization. Both adapter digests are checked by the final analyzer. |
| Medium2: wrapper retention not unique/hash-bound | Fixed retention helper uses cache-relative paths, one source read, collision rejection and per-file hashes. Future timing packets include that inventory. Current raw timing packet remains unchanged: a separately hash-linked supplement retains50 sources and semantic/policy bindings for33 cases. Original capture lacks exact executed cache-filename metadata; the documentation explicitly preserves that limit. |
| Low3: per-buffer layout map assumes one load/buffer | Safe only under the exact12-template admission. Final analyzer verifies unique loaded buffer IDs in each admitted template. Native emitter remains frozen; this is not a generic repeated-load adapter. Unsupported descriptor changes continue to reject before IR. |
| Low4: neighbor chronology relies on recorded ordering | Accepted residual evidence limit, explicitly retained. Original33 CPU gate preceded GPU source;66-case extension ran after source and before first GPU launch. No post-hoc timestamp or historical claim was added. |
| Low5: counter overwrites admitted candidate artifacts | Fixed: load the retained PTX through the existing kernel loader with `compile_ptx=False`, checking recorded PTX hash first. Focused counter rerun leaves all156 original candidate files unchanged; no new timing result is used. |
| Low6: missing capture hash gate | Fixed: validate `cpu.capture_sha256` before compiler identity or emission. The suggested removal of `kernel.directory = path` is not applied: direct retained `Kernel` lacks that attribute in the actual tested stack, as demonstrated by the preserved harness-artifact-path failure. |
| Info7: shared Cargo target path name | Kept and documented. Distinct binary name and binary hash identify this probe; reusing compiled dependencies does not change ownership or the kernel body. |
| Info8: opcode substring grouping | Fixed to opcode-boundary matching; aggregate executed counts still match each invocation's imported metric exactly. |
| Info9: measurement scope/clock honesty | Preserved. No stable pooled intrinsic cost, universal win or promotion claim; all losing samples/clock transitions remain. |

These are provenance/retention and audit corrections, without new GPU arithmetic,
address expressions, work assignment or native emitter changes. The affected
numerical/runtime and artifact checks are rerun; a second broad review is not
needed for these bounded follow-ups. Final Ruff, Rust formatting, diff-whitespace
and the full offline analyzer results are retained in `quality.log`.

Production/reduction/default/precision/warmup paths remain unchanged. The trial
is complete as an explicit qualified diagnostic; Stage5 remains active and
Stage6 remains deferred. No commit, push or remote task-tracker sync is performed.
