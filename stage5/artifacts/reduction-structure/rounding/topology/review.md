I now have a complete picture. Let me compile the review.

---

# Code Review: `plane_shared_reference512` diagnostic topology

## Verdict

**No blocking correctness or benchmark-validity defects found.** The implementation matches the stated intent: one GPU candidate reconstructing the Triton R2048/16-warp topology, grouped protected loads retained, fixed split9/final untouched, unconditionally ineligible for automatic selection, with the narrowly-scoped native geometry. Findings below are minor observations, none of which I'd block on.

## Verified against intent

**1. Work assignment / arithmetic (`descriptor3.rs`)**
- Chain assignment `r = t + 512j + 2048k` is correct: `RangeLoopOp(unit, extent, step*4)` with per-accumulator offset `i*cooperation` (`cooperation = workgroup[0] = 512`, 4 accumulators).
- Adjacent private merge `(a0+a1)+(a2+a3)` emitted only for `private_merge == "adjacent"`; the halving path for existing plans is byte-identical to before.
- Intra-plane reduction uses literal decreasing `ShuffleXorOp` distances `[16,8,4,2,1]`; inter-warp uses `[8,4,2,1]`; both add `FAddOp` per level, matching the CPU model's `butterfly` (RN32 per add, XOR pairing) and the reference's `add.rn.f32`/`shfl.sync.bfly` evidence.
- The inter-warp butterfly runs inside the `first_plane` branch where all 32 lanes of warp 0 are active, and distances ≤8 never cross the half-warp boundary, so lanes 0–15's values are never polluted by the neutral zeros the `live` guard (`plane_lane < 16`) gives lanes 16–31. Only lane 0's result is stored — bit-identical to the model's replicated-load emulation. The divergences from Triton (guarded load vs `tid&15` replication, single leader store) are documented and don't affect the owner's arithmetic.

**2. Batched protected loads / neutral-update equivalence.** The batched path adds `+0.0` to accumulators on iterations where `r ≥ E` (reference skips the update). I verified this is exactly value-preserving: accumulators initialize at `+0.0`, and in RN an accumulator can never become `−0.0` (that would require both addends `−0.0`), so `x + (+0.0) = x` always holds. The `−0.0`/NaN/±Inf specials and the 149 bitwise partial/final comparisons confirm this empirically, including the subnormal adversary (no FTZ on either side).

**3. Full-epilogue barriers/lifetime.** Publication `sync` → first-warp reduce → root publication `sync` → all-units `shared[0]` read, with an additional `sync` before any subsequent reduction's scratch reuse (line 842). No barrier sits under a unit-dependent predicate (`collective_guard` is workgroup-uniform via `CubePosX`). Global partial store is a unit-0 leader store, matching "only thread 0 stores partial[block]". Canary-backed tmp/out backings, non-default-stream repeat with allocator churn, and original-pointer assertions are all present in the probe.

**4. Validators (`descriptor5.rs`) and negatives.** The `plane_shared_reference512` arm pins `[512,1,1]`, `outputs=1`, `width=1`, `accumulators=4`, `private_merge="adjacent"`; `combined` then requires `[512,1,1]` only for this strategy (`load_batch`/`plane_shared` remain 256-max). The 512 geometry is exclusive to the new strategy. The 12 Python/Rust invalid-plan neighbors (including `private_accumulators: True`, correctly rejected by the `type(...) is not int` check) and 6 native geometry rejections cover the neighborhood well.

**5. Native geometry (`launcher.c`).** The clause admits exactly `(512,1,1)` with `shared == 64` and nothing else above 256 threads; `shared > 2048` and the 2048-bytes-only-for-(256,1,1) rules are preserved. Shared size `workgroup[0]/32 = 16` floats = 64 bytes is consistent end-to-end. Precedence/short-circuit behavior is correct (verified the key cases: 512/0, 512/32, 512/2048, (256,2,1)/64, (1024,1,1)/64 all rejected; (256,1,1)+2048 still accepted).

**6. Diagnostic-only ineligibility.** `numerical_reason` rejects `plane_shared_reference512` unconditionally on schema 3 (and `structural_reason` rejects it on non-reduction schemas via "wrong semantic family"), so the **accepted** plan list — and therefore the tuning cache key's `catalog` field — is unchanged for default paths. `qualified_for_runtime` remains compensated-only. The runtime diagnostic branch consults `structural_reason` only (explicit request bypasses numerical policy), consistent with the other diagnostics.

**7. Fixed split9/final unchanged.** The probe re-compiles the retained partial/final semantic packets and asserts the final's `kernel.cu`/`kernel.ptx` hashes equal the retained `fixed_final` artifacts — a byte-level proof, not just a plan-identity claim. `finish_model`'s halving tree reproduces the fixed final's owner arithmetic, validated bitwise on all 149 checks.

**8. Provenance/qualification/timing harnesses.** Harness hash chaining (`qualification.json` → probe), input-source hashes, input-hash agreement with the prior packet, `evidence.inventory()` GIL/native-module independence, 148-case CPU qualification reproducing all 108 applicable retained Triton outputs bitwise plus the adversarial partial buffer, round-matched graph references in the 65K band with `eager_failures_preserved` kept separate (correctly *not* treated as qualification failures), rotation C/T,T/C,C/T with the ≥2/3-round plus disjoint median±3MAD rule, 384-event strict counts, and the NCU SASS-vs-`smsp__inst_executed.sum` cross-check all hold together. The gates runner's snapshot/restore covers tracked artifacts plus the previously-untracked SOURCE packet, matching the "ignore transient artifact diffs" instruction.

## Minor findings (non-blocking)

1. **`qualification.json` field `"gpu_candidate_implemented": false`** is written by the CPU-model harness even though this pass does ship a GPU candidate; the field really means "this packet contains no GPU execution." Consider renaming (e.g., `gpu_execution_included`) or documenting it in the README to avoid future misreading of the packet.

2. **One-time re-tuning cost on default paths.** `AutotunedKernel.identity["tuning_source"]` hashes `plan.py`/`legality.py`/`runtime.py`, all of which changed. Since `key_data` includes `identity`, every persisted empirical selection key changes → first post-change `first_use` runs re-validate/re-time even though the legal candidate set and outcomes are provably identical. This is the established conservative "source hashes invalidate old timing selections" policy, but given the "no policy/default changes" requirement, it's worth stating explicitly in the README/disposition that a one-time re-selection cost (not an outcome change) is incurred.

3. **Benchmark JSON key naming.** In `component-timings.json`, the old load-batch plan is keyed `"baseline"` while the actual frozen baseline (`shared_tree_reduction_baseline`) never appears in this packet. The embedded `plan` dict disambiguates, but a future reader diffing against reports that use "baseline" for the frozen plan may trip; renaming to `"load_batch"` would match the README table labels.

4. **Gates restore scope.** `reduction_topology_gates.py` protects tracked files plus everything under `SOURCE` (excluding `OUTPUT`); any gate writing into a *different* untracked artifact directory would not be restored. This is the established mechanism from the prior pass and the 18 gates' outputs are redirected under `SOURCE → OUTPUT`, so risk is low — noting it only for completeness.

5. **Short-extent diagnostics fall back silently.** `reference512_diagnostic` (like its siblings) uses the frozen baseline for extents ≤ 4096. Pre-existing shared behavior; no signal is emitted. Fine as-is, but a one-line warning could help future diagnostic runs.

## Notes on things I specifically checked and found sound

- The model's `torch.where(coords + k*2048 < extent, ...)` exactly implements "update only when `r < E`" with zero-padded data handling the `p*E + r ≥ N` protected loads; the independent `scalar_model` in `reduction_topology_checks.py` (struct-pack RN32, no tensor reshaping) cross-validates it on 6 extents including 0, 1, 513.
- `PhysicalCandidate` round-trip and catalog membership are consistent; the new catalog entry is required for `from_dict` deserialization of the artifact packets and cannot leak into eligibility.
- The `.gitignore` additions sensibly un-ignore the curated reduction-structure/topology packets while excluding bulky `representatives/` (hashes recorded in-packet instead); `*.csv`/`*.log` raw captures stay local by the global rules, consistent with the analyze harness storing only paths + SHA-256s.
- Changed-file inventory matches the task description; the stage0–4/accumulation artifact churn is gate-runner transient state that the runner restores.

**Conclusion:** The change is additive, narrowly scoped, and well-evidenced. The numerical-equivalence reasoning for the two intentional deviations from Triton's structure (neutral batched updates, guarded half-warp loads) is sound and empirically confirmed. Recommend accepting after considering the minor naming/documentation notes (findings 1–3); no code changes are required for correctness.

---
## Review Trace
- Workspace: /home/imi/Projects/pytortch-backend-experiments
- Diff target: HEAD
- Context entries requested: 10
- Context files loaded: 10
- Files selected for initial diff: 15
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 213946 prompt tokens, 6077 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 224662 prompt tokens, 95 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 226535 prompt tokens, 60 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 228253 prompt tokens, 4932 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 236955 prompt tokens, 352 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: search_repository
- Model call: API usage 239035 prompt tokens, 1101 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 240160 prompt tokens, 115 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 241207 prompt tokens, 5819 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 248020 prompt tokens, 3985 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: list_changed_files
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 255144 prompt tokens, 271 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 255460 prompt tokens, 59 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 256389 prompt tokens, 1279 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: get_uncommitted_changes
- Model call: API usage 257672 prompt tokens, 30 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 258102 prompt tokens, 119 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 258382 prompt tokens, 8737 completion tokens
- Review complete after 15 model call(s)
