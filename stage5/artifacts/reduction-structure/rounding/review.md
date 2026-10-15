# Review: adjacent-private-merge experiment

## Verdict

No blocking correctness, benchmark-validity, or eligibility defects found. The experiment is implemented exactly as described: one arithmetic change gated to a single legal topology, explicitly diagnostic, with old identities byte-preserved and both validators symmetric. Findings below are low-severity robustness/documentation items.

## Verified against the stated intent

**Merge arithmetic and scope (`descriptor3.rs:559-580`)**
- Adjacent branch emits exactly `(values[0]+values[1]) + (values[2]+values[3])`; the retained halving loop for 4 accumulators is `(a0+a2)+(a1+a3)`. Both use 3 uncontracted `FAddOp`s — matching the "three ordinary FP32 adds" claim, mechanically confirmed downstream by `reduction_merge_analyze.retain_kernels` (exactly 2 changed `" + "` CUDA lines after entrypoint normalization; equal PTX `add.f32` counts; fixed-final CUDA/PTX byte-identical to the historical capture).
- The change is confined to the post-loop private merge; batching, guards, plane butterfly, publication, barriers and final kernel are untouched.

**Gating symmetry (`descriptor5.rs:103-115,189-198` vs `plan.py`)**
- Rust: `adjacent` requires `batched && accumulators == 4`; `combined` (which includes batched) forces `workgroup==[256,1,1]`, `width==1`, `outputs==1`, `plane_width==32`. Non-string `private_merge` values (`null`, `true`, `1`) fail at `as_str`. So adjacent is reachable only as scalar256/four-chain/load-batch — as claimed.
- Python: `from_dict` enforces exact catalog membership, so all 11 invalid neighbors in `reduction_merge_checks.py` (unknown/None/True/int merge, wrong strategy, 2 accumulators, 128 workgroup, width 2, plane 64, 2 outputs) are rejected on both sides; both rejection paths are actually executed against the real Rust binary, and the FP32 counterexample (2^24,1,−2^24,0: halving exact=1, adjacent=0) is genuine and correctly preserved as evidence of no universal dominance.

**Identity/cache preservation (`plan.py:28-44`, `runtime.py:415`, `artifact.py:385-394`)**
- `to_dict()` deletes `private_merge` when `"halving"`, so the old load-batch serialized identity and artifact-cache digest are unchanged; the adjacent candidate gets a distinct entrypoint/directory. `compile_kernel` keys on `(semantic, device, policy)`, and `tuning_source` hashes `plan.py`/`runtime.py`, so prior selections can't be silently reused across the change. `checks.py` asserts baseline-omission and identity-inequality directly.

**Automatic admission still prohibited**
- `legality.numerical_reason` rejects every `plane_shared_load_batch` plan (both merge orders) before timing; `reduction_policy.qualified_for_runtime` admits only `shared_tree_compensated`; `select()` only ever sees `candidates()` output. `checks.py:29-30` asserts both guards explicitly. The runtime diagnostic branch (`runtime.py:167-201`) applies adjacent only above `LONG_REDUCTION_THRESHOLD=4096` and re-validates structurally.

**CPU model fidelity (`reduction_rounding_probe.py`)**
- The model mirrors the executed topology exactly: split-9 zero padding (guard semantics), 4×256 chain tiling advanced by 1024 per iteration with persisted accumulators, adjacent/halving merge, increasing-XOR (1..16) plane butterfly, neutral-filled first-8 shared lanes, halving final tree. Baseline partials *and* final output are asserted bitwise against executed bytes; the qualification harness re-asserts bitwise model match for **both** plans on every one of the 148 cases, including FP16/BF16 (single final rounding barrier reproduced via `.to(x.dtype)`).

**Counts, timing rule, and claims**
- 148 = 109 original (case+input-hash matched to the historical capture) + 6 storage + 32 new held-out + 1 non-default-stream/allocator-churn repeat; 178 graph cases; 18 RMS phases (3 dtypes × 3 shapes × forward/backward); 12 timing cases (3 dtypes × 4 shapes); 3 rotated rounds with proper Latin rotation. The `stable_win`/`stable_loss` predicate is exactly ≥2/3 round medians plus disjoint pooled median±3·MAD, and `analysis.json` only asserts *no stable loss* vs the old plan — "no established regression / not acceleration" is accurate. Compensated comparison is recorded but not gated, matching the hedge.

**Historical-evidence preservation (`reduction_merge_gates.py`)**
- All 18 gate commands come from the parent packet; the corpus gate's output is redirected into this packet; tracked artifact files are snapshotted (current working-tree bytes, which is the right semantics given the uncommitted base) and restored in a `finally` even on failure, with sha256-stamped logs in ignored cache and compact captures recorded. No completion is claimed; `analysis.json` does not incorporate gate results.

## Findings

1. **Low (robustness), `descriptor3.rs:562-567`** — the adjacent branch indexes `values[2]`/`values[3]` with no local guard; safety rests entirely on `descriptor5.rs` being the sole JSON caller. A future caller of the `pub(crate) build_physical` passing `"adjacent"` with `accumulators != 4` panics on index-out-of-bounds instead of returning `Err`. A `debug_assert_eq!(private_accumulators, 4)` or an early `"adjacent merge requires four accumulators"` error would make the invariant local. (Today's two call sites are safe: `descriptor3::build` passes literal `"halving"`/1.)

2. **Low (documentation/overclaim risk), STAGE5_REPORT.md "What remains unresolved"** — "The latest investigation reran all18 regression gates successfully …" immediately follows two paragraphs about the adjacent follow-up, while the packet README (correctly) says gates are recorded "as they complete; no success is implied by a pending gate." A reader can attribute a completed 18-gate rerun to this experiment. Recommend rewording to name the load-batch pass explicitly (report file is context, not focus; the in-packet wording is fine).

3. **Low, `reduction_merge_gates.py`** — results file is a bare array with no `complete` marker or expected-count assertion, so a partial run (runner exits on first failure) is indistinguishable from a full pass without counting entries. Add `"expected": 18` / `"complete"` once finished. Also, only *tracked* artifacts are restored; untracked files a gate creates inside artifact directories would linger (mostly covered by the corpus redirect and `.gitignore`).

4. **Info, `reduction_merge_benchmark.py`** — per-plan outputs are recorded but not numerically re-verified in this harness (output verification lives in the qualification harness over identical kernels/semantics). Acceptable; noting for traceability.

5. **Info** — `descriptor5.rs` accepts an explicit `"private_merge": "halving"` on strategies that never carry it (e.g. pointwise), while Python's catalog-membership check rejects the same identity. Unreachable through the Python pipeline and semantically identical to the default; harmless asymmetry.

## Minor notes

- `reduction_merge_qualification.py` reference lookup via `next(...)` raises `StopIteration` rather than a descriptive error if a label/regime ever goes missing — fine for a fixed harness, slightly unfriendly if corpora evolve.
- The stream/allocator repeat correctly re-uses the `storage_torch.float32_8` label, so `qualify_case`'s `input_sha256` equality doubles as an RNG-integrity check for the non-default stream — good.
- NaN/Inf/−0 specials are checked against ATen exactly (`rtol=0, atol=0, equal_nan=True`) plus signbit, and deliberately excluded from the CPU-model bitwise asserts — consistent with the "NaN payloads are not a sum contract" comment in the Rust.

Overall: the scoped claim set (bounded diagnostic qualification, no default change, no FP64 target, no Triton speed claim) is faithfully encoded and mechanically re-verified by the analyze step; nothing found that would invalidate the packet's evidence.

---
## Review Trace
- Workspace: /home/imi/Projects/pytortch-backend-experiments
- Diff target: HEAD
- Context entries requested: 3
- Context files loaded: 3
- Files selected for initial diff: 13
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 16441 prompt tokens, 2189 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: list_changed_files
- Tool call: search_repository
- Model call: API usage 20712 prompt tokens, 186 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_files
- Tool call: read_file
- Model call: API usage 27677 prompt tokens, 568 completion tokens
- Model call: model requested 3 tool call(s)
- Tool call: search_repository
- Tool call: read_file
- Tool call: search_repository
- Model call: API usage 29090 prompt tokens, 773 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 32553 prompt tokens, 969 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 37555 prompt tokens, 624 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 40255 prompt tokens, 996 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 43523 prompt tokens, 45 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 44682 prompt tokens, 767 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 48786 prompt tokens, 376 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 49544 prompt tokens, 328 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 49934 prompt tokens, 87 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 50161 prompt tokens, 58 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 51447 prompt tokens, 93 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 51564 prompt tokens, 29 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 51978 prompt tokens, 32 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 52795 prompt tokens, 2389 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: list_repository_tree
- Tool call: search_repository
- Model call: API usage 55421 prompt tokens, 331 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 56526 prompt tokens, 2657 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: search_repository
- Model call: API usage 60322 prompt tokens, 568 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 63875 prompt tokens, 2677 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 67187 prompt tokens, 86 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 67323 prompt tokens, 33 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 67525 prompt tokens, 448 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 67984 prompt tokens, 30 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 68236 prompt tokens, 5784 completion tokens
- Review complete after 26 model call(s)
