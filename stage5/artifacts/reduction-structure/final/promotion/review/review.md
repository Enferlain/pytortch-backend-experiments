# Code Review: cwk.9 — `independent_planes_final9` promotion

## Summary

I traced the full promotion path: plan registration → Rust descriptor admission → codegen reuse → Python legality/selection → cache identity → per-call ABI guards, and audited the two check harnesses. **The change is sound. I found no blocking defects.** The guard symmetry, routing safety, fallback behavior, cache identity separation, and precision/storage/stream preservation all check out. Below is the evidence, followed by minor findings.

## 1. Semantic guard symmetry (Python `semantic_reason` vs Rust `validate_final9`) — ✅ symmetric

I compared both implementations clause by clause:

| Contract clause | Python | Rust | Verdict |
|---|---|---|---|
| schema==3, int64 index, exactly one positive int64 scalar, `outer`/`numel` == scalar slot 0, extent == constant 9 | ✅ | ✅ | symmetric (missing keys → reject on both) |
| effects: absent OK; non-object reject; any value ≠ `false` reject | `v is not False` | `v != json!(false)` | symmetric, including edge cases (`0`, `null`, non-object) |
| exactly 2 buffers, output dtype ∈ {fp32,fp16,bf16}, strict structural equality (offset 0, `[outer,9]`/`[9,1]`, `[outer]`/`[1]`) | ✅ | ✅ | symmetric; both strict key-for-key |
| 3 ops (FP32) / 7 ops (FP16/BF16), unique string ids excluding store, load index `add(reduction, mul(9, linear))`, fp32→ty cast, reduction (fp32 accumulator, source/semantic dtype = ty), trailing `ty→fp32→ty` cast chain, owner store (buffer 1, linear, outer scope) | ✅ | ✅ | symmetric (Rust's always-`ty` reduction dtypes collapse to the same value for `ty == "float32"`) |

The index expression `r + 9·o` is correct for the mandated contiguous layout, and dict/`Value` equality is key-order-insensitive on both sides.

**Codegen preconditions are guaranteed upstream.** `build_physical_codegen`'s `final_plane` probe gate requires `independent_planes`, `[32,4,1]`, width 1, accumulators 1, constant extent 1..32, single reduction, owner-only outer stores. The descriptor5.rs path enforces all of these before reaching the branch: the new match arm (`workgroup == [32,4,1] && outputs == 4 && width == 1`), the accumulator policy (final9 is neither `independent` nor `combined`, so `accumulators == 1` is forced), `private_merge == "halving"` forced, `elements == 1` forced, and `validate_final9` forces extent 9 and the op sequence. The branch returns before `build_physical`, so generic FSum and all partial emitters are untouched. `PhysicalCandidate.from_dict` additionally pins Python-side packets to exact catalog entries.

## 2. Automatic route / cache safety — ✅

- **Coarse guard exclusion**: `qualified_for_runtime` admits only `shared_tree_compensated` (reduction_policy.py:63-67). final9 is excluded, and `controls()` asserts it.
- **Never forced/default**: `frozen()` still defaults to `shared_tree_reduction_baseline`; `first_use` empirically tunes among `candidates()` output and keeps the frozen baseline unless a stable 3·MAD-disjoint win (runtime.py:340-352). Diagnostic policies are untouched and don't include final9.
- **Eligibility is exactly the contract**: `structural_reason` short-circuits final9 to `semantic_reason`, then falls through to the normal checks (family, `units=128 ≤ device_max_threads`, `plane_width == target["plane_width"]`); `numerical_reason` is a no-op at extent 9 (short-row policy, ⌈9/32⌉=1 ≤ 256). Non-32-lane targets are rejected both structurally (live query) and at artifact load (`Kernel.__init__` resources check), with Rust hard-pinning plane_width 32 — layered, consistent with "CUDA32 queried topology".
- **Fallback preserved**: `transitions()` proves extents 8/10 keep ordinary winners, per-extent object reuse, and exactly 3 selection keys; the `row_sum` dynamic fixture still admits `shared_tree_reduction_baseline` and never final9 (fixture has 2 scalars + dynamic extent → contract rejects).
- **Cache identity separation**: artifact keys rotate via the new `final_contract` digest plus changed `adapter` (artifact.py) and native binary digests; tuning keys rotate via `tuning_source` including `final_plan.py`. Reload asserts `COMPILER_INVOCATIONS == 0`; the graphs harness proves the rotated identity recompiles partials byte-identical to the retained prior artifacts, so the one-time full recompile is deterministic and safe.
- **Poisoned/stale-cache defense**: `Kernel.__init__` rejects any final9 artifact whose semantic fails the contract; `_enqueue` re-runs `reason()` per call (mirroring the vector path) — verified warm-cache rejections with the native launch counter unchanged.

## 3. Precision / storage / stream — ✅

The bitwise-parity claims are mathematically justified, not just empirically: the old 256-lane descending tree's stages ≥16 only add zeros, and stages 8/4/2/1 perform the *identical* nonzero association `(((p0+p8)+p4)+(p2+p6))+((p1+p5)+(p3+p7))` as the XOR32 plane tree — same operations in the same order, so identical rounding, signed-zero and NaN-propagation behavior. This makes `tensor_sha` equality between fixed_final and final9 (all 3 dtypes, NaN/±Inf/−0/subnormal/cancellation rows), the 148-case corpus parity, and the frozen-vs-promoted graph output equality valid and strong. Per-call ABI (shape/stride/dtype/device/storage-span/disjoint-output/positive scalar) stays in `_validate`; the offset-view + non-default-stream check and storage canaries pass through the unchanged launcher contract. `prepared`/`batch` and `_call_pointers` lifetime behavior is unchanged, and the harness asserts `_call_pointers is None` after ordinary launches.

## 4. Test validity — ✅ with notes

- The frozen-producer control subclasses `CubeCLScheduling` and only reassigns `self.tuning` per kernel between two *existing* policies via the official `scheduling=` extension point — no compiler monkeypatching or artifact substitution. The `_pre_fusion_custom_pass` patch composes with any prior pass.
- The same-symbol byte-comparison against the validated probe artifacts (IR/CUDA/PTX, 3 dtypes) plus SASS instruction equality after BRA/BSSY relocation is a solid reuse proof.
- Timing honesty: analysis.json matches the claims (1.888→1.535 µs isolated, 43.7895→43.4585 µs graph at 8×1M); the 64×65539 case has MAD ~32-42 µs with extreme outliers and no speedup is claimed — correctly handled.

## Minor findings (non-blocking)

1. **`graphs()`: the ordinary static9 selection is recorded but never asserted.** `result["ordinary_static9_graph"]["selected"]` captures `direct_obj.last.kernel.plan` and only *eligibility* is asserted; `final_plan_analyze` retains the directory but doesn't check the strategy either. Since the analyzer *does* enforce `promoted[-1] == independent_planes_final9` in graph-storage, this is an inconsistency in rigor — and because first_use is empirical, on different hardware the stated claim could silently fail while the harness passes. Recommend `assert direct_obj.last.kernel.plan.strategy == STRATEGY` (or an analyzer check).
2. **`semantic_reason` is not total on malformed input.** Non-dict buffer/op entries (or a non-sized `buffers`) raise `AttributeError`/`TypeError` instead of returning a reason; Rust returns a clean `Err`. It's used as a boolean predicate in routing paths (`frozen_backend`, `_enqueue`, `legality`), where a raise would crash rather than fall back. Low risk since descriptors are machine-generated and well-formed in all mutations tested, but a `try/except` or structural guard would make the symmetry exact.
3. **Mislabeled rejection evidence**: the `"alias"` ABI control for FP16/BF16 passes `y.double()` — that exercises the *dtype* guard, not output-disjointness; genuine aliasing is only tested for FP32. Cosmetic, but the recorded `rejected` rows say `"runtime": "alias"`.
4. **`final_plan_analyze.sass()` fragility**: instruction lines parsed before an `Address ` header would `NameError` (`start/end` unbound), and column drift would silently corrupt text. Acceptable for fixed-format offline analysis.
5. **`records[...][-1]` ordering assumption**: graph-storage analysis assumes `runtime.CACHE.values()` insertion order is [partial, final]. It holds (evidenced by the passing run), but a comment or an explicit strategy-lookup like the partial's would be more robust.
6. Trivial: `components.py` evaluates `semantic_reason(semantic)` twice per iteration (assert + record); per-call `reason()` for final9 in `_enqueue` is fully static (adds small host overhead each enqueue; enqueue intervals are recorded, so this is visible, not hidden).

## Conclusion

The promotion is correctly scoped: a distinct plan identity reusing the already-qualified probe emitter through ordinary cpp, admitted only through normal first_use selection under a strict, symmetrically-enforced contract, with the coarse guard excluded, fallbacks and partial plans untouched, per-call ABI guards preserved, and cache identities rotated with contract-source hashes. Numerical controls are valid (bitwise parity is structurally guaranteed by identical nonzero association). The findings above are quality-of-evidence improvements, none of which block the change.

---
## Review Trace
- Workspace: /home/imi/Projects/pytortch-backend-experiments
- Diff target: HEAD
- Context entries requested: 5
- Context files loaded: 5
- Files selected for initial diff: 10
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 33886 prompt tokens, 10624 completion tokens
- Model call: model requested 3 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 49869 prompt tokens, 105 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 52157 prompt tokens, 176 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 55814 prompt tokens, 110 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 60997 prompt tokens, 2256 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 63283 prompt tokens, 70 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 64692 prompt tokens, 100 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 66778 prompt tokens, 1153 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 69611 prompt tokens, 7169 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 76844 prompt tokens, 59 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 77079 prompt tokens, 3123 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 80237 prompt tokens, 61 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 80758 prompt tokens, 3818 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 84600 prompt tokens, 2280 completion tokens
- Review complete after 14 model call(s)
