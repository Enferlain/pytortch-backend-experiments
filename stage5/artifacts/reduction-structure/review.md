# Review: `plane_shared_load_batch` diagnostic candidate

## Summary

The mechanism change is sound and tightly bounded: I traced the IR transformation, both validators, the selection/eligibility gates, and the cache keys, and found **no correctness or eligibility bug**. The candidate cannot reach automatic selection (double-gated), old plan identities are preserved and re-verified, and the lazy protected-access contract is intact. The actionable findings are **provenance/footgun issues in the new `long_row_normalization.py` options**, which matter because the report's evidence claims lean on "preserved captures".

## Verified correct

**Mechanism (`descriptor3.rs:432-548`)**
- Guard placement is unchanged: the semantic body (`emit`) stays inside the per-accumulator `live` branch; only the store target changes (pending local instead of in-branch accumulator update). No load is hoisted or converted to a select-of-value. ✓
- Per-iteration `+0.0` zero-init of `pending` occurs before the guard, so inactive iterations contribute exactly zero. Accumulator update operand order (`old + value`) and per-chain sequencing are preserved; the balanced `(a0+a2)+(a1+a3)` merge (stride-halving at line 556-566) and the `plane_shared` collective (two `sync`s, leader publication, shared-root broadcast, lines 671-724) are byte-for-byte the same code path the old `plane_shared` uses, since only the `plane_shared` boolean was widened. ✓
- The neutral-arithmetic identity claim holds by construction, not just empirically: accumulators init to `+0.0`, and under round-to-nearest `(+0.0) + v` can never produce `-0.0`, so `acc + (+0.0)` is a bitwise identity for every value `acc` can hold (including quieted NaNs from prior FAdds). This *does* depend on the asserted no-FTZ/no-FMA/no-fast-math environment and the `+0.0` init — worth one sentence in the code comment so a future init change doesn't silently break the identity argument.
- `batch` is empty for all pre-existing strategies, so the second loop is a no-op and old plans are untouched (the analyzer's descriptor/PTX hash assertions cover this).

**Validation (`descriptor5.rs`)**
- Complete for the new identity: exactly-4 accumulators (explicit error, plus the generic policy check as backstop), `combined` forces workgroup `[256,1,1]`/width 1/outputs 1, `plane_width` must be 32, strategy added to the schema-3 allowlist, and the schema-4 pointwise path (line 216-224) rejects it. `as_u64` rejects bool/float/negative counts. The Python mirror (`from_dict` catalog membership + strict `type(...) is int`) rejects the same seven neighbors, and `reduction_structure_checks.py` exercises both validators independently, including direct Rust-compiler negatives (whose `Error:` stderr matches the assertion).

**Eligibility / cache boundaries**
- Double gate confirmed: `numerical_reason` rejects `plane_shared_load_batch` unconditionally for schema 3 (no extent condition, so "every extent" holds by construction), and `qualified_for_runtime` (`reduction_policy.py:63`) whitelists only `shared_tree_compensated`. `candidates()` can never accept it; the persisted selection replay re-validates `winner not in plans`.
- Diagnostic selection is reachable only via explicit policy strings, allowlisted in `backend.py`, `runtime.compile_kernel`, and the probe CLI; the `AutotunedKernel` diagnostic branch (`runtime.py:167-197`) applies it only to schema-3 extents > 4096, exactly as the report states, and never touches the automatic selection/cache path.
- Cache separation: `compile_kernel` keys on `(semantic, device, policy)`; `CompiledKernel`'s entrypoint/directory digest includes the physical-plan dict, so `four` and `batch` artifacts cannot collide.

**Evidence harnesses**
- The 109+6+1 = 116 bitwise comparisons count matches the report; `heldout_10017` is inside the exercised seed range and failures are recorded, not averaged away. Component win criterion (≥2/3 rounds + disjoint median±3·MAD), grid `9*rows` assertion, and curated-packet copying in the analyzer all match the described protocol.

## Findings

1. **Medium — diagnostic normalization artifact has no policy provenance** (`long_row_normalization.py:122-132`). The result JSON records `harness_sha256` of the harness file itself — identical for baseline and `load_batch_diagnostic` runs — and `reference_config` (Triton patches only). A load-batch `normalization-cubecl.json` is therefore *indistinguishable* from the baseline artifact. Contrast `reference_reduction_probe.py:119`, which correctly records `cubecl_policy`. Add e.g. `"cubecl_policy": args.cubecl_policy if args.impl == "cubecl" else None` to the result dict.

2. **Medium — default dirs can clobber the baseline evidence.** `--output-dir`/`--cache-dir` default to the baseline `artifacts/long-row` and `cache/long-row`. Running `--cubecl-policy load_batch_diagnostic` without explicit dirs overwrites `normalization-cubecl.json`, the raw `norm-cubecl-*.json` traces, the npz, and shares the Inductor cache dir. Since the whole point of the options is "preserved captures", either require explicit dirs when the policy is non-default or derive a policy-suffixed default.

3. **Low — `--cubecl-policy` is silently ignored for `--impl triton`.** A `triton --cubecl-policy load_batch_diagnostic` run silently produces a baseline Triton capture. Restrict or warn.

4. **Low/nit — the bitwise identity assertion in `reduction_structure_qualification.py` compares post-*final* outputs**, not the partial `tmp` bytes. Since the fixed final is a deterministic FP32 function of the partials, a 1-ulp partial difference could in principle be absorbed. The claim as worded ("outputs bitwise identical") is accurate; just don't escalate it to "partial buffers bitwise identical" without also hashing `tmp`.

5. **Info — the three policy allowlists** (`backend.py:59-68`, `runtime.py:402-409`, probe CLI) are duplicated string sets; a future policy must be added in all three. Acceptable for an experimental harness, but a shared constant would be safer.

6. **Info — `descriptor5.rs:105-107`** (`batched && accumulators != 4`) is redundant with the generic accumulator-policy check at 108-112, but the clearer error message justifies it.

## Conclusion

Approve the code with findings 1 and 2 addressed before the packet is treated as final evidence — both are small harness fixes that don't touch the kernel mechanism, validators, or eligibility gates. The "diagnostic only, never automatically eligible" requirement is enforced correctly and redundantly; the Stage3 first-failure handling (recorded separately, not relaxed) and the byte-identical HEAD rebuild claims are consistent with what the code and harnesses actually do.

---
## Review Trace
- Workspace: /home/imi/Projects/pytortch-backend-experiments
- Diff target: HEAD
- Context entries requested: 5
- Context files loaded: 5
- Files selected for initial diff: 8
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 16331 prompt tokens, 4262 completion tokens
- Model call: model requested 4 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 28438 prompt tokens, 2331 completion tokens
- Model call: model requested 4 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 35180 prompt tokens, 2438 completion tokens
- Model call: model requested 3 tool call(s)
- Tool call: read_file
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 38994 prompt tokens, 173 completion tokens
- Model call: model requested 3 tool call(s)
- Tool call: read_file
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 40195 prompt tokens, 1421 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 43994 prompt tokens, 2042 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 46137 prompt tokens, 90 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 47150 prompt tokens, 2209 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 49821 prompt tokens, 974 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 51463 prompt tokens, 4070 completion tokens
- Review complete after 10 model call(s)
