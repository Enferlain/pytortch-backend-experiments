I have now verified all the substantive claims. Here is my review.

---

# Review: hs6 — host-only bounded RMS selection/attribution follow-up

## Scope compliance (task constraints)

| Constraint | Status |
|---|---|
| No GPU compilation/execution/timing | ✅ Probe links helpers only; no runtime/client/tensor binding; `evidence.json` claims `new_gpu_compilation_or_execution: false`; all SASS/PTX evidence is re-analysis of the frozen 5nb packet |
| No new candidate, no production change | ✅ Only report row + host probe + verifier; `automatic_routine_winner: null` asserted per row in `inspect.py`; no sweep/fallback/promotion language anywhere |
| 5nb closed at `7f71ce8` | ✅ `preservation.json.head == 7f71ce82...`; 13,249 file hashes + upstream heads/diffs re-verified (cubek diff hash = SHA of empty string → byte-identical; cubecl retains only its known patch, non-empty `diff_sha256`) |
| Host evidence + report only | ✅ |

## Claims verified against pinned sources (all confirmed)

1. **Reduce API requires explicit routine.** `cubek-reduce/src/lib.rs:96-104` — `strategy: ReduceStrategy` is a mandatory parameter; the doc comment at lines 45–48 ("An optional `ReduceStrategy`... If omitted, a best effort is done") and the `None`-passing example at line 87 are indeed stale relative to the signature. `launch/strategy.rs:20-27` defines `RoutineStrategy::{Unit,Plane,Cube}`; `launch/base.rs:164-177` dispatches on `strategy.routine` with no comparison/benchmark; `autotune_level` only gates `unchecked_fast_paths` (base.rs:157–161). The benchmark catalog (`eval/benchmarks/strategy.rs`) enumerates unit/plane/cube × serial/parallel — a catalog, not a selector.
2. **Conditional Plane packing matches 32×8.** `routines/plane.rs:97-102` → `working_planes = reduce_count / vector_size_output = B`, `working_units = B*32`, `plane_count = 8` for both B=64 and B=16 (probe result matches; `cube_dim = (32, 8)`), `working_cubes = 8/2`. The helper genuinely ignores reduction length and SM count (verified in `launch/utils.rs:8-32`).
3. **I/O width model is faithful.** CUDA runtime sets `load_width: 128` (runtime.rs:179); `io_optimized_vector_sizes(4)` → `vector_sizes_down_from(128/(4·8)) = [4,2,1]` (properties.rs:548-599); probe's `max_vector_size: usize::MAX`, `plane 32/32`, `max_units_per_cube 1024`, `num_cpu_cores: None` mirror the CUDA runtime's `HardwareProperties` (runtime.rs:178-199). `tensor_vector_size_parallel([4,2,1], [B,N], [N,1], 1)` yields 4/1/1 exactly as in `host-policy.json`; the two stride/extent controls correctly isolate the two divisibility conditions of the real helper (core lib.rs:132-174). The FP32-virtual-operand choice for all three storages is consistently labeled and matches the 5nb descriptor contract.
4. **Retained artifact assertions.** I spot-checked every hardcoded string in `inspect.py` against the frozen packet: `v91 = v89 - v90` / `v93 = v91 + v90` round trip (kernel.cu:82-83), `rem.u64`/`rem.u32`, `mul.lo.s64 %rd7, %rd77, %rd1`, `add.s64 %rd49`, `div.rn.f32` at ptx:192 — all present.
5. **SASS ledger.** Region 29 static entries, 1536 executed / 47680 total (= raw `smsp__inst_executed.sum`), fraction 0.032214765100671144 correct; exactly one `CALL.REL` in region with 0 executions; region visibly includes high-bit dispatch (`LOP3.LUT`/`ISETP`/`BRA`) and reconvergence (`BSSY`…`BSYNC`), matching the README's caveats. No latency-decomposition claim anywhere; the report row reports instruction counts only.
6. **Future-boundary semantics.** `Sum::to_output_parallel` exists (sum.rs:77-85) and lowers to `Vector::vector_sum`, which starts at `T::from_int(0)` and folds components in ascending `extract(i)` order (cpp/shared/vector.rs:99-106) — the README's "starts +0, folds components ascending" is exact. The `128*k + 4*l + c` addressing is the correct lane×vector mapping for a width-4 parallel plane reader; the order genuinely differs from 5nb's `32*k+lane` partition, and the README correctly forbids inheriting scalar-order qualification and requires CPU-before-GPU with odd-width rejection/scalar controls.
7. **Verifier coverage matches claims.** 9 rows + 4 controls asserted; registry-package equality between the two locks (root renamed only); cubek source↔copy SHA equality; both ordinary and component IR/CUDA/PTX hash-bound (37 bound paths = `validation.json.bound_sources_and_artifacts`); 13,249 preserved hashes + upstream head/diff (validation.json: `preserved_files: 13249`). Retained setup failure (`setup-order-failure.txt`: missing manifest) matches the README's account.

## Findings

**No blockers.**

1. **Low — stale-pass marker on failed verification.** `inspect.py` writes `validation.json` only at the end of a successful run, but a failing run leaves the previous `"complete": true` file in place, so a stale pass marker survives a later drift failure. Consider removing/annotating `validation.json` on failure (or having readers compare its `evidence_sha256` against a freshly recomputed value, which it does record).

2. **Low — unbound retained-failure record.** `setup-order-failure.txt` is referenced by the README but is the only non-regenerated artifact in the directory not hash-bound in `evidence.json`. (README itself is also unbound, consistent with prior stages; the failure record is the one that functions as evidence, so it's worth binding.)

3. **Low — forward-referenced `review/` directory.** The README states "Review and disposition are retained under `review/`", but no `review/` exists yet under `reuse-first-rms-attribution/` (this review is that review). Either create it with the disposition after this pass or reword ("will be retained").

4. **Nit — "the GPU cap yields eight planes".** For both B=64 and B=16 the binding cap is CubeK's `NUM_PLANE_MAX = 8` constant (`launch/utils.rs:22`); the hardware cap (`max_units_per_cube/plane_dim = 1024/32 = 32`) never binds. Given how precise this document is elsewhere, "GPU cap" reads as a hardware-derived limit where it is actually a source constant on the GPU (non-CPU) branch.

5. **Nit — brittleness (accepted).** The SASS ledger anchors on the exact instruction string `LOP3.LUT R0, R3, R7` and fixed header-word column offsets, and verify-mode equality includes the SHA of the locally built probe binary. Both are frozen-artifact checks, so drift is caught rather than silently mis-parsed, and the README documents `--capture` as initial-snapshot only — acceptable, just noting a rebuild with a different toolchain will (by design) fail verification rather than repair.

6. **Info — lock comparison excludes path deps.** The registry-pin check compares only `"source"`-bearing packages; path-dependency entries aren't compared. Their identity is instead enforced transitively (evidence hashes of the source trees + a successful `--locked --offline` build), which is adequate.

## Conclusion

This is a carefully bounded host-only pass that does exactly what it claims. Every load-bearing source claim (mandatory `ReduceStrategy`, no automatic winner, stale optional/best-effort doc, 128-bit CUDA I/O → width 4/2/1, conditional 32×8 packing, `to_output_parallel` fold order) checks out against the pinned trees, the probe faithfully models the CUDA property surface while calling the real helpers, the verifier binds everything it asserts, and the future vector4 proposal is stated with the correct changed-order/odd-width/stop-before-GPU conditions and no winner inference. The four findings above are minor precision/robustness items; none blocks. The `.beads/issues.jsonl` change is tracker noise outside focus scope.

---
## Review Trace
- Workspace: /home/imi/Projects/pytortch-backend-experiments
- Diff target: HEAD
- Context entries requested: 4
- Context files loaded: 4
- Files selected for initial diff: 7
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 12714 prompt tokens, 463 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: list_changed_files
- Tool call: read_file
- Model call: API usage 14232 prompt tokens, 1131 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 18344 prompt tokens, 54 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 18719 prompt tokens, 286 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 19306 prompt tokens, 69 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 22709 prompt tokens, 1227 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: search_repository
- Model call: API usage 25358 prompt tokens, 281 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 26326 prompt tokens, 198 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: search_repository
- Model call: API usage 27153 prompt tokens, 209 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 27587 prompt tokens, 254 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 27935 prompt tokens, 45 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 28914 prompt tokens, 283 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 29939 prompt tokens, 258 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 30887 prompt tokens, 137 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 31064 prompt tokens, 87 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 32142 prompt tokens, 667 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 33507 prompt tokens, 2147 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 37061 prompt tokens, 1807 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 41887 prompt tokens, 392 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_files
- Tool call: search_repository
- Model call: API usage 42816 prompt tokens, 293 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: search_repository
- Model call: API usage 43660 prompt tokens, 1086 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_files
- Model call: API usage 47192 prompt tokens, 210 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 47556 prompt tokens, 171 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 47882 prompt tokens, 239 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 48666 prompt tokens, 258 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 49084 prompt tokens, 176 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 50314 prompt tokens, 96 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 50434 prompt tokens, 40 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 50836 prompt tokens, 43 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 51248 prompt tokens, 1200 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: list_repository_tree
- Model call: API usage 52914 prompt tokens, 305 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: list_repository_tree
- Tool call: read_files
- Model call: API usage 54651 prompt tokens, 202 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: list_repository_tree
- Model call: API usage 57638 prompt tokens, 2685 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 62059 prompt tokens, 108 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 62428 prompt tokens, 4011 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 66502 prompt tokens, 84 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 67695 prompt tokens, 3937 completion tokens
- Review complete after 37 model call(s)
