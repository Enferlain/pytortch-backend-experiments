# Review: 8x0 reuse-first selection packet

## Scope & verdict

This is a selection-evidence packet, not an implementation, and it holds that line consistently: no new candidate kernel, no changed arithmetic, no timing, no backend source changes. The one code artifact (`capture.py`) is a baseline-only capture harness. I verified the factual claims against pinned sources, the retained artifacts, and the corpus/harness code. **No blockers; a few provenance and robustness nits.**

## What I verified as correct

**capture.py provenance and binding** (the core reviewable artifact):
- `tuning="baseline"` is a real, unmodified mode of the existing backend (`stage5/backend.py:59–67` validates it); nothing in the compile path is patched by this file.
- Exactly one launch per call is asserted (`assert len(launches) == 1`), matching the "one fused kernel per call" claim; `baseline-capture.json` contains all 9 cases (verified by grep) and exactly 3 artifact sets, which I confirmed are the three dtypes (float32/float16/bfloat16 input buffers in the three `kernels/*/descriptor.json`).
- Binding is genuine: each record carries input/output SHA-256, seed, launch geometry (`grid`/`block`/`stream` from `artifact.py:497–506`), artifact directory + per-file hashes, compiler identity (NVRTC lib/builtins/headers, driver, device UUID, `--fmad=false`), git HEAD, and the Triton/GIL independence inventory. `stage1/evidence.py` really implements `guard/state/inventory` as claimed (Triton import blocked, GIL-disabled assertion, `/proc/self/maps` scan), so `validation.json`'s `triton_unloaded_and_GIL_disabled` is supported.
- The `matches[-1:]` lookup over all `artifact.EVENTS` (not just since `event_start`) is the right choice: cache-hit calls for shapes 2–3 still bind to the original compile artifact via the shared entrypoint.

**README factual claims vs pinned source** — all checked out:
- The captured FP32 kernel (`kernels/177394…/kernel.cu`) is exactly the claimed fused RMS: strided load → square → FP32 accumulate → shared-memory tree → `/N` → `+1e-5` → `rsqrt` → mul → store, with `"iteration_scope": "reduction"` in the descriptor. One kernel, one sum, one reduction-scope store. ✓
- `GlobalFullPlaneReduce`: public `execute` writes only from `UNIT_POS_X == 0` (plane.rs:166,175) while `reduce_single` returns the all-lane accumulator — but `reduce_single` is *private* (`fn`, line 182). The README's instruction to compose the **public** `Reader::new` / `PlaneReader::{new,length,read}` / `Sum::{null_accumulator,reduce,plane_reduce_inplace}` instead is therefore both accurate and necessary. Lazy merge (`ReduceStep::Identity` + final `plane_reduce_inplace`) matches sum.rs/plane.rs. ✓
- Eight planes: `calculate_plane_count_per_cube` caps at `NUM_PLANE_MAX = 8` (utils.rs:22); with `working_units = reduce_count × plane_size` (routines/plane.rs:97–99), both 64 and 16 rows yield 8 planes, so `ceil(B/8)` blocks is arithmetically consistent. ✓
- XOR butterfly offsets 1,2,4,8,16 for a 32-lane plane match the `plane_reduce` loop in polyfills.rs. Importantly, the existing baseline uses a *different* order (per-thread strided adds + descending halving tree), so the README's refusal to reuse the frozen descending-order qualification for the ascending CPU model is substantively correct, not just cautious. ✓
- Revision pins: cubek `4ca0235…` declares cubecl `80bb36b…` (Cargo.toml:46 verified); our compiler cubecl `6bba5f…`; old `research.json` records cubek `a680b70…`/cubecl `92f9af…` — "earlier snapshot" is accurate. Strong corroboration: source-evidence's cubecl `diff_sha256` (`9ae155ef…`) equals research.json's old `fp_patch_sha256`, independently supporting "upstream diffs unchanged". ✓
- `stage0/src/descriptor2.rs` contains no `crate::` references — the "reuse emit/index unchanged from a separate crate" feasibility claim holds. ✓
- Component-map symbols all exist as cited: `VirtualTensorOperations(Expand)` (virtual.rs:311), `copy_perpendicular`/`launch_copy_perpendicular_ref`, tile `procedural::Product`/`product_of`, `Tile::{reduce_axis,store_rows,scale_rows,rescale_rows}`, `BoundChecks::Branch`, `Mean`'s divide-by-`shape_axis_reduce` (supporting "don't silently use Mean's output conversion"). Corpus families/shapes referenced in the map (odd transpose dims, `mean_epilogue` 64×4099, etc.) match `stage5/corpus.py`. ✓
- Historical numbers (2.113µs/1 kernel inference; 6.941µs/3 backward vs triton default 4.706µs/2) match the retained records and are explicitly quarantined as dated prioritization, not mixed with fresh samples. ✓

**.gitignore**: `artifacts/*` (line 8) plus `!artifacts/reuse-first-selection/` follows the proven `!artifacts/reduction-structure/` pattern — the negation is effective because only directory *contents* are excluded, and the new packet's files (README, capture.py, 3 JSONs, 3 kernel dirs) become trackable. `cache/` (line 2) still covers the `TORCHINDUCTOR_CACHE_DIR` target. Correct.

**STAGE5_REPORT.md row**: consistent with README and capture evidence; explicitly scoped "Selection only: no new candidate, speedup or default claim"; link target exists.

**Boundary reviewability**: Yes — the trial names exact symbols, one mechanism configuration (vector-width 1, 32-lane plane, 8 planes, `ceil(B/8)` blocks, Lazy merge), pins revisions, forbids widening, requires references *before* GPU implementation, and defines concrete stop conditions. The proposed adapter is honestly flagged "proposed, not yet compiled".

## Issues (none blocking)

1. **Provenance of validation.json / source-evidence.json (moderate)**: no committed script generates them; `capture.py` writes only `baseline-capture.json`. Claims like `protected_sources_equal_HEAD`, `upstream_diffs_unchanged`, and the two hash bindings are not reproducible from committed tooling. I could corroborate several by hand (35 sources counted; dtype sets; revision pins; diff-hash continuity with research.json), but the packet should either commit the checker or state these are manually assembled.
2. **Tolerance duplication (nit)**: `capture.py:56–60` hardcodes rtol/atol instead of importing `bench.tolerances`, and omits the corpus's `equal_nan=True`. Behaviorally equivalent (stricter, and randn inputs won't produce NaN), but "unchanged corpus tolerances" is near-literal rather than exact, and the duplication can drift if bench tolerances change.
3. **"One FP32 sum / reduction-scope store" not machine-checked (nit)**: only the launch count is asserted; the sum/store characterization rests on the retained `kernel.cu` (which I verified true). A cheap text assert on the copied CUDA would make the README claim self-enforcing.
4. **Minor robustness (nit)**: `artifact.compiler_identity(0)` hardcodes device 0 while tensors go to the default device; `git rev-parse HEAD` depends on cwd (README says run from repo root — using `PROJECT` would be safer). Copied `integrity.json` is never cross-checked against the copied files (binding still holds via the independently recorded hashes).

No security concerns: the only subprocess call is a fixed-argument `git rev-parse`, and all file copies are within the repo's own artifact directories.

## Bottom line

The packet does what it claims and nothing more: the selection evidence is source-backed and accurate at every spot I checked, the capture harness is sound with real artifact binding, and the trial boundary is concrete, bounded, and honest about what remains uncompiled and unmeasured. Recommend accepting after considering nits 1–3.

---
## Review Trace
- Workspace: /home/imi/Projects/pytortch-backend-experiments
- Diff target: HEAD
- Context entries requested: 3
- Context files loaded: 3
- Files selected for initial diff: 4
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 14799 prompt tokens, 295 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: list_repository_tree
- Model call: API usage 16896 prompt tokens, 3617 completion tokens
- Model call: model requested 4 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 23827 prompt tokens, 1107 completion tokens
- Model call: model requested 4 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 28446 prompt tokens, 760 completion tokens
- Model call: model requested 3 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 31925 prompt tokens, 618 completion tokens
- Model call: model requested 3 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 34643 prompt tokens, 1372 completion tokens
- Model call: model requested 3 tool call(s)
- Tool call: read_file
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 36780 prompt tokens, 503 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 39753 prompt tokens, 705 completion tokens
- Model call: model requested 3 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Tool call: search_repository
- Model call: API usage 42244 prompt tokens, 962 completion tokens
- Model call: model requested 4 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Tool call: list_changed_files
- Tool call: search_repository
- Model call: API usage 44353 prompt tokens, 1101 completion tokens
- Model call: model requested 4 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 46513 prompt tokens, 1796 completion tokens
- Model call: model requested 3 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 48460 prompt tokens, 3245 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 52366 prompt tokens, 3373 completion tokens
- Review complete after 13 model call(s)
