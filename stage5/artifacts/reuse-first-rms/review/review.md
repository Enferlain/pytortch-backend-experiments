# Review: `5nb` bounded CubeK RMS inference trial

## Verdict

**No blockers.** The implementation is a faithful realization of the pre-existing 8x0 boundary; the numerical gate is genuinely pre-GPU and integrity-chained; the wrapper provenance, ABI mapping, dependency pins, and the negative-result reporting all check out against the retained evidence. I verified the load-bearing pieces against the actual pinned cubecl/cubek sources rather than taking the README's word. Below: what I verified, then actionable findings (all low severity).

## What I verified as correct

**Virtual metadata/replay boundary (`rms_probe/src/main.rs`)** — the trickiest part, and it is right:
- `ParallelReader::read_plane` passes the *global* offset `row*n + i` to the virtual read (`pos = vector_index*32 + UNIT_POS_X`; `batch_offset = row*n` via `coordinate(i,axis)=(i/stride)%shape` on the mode-1 `[B,1]` meta). `ReplayExpand::coordinates` recovers `column = pos − row*n`, and the `bind` rewrite (`reduction`→scalar slot 2 = column, `linear`→`row`) reproduces `col + row*N` for both the load and the disjoint store — confirmed against all three retained `semantic.json` templates (FP16/BF16 cast chains included, so the reduction operand is the original FP32 square, not a low-precision `P::EI` square).
- `BoundChecks::Branch` + `null_input` (+0) gives ascending-k chains with +0 tails; `length_plane = ceil(N/32)`; the `row < rows` plane-uniform branch precedes any virtual read; every lane stores its own columns with the all-lane plane sum (no lane-zero restriction).
- Mode asserts make the `[B,1]` output metadata genuinely write-impossible.

**Arithmetic order claim** — corroborated at source level: `Sum::plane_reduce_inplace` → `plane_sum` → `FSumOp` → shared polyfill `plane_reduce`, which is exactly a simultaneous XOR butterfly with offsets 1,2,4,8,16 (`offset *= 2` while `< min(PLANE_DIM, CUBE_DIM)=32`). The CPU model in `rms_trial_data.model` is per-lane (`factors[:, arange(n) % 32]`, no lane-0 presumption) and matches, including the separate bitwise all-lane assertion. `--fmad=false` confirmed in `stage0/bridge.py`.

**ABI/bounds/lifetime** — the existing stage4/stage5 bridge packs info scalars `[outer*256, N, B]`; the Rust reads slots 1/2 (slot 0 deliberately unused). Grid `ceil(B/8)` via schema-3 geometry with `outputs_per_workgroup=8` — corroborated by the SASS `grid (8,1,1) / block (32,8,1)`. Rows≤65/widths∈WIDTHS enforced host-side on top of unchanged layout/stride/span/disjoint-output validation; `_call_pointers` cleared after calls; prepared scope only for isolated timing.

**Gates** — I recomputed the fixture census: 66 cases, exactly **945,534** output elements. Reference (compiled-Triton, separate env) → CPU model (`cpu_gate` enforced inside `compile_kernel`, so GPU compilation cannot happen without a complete passing CPU gate) → GPU (canaries, input preservation, 21 live rejections with launch-count checks, nondefault-stream offset views, pointer-lifetime check). `qualify_case` is the unchanged `reference-or-better-v1` (tolerance OR no-worse-than-FP64, NaN/Inf class equality, no sanity-limit relaxation).

**Wrapper provenance** — the retained generated wrapper really binds `cubecl_rms_trial_0 = rms_trial_compile(descriptor, 0)` at module scope and calls it with `(N, B)` scalars; `capture()` asserts 16 CUPTI kernel events all named the bound entrypoint; the analyzer re-verifies all 120 traces (sha, census, name, durations) and the timed-vs-final code identity modulo entrypoint renaming.

**Pins / honesty** — lockfile has zero registry-source cubecl entries (patch redirects complete); spot-checked registry pins (allocator-api2 0.2.21/0.4.0, cudarc 0.19.10, identical checksums) match the retained component-probe lock. Report row and README numbers match `analysis.json` exactly; `component_winning_rounds=0` in all scopes; no fallback code exists; the missing raw barrier metric is left `null` rather than invented; no claim that the 18-gate suite was rerun. The `.gitignore` negations are correct and necessary (global `*.csv`/`*.ncu-repz` would otherwise drop the NCU evidence).

## Findings

1. **Low — evidence integrity gap in the analyzer.** `rms_trial_analyze.py` verifies `preservation.json` (12,784 files + upstream git revisions/diffs) but never re-verifies `component-sources.json` (the six exercised-copy sha256s) or the NCU provenance artifacts (`ncu-provenance.json`, `counter-command.txt`, `environment.json`). Since `stage5/cache/codegen/**` is gitignored, a drifted compatibility copy would change the binary hash (and hence the entrypoint) *silently* — nothing would fail against the recorded byte-for-byte claim. Action: add those hashes to the analyzer's assertion set.

2. **Low — dead telemetry.** `rms_trial_runtime.EVENTS` is appended on every launch (entrypoint, scalars, grid, block, stream) but is consumed by nothing (the trial and analyzer use the trace JSONs instead). Either wire it into the analyzer (it would be a cheap independent check that accepted launches used the recorded geometry/stream) or drop it.

3. **Low — `cpu_gate` binds only `rms_trial_data.py`.** The gate hash-checks `cpu-model.pt`, `reference.json`, and the model source, but not `cpu.json`'s own `source_sha256["rms_trial.py"]`; that binding exists only in the offline analyzer's `phase-sources` check. Cheap to add; matters only if someone re-runs `cpu` with a modified driver and then compiles in the same session.

4. **Info — load-bearing guard deserves a comment.** The `row < rows` branch is not only the idle-plane branch required by 8x0; it also prevents silent wrap-around in the meta coordinate mapping (`% shape(0)` would fold `row ≥ rows` onto row 0). It is present and correct, but a future "simplification" would corrupt rather than crash. A one-line comment in `rms_components` (or a debug assertion) would lock the invariant in.

5. **Nit — hardcoded manifest metadata.** `manifest.json` hardcodes `"scalar_count": 3, "bytes": 40` and must stay in sync with the existing bridge ABI (`metadata_key[1]` length). Fine today; consider deriving or cross-checking it where the manifest is consumed.

6. **Nit — repeated template I/O.** `admit()` re-reads `baseline-capture.json` plus three `semantic.json` files on every call (per `define_kernel` and per compile). Harmless at this scale; cache it if the trial is ever extended.

7. **Info — first-round drift is visible but undiscussed.** In `corpus_float16_16_4099` (and to a lesser degree BF16), the first round medians are far below the steady state (ordinary 3.779 vs ~8.0; component 12.72/12.68 vs ~27.5). The conclusion is unaffected — the component's *best* round still loses to the ordinary path's *worst* round, and `component_winning_rounds=0` is computed per round — but one sentence in the README's timing section acknowledging the within-case first-round drift would make the raw-timing analysis fully self-describing.

## Notes on scope compliance

- Tracked diff touches only `STAGE5_REPORT.md` (one row, accurate) and `stage5/.gitignore` (correct negations); no production compiler/planner/catalog/runtime source changed — the trial backend is a new isolated module mirroring the existing `stage5.backend` convention verbatim.
- The isolated `Cargo.lock`/`Cargo.toml` and the `--locked --offline` reproduction path are consistent with the documented isolated-copy procedure; the compatibility copy must be pre-populated by `reduction_codegen_setup` (documented).
- The negative result is reported honestly and the stop condition is respected: no second mechanism, no tuning sweep, no eligibility change anywhere in the reviewed code.

---
## Review Trace
- Workspace: /home/imi/Projects/pytortch-backend-experiments
- Diff target: HEAD
- Context entries requested: 4
- Context files loaded: 4
- Files selected for initial diff: 11
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 13953 prompt tokens, 354 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: list_repository_tree
- Tool call: read_file
- Model call: API usage 15263 prompt tokens, 75 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 19916 prompt tokens, 42 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 20082 prompt tokens, 2054 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_files
- Model call: API usage 26788 prompt tokens, 6325 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_files
- Model call: API usage 36512 prompt tokens, 110 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_files
- Model call: API usage 38942 prompt tokens, 2709 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: search_repository
- Model call: API usage 42548 prompt tokens, 51 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 43344 prompt tokens, 420 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 43772 prompt tokens, 104 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 43927 prompt tokens, 59 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 44201 prompt tokens, 72 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 44856 prompt tokens, 88 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 46430 prompt tokens, 99 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 46553 prompt tokens, 59 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 47400 prompt tokens, 44 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 48547 prompt tokens, 78 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 48785 prompt tokens, 45 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 50124 prompt tokens, 1869 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 52017 prompt tokens, 62 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 52109 prompt tokens, 35 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 52168 prompt tokens, 62 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: list_repository_tree
- Model call: API usage 52259 prompt tokens, 38 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: list_repository_tree
- Model call: API usage 52313 prompt tokens, 59 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 52803 prompt tokens, 45 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 54576 prompt tokens, 89 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 54689 prompt tokens, 45 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 55422 prompt tokens, 103 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 55573 prompt tokens, 45 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 56553 prompt tokens, 348 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 57236 prompt tokens, 124 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 58990 prompt tokens, 1180 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 61515 prompt tokens, 30 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 64084 prompt tokens, 1258 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 66632 prompt tokens, 1420 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 68098 prompt tokens, 102 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 68289 prompt tokens, 101 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 68750 prompt tokens, 1103 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 73016 prompt tokens, 32 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 76192 prompt tokens, 33 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 77080 prompt tokens, 1400 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 79443 prompt tokens, 2144 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 84391 prompt tokens, 2042 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 86457 prompt tokens, 29 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 87385 prompt tokens, 334 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 89328 prompt tokens, 84 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 89436 prompt tokens, 33 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 89543 prompt tokens, 70 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 90519 prompt tokens, 323 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: list_repository_tree
- Model call: API usage 90851 prompt tokens, 312 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 91736 prompt tokens, 4039 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 95799 prompt tokens, 38 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 97240 prompt tokens, 67 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 97959 prompt tokens, 258 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 98914 prompt tokens, 1658 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 101643 prompt tokens, 204 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 101952 prompt tokens, 2579 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 105299 prompt tokens, 113 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 108283 prompt tokens, 485 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 108792 prompt tokens, 32 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 108838 prompt tokens, 31 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 109624 prompt tokens, 1071 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 110820 prompt tokens, 5546 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 116565 prompt tokens, 419 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 117008 prompt tokens, 40 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 117072 prompt tokens, 77 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 117261 prompt tokens, 35 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 118447 prompt tokens, 1696 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 120220 prompt tokens, 66 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 120806 prompt tokens, 2460 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: list_repository_tree
- Model call: API usage 123738 prompt tokens, 78 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 125685 prompt tokens, 2081 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 127790 prompt tokens, 31 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 127929 prompt tokens, 3286 completion tokens
- Review complete after 74 model call(s)
