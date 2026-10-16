# Review: commit 4fd9337 — Stage5 CUPTI PM source + final bounded A/B/C memory-state pass

## Verdict

**No material correctness, admission-waiver, or provenance defects found.** The measurement/admission logic matches the stated intent, every numeric claim I checked reproduces exactly from the retained raw data, the 24 MiB failure is handled by parking (not waiving) admission, and the production/reduction surface is untouched. Findings below are minor robustness/polish items only.

## Critical checks — verified

**1. No production/reduction change.** Working tree differs from the commit only in `.beads/issues.jsonl` (verified via `list_changed_files`/`get_uncommitted_changes`). The commit's source diff touches only diagnostics, analyzers, packet artifacts and reports. Per-capture, `reduction_memory_analyze.py:46-55, 258-260` re-verifies from disk that the executed target is `cubecl5_42026f8357481503` at grid576/block512/40reg/64B-shared and that every recorded artifact hash still matches, and `reduction_memory_state.py:82-99` pins input hash, UUID and frozen artifact identities against the prior packet. Residual reliance on `all-production-and-retained-sha256.json` (1393 entries) for the remaining files is self-reported but corroborated by `prior-packet-integrity.json` (2132 prior hashes).

**2. The 24 MiB failure does not silently waive admission.** Verified end-to-end:
- Raw anchors in `footprint_25165824-07/r7-footprint_25165824.clock.json`: wall bracket = 34,582 ns (matches README), max conversion residual = 34,613 ns → guard = 1500 + 34,613 = **36,113 ns** for a 26,535 ns target → interior window empty by construction.
- `reduction_memory_analyze.py:160-170` returns `counter_window_unavailable` (no guard narrowing); `:331-342` excludes the *entire* contrast from `admitted`; `:343-359` retains all 16 raw latencies as `latency_only`; `:360-409` exclude the contrast from `summary`/`paired` (confirmed: no `footprint_25165824` entry in `paired_first_minus_second` — no favorable prefix); `:411` sets `complete=false`; `:430` writes `analysis.json` *before* `:451-453` raises `SystemExit(2)`, so evidence is retained while automation cannot see exit 0.
- Controls exercise this exact path: `controls.py:108-121` widens an anchor by 1 ms and asserts `qualified_comparison_captures == 0`, contrast parked, latencies preserved.

**3. Conservative bounds / clock guard unchanged.** `capture()` (analyzer `:96-159`) keeps the 1500 ns + measured-anchor-uncertainty guard, interior-sum lower bounds, expanded-overlap upper bounds, contiguity (`:154-159`), nonnegative finite overlapping values (`:198-201`), raw/converted column identity (`:123-129`), and decode completion/no-overflow (`:88-91`). The C10 converter lifecycle in `reduction_state_sampling.cpp` (`state_clock_begin` constructs before capture, `state_clock_finish` calls `makeConverter()` after) matches Kineto and is re-initialized per capture (`reduction_memory_state.py:294-310`); the domain check `approx_before ≤ cupti_timestamp ≤ approx_after` (`:312-327`) is verified per anchor before conversion use.

**4. Idle helper lifetime/no-GPU-work.** `reduction_memory_idle.cpp` uses only `cuLaunchHostFunc`; the callback busy-waits on `steady_clock`, writes two timestamps, and `delete`s its heap struct (also deleted on enqueue failure); no CUDA calls inside. The ctypes timestamp buffer is held by `idle_timestamps` through the capture iteration (`reduction_memory_state.py:239-250, 307, 403-412`) and read only after the `torch.equal` sync, so no use-before-execution. FFI rejects null buffer/`>2.5 ms` before touching CUDA (`controls.py:154-165` confirms). Stream ordering (callback after primers, blocking the target) matches the documented `cuLaunchHostFunc` contract.

**5. A/B equivalence and immutability.** `analyze()` `:292-311` enforces pairwise-nonoverlapping spans, 64 MiB for both A and B, and identical count (`count*4==bytes`), stride, sink pointer and 82×256 geometry for every prepared action; the harness prepares both allocations as `torch.ones` of identical size (`reduction_memory_state.py:115-155`). Plan immutability is triple-pinned: `run.py:24-29`, harness `main()` `:490-505` (rejects any invocation not matching exactly one predeclared pair, incl. iterations=65536, metrics, interval, dense reset), and analyzer `:211-212, 242-246`. Fresh process per pair; output dirs must not exist.

**6. Claims fit the data.** I recomputed from `analysis.json`: all A-table medians/bounds/ranges (21.923–23.012 vs 25.889–26.786 µs); the 1.5 MiB row including "all eight pairs improve latency *and* conclusively reduce writes" (largest candidate upper 1,658,624 B < smallest control lower 2,500,224 B; median paired Δ = −1.838 µs, range −2.552…−1.347); 3 MiB all-faster/overlapping-bounds; 6 MiB 7-of-8 slower; 12 MiB tied; all four C rows including the ~2 µs no-callback gaps and the honest 250 µs-round-7 counterevidence (upper 2,697,984 < control lower 2,707,200 while latency +0.291 µs). Counts: 96 pairs/192 captures, 176 admitted, 191 individual windows, 22 CPU controls (17 rejects + valid fixture + park + overflow + extra-work + FFI) — all consistent.

**7. Counter-packet compatibility.** The refactored `reduction_state_counter_analyze.py` still exports the `analyze`/`stats` interface consumed by `state-counters/summarize.py:3`, and its new `ValueError` guard (dense + read_write/write_read only) does not affect the `qualified-*` families that summarize.py processes.

**8. Honest disclosure.** `review-status.json` records `independent_review_completed: false`; `build-provenance.json` discloses the ELF RUNPATH rebuild difference and the absent source-to-binary attestation; these limitations are stated, not hidden.

## Minor / optional findings (non-blocking)

1. **Hardcoded SM count** — `reduction_memory_analyze.py:32` asserts `device["numSms"] == 82` instead of comparing against `plan["device_query"]["SMs"]`. Same constant is also hardcoded in the harness. Harmless for this pinned single-device plan; deriving it from the plan would avoid silent divergence if the plan is ever templated.
2. **Plan fields not cross-checked** — `plan["reset_bytes"]` and `gate_cycles"]` are never asserted against the capture (the harness hardcodes 64 MiB reset / 50 M sleep). The 5-kernel trace structure and the 64 MiB reset span in `pointer_spans` compensate; asserting the span length would close the gap cheaply.
3. **Park path ordering** — the `overlapping` coverage assertions (`:148-159`) run before the empty-`interior` park; a session with *no* samples overlapping the window would `IndexError` instead of yielding the clean park record. Unreachable in practice (PM covers the whole session), but reordering the emptiness check first would make the failure mode self-describing.
4. **Sampler polish** — `state_clock_convert` returns 0 when uninitialized (would conservatively park via a huge guard, never silently admit, but an error return would be clearer); a trailing comma in the metric list pushes an empty metric name into CUPTI config; `state_pm_close` doesn't clear `names`/`metrics` (moot given the one-init-per-process guard).
5. **Pre-capture target warmup** — `reduction_memory_state.py:225-233` runs 20 target batches plus alternate/Triton bit-checks outside the profiler before the first capture. This is inherited unchanged from the qualified counter protocol and identical across all conditions (state is re-selected per capture by reset+primers), so it is not a confound — but it would be worth one explicit sentence in the packet README since the packet otherwise emphasizes "no warmup".
6. **`run.py` schedule indexing** — `--start/--stop` are positional indexes into the schedule; safe only because the plan hash is pinned at `run.py:25-29`. A comment noting that dependency would help future reruns.

## Provenance residuals (disclosed, acceptable)

Source/binary hashes are independent provenance records, not cryptographic source-to-binary attestation (no embedded build hash); the idle-helper offline rebuild differs in ELF RUNPATH with identical `.text` (disassemblies retained). Independent review is explicitly incomplete (`review-status.json`). Both limitations are correctly disclosed in the README and STAGE5_REPORT.md rather than papered over.

**Recommendation:** Accept as-is. Items 1–3 are the only ones I would consider addressing in a future diagnostics-only touch-up; none affect the validity of the 176 admitted comparisons, the parked 24 MiB contrast, or the frozen production surface.

---
## Review Trace
- Workspace: /home/imi/Projects/pytortch-backend-experiments
- Diff target: 4fd9337^
- Context entries requested: 6
- Context files loaded: 6
- Files selected for initial diff: 11
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 186752 prompt tokens, 9662 completion tokens
- Model call: model requested 4 tool call(s)
- Tool call: list_changed_files
- Tool result truncated: get_uncommitted_changes from 23609 to 20000 chars
- Tool call: get_uncommitted_changes
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 201632 prompt tokens, 235 completion tokens
- Model call: model requested 3 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Tool call: search_repository
- Model call: API usage 203493 prompt tokens, 4004 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 211666 prompt tokens, 326 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 216139 prompt tokens, 175 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 220761 prompt tokens, 4265 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: list_repository_tree
- Tool call: read_file
- Model call: API usage 225567 prompt tokens, 2425 completion tokens
- Review complete after 7 model call(s)
