I now have everything needed. Here is the review.

# Review: d39 bounded Stage5 reuse-first transpose/layout trial

## Verdict

No blockers. The implementation matches the stated intent, the guards are real (not decorative), and the evidence chain binds source → binary → artifacts → data → traces → SASS tightly enough to support the conservative negative conclusion. Everything below the "verified" section is minor robustness observation in explicit diagnostic scope; none of it undermines the stop decision.

## What I verified (against code, not just the README)

**Address/ownership/lazy bounds**
- `transpose_model.verify` reproduces the upstream worker→`global_index`→gather→register-transpose→store mapping independently and asserts single ownership per output scalar plus equality with a dense transpose; `transpose_analyze` recomputes all 24 rows rather than trusting `cpu.json`.
- Worker math is consistent across layers: upstream `num_runs = output.len()/(num_batch·v)` with `len()` in vector units equals `n/v`; `Kernel.geometry` = `(n//v+255)//256` produces exactly that many workers, and the `ABSOLUTE_POS >= num_runs` termination precedes all loads. Grids 1/2/3 in `analysis.json` match.
- All in-kernel truncating divisions (`m/v`, `n/v`) are exact because `width()` requires `m % v == n % v == 0` and `_validate` rejects any runtime `(m,n)` whose dictated width differs from the compiled vector. No admitted shape yields width 2 or FP32 width 8.

**dtype precision**
- The write path in `main.rs` replays `ops[1..len-1]` per vector component with a fresh SSA map seeded by the gathered component, so every FP16/BF16 cast chain and the FP32 sin/add from the captured template is preserved verbatim; the descriptor's load index is realized by the gather addressing instead of being re-emitted. The `--fmad=false` count is re-checked per retained artifact.
- `check()` enforces value classes (NaN/±inf/signbit-of-zero) on top of assert_close; GPU rows additionally require bitwise byte equality with the retained ordinary PTX output, and `special-values.pt` is re-loaded and re-checked offline.

**ABI / current-stream / lifetime**
- Scalars are `(numel, n, m)`; `ReplayExpand.scalars = scalars[1..]` matches descriptor slots 0/1, and the manifest `{"scalar_count": 3, "bytes": 40}` is pinned in `Kernel.__init__`. The `physical_plan` literal in `main.rs` matches `PhysicalCandidate("scalar_pointwise_baseline").to_dict()` (confirmed against `stage5/plan.py`, including the conditional key deletions).
- The alignment guard (`data_ptr % (vector·elemsize)`) is precisely the condition the reinterpret-cast vector loads/stores need, since all element offsets are multiples of `width` (`pos*width` in both read and write). Pointer bases passed at launch are the tensor data pointers, so the guard is on the right address.
- Stream control runs the dispatch on a non-default stream with an 8-element aligned offset; disjointness, span, dtype/device/layout and capture-veto all remain owned by `ExistingKernel._validate`, which the override calls *after* its own shape admission.

**Numerical gate / artifact / data binding**
- The "too-loose initial FP32 capture" is handled honestly: `capture.json` is only bound to its `phase-sources` snapshot hash, the strict FP32 2e-6 policy lives in `transpose_qualify.check` and the nine view controls, and `stage5/runtime.compare` confirms 2e-6 is the actual pointwise policy (not a relaxation).
- Chronology is enforced, not asserted after the fact: `transpose_model.main` refuses to run if the probe source exists; `Compiled.__init__` re-checks the completed CPU gate and AST-compares the model; `transpose_analyze` re-verifies both.

**Trace census / producer snapshots / frozen source**
- 216 unique trace paths, 3456 kernel events, per-trace name/grid/block/duration-vs-sample/median recomputation, plus predeclared `timing-plan.json` written before any timing. Expected ordinary grids are recomputed from the recorded plan, not trusted.
- The six mechanical substitutions deterministically regenerate `component.rs` from the byte-compared pinned upstream file; `.gitignore` is verified as exactly the declared four-line addition over `ignore.before`; 14,263 preserved files hash-checked; SASS totals are cross-checked against independent NCU metric rows per invocation.

**Claims vs evidence**
- Report/README numbers all reconcile with the artifacts: 27/24/54/42/6/9/7 counts, 1,215,843 CPU identities, 0/108 component wins, 5 clock snapshot pairs, 18 summary rows. The "not a stock-launcher benchmark" caveat is visible in code (fixed 256-thread envelope, only the worker-grid calculation overridden).

## Findings (minor, diagnostic scope)

1. **Binary location fragility** — `transpose_runtime.py:16`: the probe binary lives in `stage5/cache/reuse-first-rms/target/...`, another trial's (gitignored, cleanable) cargo target dir. `transpose_analyze`'s `sha(BINARY)` assertion fails if that cache is cleaned even though all retained evidence is intact. The retained `source/` snapshot and build logs permit reproduction, but consider snapshotting the binary itself (or tolerating absence with a recorded-hash fallback) so the offline verifier depends only on the artifact directory.

2. **Verifier coupling to one machine/tool version** — `transpose_analyze.py:32-52` parses the NCU source page by `line.index()` of header fragments and `split()[2]`; the clock census hardcodes CSV column indices 5/6 and the GPU-UUID fragment `"510ddc27"`. Correct for the retained NCU 2026.3.1 data on this host (the `activity()` query ordering confirms indices 5/6 are SM/memory clocks), but these constants will silently mis-parse on any regeneration. Acceptable for a one-shot verifier over frozen artifacts; worth a comment at minimum.

3. **Hardcoded `paired_comparisons=108`** — `transpose_analyze.py:318`: derived from a literal rather than `len(timing["rows"]) * 6`. The surrounding census makes it safe today; deriving it would be more robust.

4. **`native()` "effects" control is not a distinct arm** — `transpose_qualify.py:278-279`: mutating `effects.mutation` also makes the semantic fail exact-template admission, so both `"semantic"` and `"effects"` reduce to the same template-mismatch rejection. The rejection is still verified; only the label overstates coverage.

5. **Unused `tuning` parameter** — `transpose_backend.py:31`: `Scheduling` is constructed with `tuning="baseline"` but the override never uses it (and drops the policy kwarg from the emitted wrapper call). Harmless since the transpose runtime always uses the frozen baseline `PLAN`, but it could mislead a reader into thinking a tuning policy applies.

6. **Partial candidate directories on compile failure** — `transpose_runtime.Compiled.__init__` writes descriptor/semantic/plan/identity JSON before invoking the probe, with no lock file or cleanup on failure (unlike `stage5.artifact.CompiledKernel`'s fcntl + integrity pattern). One-shot and serialized by `LOCK`, so minor.

7. **Bare `StopIteration` on candidate mismatch** — `transpose_measure.py:97-99`: `next(k for k in candidates if ...)` would raise an context-free StopIteration if the CACHE filter ever missed; an explicit assertion with the label would preserve debuggability. Same class of nit: `width()` can return 2 for hypothetical even shapes while the Rust probe only admits {1,4,8}; no admitted shape triggers it and failure would be a loud compile error, not silent wrongness, but a comment tying `width()` to the probe's admitted set would help future extension.

## Documentation consistency

`STAGE5_REPORT.md` new row and the artifact README are consistent with each other, with the code counts, and with the stated scope limits (no stock-launcher claim, no intrinsic pooled-cost claim across 210→1710 MHz, no exclusive-GPU claim, no production/reduction changes, review pending). The `.gitignore` addition is exactly the four declared lines and its re-include patterns cover the retained `.csv`/`.log`/`.ncu-repz` evidence without un-ignoring anything else globally. No staging/commit/push was performed during this review.

---
## Review Trace
- Workspace: /home/imi/Projects/pytortch-backend-experiments
- Diff target: HEAD
- Context entries requested: 8
- Context files loaded: 8
- Files selected for initial diff: 13
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 37797 prompt tokens, 429 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_files
- Tool call: read_files
- Model call: API usage 46220 prompt tokens, 93 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 51008 prompt tokens, 104 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_files
- Tool call: read_files
- Model call: API usage 59007 prompt tokens, 107 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 60725 prompt tokens, 92 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool result truncated: read_files from 36646 to 20000 chars
- Tool call: read_files
- Model call: API usage 67304 prompt tokens, 14416 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 81753 prompt tokens, 100 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 83078 prompt tokens, 285 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 83393 prompt tokens, 91 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 85651 prompt tokens, 248 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 87422 prompt tokens, 448 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 88713 prompt tokens, 666 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: list_repository_tree
- Tool call: search_repository
- Model call: API usage 89837 prompt tokens, 257 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: list_repository_tree
- Model call: API usage 92728 prompt tokens, 8340 completion tokens
- Review complete after 14 model call(s)
