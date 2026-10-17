# Code Review: wqc — vector4 PlaneReader/Sum RMS trial

I read the untracked implementation files, the retained CubeK/CubeCL sources they compose, the frozen semantic templates, the descriptor emitter, and the artifact packet, and traced the numerical order end-to-end.

## Verdict: No blockers. The implementation matches the stated intent, the stop condition is respected, and the evidence/honesty claims check out. Findings below are minor.

## What I verified

**Semantic adapter / indexing (main.rs)**
- `Operand` read (mode 0): `scalar_pos = 4*pos + c` (L109–126), then `column = scalar_pos − row*n` (L200–207), and the unchanged `descriptor2::emit` re-adds `linear*N` — the documented round trip. Combined with `ParallelReader::read_plane` (`pos = 32*k + lane`, `batch_offset = (row*n)/4`, exact since all admitted widths divide 4), the global element order is exactly `128*k + 4*lane + c`, matching the claimed and CPU-modeled order.
- Reader semantics: `num_chunks = n/4`, `length_plane = ceil(n/4/32)` (8 iterations at 1024); `BoundChecks::Branch` returns `null_input` (+0 vector) out-of-bounds — matching the CPU model's zero padding (only width 32 has tail lanes).
- `Sum` path is stock: `null_accumulator` = +0 `Vector<f32,4>`, `ReduceStep::Identity` element-wise accumulate (4 private chains), `plane_reduce_inplace` → per-component butterfly (SHFL 1280 = 4×320 corroborates 5 XOR steps), `to_output_parallel` → `Vector::vector_sum` (+0 ascending fold). Epilogue and `lane+32*k` store loop (L336–338) are unchanged descriptor replay; metadata-only output stays `[B,1]`-shaped with scalar shape/stride `[B,N]/[N,1]` (L208–221). `bind()`'s slot-2 substitution is consistent with the templates (defining `reduction` op has `dtype`, nested index references don't).
- Row mapping `cube_x*8 + unit_y` with `row < rows` guard, `effective_plane_dim = CUBE_DIM_X = 32`, grid `ceil(B/8)` via schema-3 `outer`/`outputs_per_workgroup` — all consistent.

**CPU model (`rms_vector_trial_data.py`)** — FP32 square, ascending-k chains from +0, XOR 1,2,4,8,16, +0 ascending component fold, `/n + 1e-5 → rsqrt`, per-lane factor broadcast `col%32`, final storage cast: matches the GPU mechanism step for step, including signed-zero-safe +0 starts.

**Guards** — `CompiledKernel.__init__` calls `cpu_gate()` before any GPU compilation, structurally enforcing CPU-before-GPU. Widths (32/256/1024/4096, all ÷4), rows 1..65, four-storage-element input alignment (L83–90), plus the inherited shape/stride/dtype/device/alias/storage-span contract produce exactly the 9×3 = 27 live negatives; the nondefault-stream offset-4 control and 12 static controls (9 Python `admit` + 3 Rust envelope: plan/mechanism/entrypoint, all rejected before IR) match the claims.

**Attribution & integrity** — The analyzer cross-binds: timed wrapper output hashes == direct qualification hashes (component and scalar), NCU kernel names == entrypoints of the timed directories, parsed SASS sums == `smsp__inst_executed.sum` (Decimal equality), scalar-control IR/CUDA/PTX == frozen 5nb modulo entrypoint, lock external pins == retained probe, 13,266 preservation hashes, environment/NVRTC/device UUID. Stale-pass clearing (`analysis.json` written `complete:false` first) is present.

**Timing honesty** — README/report tables match `analysis.json` exactly (all 9 graph rows, isolated rounds, pooled medians, per-case win counts 0-vs-ordinary everywhere, 6/0/1 vs scalar). Full isolated record retained with the cross-clock-state disclaimer; `--cache-control none --clock-control none`, `--profile-from-start off` confirmed in `counter-command.txt`; scalar-always-middle and not-fully-permuted limitations are disclosed.

**Scope** — New backend is a separate module (`rms_vector_trial_backend.py`) that only schedules through the explicit adapter with `admit()`; ordinary catalog/planner/runtime untouched. `Cargo.toml` is byte-identical to `rms_probe`'s except the root name. `.gitignore` additions mirror the 5nb pattern (dir + `*.csv`/`*.ncu-repz` re-inclusion).

## Findings (minor)

1. **`rms_vector_trial.py` L391–396 — precedence-fragile filter.** `corpus_ or (row_tail and (16,65))` is correct only because `and` binds tighter; the outer parentheses invite misreading as `(corpus_ or row_tail) and (16,65)`. Add explicit inner parens.
2. **README claims `review/` is retained, but `stage5/artifacts/reuse-first-rms-vector4/review/` does not exist yet** (this review is the pending one). Add the disposition there before closing wqc.
3. **Two README claims are retained-evidence-only, not analyzer-enforced:** "no 128-bit LDG" (the SASS parse folds `LDG.E.128` into `LDG`; equality of vector LDG count 4416 with the all-scalar control is strong indirect evidence but not a machine check) and "process query reports no external compute rows" (only returncode is asserted). Consider a targeted opcode-suffix check if this packet is relied on later.
4. **CPU-gate chronology is self-attested** (`gpu_candidate_source_existed: false` recorded by the same producer), mitigated by hash-bound `phase-sources` snapshots and the cpu.json hash baked into every compiled candidate identity — inherent to the untracked workflow, same as 5nb; no action needed.
5. **Dead-ish code in main.rs L229:** the `One`-variant `__expand_read_method` asserts `mode == 0` but no read goes through it in the vector4 path (inherited 5nb shape). Harmless; the assert message could mislead a future reader.
6. **Note:** this commit also adds the hs6 attribution row/artifacts (report + `.gitignore` lines) alongside wqc. Consistent with "preserved," just be aware they land together.

No files were modified during this review.

---
## Review Trace
- Workspace: /home/imi/Projects/pytortch-backend-experiments
- Diff target: HEAD
- Context entries requested: 4
- Context files loaded: 4
- Files selected for initial diff: 12
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 19129 prompt tokens, 410 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: list_changed_files
- Tool result truncated: get_uncommitted_changes from 29323 to 20000 chars
- Tool call: get_uncommitted_changes
- Model call: API usage 24470 prompt tokens, 143 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 29540 prompt tokens, 41 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 30817 prompt tokens, 1868 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 34356 prompt tokens, 1070 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 35513 prompt tokens, 33 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 40334 prompt tokens, 1854 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 42212 prompt tokens, 60 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: list_repository_tree
- Model call: API usage 42295 prompt tokens, 41 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 42585 prompt tokens, 77 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 44088 prompt tokens, 53 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 45483 prompt tokens, 992 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 46510 prompt tokens, 42 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 47021 prompt tokens, 71 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 47116 prompt tokens, 44 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 48142 prompt tokens, 1145 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 52248 prompt tokens, 32 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 55163 prompt tokens, 33 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 57609 prompt tokens, 1049 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 59163 prompt tokens, 1707 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 62615 prompt tokens, 587 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 63668 prompt tokens, 33 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 64237 prompt tokens, 511 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 66144 prompt tokens, 242 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 66432 prompt tokens, 80 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 68535 prompt tokens, 95 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 70967 prompt tokens, 2510 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 74329 prompt tokens, 112 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 77898 prompt tokens, 35 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 78311 prompt tokens, 8545 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 87817 prompt tokens, 1136 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_files
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 89356 prompt tokens, 113 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 90239 prompt tokens, 534 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: list_repository_tree
- Model call: API usage 92359 prompt tokens, 202 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result truncated: read_files from 35122 to 20000 chars
- Tool call: read_files
- Tool call: list_repository_tree
- Model call: API usage 103522 prompt tokens, 416 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: list_repository_tree
- Model call: API usage 104120 prompt tokens, 711 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 104878 prompt tokens, 43 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 105530 prompt tokens, 44 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 105598 prompt tokens, 33 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 105655 prompt tokens, 77 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 105756 prompt tokens, 40 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 105820 prompt tokens, 105 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: list_repository_tree
- Model call: API usage 105944 prompt tokens, 40 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: list_repository_tree
- Model call: API usage 106002 prompt tokens, 254 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 106293 prompt tokens, 43 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 107972 prompt tokens, 6722 completion tokens
- Review complete after 46 model call(s)
