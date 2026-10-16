# Review: cwk.7 explicit plane-final generated-graph qualification

## Verdict

The change set is sound and matches the stated intent. The Rust guard broadening is correctly scoped to the explicit probe route only, the codegen was already extent-generic so the widened admission is semantically justified, and the harness/analyzer are carefully constructed. I verified the critical mechanisms (wrapper module-level kernel binding, substitution scoping, baseline-vs-candidate parity, barrier-freedom of the plane path, artifact identity gates). I found **one medium isolation/provenance issue** and a few low-severity observations; nothing blocks the diagnostic claims as scoped.

## 1. Production-adjacent change (`descriptor3.rs`) — correct

- **Route isolation verified**: `final_plane: true` is only set in `main.rs:152` for the `cpp-probe-final-plane`/`-uniform` routes. The ordinary `cpp` path (`descriptor5::build`) never sets it, so the guard change cannot affect normal compiles, the planner, or defaults. `vector_width`/`private_accumulators` are hardcoded to 1 by the route, so those guard arms are belt-and-braces only.
- **The 1..32 bound is load-bearing, not arbitrary**: the plane codegen (`descriptor3.rs:711-718`) loops `RangeLoopOp(begin = lane, end = reduction_extent, step = 32)`, i.e. lane *t* loads `p[o,t]` iff `t < extent`, followed by the fixed `[16,8,4,2,1]` ShuffleXor butterfly over the 32-lane plane. Extent ≤ 32 guarantees exactly one element per lane and matches the CPU `final_model` (zero-padded 32-lane butterfly) association-for-association; extent ≥ 1 excludes the empty reduction. Guard and codegen agree.
- **Rejection semantics of the new check are right**: missing/non-object `reduction_extent` indexes to `Null`; `as_u64()` returns `None` for floats (incl. `9.0`), negatives, and null → rejected; 0 and 33+ rejected. Matches the old exact-match strictness for the integer case.
- Minor loosening (low): the old exact `json!` comparison also required the object to have *exactly* `op`/`value` keys; the new check tolerates extra unknown keys in `reduction_extent`. Harmless for a diagnostic route, but it is a (small) admission widening beyond 1..32.

## 2. `final_model` generalization — bitwise-safe for extent 9

For the only case the pre-existing cwk.7 gates exercise (9-wide partials), `lanes[..., :9] = zeros_like(partials) + partials` is unchanged — the frozen 148-case results are byte-identical by construction. For n ≤ 32 I verified the neighbor comparison is a genuine independent recomputation: `finish_model` pads to 256 lanes and descends `(128,…,1)`; distances 128/64/32 only add +0 padding, then 16..1 reduce the same 32 lanes in the same association as the plane model. Signed-zero (−0.0 → +0.0 via `+0` init), NaN propagation through the full butterfly, and the min-subnormal row are all handled identically on both sides because both models apply the identical `zeros_like + p` initialization.

## 3. `--qualification-only` — clean

`measure`'s early return is safe: `timings` is bound into `result` once and nothing else in `run()` consumes `measure` side effects; the retained sleep-based measurement (incl. `torch.cuda._sleep`) is skipped entirely, and the harness asserts `result["timings"] == []`. `getattr(args, "qualification_only", False)` keeps all existing callers unaffected. Double `evidence.guard()` (harness + `run`) is idempotent (the import wrapper closes over the original `__import__`; meta_path re-insertion is harmless).

## 4. New harness — verified mechanisms

- **Cache isolation / same-process variant separation is real.** I confirmed from retained wrappers that `define_kernel` produces *module-level* bindings (`cubecl_stage5_N = cubecl_stage5_compile(...)`, called later with `(scalars, tensors)`). The `add_import_once("# cwk.7 plane final: …")` tag is therefore the correct — and necessary — mechanism: without it, PyCodeCache would hand the second variant the first variant's module and its module-level kernel objects. Tagging at `AutotunedKernel.__init__` (rather than reading the flag at call time) is what makes `runtime.CACHE.clear()` + variant flip behave correctly, since a cleared CACHE would otherwise re-resolve the *same* semantic key to the wrong-tagged object. The per-capture entrypoint census (`names * 16`) would fail loudly on any leakage.
- **Substitution is precisely scoped**: `_cwk7_plane` ∧ policy `reference512_diagnostic` ∧ semantic `reduction_extent == {constant, 9}` ∧ plan `shared_tree_reduction_baseline`. The partial's extent is dynamic and > `LONG_REDUCTION_THRESHOLD` (4096), so it takes the `plane_shared_reference512` path and is never substituted — the frozen partial's compile route is untouched. The `__new__`-constructed `CompiledKernel` sets every attribute `__call__` uses (`kernel`, `entrypoint`).
- **Ownership/bounds/sync**: controls use canary-backed storages with real storage offsets; `Kernel._validate` enforces span/stride/disjointness; the plane codegen has no `create_shared`, no `sync` in the planes branch (the post-reduction sync at line 1026 is `if !planes`), and the outer store is leader-only (`unit == 0`) — the barrier-freedom claim matches the code. Loop bounds make OOB reads impossible (loads execute only when `coordinate < extent`).
- **No synthetic sleeps/primers** in the new harness (the only `_sleep` usage lives in the retained probe's `measure`, which qualification mode now skips). Warmup is real compilation/execution only.
- **Triton reference is honestly scoped**: `DriverKernel` is driver-API cubin launch only; its two-kernel sequence is validated bitwise against the graph output (`tensor_sha(y) == output_sha`), and only device-side `dur`/`ts` are extracted — consistent with "device-only reference, not full wrapper/host-time".
- **Frozen partial byte-identity** has a real gate: the analyzer sha-compares the timing-run partial artifacts against the retained r0 `selected_kernels[0]` files, and `controls` recompiles the retained partial/final/candidate with the rebuilt compiler and demands byte-identical IR/CUDA/PTX (27 files) — this is exactly the regression check that the guard change leaves extent-9 output unchanged.
- **Counts all cross-check**: 148 = 115 core + 33 extra; 1332 = 148×9; 324 = 3×9×6×2; 27 = 3×3×3; 178 rows; 108 = 36 graph + 72 isolated traces.
- **Failure preservation**: staged admission (CPU → controls → qualification → extras/timing → isolated → analyze), `complete: False` written early and only flipped at the end, `qualification()` writes partial `result` in its `finally`, `compile_final` uses `exist_ok=False`, per-mode harness archiving refuses reruns, and `overlay.install` is restored before the `lambda: None` could leak if `original_install()` raises.

## 5. Findings

1. **[Medium] `extra_graphs`: `TORCHINDUCTOR_CACHE_DIR` set after `install()` and after `from stage5.backend import compile_backend`** (which imports `torch._inductor.config` at module level). `timing()` does this in the correct order (env → `install()` → imports); `extra_graphs` inverts it, so the intended `cache/cwk7-extra` PyCodeCache dir may be inert depending on how the installed torch caches `cache_dir`. Correctness is unaffected (wrapper sources are distinct via the plane tag and keyed by source hash), but the cache-isolation provenance for the extras packet is weaker than intended. Move the assignment above `install()`.

2. **[Low] GPU neighbor controls are parity-only at extent n.** The 324 checks compare freshly compiled baseline (tree, route `cpp`) vs candidate (plane) on the same replaced semantic — there is no absolute comparison to the CPU-model values for `direct` inputs on GPU (input/canary/immutability checks are present). That is fine for the substitution-parity claim actually made, but "nearby-size checks" should not be read as absolute extent-n correctness. Relatedly, `replace()` rewrites *every* `{"op":"constant","value":9}` in the semantic — correct for the current fixed_final descriptor (all 9s are extent-related: buffer size/stride/index multiplier/reduction_extent) and indirectly policed by `Kernel._validate` against the actual `(count, n)` tensors, but brittle if the descriptor ever gains an unrelated constant 9. An explicit comment or count assertion would harden it.

3. **[Low] Analyzer does not independently re-assert substitution occurred.** The harness asserts `sum(r["substituted"]) > 0` before writing, but `reduction_final_graph_analyze.py` never checks `qualification["substitutions"]`. A degenerate rerun that produced identical rows with zero substitutions would be indistinguishable to the independent admission step. One added assert closes this.

4. **[Low] In-harness frozen/plane partial byte-compare may be a self-comparison**: both phases resolve to the same `cache/artifacts/<key>` directory via the integrity cache, so `read_bytes()` equality can be trivially true. The substantive gate (analyzer sha vs retained r0 artifacts) exists, so this is only worth knowing when reading the harness.

5. **[Info]** `host_enqueue_gap` in the analyzer is actually the device-timestamp gap between partial end and final start within a call (enqueue-influenced but not a host measurement); naming slightly overstates what it isolates. `isolated(counter=True)` intentionally writes no artifact. `actual == old` dict equality in `cpu_gate` would spuriously fail on NaN row values (none exist in this corpus; the convention is inherited from the prior pass).

## 6. Conclusion

Admission ordering, FP-order equivalence (descending 16..1, +0 init, lazy lane loads), ownership/bounds (canaries, offsets, leader-only store, disjointness), synchronization (barrier-free plane path, single reduction + outer-only stores enforced by the guard), cache isolation (module-source tagging + CACHE lifecycle), provenance (compiler sha, input shas tied to retained r0 runs, trace shas re-verified, retained artifact copies), and the measurement conclusions (device-only durations, rotated orders, no synthetic activity, explicit `automatic_eligibility: False`) are all implemented as described. The one actionable item is the `extra_graphs` env-var ordering; items 2–4 are hardening/clarification suggestions, not blockers.

---
## Review Trace
- Workspace: /home/imi/Projects/pytortch-backend-experiments
- Diff target: HEAD
- Context entries requested: 3
- Context files loaded: 3
- Files selected for initial diff: 5
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 15148 prompt tokens, 14113 completion tokens
- Model call: model requested 4 tool call(s)
- Tool call: read_file
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 31107 prompt tokens, 198 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 33693 prompt tokens, 173 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 36317 prompt tokens, 599 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 38420 prompt tokens, 66 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 39495 prompt tokens, 1051 completion tokens
- Model call: model requested 4 tool call(s)
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 40806 prompt tokens, 202 completion tokens
- Model call: model requested 3 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Tool call: search_repository
- Model call: API usage 42857 prompt tokens, 652 completion tokens
- Model call: model requested 3 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 46288 prompt tokens, 1175 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 47527 prompt tokens, 165 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: list_repository_tree
- Model call: API usage 47748 prompt tokens, 30 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 48852 prompt tokens, 808 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 52811 prompt tokens, 1007 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 53842 prompt tokens, 54 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: list_repository_tree
- Tool call: search_repository
- Model call: API usage 53998 prompt tokens, 30 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 55017 prompt tokens, 462 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 57416 prompt tokens, 335 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 58142 prompt tokens, 173 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 58371 prompt tokens, 59 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 60258 prompt tokens, 426 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: search_repository
- Model call: API usage 61823 prompt tokens, 58 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 63217 prompt tokens, 2636 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 65877 prompt tokens, 63 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: read_file
- Model call: API usage 66575 prompt tokens, 92 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 66717 prompt tokens, 29 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 67480 prompt tokens, 498 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 68034 prompt tokens, 30 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 68110 prompt tokens, 30 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 68861 prompt tokens, 1369 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 71320 prompt tokens, 550 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 71894 prompt tokens, 38 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 75008 prompt tokens, 1663 completion tokens
- Model call: model requested 4 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 76761 prompt tokens, 119 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result truncated: read_files from 25228 to 20000 chars
- Tool call: read_files
- Model call: API usage 82412 prompt tokens, 617 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 84561 prompt tokens, 426 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 85022 prompt tokens, 64 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 85268 prompt tokens, 32 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 85898 prompt tokens, 2960 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 89607 prompt tokens, 6700 completion tokens
- Model call: model requested 3 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 96373 prompt tokens, 32 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 97827 prompt tokens, 857 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 98708 prompt tokens, 30 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 99503 prompt tokens, 171 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 100086 prompt tokens, 6455 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 106558 prompt tokens, 2499 completion tokens
- Review complete after 45 model call(s)
