# Tasks

## 1. Scaffolding and native build

- [x] 1.1 Create the `cubecl_torch/` package skeleton (`runtime/`, `compiler/`, `inductor/`, `ops/`, `native/`, `testing/`) and a root `pyproject.toml` declaring the package, ruff settings and pytest config. Verify that `stage0/.venv/bin/python -c "import cubecl_torch"` succeeds from the repo root and that `ruff check cubecl_torch` passes.
- [x] 1.2 Copy the Rust crate, `launcher.c` and both CubeCL patch files into `cubecl_torch/native/`, and fix the Cargo path dependencies to the same `.stage0-sources/cubecl` checkout. Add `native/build.sh`, which verifies the patch state and builds the release binary and launcher library. Verify that the build succeeds and `git status stage0 fp_hardening` is clean.
- [x] 1.3 Add the layer-dependency test in `cubecl_torch/testing`: `runtime/` imports nothing from `compiler/`, `inductor/`, `ops/`, `torch._inductor` or `torch._dynamo`; `compiler/` imports only `runtime/`. Verify that it runs under `stage0/.venv/bin/python -m pytest cubecl_torch/testing`.

## 2. Runtime: artifact contract, cache and launch

- [x] 2.1 Define the `KernelArtifact` contract (PTX, entry point, launch shape, ordered parameters, tensor guards as data rules, provenance), and implement loading and launch on the current stream, built from the `stage0/bridge` launch path and `launcher.c`. Verify with a test that launches the Stage 0 programmatic `sin(x)+1` PTX (`stage0/artifacts/cpp`) through a hand-written contract on a non-default stream, matches `torch.sin(x) + 1`, and loads no `torch._inductor`, `torch._dynamo` or `compiler/` module.
- [x] 2.2 Implement contract-driven launch validation. Verify with one test per mismatch class (device, dtype, strides/offset, extent), each asserting the launch is refused with an error naming the class.
- [x] 2.3 Implement the integrity-checked artifact cache: key from producer inputs plus provenance only, root resolved from explicit config, then `CUBECL_TORCH_CACHE_DIR`, then the per-user default. Verify with tests for: custom directory used exclusively; warm reuse without invoking a compiler; a corrupted file refused; a key unchanged when only a policy file changes.
- [x] 2.4 Implement provenance recording and refusal of artifacts whose CubeCL pin or patch state isn't the audited one. Verify with a test that loads a contract with altered provenance and asserts refusal.
- [x] 2.5 Make tracing opt-in (`CUBECL_TORCH_TRACE`) and bounded, and add the GIL/Triton guard helper (adapted from `stage1/evidence`). Verify with tests: tracing off by default and recording when enabled; a 3.14t process loading and launching an artifact keeps the GIL off and never imports Triton.

## 3. Descriptor compiler

- [x] 3.1 Copy the descriptor and plan model into `compiler/` (`stage1/plan`, `stage3/plan`, `stage5/plan` types, `stage2/indexing`, `stage1/precision`) with imports rewritten. Verify the module imports and the layer-dependency test.
- [x] 3.2 Implement the Rust/NVRTC adapter producing a `KernelArtifact` from a schema-5 packet, and the binding that turns descriptor buffers, scalars and symbolic sizes, strides, offsets and element count into contract parameters and guards (the logic in `stage2/runtime.py` through `stage4/runtime.py` and `stage5/artifact.py`). Verify that a hand-authored pointwise descriptor compiles and runs without Inductor. Also verify that, for a dynamic-shape descriptor at a newly seen extent, the bound parameters equal Stage 5's for the same descriptor and tensors.
- [x] 3.3 Implement the audited-source check with a parameterised source path. Verify with a test against a temporary copy or fake git checkout containing an unexpected change: compilation is refused and no artifact is written. The real `.stage0-sources/cubecl` must not be modified.
- [x] 3.4 Verify deterministic compilation and malformed-descriptor rejection. Tests: the same descriptor compiled twice from empty caches gives identical IR/CUDA/PTX bytes; descriptors with missing or inconsistent fields fail with an error naming the problem and leave the cache empty.

## 4. Inductor front end

- [x] 4.1 Copy the overlay installer verbatim into `inductor/overlay.py`, keeping the pinned-commit and diff checks, and install it only from the backend entry point. Verify with tests: importing `runtime` or `compiler` doesn't install it; a mismatched `torch.version.git_version` (monkeypatched) is refused before any Inductor import; digests of every installed `torch/_inductor` file are identical before and after installing and compiling a graph.
- [x] 4.2 Copy the capability modules and the class chain into responsibility-named modules (`pointwise`, `semantics`, `reductions`, `effects`, `physical`), keeping the inheritance links and using `compiler/` and `runtime/`. Verify that `torch.compile(lambda x: torch.sin(x) + 1, backend=...)` runs and matches eager.
- [x] 4.3 Move candidate generation, legality and admission, reduction policy, the final/column/reference512 plans and oracle, and first-use tuning (`stage5/runtime`, `timing`, `choices`, `index_masks`, `diagnostics`) into `inductor/`. The selection cache keeps Inductor's `PersistentCache` under the configured root, with policy sources in its identity. Add a hook for injected deterministic timings. Verify by compiling a reduction case under `first_use` and `baseline`.
- [x] 4.4 Expose the backend entry point with the same policy names and options as `stage5.backend.compile_backend`, raising the existing unsupported-capability error with its reason. Verify with a test for an unsupported op (`relu`): the error type and that the message names the operation.

## 5. Equivalence and manifest

- [x] 5.1 Implement the structural equivalence harness in `testing/equivalence.py`. Per case it builds an ordered list of region records (identity, role, descriptor digest, physical plan, IR/CUDA/PTX digests) in fresh processes for `stage5.backend` (via a compiled-kernel constructor hook) and for the package. It compares the lists with multiplicity, along with candidate sets and `baseline` outputs bitwise. Verify on a 5-case subset that the harness reports per-case results, and that it detects both a deliberately altered artifact and a deliberately duplicated region mapping.
- [x] 5.2 Add injected-timing selection equivalence: replace both backends' benchmarkers with the same deterministic timing table and compare chosen candidates and selection-cache key data (without compiler provenance) per case. Verify on the subset, including detection of a deliberately perturbed timing.
- [x] 5.3 Run the full 168-case structural, output and selection equivalence. Fix any divergence in the package, never by editing the stages. Verify that all cases are identical, and keep the result JSON under the configured results location.
- [x] 5.4 Add the warm per-launch host-overhead microbenchmark (same prepared kernels, alternating rounds, both implementations). Verify that the package median is within the measured noise of Stage 5, and record the result.
- [x] 5.5 Record the structural manifest from the passing run, and add `testing/manifest_check.py`, which reports changed cases, regions and files. Verify that the check passes on the current package and flags a deliberately edited plan constant (reverted afterwards).

## 6. Historical contract inventory

- [x] 6.1 Write `testing/contract_inventory.md` listing every assertion in the Stage 5 gates (`components`, `controls`, `proof_controls`, `descriptor_controls`, `launch_controls`, `cache_probe` init/reload, `compiler_compare`), the Stage 4 gates (`verify`, `runtime_checks`, `verify_bounds`, `counter_guard_probe`, `components`) and `stage4/regressions` (stages 0–3). Classify each as a live contract or historical, with a reason for each historical one. Verify that every assertion in those scripts appears in the inventory.
- [x] 6.2 Port each live-contract assertion into package tests (runtime, compiler or Inductor front end), and link each inventory entry to its test. Verify that all ported tests pass and every live entry links to a passing test.

## 7. Custom-op front end

- [x] 7.1 Implement the `ops/` registration helper (schema, fake implementation, CUDA implementation over a `KernelArtifact`) and the proof op `cubecl_torch::sin_plus_one` from a hand-authored schema-5 descriptor with a fixed plan. Verify with a test: eager results match `torch.sin(x) + 1` on a non-default stream, and no Inductor or Dynamo module or overlay is loaded.
- [x] 7.2 Verify the mixed graph. Compile `f(x) = cos(sin_plus_one(x * 2)) + 1` with the package backend and check: one extern call for the op in the compiled graph (via the Inductor graph/kernel record); generated CubeCL kernels for the surrounding ops; graph-break counter at 0; output equal to eager.
- [x] 7.3 Verify free-threaded use. A test calls the op from 4 threads on independent tensors and streams, checks all results, and asserts the GIL stays disabled.

## 8. Integration and handover

- [ ] 8.1 Switch `scoreboard/run_one.py`'s `cubecl` path to `cubecl_torch.inductor`, and run the fail-fast and inventory modes for all workloads. Verify that region counts and rejection reasons equal the 2026-10-20 Stage 5 results (`scoreboard/results/first-failfast`, `first-inventory`), and that the GIL stays off at every phase.
- [ ] 8.2 Update AGENTS.md, CLAUDE.md, STAGE5_REPORT.md and the scoreboard README: `cubecl_torch` is the live backend, the stages are frozen history, and fixes go only to the package. Document the build, test, equivalence and manifest commands, and verify each documented command runs as written.

## Workflow follow-up

- Archive the change once the equivalence, contract inventory, host-overhead and scoreboard parity results are recorded and the user has reviewed them.
- Close beads `sil.5` and unblock `4z8.6`, `4z8.7` and `4z8.15`, which land in the package.
