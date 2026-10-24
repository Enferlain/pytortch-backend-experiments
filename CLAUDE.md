# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

Run Python that imports torch or the backend with `stage0/.venv/bin/python` (isolated free-threaded 3.14t, no Triton, pinned PyTorch). The Triton reference runs use `/home/imi/Projects/sd-scripts/.venv-wsl/bin/python`.

## Tools and subagents

- Use Sonnet Medium agents for reviewing sections.
- Fetch relevant skills when one sounds like it would fit the task at hand.
- Search with `rg`. `.stage0-sources` is gitignored, so pass its path (`rg pattern .stage0-sources/cubek`) or use `-uu`.

## Folders

- `cubecl_torch` — **the live CubeCL backend**: kernel runtime, descriptor compiler, `torch.compile` front end (`cubecl_torch.inductor`) and custom ops. Build, test, equivalence and manifest commands are in [`cubecl_torch/README.md`](cubecl_torch/README.md).
- `stage0`–`stage5`, `fp_hardening` — frozen experiment history and the equivalence reference for `cubecl_torch`. Do not fix bugs or add capabilities there; fixes go only into `cubecl_torch`.
- `scoreboard` — real-workload coverage and cost harness (SDXL, Anima, optimizer); its `cubecl` path uses `cubecl_torch.inductor`.
- `.stage0-sources` — Reference checkouts of `cubecl`, `cubek`, `pytorch`, `tilelang`, `triton` and `KohakuMossTrainer`; see [`.stage0-sources/README.md`](.stage0-sources/README.md) for per-repo summaries, pinned commits and clone dates.

## Working rules learned on this repo

Details and commands are in [`cubecl_torch/README.md`](cubecl_torch/README.md) (Running the checks, GPUs on this machine, Comparing with Triton, Inspecting CubeCL kernels).

- **Standard:** Inductor+Triton is the floor for accuracy *and* speed, judged per kernel at matched settings, with no tolerance margins. Noise-level differences are fine. A real gap found inside a task is fixed inside it, unless that would take much more effort; then stop and flag it (a bead) rather than accept it.
- **CubeK first:** before designing a kernel strategy or plan (reduction layouts, merge orders, vectorization, matmul and attention), check how CubeK (`.stage0-sources/cubek`, e.g. `crates/cubek-reduce`) already does it and reuse or adapt its pattern where it fits. Record in the design why it was or wasn't used.
- **Check before claiming:** read the code in `.stage0-sources` (CubeK, Triton, Inductor) before asserting how upstream does something, and read both legality layers (Python `inductor/legality.py`, Rust `descriptor5.rs`) before building on a plan.
- **GPUs:** `cuda:0` is the display RTX 3090 and `cuda:1` the RTX 3080 (the reverse of `nvidia-smi`). Pin devices by name. Time on the 3080 and confirm important results on the 3090 too. Ad-hoc timing needs a clock warm-up.
- **One GPU job at a time.** Never rebuild the native pieces while a GPU job or test run is in flight. Run long jobs in the background and wait for their completion notification; don't poll with `pgrep -f` loops, which match themselves.
- **Suite:** run with `-n 4`. Clean `/tmp/pytest-of-$USER` and scratch caches after heavy runs.

## Performance / GPU tooling available

- `ncu` — Nsight Compute; kernel-level profiling, hardware counters, occupancy, memory throughput, stalls, register/shared-memory analysis.
- `nsys` — Nsight Systems; CPU/GPU timeline, CUDA API calls, launch gaps, synchronization, stream activity, NVTX ranges.
- `perf` — Linux CPU sampling/profiling; use for host-side library/function attribution.
- `cProfile` — Python function-level CPU profiling.
- `NVTX` — annotate backend phases/launches so they are identifiable in Nsight traces.
- CUPTI / PyTorch profiler — actual GPU kernel durations, kernel census, CUDA activity tracing.
- CUDA events — warm GPU/event timing; distinguish queue-inclusive timing from intrinsic kernel duration.
- `nvidia-smi` — GPU clocks, utilization, power, temperature and device-state checks.
- `nvdisasm` — inspect generated SASS.
- `cuobjdump` — inspect CUDA binaries/fatbins, PTX/SASS and resource information.
- `ptxas` — offline PTX assembly/resource diagnostics.
- CUDA Driver API attributes — production kernel register/local/shared-memory and launch-resource queries.
- Compute Sanitizer — CUDA memory, race, initialization and synchronization correctness checks.

Linux-native Nsight tools should be preferred for WSL processes. Windows CUDA 13.1 `ptxas`, `nvdisasm`, and `cuobjdump` are also available through WSL interop for offline diagnostics; production compilation remains CUDA 13.0 NVRTC.

## Disk space

Space is finite (WSL's virtual disk only grows on the Windows drive). Anything that writes
large temporary output (per-case caches, traces, profiles, captured tensors) must delete it
when done unless the caller asked to keep it: harness work directories, scratch inputs,
`.ncu-rep`/`.nsys-rep` files. The corpus equivalence/manifest harnesses use ~8 GB of caches per
run and remove them unless `--work` is given. After heavy runs, check `du -sh /tmp` and the
repo's ignored caches, and clean up what is no longer needed.

<!-- codebase-memory-mcp:start -->
## Codebase Memory

This repository uses Codebase Memory for structural code exploration.
Use the installed `codebase-memory` skill for architecture, symbol discovery,
call/dependency tracing, impact analysis, and related structural queries.
Prefer its graph workflow before broad grep/glob/file exploration.
<!-- codebase-memory-mcp:end -->

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->
