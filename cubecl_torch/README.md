# cubecl_torch

The live CubeCL backend for PyTorch: a `torch.compile` backend built on Inductor, and
`torch.library` custom ops, both on one kernel runtime. It replaces the `stage5.backend`
subclass chain. The `stage*/` and `fp_hardening/` directories are frozen history and the
equivalence reference: **fixes go only into this package**.

## Layers

Dependencies point downwards only; `testing/test_layers.py` enforces it.

| Layer | Contents |
| --- | --- |
| `runtime/` | `KernelArtifact` contract, contract-driven validation, launch on the current stream, integrity-checked artifact cache, provenance check, opt-in tracing, GIL/Triton guard |
| `compiler/` | descriptor validation, schema-5 packets, indexing model, physical plans, Rust/NVRTC adapter, descriptor-to-contract binding, audited-source check |
| `inductor/` | `install()` (pinned overlay) and `compile_backend` (`backend`); admission at backend selection, pre-fusion and fusion with the shared op table (`admission`); the Inductor kernel, ops handler and scheduling (`codegen`); candidate legality, reduction policy, reference512 and final/column plans, first-use tuning |
| `ops/` | `torch.library` registration over artifacts; proof op `cubecl_torch::sin_plus_one` |
| `native/` | Rust descriptor compiler (`cubecl-stage0`), CUDA driver launcher, CubeCL FP patch stack |
| `testing/` | tests, corpus runner (`equivalence.py`), structural manifest, gate contract inventory |

The descriptor compiler in `native/src/`. The JSON `schema` numbers (1–5) are data-format
versions; `main.rs` dispatches on them.

| Module | Contents |
| --- | --- |
| `main.rs` | CLI: reads a descriptor, builds the kernel, writes `kernel.cu`/IR/manifest; offline codegen probe routes |
| `plan.rs` | schema 5: physical-plan validation (Rust legality layer) and dispatch to the builders below |
| `semantic.rs` | the semantic op layer every schema shares: index expressions, op validation, scalar replay into IR |
| `ir.rs` | IR helpers: storage dtypes, constants, guarded branches, indexed and vector pointers |
| `pointwise/` | `physical` (schema-5 scalar/vector plans), `column` (bounded column plan), `scalar` (frozen scalar256, schemas 2 and 4), `minimal` (schema 1) |
| `reduction/` | `combiner` (sum, max/min, Welford algebra), `analysis` (schema-3 validation, dependency closures), `emit` (the cooperative emitter for every strategy, `CodegenOptions` probes), `vector` (vector loads, unit-stride proof) |
| `effects.rs` | schema 4: Philox RNG, checked indirect addressing, atomics |
| `require.rs` | the device-trap precondition op used by indirect addressing |
| `lib.rs` | C ABI metadata serialization used by the launcher |

## Environment

Everything runs in the isolated free-threaded venv, `stage0/.venv` (CPython 3.14t, no Triton,
PyTorch symlinked from the pinned wheel). Run from the repository root with that interpreter.

The artifact cache root is resolved in this order: `cubecl_torch.runtime.cache.configure(path)`,
then `CUBECL_TORCH_CACHE_DIR`, then `$XDG_CACHE_HOME/cubecl_torch` (`%LOCALAPPDATA%` on
Windows). Set `CUBECL_TORCH_TRACE=1` to record bounded evidence events.
`CUBECL_TORCH_COMPILER`, `CUBECL_TORCH_LAUNCHER`, `CUBECL_TORCH_METADATA_LIB` and
`CUBECL_TORCH_NVRTC` override the native compiler binary, launcher library, metadata library
and NVRTC library; their digests are part of every artifact key.

## Numerics

Inductor+Triton is the floor, not the target: never measurably worse than Triton, and a proven,
material improvement is kept even where Triton differs (bead `4z8.17`). With default Inductor
config:

- FP16/BF16 math inside a fused kernel runs in FP32 and rounds once, at storage.
- A cast to FP16/BF16 in the graph (`x.to(torch.bfloat16)`) always rounds; Triton drops it.
- NVRTC contracts multiply-add pairs into FMA (`--fmad=true`), as Triton does.
- FP32 division and reciprocal use `div.full` (up to 2 ulp), as Triton does; correctly
  rounded division costs up to ~10% in division-bound kernels for an immaterial gain.

Inductor's default `emulate_precision_casts_on_saved_tensors` still rounds a low-precision
result saved for backward, so forward and backward see the same value.
`torch._inductor.config.emulate_precision_casts = True` restores eager's per-operation
roundings and turns contraction off; `torch._inductor.config.eager_numerics.division_rounding
= True` makes division correctly rounded. Both are the Triton switches, captured at codegen
into `compiler.nvrtc.Numerics`, part of every artifact key; Stage 5 ran with both on.

## Commands

Build the native pieces. This checks that `.stage0-sources/cubecl` is at the audited pin with
exactly `native/cubecl-fp.patch` applied (`git -C .stage0-sources/cubecl apply cubecl_torch/native/cubecl-fp.patch`
on a clean checkout at the pin applies it):

```bash
cubecl_torch/native/build.sh
```

Run the tests. `CUBECL_TORCH_SLOW=1` adds the native FP-rewrite suite, which builds `cubecl-llvm`.
`CUBECL_TORCH_GPU_TRAPS=1` adds `test_device_bounds_trap`, whose kernels trap on out-of-bounds
indices on purpose: under WSL each case stalls the GPU (the display, on `cuda:0`) while Windows
records a watchdog live dump. Run it when changing indirect indexing or bounds checks, ideally on a
GPU nothing else is using:

```bash
stage0/.venv/bin/python -m pytest cubecl_torch/testing
CUBECL_TORCH_SLOW=1 stage0/.venv/bin/python -m pytest cubecl_torch/testing -k rust_suite
CUBECL_TORCH_GPU_TRAPS=1 stage0/.venv/bin/python -m pytest cubecl_torch/testing -k device_bounds_trap
```

Use the backend:

```python
from cubecl_torch.runtime import guard

guard.install()  # optional: GIL/Triton evidence, before torch
import torch
import cubecl_torch.inductor

cubecl_torch.inductor.install()  # before anything imports torch._inductor
f = torch.compile(fn, backend=cubecl_torch.inductor.compile_backend)
```

Manifest, the structural gate over the 168-case Stage 5 corpus. After a change, `check` reports
which cases, regions and files changed. Re-record only after reviewing every difference
(`record` compiles the corpus with this package):

```bash
stage0/.venv/bin/python -m cubecl_torch.testing.manifest_check check --jobs 6
stage0/.venv/bin/python -m cubecl_torch.testing.manifest_check record --jobs 6
```

The corpus is mostly sums. `native_matrix` byte-compares two compiler binaries over a reduction
matrix the corpus doesn't reach: every schema-5 reduction plan for sum, max and Welford over
float32/bfloat16, final9 and every `cpp-probe-*` route (seconds, no GPU). A binary against itself
checks determinism across processes (the suite does this); for a refactor or CubeCL upgrade,
build the old revision in a worktree with its own `CARGO_TARGET_DIR` and compare:

```bash
stage0/.venv/bin/python -m cubecl_torch.testing.native_matrix OLD_BIN \
    cubecl_torch/native/target/release/cubecl-stage0
```

Structural equivalence against the Stage 5 backend was retired when CubeCL moved past `6bba5fdf`
(2026-10-25): the frozen Stage 5 side refuses any other CubeCL checkout. Its last full result was
336/336 identical at `6bba5fdf`; the manifest carried that state over unchanged.

Warm per-launch host overhead against Stage 5, and regeneration of the gate contract inventory:

```bash
stage0/.venv/bin/python -m cubecl_torch.testing.host_overhead \
    --out cubecl_torch/testing/results/host-overhead.json
stage0/.venv/bin/python -m cubecl_torch.testing.contract_inventory
```

## Running the checks

The suite runs in parallel with `pytest-xdist`: about 7 minutes with four workers, against 18
serially. The slowest tests compile many graphs in fresh subprocesses and compare them with
stored Triton data (`test_reduction_extrema`, `test_float_extrema`, the Welford front-end,
special-value and band tests, each 40–90 s).

```bash
stage0/.venv/bin/python -m pytest -q -n 4 cubecl_torch/testing
```

Approximate durations on this machine: manifest check and record about 10 minutes each
(`--jobs 6`, the most RAM-hungry steps, around 20 GB in WSL), the Welford band sweep
3–6 minutes per GPU and the Welford timing about 16 minutes per GPU. Run GPU jobs one at a time,
and never rebuild the native pieces (`native/build.sh`) while one runs: the compiler binary's
digest is part of every artifact key, so a mid-run rebuild mixes kernels from two compilers.
Tests leave gigabytes in `/tmp/pytest-of-$USER`; delete it after heavy runs.

Style checks, clean before every commit (ruff settings in the root `pyproject.toml`, clippy at its
default lint set with warnings as errors). They take seconds and need no GPU:

```bash
ruff format --check . && ruff check .
cargo fmt --manifest-path cubecl_torch/native/Cargo.toml --check
cargo clippy --manifest-path cubecl_torch/native/Cargo.toml --all-targets --locked -- -D warnings
```

`ty check` is configured in `pyproject.toml` too, but is not a gate until its findings are
resolved (bead `ecm`). Formatting or lint fixes in `native/` change the compiler binary's digest,
so caches rebuild; confirm with the manifest check that generated kernels did not change.

## GPUs on this machine

CUDA orders devices fastest first, `nvidia-smi` by PCI bus, so the indices are swapped:
`cuda:0` is the RTX 3090, which drives the display, and `cuda:1` is the RTX 3080 (`nvidia-smi`
index 0). Anything that does not pick a device runs on the 3090. Pin a device by name
(`--gpu 3080` in the timing tools) or with `CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=<nvidia-smi index>`. Time on the 3080; check important results on both.
Clocks swing between 210 and 1710 MHz: a timing script must warm the GPU (the timing tools run
a matmul loop first) or it reports kernels several times slower than they are.

## Comparing with Triton

Triton runs only in the reference environment, `/home/imi/Projects/sd-scripts/.venv-wsl/bin/python`
(set `PYTHONPATH` to the repository root for scripts outside the package). Use a fresh
`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` when capturing:

- Generated source: `torch._inductor.utils.run_and_get_code(compiled, *args)`.
- PTX and cubin: the `*.ptx` files under `TRITON_CACHE_DIR`.
- The launch configuration Triton actually chose (`XBLOCK`, `R0_BLOCK`, `num_warps`): wrap
  `torch._inductor.runtime.triton_heuristics.CachingAutotuner.run` and read
  `self.launchers[i].config`. It differs between static and dynamic shapes (LayerNorm
  4096×1280: `R0_BLOCK` 2048 static, 1024 dynamic), and that changes the arithmetic.

Compare at matched settings and kernel to kernel: CubeCL computes each reduction output in its
own kernel, so compare a one-statistic graph against Triton's one-statistic kernel, not graph
totals. The reference generators (`welford_reference.py`, `extrema_reference.py`) store
Triton's results, so accuracy runs need only this venv. `welford_band.py` decides accuracy per
reduced row (sign test plus paired mean difference, p < 0.01, no tolerance margin);
`welford_timing.py` alternates backends in separate processes and records per-kernel times.

## Inspecting CubeCL kernels and plan choices

With `CUBECL_TORCH_CACHE_DIR` set to a fresh directory, each compiled kernel is under
`artifacts/<key>/` (`kernel.cu`, `kernel.ptx`, `cubecl.ir`). First-use tuning writes
`tuning/<key>/selection.json`: every candidate's per-round timing samples, the winner and each
rejected plan with its reason. Early rounds can be inflated 2–5× (bead `sil.1`), so read the
rounds before trusting a selection. Tests and the timing tool pin a plan by replacing
`tuning.AutotunedKernel.select`.

Plan legality has two layers that must agree: Python (`inductor/legality.py`,
`structural_reason` / `numerical_reason`, which also checks the actual tensors' alignment and
strides) and Rust (`native/src/plan.rs`). Check both before building on a plan. A
selection policy that is about accuracy, not memory safety, belongs in `candidates()`, not in
`reason()`, which is also the launch-time contract. New plans for one reduction family go in
their own catalog (as `plan.welford_catalog()`), so the manifest's candidate lists for other
families stay unchanged.

`ncu` works under WSL. Scope it to one call with an NVTX range:
`ncu --nvtx --nvtx-include "target/" --section SpeedOfLight --section Occupancy --section LaunchStats --csv --page details <python> script.py`,
with the measured call wrapped in `torch.cuda.nvtx.range_push("target")` / `range_pop()`.
Nsight Compute fixes clocks, so its durations differ from CUPTI's; use it to compare kernels
under identical conditions (achieved occupancy, waves per SM, DRAM throughput).
