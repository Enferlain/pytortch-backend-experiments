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
| `inductor/` | `install()` (pinned overlay) and `compile_backend`; the Inductor class chain, capabilities, candidate legality, reduction policy, reference512 and final/column plans, first-use tuning |
| `ops/` | `torch.library` registration over artifacts; proof op `cubecl_torch::sin_plus_one` |
| `native/` | Rust descriptor compiler (`cubecl-stage0`), CUDA driver launcher, CubeCL FP patch stack |
| `testing/` | tests, Stage 5 equivalence harness, structural manifest, gate contract inventory |

## Environment

Everything runs in the isolated free-threaded venv, `stage0/.venv` (CPython 3.14t, no Triton,
PyTorch symlinked from the pinned wheel). Run from the repository root with that interpreter.

The artifact cache root is resolved in this order: `cubecl_torch.runtime.cache.configure(path)`,
then `CUBECL_TORCH_CACHE_DIR`, then `$XDG_CACHE_HOME/cubecl_torch` (`%LOCALAPPDATA%` on
Windows). Set `CUBECL_TORCH_TRACE=1` to record bounded evidence events.
`CUBECL_TORCH_COMPILER`, `CUBECL_TORCH_LAUNCHER`, `CUBECL_TORCH_METADATA_LIB` and
`CUBECL_TORCH_NVRTC` override the native compiler binary, launcher library, metadata library
and NVRTC library; their digests are part of every artifact key.

## Commands

Build the native pieces. This checks that `.stage0-sources/cubecl` is at the audited pin with
exactly `native/cubecl-fp.patch` applied (`python fp_hardening/patches.py` applies it):

```bash
cubecl_torch/native/build.sh
```

Run the tests. `CUBECL_TORCH_SLOW=1` adds the native FP-rewrite suite, which builds `cubecl-llvm`:

```bash
stage0/.venv/bin/python -m pytest cubecl_torch/testing
CUBECL_TORCH_SLOW=1 stage0/.venv/bin/python -m pytest cubecl_torch/testing -k rust_suite
```

Use the backend:

```python
from cubecl_torch.runtime import guard
guard.install()                      # optional: GIL/Triton evidence, before torch
import torch
import cubecl_torch.inductor
cubecl_torch.inductor.install()      # before anything imports torch._inductor
f = torch.compile(fn, backend=cubecl_torch.inductor.compile_backend)
```

Structural equivalence against the Stage 5 backend, over the 168-case corpus:

```bash
stage0/.venv/bin/python -m cubecl_torch.testing.equivalence run \
    --out cubecl_torch/testing/results/equivalence-full.json --jobs 6
```

Manifest. After a change, `check` reports which cases, regions and files changed. Re-record
only after reviewing an intended change:

```bash
stage0/.venv/bin/python -m cubecl_torch.testing.manifest_check check --jobs 6
stage0/.venv/bin/python -m cubecl_torch.testing.manifest_check record \
    --from cubecl_torch/testing/results/equivalence-full.json
```

Warm per-launch host overhead against Stage 5, and regeneration of the gate contract inventory:

```bash
stage0/.venv/bin/python -m cubecl_torch.testing.host_overhead \
    --out cubecl_torch/testing/results/host-overhead.json
stage0/.venv/bin/python -m cubecl_torch.testing.contract_inventory
```
