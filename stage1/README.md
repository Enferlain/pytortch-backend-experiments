# Stage 1: local experimental CubeCL Inductor backend

Read [STAGE1_REPORT.md](../STAGE1_REPORT.md) for evidence and limitations.
This uses the existing Stage 0 environment and compiler/launcher. The local
`compile_backend` callable is an experimental option, not a PyTorch release API.

From the repository root, build the extended bridge once:

```bash
bash stage0/setup.sh --features llvm
```

`--features llvm` preserves the independently tested LLVM alternative; Stage 1
always selects C++/NVRTC. The existing Stage 0 source/toolkit setup is required
for the LLVM regression. A C++-only build can omit the feature.

Run in fresh processes with no PYTHON_GIL setting:

```bash
stage0/.venv/bin/python stage1/source_trace.py
stage0/.venv/bin/python stage1/baseline_probe.py
stage0/.venv/bin/python stage1/verify.py
stage0/.venv/bin/python stage0/verify.py cpp
CUDA_PATH="$PWD/stage0/cuda13-toolkit/nvidia/cu13" stage0/.venv/bin/python stage0/verify.py llvm
```

Minimal application, from the repository root:

```python
import torch
from stage1.overlay import install
install()  # before any Inductor import; isolated source copy, exact wheel pinned
from stage1.backend import compile_backend


def f(x):
    return torch.sin(x) + 1

compiled = torch.compile(f, backend=compile_backend, fullgraph=True, dynamic=False)
x = torch.randn(257, device="cuda", requires_grad=True)
out = compiled(x)
torch.testing.assert_close(out, torch.sin(x) + 1)
grad, = torch.autograd.grad(out, x, torch.ones_like(out))
torch.testing.assert_close(grad, torch.cos(x))
```

Use the isolated `stage0/.venv/bin/python`, where Triton is unavailable. Tests
also block actual top-level Triton imports and record the already-optional
PyTorch attempts. Relative `torch._inductor.codegen.triton` imports are ordinary
PyTorch Python modules and are distinguished from the absent `triton` package.

Only static 1D, contiguous float32 CUDA pointwise schedules with one/two equal-sized
inputs and one disjoint output are supported. Backward accepts a contiguous
upstream gradient. Reductions (including compiling `f(x).sum()`), noncontiguous
or broadcast gradients, other devices/dtypes, non-finite constants and
unimplemented operations reject clearly. Special-float input tensors are supported.
No eager fallback is enabled. Inductor's empty-tensor elimination is retained.

The overlay changes only `Scheduler.create_backend`'s availability requirement;
shared installed PyTorch files are never written. Registration routes to CubeCL
only for the local `cuda_backend="cubecl"` compiler scope and preserves the
previous CUDA factory otherwise. Concurrent overlay installation/compilation
is outside this experiment. Use a fresh process when switching installations.
Stage 1 does not promise Inductor-managed CUDA Graph support; the independent
Stage 0 graph-capture contract and tests remain available.

Native modules and descriptor cache entries live for the process. Metadata
uploads happen on setup/cache misses. Enqueue still uses the current PyTorch
stream with `record_stream`, without device-wide synchronization or x/y copies.
Artifacts are saved under `stage1/artifacts`; generated overlays and caches are
ignored setup products.
