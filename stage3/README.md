# Stage 3 experimental cooperative reductions

Read [STAGE3_REPORT.md](../STAGE3_REPORT.md) for the executed surface and limits.
The local selector is experimental and out of tree:

```python
import torch
from stage1.overlay import install

install()  # before importing torch._inductor
from stage3.backend import compile_backend

compiled = torch.compile(
    lambda x: x.sum(-1), backend=compile_backend, fullgraph=True, dynamic=True
)
```

Use the established Triton-free, GIL-disabled CPython 3.14t environment without
`PYTHON_GIL` forcing. Install/build and Inductor suites run sequentially; the
existing overlay installer is not a concurrent setup API.

```bash
bash stage3/setup.sh
TORCH_LOGS=guards,recompiles stage0/.venv/bin/python stage3/verify.py
stage0/.venv/bin/python stage3/components.py
stage0/.venv/bin/python stage3/regressions.py
stage0/.venv/bin/python stage3/capture.py
stage0/.venv/bin/python stage3/measure.py
```

`probe.py` captures real reduction lowering before fusion/compilation, retaining
sum, column sum, mean and RMSNorm-shaped bodies for all three storage dtypes.
`verify.py` proves real scheduler fusion, dtype/layout/dynamic/masking semantics,
compiled backward, streams, allocator pressure, kernel counts and negative gates.
`components.py` also executes the same cooperative artifact at reduction lengths
0 and 1 and validates malformed/divergent physical plans before IR construction.

The semantic sum DAG has logical outer/reduction coordinates. `ReductionPlan`
chooses one 256-unit workgroup per outer output, strided per-unit accumulation,
1024 bytes of shared scratch and a binary tree with uniform barriers. Full-size
epilogues revisit the row in a second per-unit loop within the same kernel.
The existing Stage 0 compiler, InfoBuilder metadata and current-stream enqueue
remain authoritative. Naturally split Inductor reductions use ordinary wrapper
temporary tensors and kernel sequencing; no CubeCL allocation or stream is used.

No max/min/Welford/arg reductions, general split planner, atomics, RNG, indirect
indexing, templates, vectorization or tuning is added. CUDA Graph capture of the
Stage 3 wrapper remains explicitly unsupported, as in Stage 2. HIP evidence is
source generation only, without HIPRTC/device/PyTorch launcher integration.

Only representative generated evidence and compact results are eligible for
Git. Local logs, caches, raw profiler traces and unchanged dependency source
snapshots remain ignored. No commit/push is part of these scripts.
