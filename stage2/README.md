# Stage 2: experimental CubeCL pointwise integration

Read [STAGE2_REPORT.md](../STAGE2_REPORT.md) for the tested surface and limits.
The local spelling is experimental:

```python
from stage1.overlay import install

install()  # Before importing torch._inductor; pinned isolated overlay.
from stage2.backend import compile_backend

compiled = torch.compile(f, backend=compile_backend, fullgraph=True, dynamic=True)
```

Use the established Triton-free CPython 3.14t environment. Installation and
compilation are sequential fresh-process experiments, not a concurrent compiler claim.

```bash
bash stage2/setup.sh
stage0/.venv/bin/python stage2/verify.py
stage0/.venv/bin/python stage2/components.py
stage0/.venv/bin/python stage1/verify.py
stage0/.venv/bin/python stage1/verify_boundaries.py
stage0/.venv/bin/python stage0/verify.py cpp
CUDA_PATH="$PWD/stage0/cuda13-toolkit/nvidia/cu13" stage0/.venv/bin/python stage0/verify.py llvm
```

`setup.sh` applies the exact audited [floating-point patch stack](../fp_hardening/README.md)
to the pinned CubeCL checkout (including upgrade from the original Stage 2 patch), rejects other modifications, builds the existing
compiler with optional LLVM and HIP source-probe features, and builds the same
native launcher. Stage 0's clean-source setup remains independent; after Stage 2
has applied its patch, use Stage 2 setup for rebuilding the patched compiler.

CUDA half/BF16 source needs the CUDA headers already bundled with this pinned
PyTorch environment under `nvidia/cu13/include`. No external runtime owns tensor
storage or streams. Only small InfoBuilder metadata is uploaded.

The semantic description, physical plan, runtime scalar arguments and runtime
tensor arguments stay separate. Generated wrappers allocate ordinary PyTorch
outputs and pass original view pointers. Cache keys include static semantic/index
structure, storage/compute types, physical plan and device. Symbol values are
passed at runtime; Dynamo guards still determine graph reuse and specialization.

`probe.py` records live lowered nodes before fusion without executing them.
`components.py` executes real compiler/launcher tests at the descriptor boundary
for signed 64-bit floor/mod, nested protected loads and predicated stores.
They complement the genuine `ops.masked` concatenation fixture in `verify.py`.
No synthetic Dynamo graph is used for these component tests.

No reduction, indirect-indexing, RNG, atomic, GEMM/template, general fallback,
vectorization, autotuning, persistent mapping or portable PyTorch launcher is added.
Stage 0 and Stage 1.5 backend entry points remain independently supported.
