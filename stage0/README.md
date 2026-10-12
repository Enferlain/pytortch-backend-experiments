# CubeCL Stage 0

Programmatic CubeCL IR → CUDA artifact → one native launcher → PyTorch-owned
float32 tensors on the current PyTorch CUDA stream. Both compiler routes run.
See [REPORT.md](REPORT.md) and the actual [artifacts](artifacts).

This is a fixed pointwise experiment. It implements no Inductor backend,
autograd registration, reductions, tensor-storage runtime, or permanent ABI.

## Build

From the repository root, with Rust 1.97.1 and a C compiler:

```bash
bash stage0/setup.sh
```

For the optional LLVM route (CubeCL's bundler obtains LLVM 23.1.0-3):

```bash
bash stage0/setup.sh --features llvm
uv pip install --target stage0/cuda13-toolkit --no-deps 'nvidia-nvvm==13.0.88'
export CUDA_PATH="$PWD/stage0/cuda13-toolkit/nvidia/cu13"
```

The C++ route needs NVRTC, available in the existing PyTorch CUDA 13 wheel
dependencies here. Set `STAGE0_NVRTC=/absolute/path/to/libnvrtc.so` if needed.
The LLVM route needs NVIDIA libdevice independently of Triton; `CUDA_PATH`
must contain `nvvm/libdevice/libdevice.10.bc`.
For a Cargo `--release` build, set `STAGE0_CARGO_PROFILE=release` when running
the Python tools (the default is `debug`).

## Isolated Python without Triton

Use a PyTorch installation matching the Python ABI. This setup shares its
installed files through local symlinks while excluding Triton and `.pth` files;
it never edits that installation. The local verified installation was:

```bash
uv venv stage0/.venv --python 3.14t
python stage0/isolate_python.py /home/imi/Projects/sd-scripts/.venv-wsl/lib/python3.14t/site-packages
stage0/.venv/bin/python stage0/verify.py cpp
CUDA_PATH="$PWD/stage0/cuda13-toolkit/nvidia/cu13" stage0/.venv/bin/python stage0/verify.py llvm
stage0/.venv/bin/python stage0/measure.py
```

`verify.py` requires `find_spec("triton") is None`, rejects attempted imports,
checks loaded modules and GIL state, invokes the real Rust compiler, runs the
real kernels and writes `artifacts/<route>/results.json`. Do not set `PYTHON_GIL`.
`measure.py` needs both prepared PTX artifacts.

## Use

With `stage0` on `PYTHONPATH`, after generating the selected artifacts:

```python
import torch
from bridge import Kernel

k = Kernel("stage0/artifacts/cpp", device=0)  # prebuilt PTX, load once
x = torch.randn(257, device="cuda:0", dtype=torch.float32)
y = torch.empty_like(x)
k(x, y)
```

Use contiguous tensors with matching shape/device and disjoint memory. Storage
offsets are supported because `data_ptr()` addresses the first logical element.
This raw forward launcher does not participate in autograd.

The wrapper fetches the current stream at each call and records tensor storage
use with PyTorch. For inputs produced elsewhere, the caller must first establish
the normal PyTorch stream dependency. Metadata is uploaded once per N and its
readiness is checked with a nonblocking event query; incomplete uploads are
waited on the current stream outside capture. This upload is setup data,
not a copy of x or y. No CubeCL allocations or device-server launches occur.

For CUDA Graphs, call `k.prepare_capture(N)` outside capture, then warm k on the
intended capture stream. That preparation waits only for the metadata upload
event; launches never synchronize. Completion is cached per immutable metadata
allocation, so no CUDA event query is needed during capture.
Keep k, input, output and any captured consumer buffers alive through replay.
No compilation, module loading or metadata creation occurs during capture.
Native modules intentionally remain loaded for process lifetime.

## Checks

```bash
cc -std=c11 -Wall -Wextra -Werror -fPIC -shared stage0/launcher.c -ldl -o stage0/libstage0_launcher.so
cargo fmt --manifest-path stage0/Cargo.toml --check
cargo clippy --manifest-path stage0/Cargo.toml --locked --features llvm -- -D warnings
```

Generated `.so` files, Cargo build outputs, local environments and toolkit files
are ignored. `Cargo.lock`, generated IR/source/PTX/manifests and measured results
are retained as inspectable evidence. No upstream source patch is required.
