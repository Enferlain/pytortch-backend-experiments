# Local CubeCL floating-point hardening

See [FP_HARDENING_REPORT.md](../FP_HARDENING_REPORT.md) for findings, scope and evidence.
This fixes the pinned compiler used by Stage 0–2; it does not start Stage 3 or add
Inductor remainder/extrema/FastMath support.

```bash
bash stage2/setup.sh
cargo test --manifest-path stage0/Cargo.toml --locked --features llvm,hip-probe --test fp_rewrites
CUDA_PATH="$PWD/stage0/cuda13-toolkit/nvidia/cu13" stage0/.venv/bin/python fp_hardening/verify.py
stage0/.venv/bin/python fp_hardening/verify_inductor.py
stage0/.venv/bin/python fp_hardening/capture.py
```

Run Inductor suites sequentially: the existing `stage1.overlay.install()` rebuilds
one local overlay directory and is not a concurrent setup API.

`cubecl-fp.patch` is the complete diff from the audited CubeCL pin.
`cubecl-fp-upgrade.patch` upgrades exactly the original Stage 2 patch to the same
source. `patches.py` accepts only clean, original Stage 2, or exact hardened states,
and rejects other staged/untracked/source changes. The original
`stage2/cubecl-ieee.patch` remains available as historical baseline evidence.

The Rust `fp_probe` binary contains fixed, manual KernelBuilder/Scope fixtures for
otherwise unsupported operations. It uses the real CubeCL C++ or LLVM compiler.
Python reuses Stage 0 NVRTC/PTX loading and native current-stream enqueue after
validating the probe's measured three-buffer ABI. These compiler-boundary tests
do not advertise new `torch.compile` operation coverage. Runtime tensors and
metadata are owned by PyTorch. No emitted CUDA source is edited by the probe.

Implicit contraction is disabled: default InstCombine permission is false,
ordinary LLVM add/sub/multiply have no contract flags, and NVRTC retains the
existing `--fmad=false`. This does not forbid FMA inside an explicitly selected
device-library implementation of a transcendental or remainder operation.

Results, review disposition and nine representative generated kernels are
eligible for Git. Raw traces/logs, full per-case outputs, and unmodified upstream
source snapshots remain local and ignored.
