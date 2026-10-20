"""Backend readiness, separate from compilation: never invokes a compiler."""

import torch

from cubecl_torch.compiler import build, source
from cubecl_torch.runtime import native


class CubeCLBackendUnavailable(RuntimeError):
    """Missing compiler/runtime dependency, distinct from unsupported semantics."""


def check_available(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise CubeCLBackendUnavailable("CubeCL backend requires available PyTorch CUDA")
    for path in (
        build.compiler_path(),
        native.metadata_library_path(),
        native.launcher_path(),
    ):
        if not path.is_file():
            raise CubeCLBackendUnavailable(
                f"CubeCL native component missing: {path}; run cubecl_torch/native/build.sh"
            )
    try:
        source.check()
    except source.SourceMismatch as error:
        raise CubeCLBackendUnavailable(
            f"CubeCL requires the exact audited FP source stack: {error}"
        ) from error
