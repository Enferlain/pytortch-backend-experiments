"""The CUDA toolkit kernels are compiled against: PyTorch's NVRTC and headers.

Shared by the descriptor compiler (``compiler/nvrtc.py``) and the runtime
bridge (``runtime/bridge.py``), so both compile with the same NVRTC library
and the same headers.
"""

import os
from functools import cache
from pathlib import Path

import torch


@cache
def library_path():
    """The NVRTC library: ``CUBECL_TORCH_NVRTC``/``STAGE0_NVRTC`` or PyTorch's own."""
    path = os.environ.get("CUBECL_TORCH_NVRTC") or os.environ.get("STAGE0_NVRTC")
    if path:
        return path
    if torch.version.cuda is None:
        raise RuntimeError("CubeCL compilation requires a CUDA-enabled PyTorch")
    major = torch.version.cuda.split(".")[0]
    candidates = sorted(
        {
            str(p.resolve())
            for p in Path(torch.__file__).parent.parent.glob(
                f"nvidia/**/libnvrtc.so.{major}"
            )
        }
    )
    if len(candidates) != 1:
        raise RuntimeError("select the exact NVRTC library with CUBECL_TORCH_NVRTC")
    return candidates[0]


def root():
    """The toolkit directory: its ``include`` holds the CUDA headers."""
    return Path(torch.__file__).parent.parent / "nvidia/cu13"


def include_dirs():
    return [root() / "include"]
