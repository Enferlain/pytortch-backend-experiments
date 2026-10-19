"""First-use implementation check using the existing independent CPU model.

Linux x86_64 glibc fenv is saved/restored on this thread, including MXCSR.
No process-wide PyTorch thread/default-dtype/denormal policy is changed.
"""

from contextlib import contextmanager
import ctypes as C
from functools import cache
import platform


@cache
def fenv():
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or platform.libc_ver()[0] != "glibc"
    ):
        return None
    try:
        lib = C.CDLL("libm.so.6")
        for name in ("feholdexcept", "fesetenv"):
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = [C.c_void_p], C.c_int
        lib.fesetround.argtypes, lib.fesetround.restype = [C.c_int], C.c_int
    except (OSError, AttributeError):
        return None
    return lib


@contextmanager
def environment():
    import torch

    # Missing platform/API support excludes the candidate. A failure of the
    # declared-supported FP environment is a hard validation error, not a
    # performance fallback or a reason to skip implementation verification.
    lib = fenv()
    if lib is None:
        raise RuntimeError("reference512 CPU oracle requires Linux x86_64 glibc fenv")
    # glibc x86_64 fenv_t is32 bytes; overallocate with long-double alignment.
    saved = (C.c_longdouble * 32)()
    if lib.feholdexcept(C.byref(saved)):
        raise RuntimeError("could not save CPU floating-point environment")
    try:
        if lib.fesetround(0) or not torch.set_flush_denormal(False):
            raise RuntimeError("could not establish CPU model FP environment")
        with torch.inference_mode(), torch.device("cpu"):
            yield
    finally:
        if lib.fesetenv(C.byref(saved)):
            raise RuntimeError("could not restore CPU floating-point environment")


def partials(x):
    import torch
    from stage5.reduction_topology_model import model

    with environment():
        host = x.detach().cpu()
        # Per-row FP32 additions have at most9*2048=18432 elements, below
        # ATen TensorIterator's32768 grain. They execute on the scoped thread;
        # unused FP64 truth diagnostics do not participate in the oracle.
        return torch.cat([model(row.unsqueeze(0))["partials"] for row in host])


def check(actual, expected):
    import torch

    with environment():
        actual = actual.detach().cpu()
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(torch.isnan(actual), torch.isnan(expected))
        finite = ~torch.isnan(expected)
        assert torch.equal(
            actual.view(torch.int32)[finite], expected.view(torch.int32)[finite]
        ), "reference512 disagrees with frozen arithmetic model"
