"""Register kernel artifacts as ``torch.library`` custom ops.

Uses the plain ``torch.library.Library`` API (``torch.library.custom_op``
imports Dynamo). Eager calls touch neither Inductor nor Dynamo nor the overlay;
compiled graphs see each op as one opaque extern call.
"""

import threading

import torch

from cubecl_torch.compiler.build import compile_descriptor
from cubecl_torch.runtime.launch import LoadedKernel

NAMESPACE = "cubecl_torch"
_LIBRARY = None
_LOCK = threading.Lock()


def library():
    global _LIBRARY
    with _LOCK:
        if _LIBRARY is None:
            _LIBRARY = torch.library.Library(NAMESPACE, "DEF")
        return _LIBRARY


def register(name, schema, *, cuda, fake):
    """Define ``cubecl_torch::<name>`` with a CUDA implementation and a fake one."""
    lib = library()
    lib.define(f"{name}{schema}")
    lib.impl(name, cuda, "CUDA")
    torch.library.register_fake(f"{NAMESPACE}::{name}", fake, lib=lib)
    return getattr(getattr(torch.ops, NAMESPACE), name)


class DescriptorKernel:
    """A descriptor compiled with a fixed physical plan, loaded once per device."""

    def __init__(self, semantic, plan):
        self.semantic, self.plan = semantic, plan
        self._loaded = {}
        self._lock = threading.Lock()

    def on(self, device):
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        with self._lock:
            if index not in self._loaded:
                compiled = compile_descriptor(self.semantic, self.plan, index)
                self._loaded[index] = LoadedKernel(compiled.artifact, index)
            return self._loaded[index]
