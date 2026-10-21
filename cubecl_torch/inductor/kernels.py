"""A compiled physical candidate: descriptor compiler output loaded on the runtime.

Keeps the Stage 5 ``artifact.CompiledKernel`` surface used by first-use tuning
(``.kernel``, ``.directory``, ``.entrypoint``, ``.cache_hit``, calling it), with
compilation in ``compiler/`` and launch in ``runtime/``. Vector and final plans
re-check their physical legality before every launch, as Stage 5 did.
"""

import threading
import time

import torch

from cubecl_torch.compiler.build import compile_descriptor
from cubecl_torch.compiler.final_plan import STRATEGY as FINAL9
from cubecl_torch.inductor import diagnostics
from cubecl_torch.runtime.launch import LoadedKernel

EVENTS = diagnostics.TraceLog()
_LOCK = threading.Lock()
# Process-wide count of artifact cache misses that ran the compiler.
COMPILER_INVOCATIONS = 0


def _launch_check(plan, semantic):
    if plan.vector_width == 1 and plan.strategy != FINAL9:
        return None
    kind = "vector" if plan.vector_width > 1 else "final"
    state = {}

    def check(scalars, tensors):
        from cubecl_torch.inductor.legality import reason

        # Legality reads only the i64 shape scalars; float values pass through.
        scalars = tuple(
            int(s) if d["dtype"] == "int64" else s
            for s, d in zip(scalars, semantic["scalars"], strict=True)
        )
        why = reason(plan, semantic, scalars, tensors, state["resources"])
        if why:
            raise ValueError(f"unsupported {kind} physical contract: " + why)

    return check, state


class CompiledKernel:
    def __init__(self, semantic, device, plan, cache=None):
        global COMPILER_INVOCATIONS
        started = time.perf_counter()
        compiled = compile_descriptor(semantic, plan, device, cache=cache)
        self.plan = compiled.plan
        self.semantic = semantic
        self.directory = compiled.directory
        self.cache_hit = compiled.cache_hit
        self.entrypoint = compiled.artifact.entry
        hook = _launch_check(self.plan, semantic)
        self.kernel = LoadedKernel(
            compiled.artifact, device, launch_check=hook[0] if hook else None
        )
        if hook:
            hook[1]["resources"] = self.kernel.resources
        if not self.cache_hit:
            with _LOCK:
                COMPILER_INVOCATIONS += 1
        self.load_compile_ms = (time.perf_counter() - started) * 1000
        EVENTS.append(
            {
                "event": "artifact",
                "entrypoint": self.entrypoint,
                "directory": str(self.directory),
                "cache_hit": self.cache_hit,
                "elapsed_ms": self.load_compile_ms,
                "attributes": self.kernel.attributes,
                "resources": self.kernel.resources,
            }
        )

    def __call__(self, scalars, *tensors):
        return self.kernel(scalars, *tensors)


def device_index(device):
    return device.index or 0 if isinstance(device, torch.device) else device
