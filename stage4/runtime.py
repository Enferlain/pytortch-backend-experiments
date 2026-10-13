"""Reuse existing descriptor compiler, metadata, current-stream/native pipeline."""

import json
from pathlib import Path
import torch
from stage3 import runtime as bridge

ROOT = Path(__file__).resolve().parent
EVENTS = []
CACHE = {}


class Kernel(bridge.Kernel):
    metadata_function = "stage4_metadata"
    storage_dtypes = {**bridge.Kernel.storage_dtypes, "int64": torch.int64}

    def __call__(self, scalars, *tensors):
        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError("Stage4 CUDA graph capture unsupported")
        return super().__call__(scalars, *tensors)

    def launch_extent(self, scalars, tensors):
        if self.semantic.get("effects", {}).get("mutation"):
            from stage2.indexing import evaluate

            return evaluate(self.semantic["numel"], scalars)
        extent = super().launch_extent(scalars, tensors)
        if self.semantic.get("effects", {}).get("rng") and extent > (1 << 31) - 1:
            raise ValueError(
                "unsupported RNG physical extent: signed-i32 logical counter domain"
            )
        return extent


def compile_kernel(semantic, device, plan):
    key = (
        json.dumps(semantic, sort_keys=True),
        device,
        json.dumps(plan, sort_keys=True),
    )
    with bridge.LOCK:
        if key not in CACHE:
            CACHE[key] = bridge.CompiledKernel(
                semantic, device, plan, root=ROOT, kernel_type=Kernel, events=EVENTS
            )
        return CACHE[key]
