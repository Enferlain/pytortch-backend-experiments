"""Proof op ``cubecl_torch::sin_plus_one``: ``y = sin(x) + 1`` for float32.

Built from a hand-authored schema-4 descriptor with a fixed scalar pointwise
plan, compiled by the descriptor compiler and launched by the runtime; no
Inductor involvement. Inputs of any shape are flattened to a contiguous view.
"""

import torch

from cubecl_torch.compiler.plan import PhysicalCandidate
from cubecl_torch.ops.library import DescriptorKernel, register
from cubecl_torch.runtime import rules


def descriptor():
    n = rules.scalar(0)
    buffer = {
        "dtype": "float32",
        "offset": rules.constant(0),
        "size": [n],
        "stride": [rules.constant(1)],
    }
    linear = {"op": "linear"}
    typed = {"dtype": "float32", "semantic_dtype": "float32"}
    return {
        "schema": 4,
        "index_dtype": "int64",
        "scalars": [{"dtype": "int64", "id": 0, "positive": True}],
        "numel": n,
        "effects": {"mutation": False, "rng": None},
        "buffers": [{**buffer, "access": "read"}, {**buffer, "access": "write"}],
        "ops": [
            {
                "op": "load",
                "id": "tmp0",
                "args": [],
                "buffer": 0,
                "index": linear,
                **typed,
            },
            {"op": "sin", "id": "tmp1", "args": ["tmp0"], **typed},
            {"op": "constant", "id": "tmp2", "args": [], "value": 1.0, **typed},
            {"op": "add", "id": "tmp3", "args": ["tmp1", "tmp2"], **typed},
            {"op": "store", "args": ["tmp3"], "buffer": 1, "index": linear, **typed},
        ],
    }


KERNEL = DescriptorKernel(
    descriptor(), PhysicalCandidate("scalar_pointwise_baseline").to_dict()
)


def _cuda(x):
    if x.dtype != torch.float32:
        raise TypeError(f"cubecl_torch::sin_plus_one supports float32, got {x.dtype}")
    flat = x.contiguous().view(-1)
    out = torch.empty_like(flat)
    if flat.numel():
        KERNEL.on(x.device)((flat.numel(),), flat, out)
    return out.view(x.shape)


def _fake(x):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


sin_plus_one = register("sin_plus_one", "(Tensor x) -> Tensor", cuda=_cuda, fake=_fake)
