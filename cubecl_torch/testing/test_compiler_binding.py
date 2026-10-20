"""Descriptor compilation without Inductor and binding parity with Stage 5 (task 3.2)."""

import json
import os
import subprocess
import sys

import pytest
import torch

from cubecl_torch.compiler.build import compile_descriptor
from cubecl_torch.compiler.plan import PhysicalCandidate
from cubecl_torch.runtime.cache import ArtifactCache
from cubecl_torch.runtime.launch import LoadedKernel
from cubecl_torch.testing.fixtures import REPO, descriptor, tensors_for

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HAND_AUTHORED = """
import json, sys, torch
from cubecl_torch.compiler.build import compile_descriptor
from cubecl_torch.compiler.plan import PhysicalCandidate
from cubecl_torch.runtime.cache import ArtifactCache
from cubecl_torch.runtime.launch import LoadedKernel
from cubecl_torch.testing.fixtures import sin_plus_one_descriptor
compiled = compile_descriptor(
    sin_plus_one_descriptor(), PhysicalCandidate("scalar_pointwise_baseline").to_dict(), 0,
    cache=ArtifactCache(sys.argv[1]),
)
kernel = LoadedKernel(compiled.artifact)
for n in (1, 1000, 70001):
    x = torch.randn(n, device="cuda")
    y = torch.empty_like(x)
    kernel((n,), x, y)
    torch.testing.assert_close(y, torch.sin(x) + 1, rtol=0, atol=1e-6)
print(json.dumps(sorted(sys.modules)))
"""


def test_hand_authored_descriptor_without_inductor(tmp_path):
    out = subprocess.run(
        [sys.executable, "-c", HAND_AUTHORED, str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    loaded = json.loads(out.splitlines()[-1])
    forbidden = (
        "torch._inductor",
        "torch._dynamo",
        "cubecl_torch.inductor",
        "cubecl_torch.ops",
    )
    assert not [m for m in loaded if m.startswith(forbidden)]


def plan(strategy, *args, **fields):
    return PhysicalCandidate(strategy, *args, **fields).to_dict()


CASES = [
    # (descriptor, plan, newly seen scalars)
    ("pointwise_strided", plan("scalar_pointwise_baseline"), (3, 4099)),
    ("pointwise_strided", plan("scalar_pointwise", (64, 1, 1)), (2, 777)),
    ("vector", plan("vector_pointwise", (256, 1, 1), 8), (8200,)),
    ("reduction", plan("shared_tree_reduction_baseline"), (37, 129)),
    ("reduction", plan("shared_tree_compensated"), (5, 5000)),
    ("column", plan("bounded_column_pointwise"), (257, 64)),
    (
        "final9",
        plan(
            "independent_planes_final9",
            (32, 4, 1),
            outputs_per_workgroup=4,
            plane_width=32,
        ),
        (13,),
    ),
]


@pytest.fixture(scope="module")
def stage5_artifact():
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage5 import artifact

    return artifact


@pytest.mark.parametrize(
    ("name", "physical", "scalars"),
    CASES,
    ids=[c[0] + "-" + c[1]["strategy"] for c in CASES],
)
def test_binding_equals_stage5(tmp_path, stage5_artifact, name, physical, scalars):
    semantic = descriptor(name)
    tensors = tensors_for(semantic, scalars)
    reference = stage5_artifact.CompiledKernel(semantic, 0, physical).kernel
    stage5_key, stage5_n, stage5_pointers = reference._validate(scalars, tensors)
    compiled = compile_descriptor(semantic, physical, 0, cache=ArtifactCache(tmp_path))
    kernel = LoadedKernel(compiled.artifact)
    key, n, pointers, grid = kernel._validate(scalars, tensors)
    assert (key, n, pointers) == (stage5_key, stage5_n, stage5_pointers)
    assert grid == reference.geometry(stage5_key[1][1:])
    assert compiled.artifact.launch.block == reference.plan.workgroup
    assert (
        compiled.artifact.launch.shared_bytes
        == reference.manifest["dynamic_shared_bytes"]
    )
    # Same launch, bitwise-equal outputs.
    expected = tensors[-1].clone()
    reference(scalars, *tensors[:-1], expected)
    kernel(scalars, *tensors)
    torch.cuda.synchronize()
    assert torch.equal(tensors[-1].view(torch.uint8), expected.view(torch.uint8))
