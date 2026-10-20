"""Contract-driven launch validation: one refusal per mismatch class (task 2.2)."""

import pytest
import torch

from cubecl_torch.runtime import rules
from cubecl_torch.runtime.artifact import TensorParam
from cubecl_torch.runtime.launch import LaunchContractError, LoadedKernel
from cubecl_torch.testing.fixtures import EXTENT_LIMIT, stage0_sin_plus_one

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def kernel():
    return LoadedKernel(stage0_sin_plus_one())


def refused(kind, kernel, scalars, *tensors):
    with pytest.raises(LaunchContractError, match=f"{kind} mismatch") as error:
        kernel(scalars, *tensors)
    assert error.value.kind == kind
    assert kernel.launch_count == 0


def test_device(kernel):
    x = torch.randn(64)
    refused("device", kernel, (64,), x, torch.empty(64, device="cuda"))
    if torch.cuda.device_count() > 1:
        x = torch.randn(64, device="cuda:1")
        refused("device", kernel, (64,), x, torch.empty(64, device="cuda"))


def test_dtype(kernel):
    x = torch.randn(64, device="cuda", dtype=torch.float16)
    refused("dtype", kernel, (64,), x, torch.empty(64, device="cuda"))


def test_strides(kernel):
    x = torch.randn(128, device="cuda")[::2]
    refused("layout", kernel, (64,), x, torch.empty(64, device="cuda"))


def test_offset():
    # Contract reads 8 elements past the pointer: a 64-element tensor is too short.
    artifact = stage0_sin_plus_one()
    shifted = TensorParam(
        "float32", (rules.scalar(0),), (rules.constant(1),), rules.constant(8)
    )
    kernel = LoadedKernel(
        artifact.__class__(
            **{**artifact.__dict__, "tensors": (shifted, artifact.tensors[1])}
        )
    )
    x = torch.randn(64, device="cuda")
    refused("layout", kernel, (64,), x, torch.empty(64, device="cuda"))


def test_extent_range(kernel):
    n = EXTENT_LIMIT + 1
    x = torch.empty(n, device="cuda")
    refused("extent", kernel, (n,), x, torch.empty_like(x))


def test_output_extent(kernel):
    x = torch.randn(64, device="cuda")
    # Scalars and the input agree; the output is shorter than the declared numel.
    with pytest.raises(LaunchContractError) as error:
        kernel((64,), x, torch.empty(32, device="cuda"))
    assert error.value.kind == "extent"


def test_alias(kernel):
    x = torch.randn(64, device="cuda")
    refused("alias", kernel, (64,), x, x)
