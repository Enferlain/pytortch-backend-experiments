"""Producer-independent artifact launch (kernel-runtime spec, task 2.1)."""

import json
import subprocess
import sys

import pytest
import torch

from cubecl_torch.runtime.launch import LoadedKernel
from cubecl_torch.runtime.native import NATIVE
from cubecl_torch.testing.fixtures import REPO, STAGE0_CUDA, stage0_sin_plus_one

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_stage0_source_matches_audited_build(tmp_path):
    compiler = NATIVE / "target/release/cubecl-stage0"
    subprocess.run(
        [str(compiler), str(tmp_path), "cpp"], check=True, capture_output=True
    )
    assert (tmp_path / "kernel.cu").read_bytes() == STAGE0_CUDA.read_bytes()


def test_non_default_stream_ordering():
    kernel = LoadedKernel(stage0_sin_plus_one())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Producer, kernel and consumer all on the non-default current stream.
        x = torch.randn(4097, device="cuda").mul_(3)
        y = torch.empty_like(x)
        kernel((x.numel(),), x, y)
        consumer = y * 2
    stream.synchronize()
    torch.testing.assert_close(consumer, (torch.sin(x) + 1) * 2, rtol=0, atol=1e-6)
    assert kernel.launch_count == 1


def test_empty_extent_launches_nothing():
    kernel = LoadedKernel(stage0_sin_plus_one())
    x = torch.empty(0, device="cuda")
    kernel((0,), x, torch.empty_like(x))
    assert kernel.launch_count == 0


def test_contract_round_trip():
    artifact = stage0_sin_plus_one()
    again = type(artifact).from_contract(
        json.loads(artifact.contract_json()), artifact.ptx
    )
    assert again == artifact


LAYER_SCRIPT = """
import json, sys, torch
from cubecl_torch.runtime.launch import LoadedKernel
from cubecl_torch.testing.fixtures import stage0_sin_plus_one
x = torch.randn(1000, device="cuda")
y = torch.empty_like(x)
LoadedKernel(stage0_sin_plus_one())((x.numel(),), x, y)
torch.cuda.synchronize()
assert torch.allclose(y, torch.sin(x) + 1, rtol=0, atol=1e-6)
print(json.dumps(sorted(sys.modules)))
"""


def test_launch_loads_no_upper_layer():
    out = subprocess.run(
        [sys.executable, "-c", LAYER_SCRIPT],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    loaded = json.loads(out.splitlines()[-1])
    forbidden = (
        "torch._inductor",
        "torch._dynamo",
        "cubecl_torch.compiler",
        "cubecl_torch.inductor",
        "cubecl_torch.ops",
    )
    assert not [m for m in loaded if m.startswith(forbidden)]


@pytest.mark.parametrize("n", [0, 1, 255, 256, 257, 1_048_583])
def test_tails_and_output_canaries(n):
    kernel = LoadedKernel(stage0_sin_plus_one())
    x = torch.randn(n, device="cuda")
    # Sentinels around an output view detect tail writes beyond logical N.
    backing = torch.full((n + 2,), -1234.0, device="cuda")
    y = backing[1:-1]
    kernel((n,), x, y)
    torch.testing.assert_close(y, torch.sin(x) + 1, rtol=2e-6, atol=2e-6)
    assert backing[[0, n + 1]].tolist() == [-1234.0, -1234.0]


def test_special_values():
    kernel = LoadedKernel(stage0_sin_plus_one())
    special = [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 1e-30, -1e-30]
    ordinary = [-3.14159265, 3.14159265, -1000, 1000, 1e10, -1e10]
    x = torch.tensor([*special, *ordinary], device="cuda")
    y = torch.empty_like(x)
    kernel((x.numel(),), x, y)
    torch.testing.assert_close(
        y, torch.sin(x) + 1, rtol=2e-6, atol=2e-6, equal_nan=True
    )
