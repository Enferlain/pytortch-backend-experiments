"""Opt-in bounded tracing; free-threaded launch without Triton (task 2.5)."""

import json
import os
import subprocess
import sys

import pytest
import torch

from cubecl_torch.runtime import trace
from cubecl_torch.runtime.launch import LoadedKernel
from cubecl_torch.testing.fixtures import REPO, stage0_sin_plus_one

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def launch_once():
    x = torch.randn(256, device="cuda")
    LoadedKernel(stage0_sin_plus_one())((x.numel(),), x, torch.empty_like(x))


@CUDA
def test_tracing_off_by_default():
    if os.environ.get("CUBECL_TORCH_TRACE") == "1":
        pytest.skip("tracing enabled by the environment")
    before = len(trace.EVENTS)
    launch_once()
    assert not trace.enabled() and len(trace.EVENTS) == before


@CUDA
def test_tracing_records_when_enabled():
    with trace.scoped():
        before = len(trace.EVENTS)
        launch_once()
        assert len(trace.EVENTS) == before + 1
        assert trace.EVENTS[-1]["entrypoint"] == "cubecl_sin_add"


def test_trace_is_bounded():
    with trace.scoped():
        for i in range(trace.LIMIT + 5):
            trace.record({"event": "probe", "i": i})
    assert len(trace.EVENTS) == trace.LIMIT


GIL_SCRIPT = """
import json, sys
from cubecl_torch.runtime import guard
guard.install()
import torch
guard.state("after_torch")
from cubecl_torch.runtime.launch import LoadedKernel
from cubecl_torch.testing.fixtures import stage0_sin_plus_one
kernel = LoadedKernel(stage0_sin_plus_one())
guard.state("after_load")
x = torch.randn(512, device="cuda")
kernel((x.numel(),), x, torch.empty_like(x))
torch.cuda.synchronize()
guard.state("after_launch")
print(json.dumps(guard.inventory()))
"""


@CUDA
@pytest.mark.skipif(not hasattr(sys, "_is_gil_enabled"), reason="needs 3.13t+")
def test_free_threaded_launch_without_triton():
    if sys._is_gil_enabled():
        pytest.skip("not a free-threaded interpreter")
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    out = subprocess.run(
        [sys.executable, "-c", GIL_SCRIPT],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    inventory = json.loads(out.splitlines()[-1])
    assert set(inventory["gil_states"]) == {
        "before_torch",
        "after_torch",
        "after_load",
        "after_launch",
    }
    assert not any(inventory["gil_states"].values())
    assert not inventory["loaded_modules"] and not inventory["import_attempts"]
