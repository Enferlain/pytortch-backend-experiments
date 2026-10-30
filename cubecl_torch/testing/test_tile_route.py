"""Tile kernels through the production route (cubecl-tile-stage-1 group 2).

The plan is pinned: no kernel is offered tile plans until they qualify.
"""

import json
import os
import subprocess
import sys
import tomllib

import pytest
import torch

from cubecl_torch.compiler.plan import PhysicalCandidate, catalog, tile_catalog
from cubecl_torch.testing.fixtures import REPO
from cubecl_torch.testing.test_wide_workgroups import geometry_error

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

RMSNORM = """
import sys
from cubecl_torch.runtime import guard
guard.install()
import cubecl_torch.inductor as ci
ci.install()
import torch
from cubecl_torch.compiler.plan import tile_catalog
from cubecl_torch.inductor import legality, tuning

plan = tile_catalog()[0]
candidates = tuning.candidates
def forced(semantic, *args, **kwargs):
    accepted, rejected = candidates(semantic, *args, **kwargs)
    return ([plan], rejected) if semantic["schema"] == 3 else (accepted, rejected)
tuning.candidates = forced
numerical = legality.numerical_reason
legality.numerical_reason = lambda p, *a: None if p == plan else numerical(p, *a)

def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

g = torch.Generator().manual_seed(0)
x = torch.randn(int(sys.argv[1]), int(sys.argv[2]), generator=g).cuda()
w = torch.randn(int(sys.argv[2]), generator=g).cuda()
y = torch.compile(rms, backend=ci.compile_backend, dynamic=False)(x, w)
torch.cuda.synchronize()
reference = rms(x.double(), w.double())
print((y.double() - reference).abs().div(reference.abs().clamp_min(1e-30)).max().item())
"""


def pinned_rmsnorm(cache, rows, cols):
    """The worst relative error against float64, and the tile artifact's directory."""
    run = subprocess.run(
        [sys.executable, "-c", RMSNORM, str(rows), str(cols)],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO),
            "CUBECL_TORCH_CACHE_DIR": str(cache),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    plans = (cache / "artifacts").glob("*/physical-plan.json")
    tile = [p.parent for p in plans if "tile_row_sum" in p.read_text()]
    assert len(tile) == 1
    return float(run.stdout.strip().splitlines()[-1]), tile[0]


@pytest.mark.parametrize(
    ("shape", "shared"),
    [((32, 16, 1), 64), ((32, 16, 1), 6144)],
)
def test_launcher_accepts_the_tile_block(shape, shared):
    assert geometry_error(shape, shared) == "null kernel handle"


@pytest.mark.parametrize("shape", [(16, 32, 1), (32, 32, 1), (32, 8, 2)])
def test_launcher_refuses_other_wide_blocks(shape):
    assert geometry_error(shape, 64) == "unsupported launch geometry"


def test_no_kernel_is_offered_a_tile_plan_yet():
    assert not {p.strategy for p in tile_catalog()} & {p.strategy for p in catalog()}


def test_tile_plans_round_trip_but_other_blocks_do_not():
    for plan in tile_catalog():
        assert PhysicalCandidate.from_dict(plan.to_dict()) == plan
    other = {**tile_catalog()[0].to_dict(), "workgroup": [512, 1, 1]}
    with pytest.raises(ValueError, match="unsupported physical plan"):
        PhysicalCandidate.from_dict(other)


def test_compiler_links_no_device_runtime():
    with (REPO / "cubecl_torch" / "native" / "Cargo.lock").open("rb") as lock:
        packages = {p["name"]: p for p in tomllib.load(lock)["package"]}
    seen, stack = set(), ["cubecl-stage0"]
    while stack:
        name = stack.pop()
        if name not in seen and name in packages:
            seen.add(name)
            stack += [d.split()[0] for d in packages[name].get("dependencies", ())]
    assert "cubecl-tile" in seen
    assert "cubecl-cuda" not in seen


@pytest.mark.parametrize(("rows", "cols"), [(2048, 8192), (300, 4096)])
def test_rmsnorm_is_correct_with_the_existing_abi(tmp_path, rows, cols):
    worst, artifact = pinned_rmsnorm(tmp_path, rows, cols)
    # Every plane read its own steps: one plane's share misses by a factor of 4.
    assert worst < 1e-6

    source = (artifact / "kernel.cu").read_text()
    assert "__grid_constant__" not in source
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["grid_constants"] is False
    assert manifest["block"] == [32, 16, 1]
    assert manifest["arguments"][-1] == "info device pointer"
    assert manifest["metadata"]["scalar_count"] == 1
    assert manifest["dynamic_shared_bytes"] == 64
