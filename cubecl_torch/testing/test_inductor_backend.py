"""Inductor front end: pointwise, reductions under both policies, rejection (4.2-4.4)."""

import pytest
import torch

from cubecl_torch.testing.fixtures import run_python

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("inductor-cache")


def test_pointwise_matches_eager(cache_dir):
    code = """
from cubecl_torch.inductor import tuning
f = torch.compile(lambda x: torch.sin(x) + 1, backend=ci.compile_backend)
x = torch.randn(4097, device="cuda")
y = f(x)
guard.state("after_run")
print(json.dumps({
    "equal": torch.equal(y, torch.sin(x) + 1),
    "kernels": len(tuning.CACHE),
    "gil": guard.inventory()["gil_states"],
    "triton": guard.inventory()["loaded_modules"],
}))
"""
    result = run_python(code, cache_dir)
    assert result["equal"] and result["kernels"] == 1
    assert not any(result["gil"].values()) and not result["triton"]


@pytest.mark.parametrize("tuning", ["first_use", "baseline"])
def test_reduction_policies(cache_dir, tuning):
    code = f"""
from functools import partial
from cubecl_torch.inductor import tuning
backend = partial(ci.compile_backend, tuning={tuning!r})
f = torch.compile(lambda x: (x * 2).sum(-1), backend=backend, dynamic=True)
errors = []
for shape in ((37, 129), (5, 5000), (300, 64)):
    x = torch.randn(*shape, device="cuda")
    errors.append((f(x) - (x * 2).sum(-1)).abs().max().item())
kernels = list(tuning.CACHE.values())
print(json.dumps({{
    "errors": errors,
    "policies": sorted({{k.policy for k in kernels}}),
    "reductions": sum(k.semantic["schema"] == 3 for k in kernels),
    "selected": [p.strategy for k in kernels for p in k.selections.values()],
}}))
"""
    result = run_python(code, cache_dir)
    assert max(result["errors"]) < 1e-3
    assert result["policies"] == [tuning] and result["reductions"] >= 1
    if tuning == "baseline":
        assert result["selected"] == []


def test_unsupported_operation_names_it(cache_dir):
    code = """
from cubecl_torch.inductor import UnsupportedCubeCL
f = torch.compile(lambda x: torch.relu(x) * 2, backend=ci.compile_backend)
try:
    f(torch.randn(64, device="cuda"))
    found = None
except Exception as error:
    found = error
    while found is not None and not isinstance(found, UnsupportedCubeCL):
        found = getattr(found, "inner_exception", None) or found.__cause__
print(json.dumps({"type": type(found).__name__, "message": str(found)}))
"""
    result = run_python(code, cache_dir)
    assert result["type"] == "UnsupportedCubeCL"
    assert "relu" in result["message"]
