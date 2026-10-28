"""Ported live availability and scheduler-boundary contracts.

Source: stage1/verify_boundaries.py, stage4/components.py (missing native ABI).
"""

import pytest
import torch

from cubecl_torch.testing.fixtures import run_python

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_missing_native_component_fails_availability(tmp_path):
    code = """
from cubecl_torch.inductor.availability import CubeCLBackendUnavailable, check_available
check_available(torch.device("cuda", 0))
outcomes = []
for name in ("CUBECL_TORCH_COMPILER", "CUBECL_TORCH_LAUNCHER", "CUBECL_TORCH_METADATA_LIB"):
    os.environ[name] = "/nonexistent/component"
    try:
        check_available(torch.device("cuda", 0))
        outcomes.append("accepted")
    except CubeCLBackendUnavailable as error:
        outcomes.append("/nonexistent/component" in str(error) and "build.sh" in str(error))
    del os.environ[name]
print(json.dumps(outcomes))
"""
    assert run_python(code, tmp_path) == [True, True, True]


def test_default_inductor_still_requires_triton(tmp_path):
    code = """
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.codegen.xpu.xpu_combined_scheduling import XPUCombinedScheduling
from torch._inductor.exc import TritonMissing
from torch._inductor.scheduler import BaseScheduling
# Register the package backend first (a supported compile), then use default Inductor.
torch.compile(lambda x: x + 1, backend=ci.compile_backend)(torch.randn(8, device="cuda"))
try:
    torch.compile(lambda x: torch.sin(x) + 1, backend="inductor", fullgraph=True)(torch.randn(257, device="cuda"))
    default = "accepted"
except Exception as error:
    default = isinstance(error, TritonMissing) or "TritonMissing" in str(error)
gates = {}
for name, backend, device in [
    ("triton_cpu", TritonScheduling(None), torch.device("cpu")),
    ("generic_cpu", BaseScheduling(None), torch.device("cpu")),
    ("xpu_combined", XPUCombinedScheduling(None), torch.device("xpu", 0)),
]:
    try:
        backend.check_backend_available(device)
        gates[name] = "unchanged"
    except Exception as error:
        gates[name] = type(error).__name__
print(json.dumps({"default": default, "gates": gates}))
"""
    result = run_python(code, tmp_path)
    assert result["default"] is True
    assert result["gates"] == {
        "triton_cpu": "unchanged",
        "generic_cpu": "unchanged",
        "xpu_combined": "TritonMissing",
    }


def test_two_output_fusion_decided_before_codegen(tmp_path):
    """The fusion decision is admission's, made before codegen; two outputs
    fuse since multi-output-kernels (Stage 5 vetoed them)."""
    code = """
from torch._inductor import config
from cubecl_torch.runtime import trace
class FusionObserved(RuntimeError):
    pass
def stop_after_fusion(nodes):
    raise FusionObserved("fusion veto observed before codegen")
def branching(x):
    a = torch.sin(x)
    return a + 1, a * 2
with trace.scoped():
    start = len(trace.EVENTS)
    try:
        with config.patch({"_post_fusion_custom_pass": stop_after_fusion}):
            torch.compile(branching, backend=ci.compile_backend, fullgraph=True, dynamic=False)(torch.randn(257, device="cuda"))
        observed = False
    except Exception as error:
        observed = "FusionObserved" in str(error)
    events = list(trace.EVENTS)[start:]
print(json.dumps({
    "observed": observed,
    "decided": any(e.get("boundary") == "fusion" and e.get("decision") == "accept" for e in events),
    # Stage 5 rejected exactly this two-output pair.
    "rejected": any(e.get("boundary") == "fusion" and e.get("decision") == "reject" for e in events),
    "kernels": sum(e["event"] == "kernel" for e in events),
}))
"""
    assert run_python(code, tmp_path) == {
        "observed": True,
        "decided": True,
        "rejected": False,
        "kernels": 0,
    }
