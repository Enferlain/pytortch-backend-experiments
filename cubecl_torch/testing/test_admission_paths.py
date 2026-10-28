"""Admission paths differ only in their kernel kind's limits (``admission.PATHS``).

Ops and index values admitted on one path are admitted on all; fusion checks
each node against its own path, plus the reduction kernel's storage limits
when the fused group contains a reduction.
"""

import pytest
import torch

from cubecl_torch.testing.fixtures import run_python

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HELPERS = """
from cubecl_torch.runtime import trace

def compiled(fn, *args):
    torch._dynamo.reset()
    start = len(trace.EVENTS)
    with trace.scoped():
        out = torch.compile(fn, backend=ci.compile_backend, dynamic=False)(*args)
    events = list(trace.EVENTS)[start:]
    fusion = sorted(
        {
            e["reason"]
            for e in events
            if e["event"] == "capability"
            and e.get("boundary") == "fusion"
            and e["decision"] == "reject"
        }
    )
    kernels = sum(e["event"] == "kernel" for e in events)
    return out, fn(*args), kernels, fusion
"""


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("admission-cache")


def run(code, cache_dir):
    return run_python(HELPERS + code, cache_dir)


def test_indirect_kernels_admit_rsqrt_and_float_index_values(cache_dir):
    """Both were refused before fusion on the indirect path only."""
    result = run(
        """
w = torch.rand(100, 16, device="cuda") + 0.5
idx = torch.randint(0, 100, (64,), device="cuda")
cases = {
    "rsqrt": lambda w, idx: w[idx].rsqrt(),
    "float index": lambda w, idx: w[idx] * torch.arange(16, device=w.device).float(),
}
report = {}
for name, fn in cases.items():
    out, ref, kernels, _ = compiled(fn, w, idx)
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=0)
    report[name] = kernels
print(json.dumps(report))
""",
        cache_dir,
    )
    assert result == {"rsqrt": 1, "float index": 1}


def test_fusion_checks_each_node_against_its_own_path(cache_dir):
    """Reduction kernels have no int64 storage: pointwise int64 nodes fuse (the
    cat's two slices of one buffer become one two-output kernel), a reduction
    group refuses them by name."""
    result = run(
        """
x = torch.arange(-500, 500, device="cuda", dtype=torch.int64)
f = torch.randn(64, 128, device="cuda")
def concatenated(x):
    y = x * 3
    return torch.cat([y, y + 1]) * 2

cases = {
    "pointwise": (concatenated, x),
    "reduction": (lambda f: ((f > 0).long() * 3, f.sum(dim=1)), f),
}
report = {}
for name, (fn, arg) in cases.items():
    out, ref, _, fusion = compiled(fn, arg)
    torch.testing.assert_close(out, ref)
    report[name] = fusion
print(json.dumps(report))
""",
        cache_dir,
    )
    assert result["pointwise"] == []
    assert result["reduction"] == [
        "tensor storage dtype torch.int64 unsupported in reduction kernels"
    ]


def test_direct_check_bounds_without_requested_bounds_is_admitted(cache_dir):
    """Indices built in the node (arange) become direct indexing plus a
    check_bounds; with no bound left to check it emits nothing, as in Triton
    (bead 4z8.27). A bound Inductor cannot prove stays rejected."""
    result = run(
        """
x = torch.randn(64, device="cuda")
mask = torch.ones(2, 16, device="cuda", dtype=torch.int64)
mask[1, 12:] = 0

def gather(x):
    return x[torch.arange(32, device=x.device) * 2] + 1

def causal_padding_mask(mask):
    # transformers' index-based sdpa mask: attention_mask[batch_idx, kv_idx].
    b, n = mask.shape
    bi = torch.arange(b, device=mask.device)[:, None, None, None]
    qi = torch.arange(n, device=mask.device)[None, None, :, None]
    ki = torch.arange(n, device=mask.device)[None, None, None, :]
    allowed = (ki <= qi) & mask[bi, ki].bool()
    return torch.where(allowed, 0.0, torch.finfo(torch.float32).min)

report = {}
for name, (fn, arg) in {"gather": (gather, x), "mask": (causal_padding_mask, mask)}.items():
    out, ref, kernels, _ = compiled(fn, arg)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    report[name] = kernels

def past_end(x):
    return x[torch.arange(70, device=x.device)] + 1

torch._dynamo.reset()
try:
    torch.compile(past_end, backend=ci.compile_backend, dynamic=False)(x)
    report["past end"] = "compiled"
except Exception as e:
    report["past end"] = str(e)
print(json.dumps(report))
""",
        cache_dir,
    )
    assert result["gather"] == result["mask"] == 1
    assert "check_bounds with requested bounds unsupported" in result["past end"]
