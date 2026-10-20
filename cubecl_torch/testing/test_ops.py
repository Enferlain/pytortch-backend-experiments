"""Custom-op front end: eager use, mixed compiled graph, free threading (7.1-7.3)."""

import pytest
import torch

from cubecl_torch.testing.fixtures import run_python

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

EAGER_PRELUDE = """
import json, sys
from cubecl_torch.runtime import guard
guard.install()
import torch
"""


def test_eager_on_non_default_stream_without_inductor(tmp_path):
    code = """
import cubecl_torch.ops.sin_plus_one
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    x = torch.randn(3, 1001, device="cuda")
    y = torch.ops.cubecl_torch.sin_plus_one(x)
    z = y * 2
stream.synchronize()
from cubecl_torch.inductor import overlay
print(json.dumps({
    "equal": torch.allclose(z, (torch.sin(x) + 1) * 2, rtol=0, atol=1e-6),
    "shape": list(y.shape),
    "loaded": [m for m in sys.modules if m.startswith(("torch._inductor", "torch._dynamo"))],
    "overlay": overlay.installed(),
}))
"""
    result = run_python(code, tmp_path, prelude=EAGER_PRELUDE)
    assert result == {"equal": True, "shape": [3, 1001], "loaded": [], "overlay": False}


def test_mixed_graph(tmp_path):
    code = """
import cubecl_torch.ops.sin_plus_one
from torch._dynamo.utils import counters
from torch._inductor.utils import run_and_get_code
from cubecl_torch.inductor import tuning

def f(x):
    return torch.cos(torch.ops.cubecl_torch.sin_plus_one(x * 2)) + 1

compiled = torch.compile(f, backend=ci.compile_backend, fullgraph=True)
x = torch.randn(4097, device="cuda")
y, sources = run_and_get_code(compiled, x)
wrapper = "\\n".join(sources)
print(json.dumps({
    "equal": torch.equal(y, f(x)),
    "extern_calls": wrapper.count("torch.ops.cubecl_torch.sin_plus_one.default("),
    "generated": len(tuning.CACHE),
    "graph_breaks": sum(counters["graph_break"].values()),
}))
"""
    result = run_python(code, tmp_path)
    assert result == {
        "equal": True,
        "extern_calls": 1,
        "generated": 2,
        "graph_breaks": 0,
    }


def test_concurrent_threads(tmp_path):
    code = """
import threading
import cubecl_torch.ops.sin_plus_one
torch.ops.cubecl_torch.sin_plus_one(torch.ones(1, device="cuda"))  # compile once
errors = []

def work(i):
    try:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for n in (17, 4096, 70001):
                x = torch.randn(n, device="cuda") + i
                y = torch.ops.cubecl_torch.sin_plus_one(x)
                stream.synchronize()
                if not torch.allclose(y, torch.sin(x) + 1, rtol=0, atol=1e-6):
                    errors.append((i, n))
    except Exception as error:
        errors.append(repr(error))

threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
guard.state("after_threads")
print(json.dumps({"errors": errors, "gil": guard.inventory()["gil_states"]}))
"""
    result = run_python(code, tmp_path, prelude=EAGER_PRELUDE)
    assert result["errors"] == []
    assert not any(result["gil"].values())
