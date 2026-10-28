"""Record mode (``rejections="all"``) lists every rejection reason; normal mode stops at the first."""

import pytest
import torch

from cubecl_torch.testing.fixtures import run_python

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HELPERS = """
import torch._functorch.config
from cubecl_torch.inductor import kernels
from cubecl_torch.runtime import trace

# As in the scoreboard: lower backward inside the backend call.
torch._functorch.config.force_non_lazy_backward_lowering = True

def outcome(fn, *args, rejections):
    errors = []

    def backend(gm, inputs):
        try:
            return ci.compile_backend(gm, inputs, rejections=rejections)
        except Exception as e:
            errors.append(e)
            return gm.forward

    torch._dynamo.reset()
    out = torch.compile(fn, backend=backend)(*args)
    first = out[0] if isinstance(out, tuple) else out
    if first.requires_grad:
        first.sum().backward()
    return out, errors

# erf is rejected in the forward graph; layer_norm compiles both ways.
LN_WEIGHT = torch.randn(32, device="cuda", requires_grad=True)
LN_BIAS = torch.randn(32, device="cuda", requires_grad=True)
def forward_and_backward(x):
    return torch.erf(x) + torch.nn.functional.layer_norm(x, (32,), LN_WEIGHT, LN_BIAS)
"""


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("rejections-cache")


def run(code, cache_dir):
    return run_python(HELPERS + code, cache_dir)


def test_first_mode_rejects_on_the_first_reason(cache_dir):
    result = run(
        """
x = torch.randn(8, 32, device="cuda", requires_grad=True)
_, errors = outcome(forward_and_backward, x, rejections="first")
print(json.dumps([[type(e).__name__, str(e), hasattr(e, "reasons")] for e in errors]))
""",
        cache_dir,
    )
    [(kind, message, has_reasons)] = result
    assert kind == "InductorError" and not has_reasons
    assert "semantic operation erf unsupported" in message
    assert "direct inputs" not in message


def test_record_mode_lists_every_reason_forward_and_backward(cache_dir):
    result = run(
        """
x = torch.randn(64, 32, device="cuda", requires_grad=True)
with trace.scoped():
    _, errors = outcome(
        lambda x: (forward_and_backward(x).sum(), (x * 3).to(torch.float64)),
        x, rejections="all",
    )
    graphs = [e["is_backward"] for e in trace.EVENTS if e.get("event") == "capability_graph"]
print(json.dumps({"errors": [[type(e).__name__, getattr(e, "reasons", None)] for e in errors],
                  "graphs": graphs}))
""",
        cache_dir,
    )
    [(kind, reasons)] = result["errors"]
    assert kind == "UnsupportedCubeCL"
    prefix = "CubeCL unsupported before fusion: "
    assert {
        prefix + "semantic operation erf unsupported",
        prefix + "dtype for to_dtype unsupported: torch.float64",
        prefix + "one logical reduction dimension required",
        # Only in the backward graph (the float64 output's gradient), scanned
        # although the forward was rejected.
        prefix + "tensor storage dtype torch.float64 unsupported",
    } <= set(reasons)
    assert result["graphs"] == [False, True]


def test_record_mode_compiles_an_admitted_region(cache_dir):
    result = run(
        """
x = torch.randn(8, 33, device="cuda", requires_grad=True)
before = kernels.COMPILER_INVOCATIONS
out, errors = outcome(lambda x: torch.sqrt(x * x + 1), x, rejections="all")
grad = x.grad.clone()
x.grad = None
torch.sqrt(x * x + 1).sum().backward()
print(json.dumps({"errors": [str(e) for e in errors],
                  "compiled": kernels.COMPILER_INVOCATIONS > before,
                  "match": torch.allclose(out, torch.sqrt(x * x + 1))
                  and torch.allclose(grad, x.grad)}))
""",
        cache_dir,
    )
    assert result == {"errors": [], "compiled": True, "match": True}


def test_record_mode_stops_without_cuda_inputs(cache_dir):
    result = run(
        """
_, errors = outcome(lambda x: torch.erf(x) + 1, torch.randn(8), rejections="all")
print(json.dumps([getattr(e, "reasons", str(e)) for e in errors]))
""",
        cache_dir,
    )
    assert result == [
        {"CubeCL unsupported input contract: CUDA tensor inputs required": 1}
    ]
