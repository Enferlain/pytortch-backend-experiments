"""Contraction policy in real AOTAutograd replay, including lowp round barriers."""

import ctypes as C
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache")
from stage1 import evidence

evidence.guard()
import torch
from stage1.overlay import install

install()
from stage2.backend import compile_backend, TRACE
from stage2 import runtime


def f(x, y):
    return x * y - 1


result = {"checks": [], "unsupported": []}
for dtype, epsilon in [
    (torch.float32, 2**-23),
    (torch.float16, 2**-10),
    (torch.bfloat16, 2**-7),
]:
    x = torch.full((257,), 1 + epsilon, dtype=dtype, device="cuda", requires_grad=True)
    y = torch.full_like(x, 1 - epsilon, requires_grad=True)
    compiled = torch.compile(f, backend=compile_backend, fullgraph=True, dynamic=True)
    before = runtime.KERNEL_LAUNCHES
    out = compiled(x, y)
    assert runtime.KERNEL_LAUNCHES - before == 1
    torch.testing.assert_close(out, f(x, y), rtol=0, atol=0)
    assert torch.count_nonzero(out) == 0
    upstream = torch.randn_like(out)
    actual = torch.autograd.grad(out, (x, y), upstream)
    reference = torch.autograd.grad(f(x, y), (x, y), upstream)
    for got, expected in zip(actual, reference):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        compiled(x, y)
        torch.cuda.current_stream().synchronize()
    path = (
        ROOT
        / "artifacts"
        / (str(dtype).removeprefix("torch.") + "-contraction-profiler.json")
    )
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
    assert len(kernels) == 1 and kernels[0].startswith("cubecl2_"), kernels
    result["checks"].append(
        {
            "dtype": str(dtype),
            "epsilon": epsilon,
            "separately_rounded_result": "exact zero",
            "backward": "exact eager match with arbitrary upstream",
            "profiler_kernels": kernels,
        }
    )
negative_x = torch.randn(257, device="cuda", dtype=torch.float32)
negative_y = torch.ones_like(negative_x)
for name, operation in [("fmod", torch.fmod), ("remainder", torch.remainder)]:
    before = runtime.COMPILER_INVOCATIONS
    start = len(TRACE)
    try:
        torch.compile(operation, backend=compile_backend, fullgraph=True, dynamic=True)(
            negative_x, negative_y
        )
    except Exception as error:
        assert "unsupported" in str(error)
        assert any(
            e.get("boundary") == "pre_fusion" and e.get("decision") == "reject"
            for e in TRACE[start:]
        )
    else:
        raise AssertionError("audit probes accidentally broadened Inductor coverage")
    assert runtime.COMPILER_INVOCATIONS == before
    result["unsupported"].append(
        {"op": name, "boundary": "pre_fusion", "compiler_invocations": 0}
    )
evidence.state("after_audit_inductor")
result["independence"] = evidence.inventory()
result["compilers"] = runtime.COMPILER_INVOCATIONS
result["launches"] = runtime.KERNEL_LAUNCHES
native = C.CDLL(str(ROOT.parent / "stage0/libstage0_launcher.so"))
native.stage0_kernel_launches.restype = C.c_uint64
result["native_launches"] = native.stage0_kernel_launches()
assert result["native_launches"] == result["launches"]
result["replay"] = TRACE
(ROOT / "artifacts/inductor.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result["checks"], indent=2))
