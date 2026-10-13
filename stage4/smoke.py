"""Executed smoke fixtures during development; acceptance is in verify.py."""

import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from stage1 import evidence

evidence.guard()
import torch
from stage1.overlay import install

install()
from stage4.backend import compile_backend, TRACE
from stage4.runtime import EVENTS

mode = sys.argv[1] if len(sys.argv) > 1 else "gather"
x = torch.arange(257, device="cuda", dtype=torch.float32) / 10 + 1
idx = torch.tensor([0, 2, 2, 256], dtype=torch.int64, device="cuda")
if mode == "gather":
    fn = lambda x, i: torch.gather(x, 0, i)
    args = (x, idx)
elif mode == "index":
    fn = lambda x, i: x[i]
    args = (x, torch.tensor([0, -1, -257, 2], device="cuda"))
elif mode == "atomic":
    fn = lambda x, i: torch.zeros(257, device=x.device).scatter_add(
        0, i, x[: i.numel()]
    )
    args = (x, idx)
elif mode == "embedding":
    fn = lambda x, i: torch.nn.functional.embedding(i, x)
    args = (torch.randn(257, 7, device="cuda", requires_grad=True), idx)
else:
    fn = lambda x: torch.nn.functional.dropout(x, 0.25, True)
    args = (x.requires_grad_(),)
try:
    f = torch.compile(fn, backend=compile_backend, fullgraph=True, dynamic=True)
    y = f(*args)
    if mode not in ("dropout",):
        torch.testing.assert_close(y, fn(*args))
    if args[0].requires_grad:
        upstream = torch.randn_like(y)
        y.backward(upstream)
        if mode == "embedding":
            xr = args[0].detach().clone().requires_grad_()
            fn(xr, args[1]).backward(upstream)
            torch.testing.assert_close(args[0].grad, xr.grad)
        else:
            mask = y.detach() != 0
            expected = upstream * mask / 0.75
            torch.testing.assert_close(args[0].grad, expected, rtol=2e-6, atol=2e-6)
    print(mode, "PASS", tuple(y.shape))
finally:
    (ROOT / "artifacts/smoke-trace.json").write_text(
        json.dumps(TRACE, indent=2, default=str)
    )
    (ROOT / "artifacts/smoke-events.json").write_text(
        json.dumps(EVENTS, indent=2, default=str)
    )
