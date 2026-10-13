"""Run each invalid-device-index case in a disposable process/context."""

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from stage1 import evidence

evidence.guard()
import torch
from stage1.overlay import install

install()
from stage4.backend import compile_backend
from stage4.runtime import EVENTS

mode = sys.argv[1]
x = torch.arange(7, device="cuda", dtype=torch.float32)
idx = torch.tensor(
    [
        7
        if mode == "upper"
        else -1
        if mode == "negative_gather"
        else 1 << 32
        if mode == "large"
        else -(1 << 63)
        if mode == "min_i64"
        else -8
    ],
    device="cuda",
)
fn = (
    (lambda x, idx: x[idx])
    if mode == "negative_index"
    else (lambda x, idx: torch.gather(x, 0, idx))
)
f = torch.compile(fn, backend=compile_backend, fullgraph=True, dynamic=True)
try:
    f(x, idx)
    torch.cuda.synchronize()
except torch._dynamo.exc.BackendCompilerFailed:
    raise
except RuntimeError as error:
    print(
        json.dumps(
            {
                "mode": mode,
                "error": str(error),
                "launched": any(e["event"] == "launch" for e in EVENTS),
                "gil": sys._is_gil_enabled(),
                "independence": evidence.inventory(),
            }
        )
    )
    assert any(e["event"] == "launch" for e in EVENTS), (
        "compile rejection is not a device bounds check"
    )
    sys.exit(0)
raise AssertionError("invalid index accepted")
