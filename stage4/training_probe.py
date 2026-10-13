"""Real mixed training fixture; optimizer boundary remains explicit eager PyTorch."""

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
from stage3 import runtime as bridge


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(17, 7))
        self.linear = torch.nn.Linear(7, 9, bias=False)

    def forward(self, idx):
        x = self.linear(torch.nn.functional.embedding(idx, self.weight))
        x = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5)
        x = torch.nn.functional.dropout(x, 0.25, True)
        return x


model = Model().cuda()
idx = torch.tensor([0, 2, 2, 16, 1], device="cuda")
c = torch.compile(
    lambda idx: (model(idx) ** 2).flatten().mean(),
    backend=compile_backend,
    fullgraph=True,
    dynamic=True,
)
opt = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, foreach=False, fused=False
)
try:
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        loss = c(idx)
        loss.backward()
        opt.step()
        print("STEP", step, loss.item(), flush=True)
finally:
    (ROOT / "artifacts/training-probe-trace.json").write_text(
        json.dumps(TRACE, indent=2, default=str)
    )
    (ROOT / "artifacts/training-probe-events.json").write_text(
        json.dumps(EVENTS, indent=2, default=str)
    )
    print(
        "GIL",
        sys._is_gil_enabled(),
        "compiles",
        bridge.COMPILER_INVOCATIONS,
        "launches",
        bridge.KERNEL_LAUNCHES,
    )
