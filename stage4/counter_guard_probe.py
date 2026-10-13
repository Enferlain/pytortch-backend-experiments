"""Reject oversized live RNG lowering without a multi-gigabyte allocation."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from stage1 import evidence

evidence.guard()
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from stage1.overlay import install

install()
from stage4.backend import compile_backend, TRACE
from stage3 import runtime as bridge

torch.cuda.init()
state = torch.cuda.get_rng_state().clone()
with FakeTensorMode():
    fake = torch.empty(1 << 31, device="cuda")
    try:
        torch.compile(
            lambda x: torch.rand_like(x),
            backend=compile_backend,
            fullgraph=True,
            dynamic=True,
        )(fake)
    except Exception as error:
        assert "unsupported RNG physical extent" in str(error), str(error)
    else:
        raise AssertionError("oversized counter domain accepted")
assert bridge.COMPILER_INVOCATIONS == 0
assert torch.equal(state, torch.cuda.get_rng_state())
rejection = next(
    e for e in TRACE if e.get("boundary") == "pre_fusion" and e["decision"] == "reject"
)
evidence.state("after_counter_guard")
(ROOT / "artifacts/counter-guard.json").write_text(
    json.dumps(
        {
            "n": 1 << 31,
            "tensor": "FakeTensor CUDA; no physical data allocation",
            "rejection": rejection,
            "compiles": 0,
            "rng_state_unchanged": True,
            "independence": evidence.inventory(),
        },
        indent=2,
    )
)
print("Live counter domain guard PASS")
