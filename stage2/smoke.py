import sys, os, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(__file__).parent / "cache")
from stage1 import evidence

evidence.guard()
import torch
from stage1.overlay import install

install()
from stage2.backend import compile_backend, TRACE

try:
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(257, device="cuda", dtype=dtype)
        y = torch.compile(
            lambda x: torch.sin(x) + 1,
            backend=compile_backend,
            fullgraph=True,
            dynamic=True,
        )(x)
        torch.testing.assert_close(
            y,
            torch.sin(x) + 1,
            rtol=2e-6 if dtype == torch.float32 else 0.01,
            atol=2e-6 if dtype == torch.float32 else 0.01,
            equal_nan=True,
        )
        print(dtype, "pass")
finally:
    Path("stage2/artifacts/smoke-replay.json").write_text(json.dumps(TRACE, indent=2))
