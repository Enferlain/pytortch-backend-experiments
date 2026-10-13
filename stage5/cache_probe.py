"""Fresh-process artifact reload and empirical selection reuse evidence."""

import json
from pathlib import Path
import sys
from stage1 import evidence

evidence.guard()
import torch
from stage5.fixtures import captured
from stage5 import artifact, runtime

ROOT = Path(__file__).resolve().parent
c = runtime.compile_kernel(captured("add", "float32"), 0)
x = torch.randn(1001, device="cuda")
y = torch.empty_like(x)
c((1001,), x, y)
torch.testing.assert_close(y, x + 1)
evidence.state("after_cache_probe")
result = {
    "compiler_invocations": artifact.COMPILER_INVOCATIONS,
    "events": runtime.EVENTS,
    "independence": evidence.inventory(),
}
(ROOT / "artifacts" / ("cache-probe-" + sys.argv[1] + ".json")).write_text(
    json.dumps(result, indent=2)
)
if sys.argv[1] == "reload":
    assert artifact.COMPILER_INVOCATIONS == 0
    assert any(e["event"] == "tuning_cache_hit" for e in runtime.EVENTS)
    assert all(e["cache_hit"] for e in runtime.EVENTS if e["event"] == "artifact")
print("Cache probe", sys.argv[1], "PASS", flush=True)
