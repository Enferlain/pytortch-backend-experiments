"""One gross timing sanity probe, including host enqueue overhead; no tuning."""

import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache")
from stage1 import evidence

evidence.guard()
import torch
from stage1.overlay import install

install()
from stage3.backend import compile_backend
from stage3 import runtime

x = torch.randn(128, 4099, device="cuda")


def f(x):
    return x.sum(-1)


c = torch.compile(f, backend=compile_backend, fullgraph=True, dynamic=True)
torch.testing.assert_close(c(x), f(x), rtol=2e-5, atol=2e-5)
for _ in range(5):
    c(x)
    f(x)
values = {}
for name, fn in [("CubeCL", lambda: c(x)), ("PyTorch_eager", lambda: f(x))]:
    samples = []
    for _ in range(15):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    values[name] = {"median_us": statistics.median(samples), "samples_us": samples}
values["scope"] = (
    "CUDA event interval around one Python invocation; includes host enqueue gaps. Not a pure-device benchmark or performance ranking."
)
values["shape"] = list(x.shape)
values["attributes"] = [k.kernel.attributes for k in runtime.CACHE.values()]
values["profiler_kernels"] = {}
for name, fn in [("CubeCL", lambda: c(x)), ("PyTorch_eager", lambda: f(x))]:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        fn()
    path = ROOT / "artifacts" / ("timing-" + name + "-profiler.json")
    p.export_chrome_trace(str(path))
    values["profiler_kernels"][name] = [
        {"name": e["name"], "duration_us": e["dur"]}
        for e in json.loads(path.read_text())["traceEvents"]
        if e.get("cat") == "kernel"
    ]
evidence.state("after_measurement")
values["independence"] = evidence.inventory()
(ROOT / "artifacts/timing.json").write_text(json.dumps(values, indent=2))
print(json.dumps(values, indent=2))
