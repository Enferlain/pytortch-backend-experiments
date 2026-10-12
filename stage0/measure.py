"""Small route comparison: kernel batches in warmed CUDA Graphs."""
import json
from pathlib import Path
import statistics
import sys
import torch
from bridge import Kernel, ROOT

assert not any(m == "triton" or m.startswith("triton.") for m in sys.modules)
rows = []
for route in ("cpp", "llvm"):
    k = Kernel(ROOT / "artifacts" / route)
    x = torch.randn(1_048_583, device=k.device)
    y = torch.empty_like(x)
    s = torch.cuda.Stream(device=k.device)
    s.wait_stream(torch.cuda.current_stream(k.device))
    with torch.cuda.stream(s):
        for _ in range(3): k(x, y)
    torch.cuda.current_stream(k.device).wait_stream(s)
    k.prepare_capture(x.numel())
    g = torch.cuda.CUDAGraph()
    count = 128
    with torch.cuda.graph(g, stream=s):
        for _ in range(count): k(x, y)
    g.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(7):
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / count)
    torch.testing.assert_close(y, torch.sin(x) + 1, rtol=2e-6, atol=2e-6)
    rows.append({"route": route, "n": x.numel(), "kernels_per_graph": count,
                 "us_per_kernel_samples": samples, "median_us_per_kernel": statistics.median(samples)})
result = {"device": torch.cuda.get_device_name(k.device), "method": "7 event-timed replays, each 128 sequential warmed kernels; includes graph scheduling",
          "rows": rows, "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: None)()}
(ROOT / "artifacts" / "comparison.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
