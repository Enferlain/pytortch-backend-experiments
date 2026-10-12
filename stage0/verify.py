"""Run in a fresh process. All checks use the real CubeCL-generated kernel."""
import importlib.abc
import importlib.util
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

class NoTriton(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            self.attempts.append(fullname)
        return None

blocker = NoTriton()
sys.meta_path.insert(0, blocker)
assert importlib.util.find_spec("triton") is None, "use the isolated environment without Triton"
import_attempts = []
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "triton" or name.startswith("triton."):
        import_attempts.append(name)
        raise ImportError("Triton deliberately unavailable in Stage 0")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
assert not os.environ.get("PYTHON_GIL"), "run without forced GIL configuration"
gil = lambda: getattr(sys, "_is_gil_enabled", lambda: None)()
states = {"before_torch": gil()}
import torch
states["after_torch"] = gil()
from bridge import Kernel, ROOT, native_path
states["after_bridge_import"] = gil()

route = sys.argv[1] if len(sys.argv) > 1 else "cpp"
artifact_dir = ROOT / "artifacts" / route
subprocess.run([str(native_path("cubecl-stage0")), str(artifact_dir), route], check=True)
states["after_cubecl_compile"] = gil()
k = Kernel(artifact_dir, compile_ptx=route == "cpp")
states["after_native_load"] = gil()
device = k.device
torch.manual_seed(7)
results = {"route": route, "python": sys.version, "torch": torch.__version__,
           "torch_runtime_commit": torch.version.git_version, "cuda": torch.version.cuda,
           "device": torch.cuda.get_device_name(device), "load_ms": k.load_ms, "checks": []}

def check(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6, equal_nan=True)

sizes = [0, 1, 255, 256, 257, 1_048_583]
for n in sizes:
    x = torch.randn(n, device=device)
    # Sentinels around an output view detect tail writes beyond logical N.
    backing = torch.full((n + 2,), -1234.0, device=device)
    y = backing[1:-1]
    k(x, y)
    check(y, torch.sin(x) + 1)
    check(backing[[0, n + 1]], torch.full((2,), -1234.0, device=device))
results["checks"].append({"tails_and_output_canaries": sizes})
x = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 0.0,
                  1e-30, -1e-30, -3.14159265, 3.14159265, -1000, 1000, 1e10, -1e10], device=device)
check(k(x, torch.empty_like(x)), torch.sin(x) + 1)
results["checks"].append("NaN infinities signed zero and ordinary/extreme values")

# A single module is reused across stream changes. Delay each producer to make
# accidentally launching on stream zero observable. No device synchronization.
streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
stream_ids = []
pending = []
for j, s in enumerate(streams):
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        torch.cuda._sleep(5_000_000)
        producer = torch.arange(1_048_583, device=device, dtype=torch.float32) * (0.001 + j * 0.0001)
        out = torch.empty_like(producer)
        k(producer, out)
        consumer = out * 3 - 2
        expected = (torch.sin(producer) + 1) * 3 - 2
        pending.append((consumer, expected))
        stream_ids.append(torch.cuda.current_stream(device).cuda_stream)
    torch.cuda.current_stream(device).wait_stream(s)
for actual, expected in pending:
    check(actual, expected)
assert len(set(stream_ids)) == 2 and all(stream_ids)
results["checks"].append({"non_default_producer_kernel_consumer_streams": stream_ids})

# Allocate on default, consume on delayed side stream, release both x and y,
# then force allocations/writes on their original allocation stream.
n = 2_097_159
k.prepare(n)
s = streams[0]
reuse_counts = []
early_reuse_counts = []
for trial in range(5):
    x = torch.randn(n, device=device)
    expected = torch.sin(x) + 1
    y = torch.empty_like(x)
    old_ptrs = {x.data_ptr(), y.data_ptr()}
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        torch.cuda._sleep(30_000_000)
        k(x, y)
        retained_consumer = y.clone()
        completed = torch.cuda.Event()
        completed.record(s)
    del x, y
    pressure = []
    early_reuses = 0
    for j in range(16):
        tensor = torch.full((n,), 123.0 + j, device=device)
        if tensor.data_ptr() in old_ptrs and not completed.query():
            early_reuses += 1
        pressure.append(tensor)
    reuse_counts.append(sum(t.data_ptr() in old_ptrs for t in pressure))
    early_reuse_counts.append(early_reuses)
    assert early_reuses == 0, "recorded tensor storage recycled before side work completed"
    torch.cuda.current_stream(device).wait_stream(s)
    check(retained_consumer, expected)
    del pressure
results["checks"].append({"allocator_pressure_trials": 5, "recycled_addresses": reuse_counts,
                         "reuses_observed_before_completion": early_reuse_counts,
                         "pressure_bytes_per_trial": n * 4 * 16})

# Warm metadata and module before capture; replay with changed values while
# input/output/metadata/module remain live.
n = 257
x = torch.randn(n, device=device)
y = torch.empty_like(x)
k.prepare_capture(n)
s.wait_stream(torch.cuda.current_stream(device))
with torch.cuda.stream(s):
    for _ in range(3):
        k(x, y)
torch.cuda.current_stream(device).wait_stream(s)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=s):
    k(x, y)
    captured_consumer = y * 2
for value in [0.2, -2.0, 9.0]:
    x.fill_(value)
    graph.replay()
    check(captured_consumer, (torch.sin(x) + 1) * 2)
results["checks"].append("CUDA Graph capture and 3 replays with changed input")

# Validation fails before any driver launch.
invalid = [(torch.ones(3, device=device, dtype=torch.float64), torch.empty(3, device=device)),
           (torch.ones(3, device=device), torch.empty(4, device=device)),
           (torch.ones((2, 3), device=device).t(), torch.empty((3, 2), device=device))]
for a, b in invalid:
    try:
        k(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid input accepted")
try:
    k(x, x)
except ValueError:
    pass
else:
    raise AssertionError("alias accepted")
results["checks"].append("dtype shape layout and alias rejections")

# Bounded barrier test: independent tensor ownership and thread-local streams,
# all compilation/cache preparation finished before threads start.
n = 65_537
k.prepare(n)
barrier = threading.Barrier(2)
errors = []
def worker(j):
    try:
        with torch.cuda.device(device):
            local = torch.cuda.Stream(device=device)
            barrier.wait(timeout=30)
            with torch.cuda.stream(local):
                for _ in range(10):
                    a = torch.randn(n, device=device) + j
                    b = torch.empty_like(a)
                    k(a, b)
                    check(b, torch.sin(a) + 1)
    except BaseException as error:
        errors.append(repr(error))
threads = [threading.Thread(target=worker, args=(j,)) for j in range(2)]
for t in threads: t.start()
for t in threads: t.join(timeout=60)
assert not any(t.is_alive() for t in threads) and not errors, errors
results["checks"].append("two threads barrier-started 20 launches on independent streams")

# GPU event timing, separated from compilation/load and from Python call cost.
n = 1_048_583
x = torch.randn(n, device=device)
y = torch.empty_like(x)
k(x, y)
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
begin = time.perf_counter()
for _ in range(100): k(x, y)
host_us = (time.perf_counter() - begin) * 1e4
end.record()
end.synchronize()  # Measurement endpoint only, stream event, not device-wide.
results["timing"] = {"n": n, "iterations": 100, "event_interval_us_per_launch": start.elapsed_time(end) * 10,
                     "host_us_per_launch": host_us, "note": "event interval includes host submission gaps; no throughput claim"}
states["after_all_checks"] = gil()
results["gil_states"] = states
results["triton_availability_probes"] = blocker.attempts
results["triton_import_attempts"] = import_attempts
results["triton_loaded"] = [name for name in sys.modules if name == "triton" or name.startswith("triton.")]
assert not results["triton_loaded"] and not import_attempts
if "free-threading" in sys.version:
    assert all(state is False for state in states.values()), states
(artifact_dir / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
