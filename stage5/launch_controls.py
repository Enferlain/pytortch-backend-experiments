"""Warm-cache negative contracts, scoped tracing and concurrent stream calls."""

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import copy
import ctypes as C
import gc
import json
import os
from pathlib import Path
import threading
import weakref

from stage1 import evidence

os.environ["CUBECL_STAGE5_TRACE"] = "0"  # This gate tests counter-only defaults.
evidence.guard()
import torch
from stage1.overlay import install

install()
from stage5 import artifact, runtime
from stage5.diagnostics import detailed_trace, tracing
from stage5.fixtures import captured
from stage5.plan import PhysicalCandidate

ROOT = Path(__file__).resolve().parent
native = C.CDLL(str(ROOT.parent / "stage0/libstage0_launcher.so"))
native.stage0_kernel_launches.restype = C.c_uint64
identity = artifact.compiler_identity(0)
rows = []
semantic = captured("add", "float32")
scalar = artifact.CompiledKernel(
    semantic, 0, PhysicalCandidate("scalar_pointwise_baseline").to_dict(), identity
)
vector = artifact.CompiledKernel(
    semantic,
    0,
    PhysicalCandidate("vector_pointwise", (128, 1, 1), 4).to_dict(),
    identity,
)


before = native.stage0_kernel_launches()
count = scalar.kernel.launch_count
scalar((0,), torch.empty(0, device="cuda"), torch.empty(0, device="cuda"))
assert scalar.kernel.launch_count == count and native.stage0_kernel_launches() == before
rows.append({"case": "zero grid is a successful uncounted no-op", "passed": True})


def reject(fn, text):
    before = native.stage0_kernel_launches()
    try:
        fn()
    except ValueError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError("warm contract accepted: " + text)
    assert before == native.stage0_kernel_launches()


for n in (0, 1, 255, 256, 257, 4099, *range(270, 340)):
    x = torch.randn(n + 3, device="cuda")[3:]
    y = torch.empty_like(x)
    scalar((n,), x, y)
    torch.testing.assert_close(y, x + 1)
assert len(scalar.kernel._layouts) == 64
assert scalar.kernel._call_pointers is None
x = torch.randn(257, device="cuda")
y = torch.empty_like(x)
scalar((257,), x, y)
reject(lambda: scalar((257,), x[:256], y), "layout")
reject(lambda: scalar((257,), x, y.double()), "dtype")
reject(lambda: scalar((257,), x.cpu(), y), "device")
reject(lambda: scalar((257,), x, x), "disjoint")
reject(
    lambda: scalar((257,), torch.empty_strided((257,), (2,), device="cuda"), y),
    "layout",
)
reject(lambda: scalar((1 << 63,), x, y), "outside i64")
reject(lambda: scalar((), x, y), "argument count")
vector((257,), x, y)
reject(lambda: vector((257,), torch.randn(258, device="cuda")[1:], y), "misaligned")
assert vector.kernel._call_pointers is None
rows.append(
    {
        "case": "changing size, eviction, dtype/device/layout/alias/alignment",
        "rejected_before_native_launch": True,
        "layout_cache_bound": 64,
    }
)

# Narrow runtime-validation component: intentionally inconsistent descriptors
# must fail even if a caller bypasses the independent Rust schema validator.
bad = object.__new__(artifact.Kernel)
bad.device = scalar.kernel.device
bad.semantic = copy.deepcopy(captured("add", "float32"))
bad.semantic["numel"] = {"op": "constant", "value": 258}
bad._buffers = tuple(bad.semantic["buffers"])
bad._store_scopes = frozenset()
bad._layouts = OrderedDict()
reject(lambda: bad._validate((257,), (x, y)), "pointwise output extent mismatch")
from stage2.runtime import Kernel as Stage2Kernel

try:
    Stage2Kernel.launch_extent(bad, (257,), (x, y))
except ValueError as error:
    assert "pointwise output extent mismatch" in str(error)
else:
    raise AssertionError("reference extent guard accepted inconsistent output")
rows.append(
    {"case": "pointwise extent guard matches reference", "rejected_before_launch": True}
)

from stage5.backend import compile_backend

rank0 = torch.compile(
    lambda z: z + 1, backend=compile_backend, fullgraph=True, dynamic=True
)
z = torch.randn((), device="cuda")
before = native.stage0_kernel_launches()
torch.testing.assert_close(rank0(z), z + 1)
assert native.stage0_kernel_launches() > before
with runtime.LOCK:
    hits = sum(c.selection_hits for c in runtime.CACHE.values())
rank0(z)
with runtime.LOCK:
    assert sum(c.selection_hits for c in runtime.CACHE.values()) > hits
rows.append({"case": "real rank-zero capture and selection counter", "passed": True})

# Same warm scalar key and shape, different storage capacity. Descriptor index
# and declared storage offset both require one extra element beyond the view.
offset = copy.deepcopy(semantic)
offset["buffers"][0]["offset"] = {"op": "constant", "value": 1}
for op in offset["ops"]:
    if op["op"] == "load":
        op["index"] = {
            "op": "add",
            "args": [op["index"], {"op": "constant", "value": 1}],
        }
o = artifact.CompiledKernel(
    offset, 0, PhysicalCandidate("scalar_pointwise_baseline").to_dict(), identity
)
backing = torch.randn(258, device="cuda")
o((257,), backing[:257], y)
torch.testing.assert_close(y, backing[1:] + 1)
reject(lambda: o((257,), torch.randn(257, device="cuda"), y), "original storage")
rows.append(
    {"case": "warm key does not cache original storage capacity", "passed": True}
)

# Caller mutation of a descriptor after compilation cannot change its guards.
semantic["buffers"][0]["stride"][0] = {"op": "constant", "value": 2}
scalar((257,), x, y)
torch.testing.assert_close(y, x + 1)
rows.append({"case": "compiled descriptor is a snapshot", "passed": True})

before = native.stage0_kernel_launches()
count = scalar.kernel.launch_count
start = len(artifact.EVENTS)
with detailed_trace(False):
    scalar((257,), x, y)
assert len(artifact.EVENTS) == start
with detailed_trace(True):
    scalar((257,), x, y)
launch = artifact.EVENTS[-1]
assert launch["event"] == "launch" and launch["pointers"] == [
    x.data_ptr(),
    y.data_ptr(),
]
assert launch["stream"] == torch.cuda.current_stream().cuda_stream
assert (
    scalar.kernel.launch_count - count == native.stage0_kernel_launches() - before == 2
)
start = len(artifact.EVENTS)
try:
    with detailed_trace(True):
        with detailed_trace(False):
            assert not tracing()
        assert tracing()
        raise RuntimeError("test scope unwind")
except RuntimeError:
    pass
assert not tracing()
rows.append(
    {"case": "trace on/off, counters, nested exception restoration", "passed": True}
)

# Distinct workers share the loaded module and cache, never a mutable invocation
# pointer array. Each worker records only its own trace preference and stream.
barrier = threading.Barrier(2)
count = scalar.kernel.launch_count
before = native.stage0_kernel_launches()
start = len(artifact.EVENTS)


def worker(detail):
    stream = torch.cuda.Stream()
    with detailed_trace(detail), torch.cuda.stream(stream):
        barrier.wait()
        for n in (255, 256, 257, 259, 1001) * 4:
            inp = torch.randn(n, device="cuda")
            out = torch.empty_like(inp)
            scalar((n,), inp, out)
            torch.testing.assert_close(out, inp + 1)
        stream.synchronize()
        return stream.cuda_stream


with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(worker, detail) for detail in (False, True)]
    streams = [f.result(timeout=120) for f in futures]
assert (
    scalar.kernel.launch_count - count == native.stage0_kernel_launches() - before == 40
)
events = [e for e in artifact.EVENTS[start:] if e["event"] == "launch"]
assert len(events) == 20 and all(e["stream"] == streams[1] for e in events)
assert not tracing()
rows.append(
    {
        "case": "free-threaded shared kernel, independent streams and trace scopes",
        "calls": 40,
        "detailed_records": 20,
    }
)

stream = torch.cuda.Stream()
producer = torch.cuda.current_stream()
inp = torch.randn(4099, device="cuda")
stream.wait_stream(producer)
with torch.cuda.stream(stream):
    out = torch.empty_like(inp)
    scalar((4099,), inp, out)
    expected = inp + 1
    for _ in range(20):
        torch.empty_like(inp).fill_(13)
    torch.testing.assert_close(out, expected)
stream.synchronize()
ref = weakref.ref(inp)
del inp
gc.collect()
assert ref() is None and scalar.kernel._call_pointers is None
rows.append(
    {
        "case": "producer readiness, non-default stream, pressure, no tensor retention",
        "passed": True,
    }
)

with scalar.kernel.prepared((257,), x, y):
    scalar.kernel.batch(2)
assert scalar.kernel._prepared is None
torch.testing.assert_close(y, x + 1)
rows.append({"case": "prepared diagnostic remains scoped", "passed": True})
evidence.state("after_launch_controls")
(ROOT / "artifacts/launch").mkdir(parents=True, exist_ok=True)
(ROOT / "artifacts/launch/controls.json").write_text(
    json.dumps({"checks": rows, "independence": evidence.inventory()}, indent=2)
)
print("Stage5 warm launch controls PASS", flush=True)
