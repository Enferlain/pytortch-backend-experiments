"""The runtime bridge on PyTorch's memory and streams (spec: runtime-bridge).

Ordering, lifetime, allocation and capture through ``runtime/bridge.py``, with
the bridge's own test kernel (``test_add_one``) and CubeK's row sum. Nothing
here synchronizes the host between the work it checks.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

from cubecl_torch.runtime import bridge, native
from cubecl_torch.testing import launch_cost
from cubecl_torch.testing.fixtures import REPO

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

N = 1 << 22  # Large enough that an unordered launch would be caught mid-flight.


def assert_row_sums(out, a):
    """Within a summation-order-independent bound of the exact row sums: 64
    float32 ulps of the sum of magnitudes (accuracy against Triton is the
    numerical band's to judge, not this test's)."""
    exact = a.double().sum(1)
    bound = 64 * torch.finfo(torch.float32).eps * a.double().abs().sum(1)
    assert ((out.double() - exact).abs() <= bound).all()


def test_load_creates_no_context():
    """Loading the library touches no device: only a launch does."""
    script = f"""
import ctypes as C, json
C.CDLL({str(bridge.library_path())!r})
cuda = C.CDLL("libcuda.so.1")
assert cuda.cuInit(0) == 0
dev, flags, active = C.c_int(), C.c_uint(), C.c_int()
assert cuda.cuDeviceGet(C.byref(dev), 0) == 0
assert cuda.cuDevicePrimaryCtxGetState(dev, C.byref(flags), C.byref(active)) == 0
print(json.dumps(active.value))
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout) == 0


def test_one_nvrtc_and_pytorchs():
    """The bridge compiles with the NVRTC PyTorch ships, the one the
    descriptor compiler uses, and no other is loaded."""
    script = f"""
import json, sys, torch
sys.path.insert(0, {str(REPO)!r})
from cubecl_torch.runtime import bridge, toolkit
x = torch.ones(8, device="cuda"); y = torch.empty_like(x)
bridge.test_add_one(x, y); torch.cuda.synchronize()
maps = {{line.split()[-1] for line in open("/proc/self/maps") if "libnvrtc.so" in line}}
print(json.dumps([sorted(maps), toolkit.library_path()]))
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    maps, expected = json.loads(out.stdout)
    assert maps == [expected]


def test_external_stream_refused():
    x = torch.ones(8, device="cuda")
    y = torch.empty_like(x)
    raw = torch.cuda.Stream()
    external = torch.cuda.ExternalStream(raw.cuda_stream)
    with (
        torch.cuda.stream(external),
        pytest.raises(bridge.BridgeError, match="external"),
    ):
        bridge.test_add_one(x, y)


@pytest.mark.parametrize("side", [False, True], ids=["default", "side"])
def test_same_stream_producer_and_consumer(side):
    """PyTorch writes, the bridge reads and writes, PyTorch reads, on one
    stream with no synchronization: every iteration sees the kernel's result."""
    stream = torch.cuda.Stream() if side else torch.cuda.current_stream()
    x = torch.empty(N, device="cuda")
    y = torch.empty(N, device="cuda")
    results = []
    with torch.cuda.stream(stream):
        for i in range(16):
            x.fill_(float(i))
            bridge.test_add_one(x, y)
            results.append(y.mul(2).sum())
    torch.cuda.synchronize()
    assert [r.item() for r in results] == [2.0 * (i + 1) * N for i in range(16)]


def test_cross_stream_use():
    """Produced on one stream, consumed by the bridge on another after the
    caller makes it wait: the kernel reads the produced values."""
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    x = torch.empty(N, device="cuda")
    y = torch.empty(N, device="cuda")
    with torch.cuda.stream(producer):
        x.copy_(torch.arange(N, device="cuda", dtype=torch.float32))
    consumer.wait_stream(producer)
    with torch.cuda.stream(consumer):
        bridge.test_add_one(x, y)
        total = y.sum(dtype=torch.float64)
    torch.cuda.synchronize()
    assert total.item() == N * (N - 1) / 2 + N


@pytest.mark.parametrize("recorder", ["bridge", "caller"])
def test_freed_while_queued(recorder):
    """An input allocated on the default stream is used by a launch on a side
    stream, and the caller drops it right after: the default stream must not
    reuse its memory before the side stream has run the kernel. Without a
    record on the side stream, the next default-stream allocation of its size
    takes its block at once. The bridge records it when asked (opt-in);
    by default, as with PyTorch's own ops, the caller's ``record_stream``
    suffices."""
    side = torch.cuda.Stream()
    outputs = []
    for i in range(16):
        x = torch.full((N,), float(i), device="cuda")
        y = torch.empty(N, device="cuda")
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            torch.cuda._sleep(2_000_000)  # Keep the side stream behind the host.
            bridge.test_add_one(x, y, record_stream=recorder == "bridge")
        if recorder == "caller":
            x.record_stream(side)
        del x
        # Same size, same stream as x was allocated on: its block if free.
        torch.full((N,), -7.0, device="cuda")
        outputs.append((i, y))
    torch.cuda.synchronize()
    assert all(torch.equal(y, torch.full_like(y, i + 1.0)) for i, y in outputs)


def test_record_stream_default_from_environment():
    """``CUBECL_TORCH_BRIDGE_RECORD_STREAM=1`` at load makes recording the
    default; without it, launches do not record."""
    script = "from cubecl_torch.runtime import bridge; print(bridge.RECORD_STREAM)"

    def default(env):
        out = subprocess.run(
            [sys.executable, "-c", script],
            env={**env, "PYTHONPATH": str(REPO)},
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()

    env = {
        k: v for k, v in os.environ.items() if k != "CUBECL_TORCH_BRIDGE_RECORD_STREAM"
    }
    assert default(env) == "False"
    assert default({**env, "CUBECL_TORCH_BRIDGE_RECORD_STREAM": "1"}) == "True"


@pytest.mark.parametrize("parallel_output", [False, True])
@pytest.mark.parametrize("routine", sorted(bridge.ROUTINES))
def test_sum_rows_routines(routine, parallel_output):
    """Every CubeK routine, with and without parallel output vectorization."""
    a = torch.randn(300, 4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(300, device="cuda")
    bridge.sum_rows(a, out, routine, parallel_output=parallel_output)
    assert_row_sums(out, a)


def test_sum_rows_refuses_misaligned_tensors():
    """CubeK vectorizes by shape and strides alone: an address off its widest
    vector would fault on the device, so the bridge refuses it."""
    a = torch.randn(64 * 1024 + 1, device="cuda")[1:].view(64, 1024)
    with pytest.raises(bridge.BridgeError, match="aligned"):
        bridge.sum_rows(a, torch.empty(64, device="cuda"), "unit")
    b = torch.randn(64, 1024, device="cuda")
    with pytest.raises(bridge.BridgeError, match="aligned"):
        bridge.sum_rows(b, torch.empty(65, device="cuda")[1:], "unit")


def test_bridge_allocations_are_pytorchs():
    """Memory the bridge needs for a new shape (the unit routine's launch
    metadata) comes from PyTorch's caching allocator on the launch's stream."""
    device = torch.cuda.current_device()

    def allocations():
        return torch.cuda.memory_stats(device)["allocation.all.allocated"]

    a = torch.randn(301, 4093, device="cuda")
    out = torch.empty(301, device="cuda")
    before = allocations()
    bridge.sum_rows(a, out, "unit")
    assert allocations() > before
    assert_row_sums(out, a)


@pytest.mark.parametrize(
    "seen_before", [False, True], ids=["first_in_capture", "eager_first"]
)
def test_capture_and_replay(seen_before):
    """Bridged launches capture into a PyTorch graph and replay against the
    captured storage: for a shape first seen in the capture and one used
    eagerly on the capture stream before; with eager use of the same shape on
    that stream after the capture, before the first replay; and with a second
    graph of the same shape replayed on its own."""
    rows = 257 if seen_before else 263
    a = torch.randn(rows, 4099, device="cuda")
    out = torch.empty(rows, device="cuda")
    x = torch.randn(N, device="cuda")
    y = torch.empty_like(x)
    capture = torch.cuda.Stream()
    capture.wait_stream(torch.cuda.current_stream())
    if seen_before:
        with torch.cuda.stream(capture):
            bridge.sum_rows(a, out, "unit")
        capture.synchronize()

    def record():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture):
            bridge.sum_rows(a, out, "unit")
            bridge.test_add_one(x, y)
        return graph

    first = record()
    with torch.cuda.stream(capture):
        eager = torch.empty(rows, device="cuda")
        bridge.sum_rows(a, eager, "unit")
    capture.synchronize()
    assert_row_sums(eager, a)
    second = record()
    for graph in (first, second, first):
        a.copy_(torch.randn_like(a))
        x.copy_(torch.randn_like(x))
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert_row_sums(out, a)
        assert torch.equal(y, x + 1)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_each_device():
    """A tensor's device index names the same GPU to CubeCL, and a launch on
    any device leaves the thread's current device as it was, as a PyTorch
    operator does."""
    current = torch.cuda.current_device()
    for index in range(torch.cuda.device_count()):
        device = torch.device("cuda", index)
        x = torch.randn(4097, device=device)
        y = torch.empty_like(x)
        bridge.test_add_one(x, y)
        assert torch.cuda.current_device() == current
        assert torch.equal(y, x + 1)


def test_bridge_is_built_from_the_audited_state():
    assert bridge.library_path().is_file()
    assert native.patch_stack(native.PATCHES)[-1].name == "bridge.patch"


def test_concurrent_threads():
    """Several threads issue bridged launches at once, each on its own stream,
    with no interpreter lock: every output is right, and no launch is lost or
    misordered within its stream."""
    import threading

    threads, rounds, n = 4, 64, 1 << 18
    assert not sys._is_gil_enabled()
    errors = []
    start = threading.Barrier(threads)

    def work(k):
        stream = torch.cuda.Stream()
        x = torch.empty(n, device="cuda")
        y = torch.empty(n, device="cuda")
        totals = []
        try:
            with torch.cuda.stream(stream):
                start.wait()
                for i in range(rounds):
                    x.fill_(float(k * 1000 + i))
                    bridge.test_add_one(x, y)
                    totals.append(y.sum(dtype=torch.float64))
            stream.synchronize()
            expected = [float(k * 1000 + i + 1) * n for i in range(rounds)]
            if [t.item() for t in totals] != expected:
                errors.append(f"thread {k}: wrong or misordered results")
        except Exception as error:  # noqa: BLE001 - reported to the main thread.
            errors.append(f"thread {k}: {error!r}")

    pool = [threading.Thread(target=work, args=(k,)) for k in range(threads)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()
    assert not errors, errors


@pytest.mark.skipif(shutil.which("perf") is None, reason="needs perf")
def test_launch_host_cost_against_triton():
    """A warm bridged launch retires no more user instructions than Triton's
    raw launch with the same arguments (``results/launch-cost-instructions.json``,
    counted by ``launch_cost.py --instructions`` in Triton's environment). The
    counts vary between processes by about 15%, as time does, so both sides are
    the fewest of several processes; time is recorded in
    ``results/torch-runtime-bridge.md``."""
    reference = json.loads(
        (
            REPO / "cubecl_torch/testing/results/launch-cost-instructions.json"
        ).read_text()
    )
    triton = min(reference["instructions_per_launch"]["triton"])
    bridged = min(launch_cost.instructions("bridge") for _ in range(3))
    assert bridged <= triton, (bridged, triton)
