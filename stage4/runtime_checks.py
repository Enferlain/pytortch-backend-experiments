"""Effectful current-stream/lifetime/determinism checks in a fresh process."""

import ctypes as C
import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache")
from stage1 import evidence

evidence.guard()
import torch
from stage1.overlay import install

install()
from stage4.backend import compile_backend, TRACE
from stage4 import runtime
from stage3 import runtime as bridge
from stage4.reference import uniform
from stage0.bridge import ROOT as STAGE0_ROOT

native = C.CDLL(str(STAGE0_ROOT / "libstage0_launcher.so"))
native.stage0_kernel_launches.restype = C.c_uint64
begin = native.stage0_kernel_launches()
torch._dynamo.config.recompile_limit = 128
results = []


def compile(fn):
    return torch.compile(fn, backend=compile_backend, fullgraph=True, dynamic=True)


gather = compile(lambda x, i: torch.gather(x, 0, i))
scatter = compile(lambda x, i, d: torch.zeros(d, device=x.device).scatter_add(0, i, x))
random = compile(lambda x: torch.rand_like(x))
drop = compile(lambda x: torch.nn.functional.dropout(x, 0.25, True))
# Prepare code/modules before lifetime/concurrency probes.
x = torch.arange(257, device="cuda", dtype=torch.float32)
i = torch.arange(257, device="cuda") % 17
gather(x, i)
scatter(x, i, 17)
random(x)
drop(x)
for trial in range(3):
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    base = torch.arange(1048583, device="cuda", dtype=torch.float32)
    index = (torch.arange(65539, device="cuda") % base.numel()).clone()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        torch.cuda._sleep(10000000)
        base.add_(trial + 1)
        index.add_(1)
    event = producer.record_event()
    consumer.wait_event(event)
    with torch.cuda.stream(consumer):
        before = len(runtime.EVENTS)
        y = gather(base, index)
        z = scatter(y, index % 257, 257)
        copied = z.clone()
        done = consumer.record_event()
        launches = [e for e in runtime.EVENTS[before:] if e["event"] == "launch"]
        assert launches and all(e["stream"] == consumer.cuda_stream for e in launches)
        assert any(
            base.data_ptr() in e["pointers"] and index.data_ptr() in e["pointers"]
            for e in launches
        )
        addresses = {base.data_ptr(), index.data_ptr(), y.data_ptr(), z.data_ptr()}
    del base, index, y, z
    pressure = [torch.full((1048583,), -1000.0, device="cuda") for _ in range(8)]
    assert not any(t.data_ptr() in addresses for t in pressure) or done.query()
    done.synchronize()
    indices = (torch.arange(65539, device="cuda") % 1048583) + 1
    expected = torch.zeros(257, device="cuda").scatter_add(
        0, indices % 257, (indices + trial + 1).float()
    )
    torch.testing.assert_close(copied, expected, rtol=2e-5, atol=2e-5)
    results.append(
        {
            "case": "delayed_producers_allocator_pressure",
            "trial": trial,
            "current_stream": consumer.cuda_stream,
            "original_data_index_pointers": True,
        }
    )
# Serial side-stream RNG calls follow PyTorch generator ordering/state exactly.
for stream in (torch.cuda.Stream(), torch.cuda.Stream()):
    with torch.cuda.stream(stream):
        torch.manual_seed(19)
        state = torch.cuda.get_rng_state().clone()
        y = random(x)
    stream.synchronize()
    after = torch.cuda.get_rng_state().clone()
    torch.cuda.set_rng_state(state)
    limits = torch.iinfo(torch.int64)
    seed = int(torch.randint(limits.min, limits.max, (1,), device="cuda").item())
    expected_state = torch.cuda.get_rng_state().clone()
    assert torch.equal(after, expected_state)
    expected = torch.tensor([uniform(seed, j) for j in range(257)], device="cuda")
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    results.append(
        {
            "case": "rng_nondefault_stream",
            "stream": stream.cuda_stream,
            "state_exact": True,
        }
    )
# Prepared wrappers on two independent streams; no concurrent compilation claim.
barrier = threading.Barrier(2)
errors = []
outputs = []
streams = [torch.cuda.Stream(), torch.cuda.Stream()]
state = torch.cuda.get_rng_state().clone()


def worker(stream):
    try:
        torch.cuda.set_device(0)
        with torch.cuda.stream(stream):
            barrier.wait()
            outputs.append(random(x))
    except Exception as e:
        errors.append(str(e))


threads = [threading.Thread(target=worker, args=(s,)) for s in streams]
for t in threads:
    t.start()
for t in threads:
    t.join()
for s in streams:
    s.synchronize()
assert not errors and len(outputs) == 2 and not torch.equal(outputs[0], outputs[1])
after = torch.cuda.get_rng_state()
torch.cuda.set_rng_state(state)
limits = torch.iinfo(torch.int64)
for _ in range(2):
    torch.randint(limits.min, limits.max, (1,), device="cuda")
assert torch.equal(after, torch.cuda.get_rng_state())
results.append(
    {
        "case": "prepared_two_thread_two_stream_rng",
        "launch_packing": "per-call",
        "state_exact": True,
        "compilation_concurrency": False,
    }
)
# IEEE classifications and zero signs with the same independently predicted mask.
values = torch.tensor(
    [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1.0, -1.0] * 37,
    device="cuda",
)
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    v = values.to(dtype)
    torch.manual_seed(13)
    state = torch.cuda.get_rng_state()
    y = drop(v)
    torch.cuda.set_rng_state(state)
    limits = torch.iinfo(torch.int64)
    seed = int(torch.randint(limits.min, limits.max, (1,), device="cuda").item())
    mask = (
        torch.tensor([uniform(seed, j) for j in range(v.numel())], device="cuda").to(
            dtype
        )
        > 0.25
    ).to(dtype)
    expected = v * mask * (1 / 0.75)
    torch.testing.assert_close(
        y,
        expected,
        rtol=0.016
        if dtype == torch.bfloat16
        else 0.002
        if dtype == torch.float16
        else 2e-6,
        atol=0.008
        if dtype == torch.bfloat16
        else 0.001
        if dtype == torch.float16
        else 2e-6,
        equal_nan=True,
    )
    assert torch.equal(torch.isnan(y), torch.isnan(expected)) and torch.equal(
        torch.isinf(y), torch.isinf(expected)
    )
    zero = expected == 0
    assert torch.equal(torch.signbit(y[zero]), torch.signbit(expected[zero]))
    results.append(
        {
            "case": "dropout_special_values",
            "dtype": str(dtype),
            "classification_zero_sign_exact": True,
        }
    )
# Deterministic mode: use the lowerer's natural external alternative, never atomics.
state = torch.cuda.get_rng_state()
torch.use_deterministic_algorithms(True)
try:
    toggle_start = len(TRACE)
    toggled = scatter(x, i, 17)
    toggled_reference = torch.zeros(17, device="cuda").scatter_add(0, i, x)
    torch.testing.assert_close(toggled, toggled_reference, rtol=0, atol=0)
    toggle_descriptors = [
        e["descriptor"] for e in TRACE[toggle_start:] if e["event"] == "kernel"
    ]
    toggle_extern = [
        e for e in TRACE[toggle_start:] if e.get("execution_path") == "extern"
    ]
    assert toggle_extern and not any(
        op["op"] == "atomic_add" for d in toggle_descriptors for op in d["ops"]
    ), "cached atomic graph must reselect deterministic alternative"
    results.append(
        {
            "case": "deterministic_toggle_existing_graph",
            "reselected_extern": toggle_extern,
            "no_generated_atomic": True,
        }
    )
    start = len(TRACE)
    events = len(runtime.EVENTS)
    deterministic = compile(lambda x, i: torch.zeros_like(x).scatter_add(0, i, x))
    src = torch.linspace(-1, 1, 257, device="cuda")
    idx = torch.arange(257, device="cuda") % 17
    y = deterministic(src, idx)
    ref = torch.zeros_like(src).scatter_add(0, idx, src)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)
    compiled_descriptors = [
        e["descriptor"] for e in TRACE[start:] if e["event"] == "kernel"
    ]
    assert not any(
        op["op"] == "atomic_add" for d in compiled_descriptors for op in d["ops"]
    )
    extern = [e for e in TRACE[start:] if e.get("execution_path") == "extern"]
    assert extern, "deterministic alternative must have an actual extern selection"
    results.append(
        {
            "case": "deterministic_scatter",
            "selection": extern,
            "no_generated_atomic": True,
        }
    )
finally:
    torch.use_deterministic_algorithms(False)
evidence.state("after_runtime_checks")
result = {
    "checks": results,
    "independence": evidence.inventory(),
    "compiles": bridge.COMPILER_INVOCATIONS,
    "launches": bridge.KERNEL_LAUNCHES,
    "native_launches": native.stage0_kernel_launches() - begin,
}
assert result["launches"] == result["native_launches"]
(ROOT / "artifacts/runtime-checks.json").write_text(json.dumps(result, indent=2))
print("Stage4 runtime checks PASS", len(results))
