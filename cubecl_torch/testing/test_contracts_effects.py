"""Ported live effect/capability contracts through the package backend.

Sources: stage4/verify.py, stage4/runtime_checks.py, stage4/verify_bounds.py
(stage4/bounds_probe.py), stage4/counter_guard_probe.py, stage3/verify.py
(negatives). Each test runs in a fresh process with the overlay installed.
"""

import os

import pytest
import torch

from cubecl_torch.testing.fixtures import run_python

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HELPERS = """
import functools
from cubecl_torch.inductor import kernels, tuning
from cubecl_torch.testing.philox import uniform
torch._dynamo.config.recompile_limit = 128

def compiled(fn, policy="generated", dynamic=True):
    return torch.compile(fn, backend=functools.partial(ci.compile_backend, rng_policy=policy),
                         fullgraph=True, dynamic=dynamic)

def seeds_from_state(state, count):
    torch.cuda.set_rng_state(state)
    bounds = torch.iinfo(torch.int64)
    result = torch.randint(bounds.min, bounds.max, (count,), device="cuda", dtype=torch.int64)
    return list(map(int, result.cpu())), torch.cuda.get_rng_state().clone()

def reference_random(seed, n):
    return torch.tensor([uniform(seed, i) for i in range(n)], device="cuda", dtype=torch.float32)

def tol(dtype):
    return {torch.bfloat16: (0.016, 0.008), torch.float16: (0.002, 0.001)}.get(dtype, (2e-6, 2e-6))
"""


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("effects-cache")


def run(code, cache_dir, **kwargs):
    return run_python(HELPERS + code, cache_dir, **kwargs)


def test_uniform_rng_exact_state_and_reuse(cache_dir):
    code = """
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    f = compiled(lambda x: torch.rand_like(x))
    base = None
    for n in (0, 1, 255, 256, 257, 65539):
        x = torch.ones(n, device="cuda", dtype=dtype)
        torch.manual_seed(321)
        state = torch.cuda.get_rng_state().clone()
        cold = f(x)
        after = torch.cuda.get_rng_state().clone()
        before = kernels.COMPILER_INVOCATIONS
        torch.cuda.set_rng_state(state)
        warm = f(x)
        assert torch.equal(cold, warm) and torch.equal(after, torch.cuda.get_rng_state())
        assert kernels.COMPILER_INVOCATIONS == before
        if n:
            seeds, expected_state = seeds_from_state(state, 1)
            assert torch.equal(after, expected_state)
            m = min(n, 1024)
            assert torch.equal(cold[:m], reference_random(seeds[0], m).to(dtype))
            assert cold.dtype == dtype and bool(((cold >= 0) & (cold <= 1)).all())
            if n == 65539:
                assert abs(cold.float().mean().item() - 0.5) < 0.012
            torch.cuda.set_rng_state(after)
            if n > 1:
                assert not torch.equal(cold, f(x))
        else:
            assert torch.equal(state, after)
        if n == 255:
            base = before
        if n in (256, 257, 65539):
            assert before == base, "compatible dynamic artifact recompiled"
f = compiled(lambda x: torch.rand_like(x) + torch.rand_like(x))
x = torch.ones(257, device="cuda")
torch.manual_seed(22)
state = torch.cuda.get_rng_state()
y = f(x)
after = torch.cuda.get_rng_state()
seeds, expected = seeds_from_state(state, 2)
assert seeds[0] != seeds[1] and torch.equal(after, expected)
assert torch.equal(y, reference_random(seeds[0], 257) + reference_random(seeds[1], 257))
print(json.dumps({"ok": True}))
"""
    assert run(code, cache_dir) == {"ok": True}


def test_dropout_forward_backward(cache_dir):
    code = """
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    rtol, atol = tol(dtype)
    f = compiled(lambda x: torch.nn.functional.dropout(x, 0.25, True))
    for n in (1, 255, 256, 257, 65539):
        x = torch.ones(n, device="cuda", dtype=dtype, requires_grad=True)
        torch.manual_seed(45)
        state = torch.cuda.get_rng_state().clone()
        y = f(x)
        after = torch.cuda.get_rng_state()
        seeds, expected = seeds_from_state(state, 1)
        assert torch.equal(after, expected)
        keep = reference_random(seeds[0], n).to(dtype) > 0.25
        xr = x.detach().clone().requires_grad_()
        ref = (keep.to(dtype) * xr) / 0.75
        torch.testing.assert_close(y, ref, rtol=rtol, atol=atol)
        upstream = torch.linspace(-1, 1, n, device="cuda", dtype=dtype)
        y.backward(upstream)
        ref.backward(upstream)
        torch.testing.assert_close(x.grad, xr.grad, rtol=rtol, atol=atol)
        if n == 65539:
            assert abs(keep.float().mean().item() - 0.75) < 0.02
    for p, train in ((0.0, True), (1.0, True), (0.25, False)):
        x = torch.ones(257, device="cuda", dtype=dtype, requires_grad=True)
        state = torch.cuda.get_rng_state().clone()
        g = compiled(
            (lambda x: torch.nn.functional.dropout(x, 0.0, True)) if p == 0
            else (lambda x: torch.nn.functional.dropout(x, 1.0, True)) if p == 1
            else (lambda x: torch.nn.functional.dropout(x, 0.25, False))
        )
        y = g(x)
        torch.testing.assert_close(y, torch.nn.functional.dropout(x, p, train))
        assert torch.equal(state, torch.cuda.get_rng_state())
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 0 if p == 1 and train else 1))
print(json.dumps({"ok": True}))
"""
    assert run(code, cache_dir) == {"ok": True}


def test_indirect_atomic_embedding(cache_dir):
    code = """
gather = compiled(lambda x, i: torch.gather(x, 0, i))
index = compiled(lambda x, i: x[i])
for n, m in ((7, 1), (255, 255), (256, 256), (257, 257), (513, 65539)):
    base = torch.arange(n * 2 + 3, device="cuda", dtype=torch.float32)
    x = base[3::2]
    i = torch.arange(m, device="cuda") % n
    if m > 1:
        i[-1] = n - 1
    assert torch.equal(gather(x, i), torch.gather(x, 0, i))
    neg = i.clone()
    neg[::2] -= n
    assert torch.equal(index(x, neg), x[neg])
for n in (0, 7):
    x = torch.empty(n, device="cuda")
    i = torch.empty(0, device="cuda", dtype=torch.int64)
    assert gather(x, i).numel() == index(x, i).numel() == 0
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    f = compiled(lambda x, i, d: torch.zeros(d, device=x.device, dtype=torch.float32).scatter_add(0, i, x.float()))
    for n in (0, 1, 255, 256, 257, 4099):
        for form in ("distinct", "two_way", "many_to_one", "sparse"):
            d = n + 7 if form == "distinct" else 17
            i = (torch.arange(n, device="cuda") if form == "distinct"
                 else torch.arange(n, device="cuda") // 2 % 17 if form == "two_way"
                 else torch.zeros(n, device="cuda", dtype=torch.int64) if form == "many_to_one"
                 else (torch.arange(n, device="cuda") % 4) * 4)
            x = ((torch.arange(n, device="cuda") % 9) - 4).to(dtype) / 8
            ref = torch.zeros(d, device="cuda").scatter_add(0, i, x.float())
            assert torch.equal(f(x, i, d), ref)
f = compiled(lambda w, i: torch.nn.functional.embedding(i, w))
for n, m in ((7, 1), (17, 255), (19, 256), (23, 257)):
    w = torch.linspace(-1, 1, n * 7, device="cuda").reshape(n, 7).requires_grad_()
    i = torch.arange(m, device="cuda") % n
    y = f(w, i)
    up = torch.linspace(-2, 2, y.numel(), device="cuda").reshape(y.shape)
    y.backward(up)
    wr = w.detach().clone().requires_grad_()
    yr = torch.nn.functional.embedding(i, wr)
    yr.backward(up)
    assert torch.equal(y, yr)
    torch.testing.assert_close(w.grad, wr.grad, rtol=2e-5, atol=2e-5)
print(json.dumps({"ok": True}))
"""
    assert run(code, cache_dir) == {"ok": True}


def test_training_steps_match_reference(cache_dir):
    code = """
import copy
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(17, 7))
        self.linear = torch.nn.Linear(7, 9, bias=False)
    def forward(self, idx, mask=None):
        x = self.linear(torch.nn.functional.embedding(idx, self.weight))
        x = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5)
        return (torch.nn.functional.dropout(x, 0.25, True) if mask is None
                else x * mask.to(x.dtype) * (1 / 0.75))
torch.manual_seed(313)
model = Model().cuda()
ref = copy.deepcopy(model)
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, foreach=False, fused=False)
refopt = torch.optim.SGD(ref.parameters(), lr=0.01, momentum=0.9, foreach=False, fused=False)
f = compiled(lambda i: (model(i) ** 2).flatten().mean())
i = torch.tensor([0, 2, 2, 16, 1], device="cuda")
close = dict(rtol=2e-5, atol=2e-5)
for step in range(4):
    opt.zero_grad(set_to_none=True)
    refopt.zero_grad(set_to_none=True)
    state = torch.cuda.get_rng_state().clone()
    seeds, expected = seeds_from_state(state, 1)
    torch.cuda.set_rng_state(state)
    loss = f(i)
    assert torch.equal(torch.cuda.get_rng_state(), expected)
    mask = reference_random(seeds[0], i.numel() * 9).reshape(i.numel(), 9) > 0.25
    loss.backward()
    reference = (ref(i, mask) ** 2).flatten().mean()
    reference.backward()
    torch.testing.assert_close(loss, reference, **close)
    for a, b in zip(model.parameters(), ref.parameters()):
        torch.testing.assert_close(a.grad, b.grad, **close)
    opt.step()
    refopt.step()
    for a, b in zip(model.parameters(), ref.parameters()):
        torch.testing.assert_close(a, b, **close)
        torch.testing.assert_close(opt.state[a]["momentum_buffer"], refopt.state[b]["momentum_buffer"], **close)
f = compiled(lambda x: torch.randn_like(x) + 1, policy="aten_extern")
x = torch.ones(257, device="cuda")
torch.manual_seed(777)
state = torch.cuda.get_rng_state().clone()
y = f(x)
after = torch.cuda.get_rng_state()
torch.cuda.set_rng_state(state)
assert torch.equal(y, torch.randn_like(x) + 1)
assert torch.equal(after, torch.cuda.get_rng_state())
print(json.dumps({"ok": True}))
"""
    assert run(code, cache_dir) == {"ok": True}


def test_unsupported_capabilities_rejected_without_compiling(cache_dir):
    code = """
from torch._inductor import config
from cubecl_torch.inductor import UnsupportedCubeCL
row = lambda x: x.sum(-1)
cases = [
    ("randn_generated", lambda x: torch.randn_like(x), (torch.ones(257, device="cuda"),), True),
    ("index_int32", lambda x, i: x[i], (torch.ones(257, device="cuda"), torch.arange(257, device="cuda", dtype=torch.int32)), True),
    ("atomic_half", lambda x, i: torch.zeros_like(x).scatter_add(0, i, x), (torch.ones(257, device="cuda", dtype=torch.float16), torch.arange(257, device="cuda")), True),
    ("user_mutation", lambda x: x.add_(1), (torch.ones(257, device="cuda"),), True),
    ("out_mutation", lambda x: torch.add(x, 1, out=x), (torch.ones(257, device="cuda"),), True),
    ("inplace_keyword", lambda x: torch.nn.functional.dropout(x, 0.25, True, inplace=True), (torch.ones(257, device="cuda"),), True),
    ("prod", lambda x: x.prod(-1), (torch.randn(3, 257, device="cuda"),), False),
    ("welford", lambda x: torch.var(x, -1), (torch.randn(3, 257, device="cuda"),), False),
    ("float64", row, (torch.randn(3, 257, device="cuda", dtype=torch.float64),), False),
    ("reduction_rank", lambda x: x.sum((1, 2)), (torch.randn(3, 5, 257, device="cuda"),), False),
]
rejected = {}
for name, fn, args, dynamic in cases:
    before = kernels.COMPILER_INVOCATIONS
    state = torch.cuda.get_rng_state().clone()
    try:
        compiled(fn, dynamic=dynamic)(*args)
        rejected[name] = "accepted"
    except Exception as error:
        assert "unsupported" in str(error).lower(), str(error)
        assert kernels.COMPILER_INVOCATIONS == before
        assert torch.equal(state, torch.cuda.get_rng_state())
        rejected[name] = "rejected"
with config.patch(numerics="strict"):
    before = kernels.COMPILER_INVOCATIONS
    try:
        compiled(row, dynamic=False)(torch.randn(3, 259, device="cuda"))
        rejected["strict"] = "accepted"
    except Exception as error:
        assert "exact ordering" in str(error) or "exact eager reduction" in str(error), str(error)
        assert kernels.COMPILER_INVOCATIONS == before
        rejected["strict"] = "rejected"
print(json.dumps(rejected))
"""
    result = run(code, cache_dir)
    # Welford (var) is admitted since OpenSpec welford-reductions; it stays
    # here as the acceptance counterpart of the reductions still rejected.
    assert result.pop("welford") == "accepted", result
    assert set(result.values()) == {"rejected"}, result


def test_stream_lifetime_rng_streams_special_values_determinism(cache_dir):
    code = """
import threading
from cubecl_torch.runtime import trace
gather = compiled(lambda x, i: torch.gather(x, 0, i))
scatter = compiled(lambda x, i, d: torch.zeros(d, device=x.device).scatter_add(0, i, x))
random = compiled(lambda x: torch.rand_like(x))
drop = compiled(lambda x: torch.nn.functional.dropout(x, 0.25, True))
x = torch.arange(257, device="cuda", dtype=torch.float32)
i = torch.arange(257, device="cuda") % 17
gather(x, i); scatter(x, i, 17); random(x); drop(x)
for trial in range(3):
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    base = torch.arange(1048583, device="cuda", dtype=torch.float32)
    index = (torch.arange(65539, device="cuda") % base.numel()).clone()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        torch.cuda._sleep(10000000)
        base.add_(trial + 1)
        index.add_(1)
    consumer.wait_event(producer.record_event())
    with torch.cuda.stream(consumer), trace.scoped():
        start = len(trace.EVENTS)
        y = gather(base, index)
        z = scatter(y, index % 257, 257)
        copied = z.clone()
        done = consumer.record_event()
        launches = [e for e in list(trace.EVENTS)[start:] if e["event"] == "launch"]
        assert launches and all(e["stream"] == consumer.cuda_stream for e in launches)
        assert any(base.data_ptr() in e["pointers"] and index.data_ptr() in e["pointers"] for e in launches)
        addresses = {base.data_ptr(), index.data_ptr(), y.data_ptr(), z.data_ptr()}
    del base, index, y, z
    pressure = [torch.full((1048583,), -1000.0, device="cuda") for _ in range(8)]
    assert not any(t.data_ptr() in addresses for t in pressure) or done.query()
    done.synchronize()
    indices = (torch.arange(65539, device="cuda") % 1048583) + 1
    expected = torch.zeros(257, device="cuda").scatter_add(0, indices % 257, (indices + trial + 1).float())
    torch.testing.assert_close(copied, expected, rtol=2e-5, atol=2e-5)
for stream in (torch.cuda.Stream(), torch.cuda.Stream()):
    with torch.cuda.stream(stream):
        torch.manual_seed(19)
        state = torch.cuda.get_rng_state().clone()
        y = random(x)
    stream.synchronize()
    after = torch.cuda.get_rng_state().clone()
    seeds, expected_state = seeds_from_state(state, 1)
    assert torch.equal(after, expected_state)
    assert torch.equal(y, reference_random(seeds[0], 257))
barrier = threading.Barrier(2)
errors, outputs = [], []
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
for t in threads: t.start()
for t in threads: t.join()
for s in streams: s.synchronize()
assert not errors and len(outputs) == 2 and not torch.equal(outputs[0], outputs[1])
after = torch.cuda.get_rng_state()
torch.cuda.set_rng_state(state)
limits = torch.iinfo(torch.int64)
for _ in range(2):
    torch.randint(limits.min, limits.max, (1,), device="cuda")
assert torch.equal(after, torch.cuda.get_rng_state())
values = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1.0, -1.0] * 37, device="cuda")
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    v = values.to(dtype)
    torch.manual_seed(13)
    state = torch.cuda.get_rng_state()
    y = drop(v)
    seeds, _ = seeds_from_state(state, 1)
    mask = (reference_random(seeds[0], v.numel()).to(dtype) > 0.25).to(dtype)
    expected = v * mask * (1 / 0.75)
    rtol, atol = tol(dtype)
    torch.testing.assert_close(y, expected, rtol=rtol, atol=atol, equal_nan=True)
    assert torch.equal(torch.isnan(y), torch.isnan(expected)) and torch.equal(torch.isinf(y), torch.isinf(expected))
    zero = expected == 0
    assert torch.equal(torch.signbit(y[zero]), torch.signbit(expected[zero]))
torch.use_deterministic_algorithms(True)
try:
    with trace.scoped():
        start = len(trace.EVENTS)
        toggled = scatter(x, i, 17)
        assert torch.equal(toggled, torch.zeros(17, device="cuda").scatter_add(0, i, x))
        events = list(trace.EVENTS)[start:]
        descriptors = [e["descriptor"] for e in events if e["event"] == "kernel"]
        assert any(e.get("execution_path") == "extern" for e in events)
        assert not any(op["op"] == "atomic_add" for d in descriptors for op in d["ops"])
        start = len(trace.EVENTS)
        deterministic = compiled(lambda x, i: torch.zeros_like(x).scatter_add(0, i, x))
        src = torch.linspace(-1, 1, 257, device="cuda")
        idx = torch.arange(257, device="cuda") % 17
        assert torch.equal(deterministic(src, idx), torch.zeros_like(src).scatter_add(0, idx, src))
        events = list(trace.EVENTS)[start:]
        descriptors = [e["descriptor"] for e in events if e["event"] == "kernel"]
        assert not any(op["op"] == "atomic_add" for d in descriptors for op in d["ops"])
        assert any(e.get("execution_path") == "extern" for e in events)
finally:
    torch.use_deterministic_algorithms(False)
guard.state("after_runtime_checks")
print(json.dumps({"ok": True}))
"""
    assert run(code, cache_dir) == {"ok": True}


@pytest.mark.skipif(
    os.environ.get("CUBECL_TORCH_GPU_TRAPS") != "1",
    reason="faults the GPU on purpose (a display stall and a Windows watchdog "
    "dump per case under WSL); set CUBECL_TORCH_GPU_TRAPS=1",
)
@pytest.mark.parametrize(
    "mode", ["upper", "negative_gather", "negative_index", "large", "min_i64"]
)
def test_device_bounds_trap(cache_dir, mode):
    code = f"""
from cubecl_torch.testing.fixtures import native_launches
mode = {mode!r}
x = torch.arange(7, device="cuda", dtype=torch.float32)
value = {{"upper": 7, "negative_gather": -1, "large": 1 << 32, "min_i64": -(1 << 63)}}.get(mode, -8)
idx = torch.tensor([value], device="cuda")
fn = (lambda x, idx: x[idx]) if mode == "negative_index" else (lambda x, idx: torch.gather(x, 0, idx))
f = torch.compile(fn, backend=ci.compile_backend, fullgraph=True, dynamic=True)
before = native_launches()
try:
    f(x, idx)
    torch.cuda.synchronize()
    outcome = "accepted"
except torch._dynamo.exc.BackendCompilerFailed:
    raise
except RuntimeError as error:
    outcome = "trapped"
print(json.dumps({{"outcome": outcome, "launched": native_launches() > before, "gil": sys._is_gil_enabled()}}))
"""
    result = run(code, cache_dir)
    assert result == {"outcome": "trapped", "launched": True, "gil": False}


def test_oversized_rng_counter_domain_rejected(cache_dir):
    code = """
from torch._subclasses.fake_tensor import FakeTensorMode
from cubecl_torch.runtime import trace
torch.cuda.init()
state = torch.cuda.get_rng_state().clone()
with FakeTensorMode(), trace.scoped():
    fake = torch.empty(1 << 31, device="cuda")
    try:
        compiled(lambda x: torch.rand_like(x))(fake)
        message = None
    except Exception as error:
        message = str(error)
    rejection = any(e.get("boundary") == "pre_fusion" and e.get("decision") == "reject" for e in trace.EVENTS)
print(json.dumps({"rejected": message is not None and "unsupported RNG physical extent" in message,
                  "compiles": kernels.COMPILER_INVOCATIONS,
                  "rng": torch.equal(state, torch.cuda.get_rng_state()),
                  "traced": rejection}))
"""
    assert run(code, cache_dir) == {
        "rejected": True,
        "compiles": 0,
        "rng": True,
        "traced": True,
    }


def test_profiler_census_generated_kernels_only(cache_dir):
    code = """
from cubecl_torch.runtime import trace
from cubecl_torch.testing.fixtures import native_launches
f = compiled(lambda x: torch.nn.functional.dropout(x, 0.25, True))
x = torch.ones(257, device="cuda", requires_grad=True)
up = torch.linspace(-1, 1, 257, device="cuda")
f(x).backward(up)
torch.cuda.synchronize()
before = native_launches()
with trace.scoped():
    start = len(trace.EVENTS)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        x.grad = None
        f(x).backward(up)
        torch.cuda.synchronize()
    launches = [e["entrypoint"] for e in list(trace.EVENTS)[start:] if e["event"] == "launch" and e["grid"][0] > 0]
names = [e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
generated = [n for n in names if n in launches]
print(json.dumps({"counts": [len(generated), len(launches), native_launches() - before],
                  "triton": [n for n in names if "triton" in n.lower()]}))
"""
    result = run(code, cache_dir)
    generated, launches, native = result["counts"]
    assert generated == launches == native > 0
    assert result["triton"] == []
