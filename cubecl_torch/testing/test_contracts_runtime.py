"""Ported live runtime contracts from the frozen gates (see contract_inventory.md).

Sources: stage5/launch_controls.py, stage5/controls.py, stage5/proof_controls.py,
stage5/components.py, stage4/components.py, stage2/components.py, stage0/verify.py.
"""

import copy
import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from cubecl_torch.runtime import trace
from cubecl_torch.runtime.launch import LaunchContractError
from cubecl_torch.testing.fixtures import captured, native_launches, physical
from cubecl_torch.testing.philox import uniform

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def scalar():
    return physical(captured("add-float32"), "scalar_pointwise_baseline")


@pytest.fixture(scope="module")
def vector():
    return physical(captured("add-float32"), "vector_pointwise", (128, 1, 1), 4)


def refused(fn, text):
    before = native_launches()
    with pytest.raises(ValueError, match=text):
        fn()
    assert native_launches() == before


def test_zero_grid_is_uncounted_noop(scalar):
    before, count = native_launches(), scalar.kernel.launch_count
    scalar((0,), torch.empty(0, device="cuda"), torch.empty(0, device="cuda"))
    assert scalar.kernel.launch_count == count and native_launches() == before


def test_layout_cache_is_bounded(scalar):
    for n in (0, 1, 255, 256, 257, 4099, *range(270, 340)):
        x = torch.randn(n + 3, device="cuda")[3:]
        y = torch.empty_like(x)
        scalar((n,), x, y)
        torch.testing.assert_close(y, x + 1)
    assert len(scalar.kernel._layouts) == 64
    assert scalar.kernel._call_pointers is None


def test_warm_rejections_before_launch(scalar, vector):
    x = torch.randn(257, device="cuda")
    y = torch.empty_like(x)
    scalar((257,), x, y)
    refused(lambda: scalar((257,), x[:256], y), "layout")
    refused(lambda: scalar((257,), x, y.double()), "dtype")
    refused(lambda: scalar((257,), x.cpu(), y), "device")
    refused(lambda: scalar((257,), x, x), "disjoint")
    refused(
        lambda: scalar((257,), torch.empty_strided((257,), (2,), device="cuda"), y),
        "layout",
    )
    refused(lambda: scalar((1 << 63,), x, y), "outside i64")
    refused(lambda: scalar((), x, y), "argument count")
    vector((257,), x, y)
    refused(
        lambda: vector((257,), torch.randn(258, device="cuda")[1:], y), "misaligned"
    )
    assert vector.kernel._call_pointers is None


def test_inconsistent_descriptor_extent_refused():
    semantic = captured("add-float32")
    semantic["numel"] = {"op": "constant", "value": 258}
    kernel = physical(semantic, "scalar_pointwise_baseline")
    x = torch.randn(257, device="cuda")
    refused(
        lambda: kernel((257,), x, torch.empty_like(x)),
        "pointwise output extent mismatch",
    )


def test_warm_key_rechecks_storage_capacity():
    semantic = captured("add-float32")
    semantic["buffers"][0]["offset"] = {"op": "constant", "value": 1}
    for op in semantic["ops"]:
        if op["op"] == "load":
            op["index"] = {
                "op": "add",
                "args": [op["index"], {"op": "constant", "value": 1}],
            }
    kernel = physical(semantic, "scalar_pointwise_baseline")
    backing = torch.randn(258, device="cuda")
    y = torch.empty(257, device="cuda")
    kernel((257,), backing[:257], y)
    torch.testing.assert_close(y, backing[1:] + 1)
    refused(
        lambda: kernel((257,), torch.randn(257, device="cuda"), y), "original storage"
    )


def test_compiled_descriptor_is_a_snapshot():
    semantic = captured("add-float32")
    kernel = physical(semantic, "scalar_pointwise_baseline")
    semantic["buffers"][0]["stride"][0] = {"op": "constant", "value": 2}
    x = torch.randn(257, device="cuda")
    y = torch.empty_like(x)
    kernel((257,), x, y)
    torch.testing.assert_close(y, x + 1)


def test_trace_scopes_and_counters(scalar):
    x = torch.randn(257, device="cuda")
    y = torch.empty_like(x)
    before, count = native_launches(), scalar.kernel.launch_count
    start = len(trace.EVENTS)
    with trace.scoped(False):
        scalar((257,), x, y)
    assert len(trace.EVENTS) == start
    with trace.scoped(True):
        scalar((257,), x, y)
    launch = trace.EVENTS[-1]
    assert launch["event"] == "launch" and launch["pointers"] == [
        x.data_ptr(),
        y.data_ptr(),
    ]
    assert launch["stream"] == torch.cuda.current_stream().cuda_stream
    assert scalar.kernel.launch_count - count == native_launches() - before == 2
    with pytest.raises(RuntimeError, match="unwind"), trace.scoped(True):
        with trace.scoped(False):
            assert not trace.enabled()
        assert trace.enabled()
        raise RuntimeError("test scope unwind")
    assert not trace.enabled()


def test_shared_kernel_independent_streams_and_trace_scopes(scalar):
    barrier = threading.Barrier(2)
    count, before = scalar.kernel.launch_count, native_launches()
    start = len(trace.EVENTS)

    def worker(detail):
        stream = torch.cuda.Stream()
        with trace.scoped(detail), torch.cuda.stream(stream):
            barrier.wait()
            for n in (255, 256, 257, 259, 1001) * 4:
                inp = torch.randn(n, device="cuda")
                out = torch.empty_like(inp)
                scalar((n,), inp, out)
                torch.testing.assert_close(out, inp + 1)
            stream.synchronize()
            return stream.cuda_stream

    with ThreadPoolExecutor(max_workers=2) as pool:
        streams = [
            f.result(timeout=120)
            for f in [pool.submit(worker, d) for d in (False, True)]
        ]
    assert scalar.kernel.launch_count - count == native_launches() - before == 40
    events = [e for e in list(trace.EVENTS)[start:] if e["event"] == "launch"]
    assert len(events) == 20 and all(e["stream"] == streams[1] for e in events)
    assert not trace.enabled()


def test_producer_readiness_and_no_retention(scalar):
    stream = torch.cuda.Stream()
    inp = torch.randn(4099, device="cuda")
    stream.wait_stream(torch.cuda.current_stream())
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


def test_prepared_scope_releases_tensors(scalar):
    x = torch.randn(1001, device="cuda")
    y = torch.empty_like(x)
    scalar((1001,), x, y)
    assert scalar.kernel._prepared is None
    with scalar.kernel.prepared((1001,), x, y):
        scalar.kernel.batch(2)
    assert scalar.kernel._prepared is None
    torch.testing.assert_close(y, x + 1)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("width", [2, 4, 8])
def test_vector_special_values_and_canaries(dtype, width):
    if dtype == "float32" and width == 8:
        pytest.skip("float32 vector8 exceeds the 16-byte transaction")
    kernel = physical(captured(f"add-{dtype}"), "vector_pointwise", (128, 1, 1), width)
    x = torch.tensor(
        [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1e-30, -1e-30] * 37,
        device="cuda",
        dtype=getattr(torch, dtype),
    )
    backing = torch.full((x.numel() + 2 * width,), 123, device="cuda", dtype=x.dtype)
    y = backing[width:-width]
    kernel((x.numel(),), x, y)
    torch.testing.assert_close(y, x + 1, rtol=0, atol=0, equal_nan=True)
    canary = torch.full((width,), 123, device="cuda", dtype=x.dtype)
    assert torch.equal(backing[:width], canary) and torch.equal(
        backing[-width:], canary
    )


def test_vector_misalignment_refused_before_launch(vector):
    x = torch.randn(258, device="cuda")[1:]
    refused(lambda: vector((257,), x, torch.empty_like(x)), "misaligned")


@pytest.mark.parametrize(
    ("strategy", "workgroup"),
    [
        ("scalar_pointwise_baseline", (256, 1, 1)),
        ("scalar_pointwise", (64, 1, 1)),
        ("scalar_pointwise", (128, 1, 1)),
    ],
)
def test_rng_candidates_exact_and_not_repeatable(strategy, workgroup):
    kernel = physical(captured("rng"), strategy, workgroup)
    for seed in (0, -1, 123456):
        buffers = [
            torch.tensor([seed], device="cuda", dtype=torch.int64),
            torch.empty(257, device="cuda"),
        ]
        state = torch.cuda.get_rng_state().clone()
        kernel((257,), *buffers)
        expected = torch.tensor([uniform(seed, i) for i in range(257)], device="cuda")
        torch.testing.assert_close(buffers[-1], expected, rtol=0, atol=0)
        assert torch.equal(state, torch.cuda.get_rng_state())
        with pytest.raises(ValueError, match="pure"):
            kernel.kernel.batch(2)


@pytest.mark.parametrize(
    ("strategy", "workgroup"),
    [
        ("scalar_pointwise_baseline", (256, 1, 1)),
        ("scalar_pointwise", (64, 1, 1)),
        ("scalar_pointwise", (128, 1, 1)),
    ],
)
def test_atomic_candidates_exact_and_not_repeatable(strategy, workgroup):
    from cubecl_torch.runtime.rules import evaluate

    semantic = captured("atomic")
    kernel = physical(semantic, strategy, workgroup)
    scalars = tuple(257 for _ in semantic["scalars"])
    tensors = []
    for buf in semantic["buffers"]:
        size = tuple(evaluate(e, scalars) for e in buf["size"])
        numel = torch.Size(size).numel()
        if buf["access"] == "write":
            tensors.append(torch.zeros(size, device="cuda"))
        elif buf["dtype"] == "int64":
            tensors.append(torch.arange(numel, device="cuda").reshape(size) % 17)
        else:
            tensors.append(
                ((torch.arange(numel, device="cuda") % 17 - 8).float() / 8).reshape(
                    size
                )
            )
    index, values = (
        (tensors[0], tensors[1])
        if tensors[0].dtype == torch.int64
        else (tensors[1], tensors[0])
    )
    expected = torch.zeros_like(tensors[-1]).scatter_add_(0, index, values)
    for _ in range(3):
        tensors[-1].zero_()
        kernel(scalars, *tensors)
        torch.testing.assert_close(tensors[-1], expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="pure"):
        kernel.kernel.batch(2)


def rng_descriptor():
    """stage4/components.py descriptor(): seed buffer, linear i32 counter, Philox."""
    const = lambda n: {"op": "constant", "value": n}
    typed = lambda d: {"dtype": d, "semantic_dtype": d}
    return {
        "schema": 4,
        "index_dtype": "int64",
        "numel": {"op": "scalar", "slot": 0},
        "scalars": [{"id": 0, "dtype": "int64", "positive": False}],
        "buffers": [
            {
                "access": "read",
                "dtype": "int64",
                "size": [const(1)],
                "stride": [const(1)],
                "offset": const(0),
            },
            {
                "access": "write",
                "dtype": "float32",
                "size": [{"op": "scalar", "slot": 0}],
                "stride": [const(1)],
                "offset": const(0),
            },
        ],
        "effects": {"rng": "explicit_seed_counter", "mutation": False},
        "ops": [
            {
                "id": "seed",
                "op": "load",
                "args": [],
                **typed("int64"),
                "buffer": 0,
                "index": const(0),
            },
            {
                "id": "wide",
                "op": "index",
                "args": [],
                **typed("int64"),
                "index": {"op": "linear"},
            },
            {"id": "counter", "op": "cast", "args": ["wide"], **typed("int32")},
            {
                "id": "rng",
                "op": "rand",
                "args": ["seed", "counter"],
                **typed("float32"),
                "algorithm": "philox4x32_10_top24",
            },
            {
                "op": "store",
                "args": ["rng"],
                **typed("float32"),
                "buffer": 1,
                "index": {"op": "linear"},
            },
        ],
    }


def test_rng_descriptor_exact_with_canaries():
    kernel = physical(rng_descriptor(), "scalar_pointwise_baseline")
    for seed in (0, 1, -1, -(1 << 63), (1 << 63) - 1, 0x12345678ABCDEF):
        for n in (0, 1, 255, 256, 257):
            backing = torch.full((n + 2,), 123.0, device="cuda")
            out = backing[1 : n + 1]
            kernel((n,), torch.tensor([seed], device="cuda"), out)
            expected = torch.tensor([uniform(seed, i) for i in range(n)], device="cuda")
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            assert backing[0].item() == backing[-1].item() == 123


@pytest.mark.parametrize("counter", [0, -1, -(1 << 31), (1 << 31) - 1])
def test_rng_counter_bits(counter):
    semantic = rng_descriptor()
    semantic["ops"][2] = {
        "id": "counter",
        "op": "constant",
        "args": [],
        "dtype": "int32",
        "semantic_dtype": "int32",
        "value": counter,
    }
    kernel = physical(semantic, "scalar_pointwise_baseline")
    y = torch.empty(1, device="cuda")
    kernel((1,), torch.tensor([0], device="cuda"), y)
    expected = torch.tensor([uniform(0, counter & ((1 << 32) - 1))], device="cuda")
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


def test_rng_runtime_counter_domain_guard():
    kernel = physical(rng_descriptor(), "scalar_pointwise_baseline")
    with pytest.raises(LaunchContractError, match="unsupported RNG physical extent"):
        kernel.kernel._layout((1 << 31,))


def masked_indirect_descriptor(nested):
    """stage4/components.py: runtime lazy regions guard invalid tensor indices."""
    const = lambda n: {"op": "constant", "value": n}
    typed = lambda d: {"dtype": d, "semantic_dtype": d}
    n = {"op": "scalar", "slot": 0}
    body = [
        {
            "id": "index",
            "op": "load",
            "args": [],
            **typed("int64"),
            "buffer": 1,
            "index": {"op": "linear"},
        },
        {
            "id": "checked",
            "op": "indirect",
            "args": ["index"],
            **typed("int64"),
            "size": const(7),
            "bounds": "require_in_range",
        },
        {
            "id": "loaded",
            "op": "load",
            "args": [],
            **typed("float32"),
            "buffer": 0,
            "index": {"op": "value", "id": "checked"},
        },
    ]
    mask = {
        "id": "mask",
        "op": "masked",
        "args": ["pred"],
        **typed("float32"),
        "body": body,
        "result": "loaded",
        "other": 7.0,
    }
    if nested:
        inner = {**copy.deepcopy(mask), "id": "inner", "args": ["inner_pred"]}
        mask["body"] = [
            {
                "id": "inner_bound",
                "op": "constant",
                "args": [],
                **typed("int64"),
                "value": 64,
            },
            {
                "id": "inner_pred",
                "op": "lt",
                "args": ["position", "inner_bound"],
                **typed("bool"),
            },
            inner,
        ]
        mask["result"] = "inner"
    return {
        "schema": 4,
        "index_dtype": "int64",
        "numel": n,
        "scalars": [{"id": 0, "dtype": "int64", "positive": False}],
        "effects": {"rng": None, "mutation": False},
        "buffers": [
            {
                "access": "read",
                "dtype": "float32",
                "size": [const(7)],
                "stride": [const(1)],
                "offset": const(0),
            },
            {
                "access": "read",
                "dtype": "int64",
                "size": [n],
                "stride": [const(1)],
                "offset": const(0),
            },
            {
                "access": "write",
                "dtype": "float32",
                "size": [n],
                "stride": [const(1)],
                "offset": const(0),
            },
        ],
        "ops": [
            {
                "id": "position",
                "op": "index",
                "args": [],
                **typed("int64"),
                "index": {"op": "linear"},
            },
            {
                "id": "bound",
                "op": "constant",
                "args": [],
                **typed("int64"),
                "value": 128,
            },
            {"id": "pred", "op": "lt", "args": ["position", "bound"], **typed("bool")},
            mask,
            {
                "op": "store",
                "args": ["mask"],
                **typed("float32"),
                "buffer": 2,
                "index": {"op": "linear"},
            },
        ],
    }


@pytest.mark.parametrize("nested", [False, True])
def test_masked_indirect_never_touches_invalid_indices(nested):
    kernel = physical(masked_indirect_descriptor(nested), "scalar_pointwise_baseline")
    limit = 64 if nested else 128
    x = torch.arange(7, device="cuda", dtype=torch.float32)
    indices = torch.arange(257, device="cuda") % 7
    indices[limit:] = 1 << 32
    backing = torch.full((259,), 123.0, device="cuda")
    y = backing[1:258]
    kernel((257,), x, indices, y)
    expected = torch.full_like(y, 7)
    expected[:limit] = x[indices[:limit]]
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    assert backing[0].item() == backing[-1].item() == 123


def test_positive_index_domain_refused():
    semantic = captured("add-float32")
    semantic["scalars"][0]["positive"] = True
    kernel = physical(semantic, "scalar_pointwise_baseline")
    x = torch.empty(0, device="cuda")
    refused(lambda: kernel((0,), x, torch.empty_like(x)), "positive index domain")


def test_record_stream_prevents_early_reuse(scalar):
    """stage0/verify.py: storage recorded on a side stream is not recycled early."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        torch.cuda._sleep(20_000_000)
        x = torch.randn(1 << 20, device="cuda")
        y = torch.empty_like(x)
        scalar((x.numel(),), x, y)
        expected = x + 1
    addresses = {x.data_ptr(), y.data_ptr()}
    retained = y
    del x
    reused = [
        t.data_ptr() for t in (torch.empty(1 << 20, device="cuda") for _ in range(8))
    ]
    assert not addresses & set(reused) - {retained.data_ptr()}
    side.synchronize()
    torch.testing.assert_close(retained, expected)


def test_capture_refused(scalar):
    x = torch.randn(257, device="cuda")
    y = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(NotImplementedError, match="capture"), torch.cuda.graph(graph):
        scalar((257,), x, y)
