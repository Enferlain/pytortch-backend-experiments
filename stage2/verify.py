"""Real Stage 2 semantic categories, compiler/launch evidence, no Triton."""

import ctypes as C
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache")
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "0"
from stage1 import evidence

evidence.guard()
import torch

evidence.state("stage2_after_torch")
from stage1.overlay import install

install()
from stage2.backend import compile_backend, TRACE
from stage2 import runtime

evidence.state("stage2_after_inductor_import")
from stage2.indexing import evaluate, encode
import sympy
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

torch._dynamo.config.recompile_limit = 64
torch.manual_seed(23)
RESULT = {
    "torch": torch.__version__,
    "torch_commit": torch.version.git_version,
    "checks": [],
    "failures": [],
    "tolerances": {
        "float32": [2e-6, 2e-6],
        "float16": [0.002, 0.001],
        "bfloat16": [0.016, 0.008],
    },
}


def check(a, b):
    rtol, atol = RESULT["tolerances"].get(str(a.dtype).removeprefix("torch."), [0, 0])
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, equal_nan=True)


def run(name, fn):
    start = len(TRACE)
    launch = runtime.KERNEL_LAUNCHES
    compile = runtime.COMPILER_INVOCATIONS
    try:
        detail = fn()
        RESULT["checks"].append(
            {
                "name": name,
                "detail": detail,
                "launches": runtime.KERNEL_LAUNCHES - launch,
                "compilations": runtime.COMPILER_INVOCATIONS - compile,
                "trace_start": start,
                "trace_end": len(TRACE),
            }
        )
        print(name, "PASS", flush=True)
    except Exception as e:
        import traceback

        RESULT["failures"].append(
            {
                "name": name,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "trace_start": start,
                "trace_end": len(TRACE),
            }
        )
        print(name, "FAIL", str(e)[-1000:], flush=True)


def compile(f, dynamic=True):
    return torch.compile(f, backend=compile_backend, fullgraph=True, dynamic=dynamic)


def sine(x):
    return torch.sin(x) + 1


try:
    for dtype in (torch.float32, torch.float16, torch.bfloat16):

        def shape_suite(dtype=dtype):
            f = compile(sine)
            records = []
            for n in (255, 256, 257, 1_048_583, 1, 0, 257):
                x = torch.randn(n, device="cuda", dtype=dtype, requires_grad=True)
                begin = runtime.KERNEL_LAUNCHES
                comp = runtime.COMPILER_INVOCATIONS
                graphs = sum(t["event"] == "graph" for t in TRACE)
                y = f(x)
                check(y, sine(x))
                assert runtime.KERNEL_LAUNCHES - begin == int(n > 0)
                if n:
                    assert runtime.EVENTS[-1]["pointers"] == [
                        x.data_ptr(),
                        y.data_ptr(),
                    ]
                tangent = torch.randn_like(y)
                begin = runtime.KERNEL_LAUNCHES
                (gradient,) = torch.autograd.grad(y, x, tangent)
                ref = x.detach().requires_grad_()
                (expected,) = torch.autograd.grad(sine(ref), ref, tangent)
                check(gradient, expected)
                assert runtime.KERNEL_LAUNCHES - begin == int(n > 0)
                if n:
                    assert runtime.EVENTS[-1]["pointers"] == [
                        tangent.data_ptr(),
                        x.data_ptr(),
                        gradient.data_ptr(),
                    ]
                records.append(
                    {
                        "n": n,
                        "compiled_delta": runtime.COMPILER_INVOCATIONS - comp,
                        "graph_delta": sum(t["event"] == "graph" for t in TRACE)
                        - graphs,
                    }
                )
            assert all(
                r["compiled_delta"] == 0 and r["graph_delta"] == 0 for r in records[1:4]
            ), records
            x = torch.tensor(
                [
                    float("nan"),
                    float("inf"),
                    -float("inf"),
                    -0.0,
                    0.0,
                    1e-30,
                    -1e-30,
                    -3.14,
                    3.14,
                    -1000,
                    1000,
                ],
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )
            y = f(x)
            check(y, sine(x))
            t = torch.randn_like(y)
            (g,) = torch.autograd.grad(y, x, t)
            ref = x.detach().requires_grad_()
            (e,) = torch.autograd.grad(sine(ref), ref, t)
            check(g, e)
            return records

        run("dtype_dynamic_forward_backward_" + str(dtype), shape_suite)

    def binary(x, y):
        return (x - y) / (y + 2)

    def unary(x):
        return torch.where(x <= 0, -x, torch.sqrt(torch.exp(x) - 1))

    def casts(x):
        return (torch.sin(x).to(torch.float32) * 0.3333333333333333).to(torch.bfloat16)

    for label, f in [
        ("binary_true_division", binary),
        ("unary_comparison_select", unary),
        ("explicit_casts", casts),
    ]:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):

            def ops_suite(f=f, dtype=dtype):
                args = [torch.randn(257, device="cuda", dtype=dtype)]
                if f is binary:
                    args.append(torch.randn(257, device="cuda", dtype=dtype) * 0.25)
                y = compile(f)(*args)
                check(y, f(*args))
                if f is binary:
                    refs = [a.detach().requires_grad_() for a in args]
                    live = [a.detach().requires_grad_() for a in args]
                    output = compile(f)(*live)
                    reference = f(*refs)
                    tangent = torch.randn_like(output)
                    actual = torch.autograd.grad(output, live, tangent)
                    expected = torch.autograd.grad(reference, refs, tangent)
                    for a, b in zip(actual, expected):
                        check(a, b)
                if f is not casts:
                    specials = torch.tensor(
                        [
                            float("nan"),
                            float("inf"),
                            -float("inf"),
                            -0.0,
                            0.0,
                            1.0,
                            -1.0,
                        ],
                        device="cuda",
                        dtype=dtype,
                    )
                    a = [specials, specials] if f is binary else [specials]
                    check(compile(f)(*a), f(*a))
                return "random and special float values"

            run(label + "_" + str(dtype), ops_suite)
    for cmp in ("eq", "ne", "lt", "le", "gt", "ge"):

        def compare_suite(cmp=cmp):
            f = lambda x: getattr(torch, cmp)(x, 0)
            x = torch.tensor(
                [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, -1, 1],
                device="cuda",
            )
            a = compile(f)(x)
            torch.testing.assert_close(a, f(x), rtol=0, atol=0)
            return "boolean storage/result and NaN comparisons"

        run("comparison_" + cmp, compare_suite)

    def signed_zero_and_ieee():
        records = []
        for name, f in [
            ("sub_self", lambda x: x - x),
            ("add_zero", lambda x: x + 0.0),
            ("mul_zero", lambda x: x * 0.0),
            ("zero_div", lambda x: 0.0 / x),
            ("neg", lambda x: -x),
            ("nonfinite_constant", lambda x: x + float("inf")),
        ]:
            x = torch.tensor(
                [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, -1, 1],
                device="cuda",
            )
            a = compile(f)(x)
            b = f(x)
            check(a, b)
            finite = torch.isfinite(b)
            assert torch.equal(torch.signbit(a[finite]), torch.signbit(b[finite])), (
                name,
                a,
                b,
            )
            records.append(name)
        return records

    run("IEEE_nan_infinity_signed_zero", signed_zero_and_ieee)

    def mixed_types():
        records = []
        for left, right in [
            (torch.float16, torch.float32),
            (torch.float16, torch.bfloat16),
        ]:
            f = lambda a, b: a + b
            a = torch.randn(257, device="cuda", dtype=left)
            b = torch.randn(257, device="cuda", dtype=right)
            y = compile(f)(a, b)
            check(y, f(a, b))
            assert y.dtype == torch.float32
            records.append([str(left), str(right), str(y.dtype)])
        return records

    run("generic_mixed_dtype_promotion", mixed_types)

    def differentiable():
        records = []
        for label, f in [
            ("exp_neg", lambda x: -torch.exp(x)),
            ("sqrt", lambda x: torch.sqrt(x)),
            ("cast", lambda x: torch.sin(x).float()),
            ("where", unary),
        ]:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                x = (torch.rand(257, device="cuda", dtype=dtype) + 0.5).requires_grad_()
                ref = x.detach().requires_grad_()
                y = compile(f)(x)
                e = f(ref)
                t = torch.randn_like(y)
                (actual,) = torch.autograd.grad(y, x, t)
                (expected,) = torch.autograd.grad(e, ref, t)
                check(y, e)
                check(actual, expected)
                records.append([label, str(dtype)])
        return records

    run("additional_arbitrary_upstream_gradients", differentiable)

    def fusion_veto():
        f = lambda x: (torch.sin(x) + 1, torch.sin(x) * 2)
        start = len(TRACE)
        x = torch.randn(257, device="cuda")
        before = runtime.KERNEL_LAUNCHES
        a, b = compile(f, False)(x)
        ea, eb = f(x)
        check(a, ea)
        check(b, eb)
        veto = [
            e
            for e in TRACE[start:]
            if e.get("boundary") == "fusion" and e["decision"] == "reject"
        ]
        assert veto and runtime.KERNEL_LAUNCHES - before == 2, TRACE[start:]
        return veto

    run("real_multioutput_fusion_veto", fusion_veto)

    def boolean_storage():
        records = []
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            mask = torch.arange(257, device="cuda") % 2 == 0
            x = torch.randn(257, device="cuda", dtype=dtype)
            f = lambda mask, x: torch.where(mask, x, -x)
            output = compile(f)(mask, x)
            check(output, f(mask, x))
            records.append(str(dtype))
        x = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, -1, 1], device="cuda"
        )
        f = lambda x: x.to(torch.bool)
        torch.testing.assert_close(compile(f)(x), f(x), rtol=0, atol=0)
        return records

    run("boolean_load_where_cast_storage", boolean_storage)

    def layouts():
        cases = [
            ("contiguous_2d", torch.randn(3, 257, device="cuda")),
            ("transpose", torch.randn(3, 257, device="cuda").T),
            ("slice_offset", torch.randn(517, device="cuda")[3::2]),
            ("zero_stride", torch.randn(1, 257, device="cuda").expand(3, 257)),
        ]
        records = []
        for name, x in cases:
            begin = runtime.KERNEL_LAUNCHES
            y = compile(sine)(x)
            check(y, sine(x))
            assert runtime.KERNEL_LAUNCHES - begin == 1
            assert runtime.EVENTS[-1]["pointers"] == [x.data_ptr(), y.data_ptr()]
            records.append(
                {
                    "name": name,
                    "shape": list(x.shape),
                    "stride": list(x.stride()),
                    "offset": x.storage_offset(),
                }
            )
        # symbolic stride changes reuse a graph where the guard permits it.
        f = compile(sine)
        records2 = []
        for step in (2, 3, 5):
            x = torch.randn(1501, device="cuda")[1::step]
            before = runtime.COMPILER_INVOCATIONS
            graphs = sum(t["event"] == "graph" for t in TRACE)
            check(f(x), sine(x))
            records2.append(
                {
                    "step": step,
                    "compilations": runtime.COMPILER_INVOCATIONS - before,
                    "graphs": sum(t["event"] == "graph" for t in TRACE) - graphs,
                }
            )
        assert records2[1]["graphs"] == records2[2]["graphs"] == 0, records2
        return {"layouts": records, "dynamic_stride": records2}

    run("direct_layouts", layouts)

    def strided_backward():
        records = []
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for layout in ("transpose", "offset_slice", "zero_stride"):
                if layout == "transpose":
                    x = (
                        torch.randn(3, 257, device="cuda", dtype=dtype)
                        .T.detach()
                        .requires_grad_()
                    )
                elif layout == "offset_slice":
                    x = (
                        torch.randn(515, device="cuda", dtype=dtype)[1::2]
                        .detach()
                        .requires_grad_()
                    )
                else:
                    x = (
                        torch.randn(1, 257, device="cuda", dtype=dtype)
                        .expand(3, 257)
                        .detach()
                        .requires_grad_()
                    )
                ref = x.detach().requires_grad_()
                f = compile(sine)
                output = f(x)
                tangent = torch.randn_like(output)
                (gradient,) = torch.autograd.grad(output, x, tangent)
                (expected,) = torch.autograd.grad(sine(ref), ref, tangent)
                check(output, sine(x))
                check(gradient, expected)
                records.append([str(dtype), layout])
        return records

    run("layout_dtype_arbitrary_backward", strided_backward)

    def broadcasts():
        records = []
        for column in (False, True):
            f = compile(lambda x, b: torch.sin(x) + b)
            for b, n in ((3, 255), (5, 256), (2, 257), (4, 513)):
                x = torch.randn(b, n, device="cuda")
                bias = torch.randn((b, 1) if column else (n,), device="cuda")
                before = runtime.COMPILER_INVOCATIONS
                graphs = sum(t["event"] == "graph" for t in TRACE)
                launches = runtime.KERNEL_LAUNCHES
                y = f(x, bias)
                check(y, torch.sin(x) + bias)
                assert runtime.KERNEL_LAUNCHES - launches == 1
                assert runtime.EVENTS[-1]["pointers"] == [
                    x.data_ptr(),
                    bias.data_ptr(),
                    y.data_ptr(),
                ]
                records.append(
                    {
                        "column": column,
                        "shape": [b, n],
                        "compilations": runtime.COMPILER_INVOCATIONS - before,
                        "graphs": sum(t["event"] == "graph" for t in TRACE) - graphs,
                    }
                )
            assert all(r["graphs"] == 0 for r in records[-3:]), records
        return records

    run("broadcast_dynamic_rows_columns", broadcasts)

    def square_symbols():
        f = compile(lambda x, b: x + b)
        records = []
        for size in (3, 5, 17):
            x = torch.randn(size, size, device="cuda")
            b = torch.randn(size, device="cuda")
            count = runtime.COMPILER_INVOCATIONS
            graphs = sum(t["event"] == "graph" for t in TRACE)
            y = f(x, b)
            check(y, x + b)
            records.append(
                {
                    "size": size,
                    "compiler_delta": runtime.COMPILER_INVOCATIONS - count,
                    "graph_delta": sum(t["event"] == "graph" for t in TRACE) - graphs,
                }
            )
        assert all(r["compiler_delta"] == r["graph_delta"] == 0 for r in records[1:]), (
            records
        )
        return records

    run("dynamic_repeated_dimension_symbols", square_symbols)

    def masked():
        f = compile(lambda x: torch.cat([x, x[:1]]) + 1)
        out = []
        for n in (255, 256, 257, 1025):
            x = torch.randn(n, device="cuda")
            start = runtime.KERNEL_LAUNCHES
            y = f(x)
            check(y, torch.cat([x, x[:1]]) + 1)
            assert runtime.KERNEL_LAUNCHES - start == 1
            out.append(n)
        assert any(
            op["op"] == "masked"
            for e in TRACE
            if e["event"] == "kernel"
            for op in e["descriptor"]["ops"]
        )
        return out

    run("real_lazy_semantic_masks", masked)

    def fusion(x):
        a = x
        for _ in range(20):
            a = torch.sin(a) + 0.01
        return a * a + a

    def fusion_suite():
        start = len(TRACE)
        f = compile(fusion, False)
        x = torch.randn(257, device="cuda")
        launches = runtime.KERNEL_LAUNCHES
        y = f(x)
        check(y, fusion(x))
        kernels = [e for e in TRACE[start:] if e["event"] == "kernel"]
        assert any(len(e["nodes"]) > 1 for e in kernels), [
            (e["nodes"]) for e in kernels
        ]
        assert runtime.KERNEL_LAUNCHES - launches == 1
        assert any(
            e["event"] == "fusion_delegate" and e["accepted"] for e in TRACE[start:]
        )
        return {"nodes": kernels[0]["nodes"], "trace": TRACE[start:]}

    run("genuine_multinode_fusion", fusion_suite)

    def profiler():
        f = compile(sine)
        x = torch.randn(257, device="cuda")
        f(x)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            y = f(x)
        torch.cuda.synchronize()
        check(y, sine(x))
        prof.export_chrome_trace(str(ROOT / "artifacts/forward-profiler.json"))
        events = json.loads((ROOT / "artifacts/forward-profiler.json").read_text())[
            "traceEvents"
        ]
        kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
        assert len(kernels) == 1 and kernels[0].startswith("cubecl2_"), kernels
        return kernels

    run("profiler_one_cubecl_kernel", profiler)

    def profile_categories():
        records = []
        cases = [
            (
                "broadcast",
                lambda x, b: torch.sin(x) + b,
                [torch.randn(3, 257, device="cuda"), torch.randn(257, device="cuda")],
            ),
            ("slice", sine, [torch.randn(515, device="cuda")[1::2]]),
            (
                "mask",
                lambda x: torch.cat([x, x[:1]]) + 1,
                [torch.randn(257, device="cuda")],
            ),
            ("fusion", fusion, [torch.randn(257, device="cuda")]),
        ]
        for name, fn, args in cases:
            f = compile(fn, False if name == "fusion" else True)
            f(*args)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                output = f(*args)
            torch.cuda.synchronize()
            check(output, fn(*args))
            path = ROOT / "artifacts" / (name + "-profiler.json")
            prof.export_chrome_trace(str(path))
            kernels = [
                e["name"]
                for e in json.loads(path.read_text())["traceEvents"]
                if e.get("cat") == "kernel"
            ]
            assert len(kernels) == 1 and kernels[0].startswith("cubecl2_"), kernels
            records.append({"category": name, "kernels": kernels})
        return records

    run("representative_profiler_one_kernel", profile_categories)

    def streams_pressure():
        f = compile(sine)
        x = torch.randn(257, device="cuda", dtype=torch.float16)
        f(x)
        for trial in range(3):
            x = torch.randn(1_048_583, device="cuda", dtype=torch.float16)
            expected = sine(x)
            allocation = x.data_ptr()
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda._sleep(30_000_000)
                y = f(x)
                saved = y.clone()
                event = torch.cuda.Event()
                event.record()
                assert runtime.EVENTS[-1]["stream"] == stream.cuda_stream
            del x, y
            pressure = [
                torch.empty(1_048_583, device="cuda", dtype=torch.float16).fill_(37)
                for _ in range(16)
            ]
            assert not any(
                p.data_ptr() == allocation and not event.query() for p in pressure
            )
            stream.synchronize()
            check(saved, expected)
        return "three delayed non-default stream allocator-pressure trials"

    run("streams_allocator_pressure", streams_pressure)

    def negatives():
        out = []
        cases = [
            (
                "float64",
                lambda x: x + 1,
                torch.randn(257, device="cuda", dtype=torch.float64),
                "backend_selection",
            ),
            (
                "integer_tensor",
                lambda x: x + 1,
                torch.ones(257, device="cuda", dtype=torch.int64),
                "backend_selection",
            ),
            (
                "rank5",
                sine,
                torch.randn(2, 2, 2, 2, 2, device="cuda"),
                "backend_selection",
            ),
            (
                "reduction",
                lambda x: x.sum(),
                torch.randn(257, device="cuda"),
                "pre_fusion",
            ),
            (
                "unknown_tan",
                lambda x: torch.tan(x),
                torch.randn(257, device="cuda"),
                "pre_fusion",
            ),
            (
                "masked_unknown_tan",
                lambda x: torch.cat([torch.tan(x), x[:1]]) + 1,
                torch.randn(257, device="cuda"),
                "pre_fusion",
            ),
            (
                "indirect",
                lambda x: x[torch.ones_like(x, dtype=torch.int64)],
                torch.randn(257, device="cuda"),
                "pre_fusion",
            ),
        ]
        for name, f, x, boundary in cases:
            start = len(TRACE)
            count = runtime.COMPILER_INVOCATIONS
            try:
                compile(f)(x)
            except Exception as e:
                assert "unsupported" in str(e) or "required" in str(e), str(e)
                assert any(
                    t.get("boundary") == boundary and t.get("decision") == "reject"
                    for t in TRACE[start:]
                ), TRACE[start:]
                assert (
                    not any(t["event"] == "kernel" for t in TRACE[start:])
                    and runtime.COMPILER_INVOCATIONS == count
                )
                out.append({"name": name, "boundary": boundary})
            else:
                raise AssertionError("unsupported case succeeded: " + name)
        for name, fn, args in [
            (
                "too_many_inputs",
                lambda a, b, c, d, e: a + b + c + d + e,
                [torch.randn(257, device="cuda") for _ in range(5)],
            ),
            (
                "bool_arithmetic",
                lambda x: x + x,
                [torch.ones(257, device="cuda", dtype=torch.bool)],
            ),
            (
                "float64_cast",
                lambda x: x.to(torch.float64),
                [torch.randn(257, device="cuda")],
            ),
            (
                "numeric_runtime_scalar",
                lambda x, scale: x * scale,
                [torch.randn(257, device="cuda"), 3],
            ),
            (
                "minimum",
                lambda x: torch.minimum(x, x),
                [torch.randn(257, device="cuda")],
            ),
        ]:
            start = len(TRACE)
            count = runtime.COMPILER_INVOCATIONS
            try:
                compile(fn)(*args)
            except Exception as error:
                assert "unsupported" in str(error) or "required" in str(error), str(
                    error
                )
                assert any(
                    t.get("boundary") == "pre_fusion" and t.get("decision") == "reject"
                    for t in TRACE[start:]
                ), TRACE[start:]
                assert runtime.COMPILER_INVOCATIONS == count and not any(
                    t["event"] == "kernel" for t in TRACE[start:]
                )
                out.append({"name": name, "boundary": "pre_fusion"})
            else:
                raise AssertionError("unsupported case succeeded: " + name)
        return out

    run("early_negative_capabilities", negatives)

    def width_tests():
        i = sympy.Symbol("i", integer=True)
        n = sympy.Symbol("n", positive=True, integer=True)
        exprs = [
            i * 3 + 2,
            FloorDiv(i - 9, 7),
            sympy.Mod(i - 9, 7),
            ModularIndexing(i, 2, n),
        ]
        out = []
        for expr in exprs:
            tree = encode(expr, lambda s: 0, i)
            for v in (-(2**32) - 1, -9, -1, 0, 2**31 + 1, 2**32 + 257):
                got = evaluate(tree, [257], v)
                expected = int(expr.subs({i: v, n: 257}))
                assert got == expected, (expr, v, got, expected)
            out.append({"expression": str(expr), "tree": tree})
        from stage1.capabilities import UnsupportedCubeCL

        unknown = sympy.Symbol("modulus", integer=True)
        for expr in (ModularIndexing(i, 2, unknown), FloorDiv(i, -n)):
            try:
                encode(expr, lambda s: 0, i)
            except UnsupportedCubeCL as error:
                assert "positive" in str(error)
                out.append({"rejected_expression": str(expr), "reason": str(error)})
            else:
                raise AssertionError("unproven-positive index divisor accepted")
        return out

    run("signed_i64_floor_mod_component", width_tests)
    evidence.state("stage2_after_all_checks")
finally:
    RESULT["compiler_invocations"] = runtime.COMPILER_INVOCATIONS
    RESULT["kernel_launches"] = runtime.KERNEL_LAUNCHES
    lib = C.CDLL(str(ROOT.parent / "stage0/libstage0_launcher.so"))
    lib.stage0_kernel_launches.restype = C.c_uint64
    lib.stage0_launcher_calls.restype = C.c_uint64
    RESULT["native_kernel_launches"] = lib.stage0_kernel_launches()
    RESULT["native_launcher_calls"] = lib.stage0_launcher_calls()
    RESULT["triton_probes"] = evidence.PROBE.probes
    RESULT["triton_import_attempts"] = evidence.ATTEMPTS
    RESULT["gil"] = evidence.GIL
    RESULT["loaded_triton"] = [
        n for n in sys.modules if n == "triton" or n.startswith("triton.")
    ]
    RESULT["native_triton_maps"] = [
        l
        for l in Path("/proc/self/maps").read_text().splitlines()
        if "triton" in l.lower()
    ]
    assert runtime.KERNEL_LAUNCHES == RESULT["native_kernel_launches"]
    assert (
        not RESULT["loaded_triton"]
        and not RESULT["native_triton_maps"]
        and not any(evidence.GIL.values())
    )
    for name, value in [
        ("results", RESULT),
        ("schedule-replay", TRACE),
        ("runtime-events", runtime.EVENTS),
    ]:
        (ROOT / "artifacts" / f"{name}.json").write_text(json.dumps(value, indent=2))
if RESULT["failures"]:
    raise SystemExit(
        "Stage 2 failures: " + ", ".join(f["name"] for f in RESULT["failures"])
    )
print("Stage 2 CUDA matrix passed", flush=True)
