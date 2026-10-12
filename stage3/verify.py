"""Bounded cooperative-reduction gate with real capture, counters and profiling."""

import ctypes as C
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache")
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "0"
from stage1 import evidence

evidence.guard()
import torch

evidence.state("after_torch")
from stage1.overlay import install

install()
from stage3 import backend, runtime
from stage3.plan import ReductionPlan
from stage0.bridge import ROOT as STAGE0_ROOT

evidence.state("after_inductor")
torch.manual_seed(71)
torch._dynamo.config.recompile_limit = 128
evidence.SELECTED |= {
    "CubeCLKernel.reduction",
    "CubeCLKernel.store_reduction",
    "SIMDScheduling.generate_node_schedule",
    "CSEProxy.reduction",
    "CSEProxy.store_reduction",
}
native = C.CDLL(str(STAGE0_ROOT / "libstage0_launcher.so"))
native.stage0_kernel_launches.restype = C.c_uint64
native.stage0_launcher_calls.restype = C.c_uint64
native.stage3_kernel_launches.restype = C.c_uint64
native.stage3_total_shared_bytes.restype = C.c_uint64
native_begin = native.stage0_kernel_launches()
native_calls_begin = native.stage0_launcher_calls()
reduction_begin = native.stage3_kernel_launches()
shared_begin = native.stage3_total_shared_bytes()
RESULT = {
    "checks": [],
    "failures": [],
    "torch": torch.__version__,
    "torch_commit": torch.version.git_version,
    "tolerances": {
        "float32": [2e-5, 2e-5],
        "float16": [0.002, 0.001],
        "bfloat16": [0.016, 0.008],
    },
}


def compile(f, dynamic=True):
    return torch.compile(
        f, backend=backend.compile_backend, fullgraph=True, dynamic=dynamic
    )


def close(a, b):
    rtol, atol = RESULT["tolerances"][str(a.dtype).removeprefix("torch.")]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, equal_nan=True)
    assert a.dtype == b.dtype
    assert torch.equal(torch.isnan(a), torch.isnan(b))
    assert torch.equal(torch.isposinf(a), torch.isposinf(b))
    assert torch.equal(torch.isneginf(a), torch.isneginf(b))


def run(name, function):
    start = len(backend.TRACE)
    before_compile, before_launch = (
        runtime.COMPILER_INVOCATIONS,
        runtime.KERNEL_LAUNCHES,
    )
    try:
        detail = function()
        RESULT["checks"].append(
            {
                "name": name,
                "detail": detail,
                "compiles": runtime.COMPILER_INVOCATIONS - before_compile,
                "launches": runtime.KERNEL_LAUNCHES - before_launch,
                "trace_range": [start, len(backend.TRACE)],
            }
        )
        print(name, "PASS", flush=True)
    except Exception as error:
        RESULT["failures"].append(
            {
                "name": name,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "trace_range": [start, len(backend.TRACE)],
            }
        )
        print(name, "FAIL", str(error)[-1200:], flush=True)


def row(x):
    return x.sum(-1)


def column(x):
    return x.sum(0)


def mean(x):
    return x.mean(-1) + 1


def rms(x):
    # Ordinary PyTorch opmath authoring with lowp input/output storage.
    f = x.float()
    return (f * torch.rsqrt((f * f).mean(-1, keepdim=True) + 1e-5)).to(x.dtype)


def profile(name, function, expected):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        function()
    path = ROOT / "artifacts" / (name + "-profiler.json")
    p.export_chrome_trace(str(path))
    data = json.loads(path.read_text())
    kernels = [e for e in data["traceEvents"] if e.get("cat") == "kernel"]
    assert len(kernels) == expected, [e["name"] for e in kernels]
    assert all(e["name"].startswith(("cubecl3_", "cubecl2_")) for e in kernels)
    return [
        {"name": e["name"], "args": e.get("args"), "duration_us": e.get("dur")}
        for e in kernels
    ]


sys.setprofile(evidence.profile)
try:
    for dtype in (torch.float32, torch.float16, torch.bfloat16):

        def row_shapes(dtype=dtype):
            f = compile(row)
            shapes = [
                (3, 31),
                (4, 32),
                (2, 33),
                (5, 255),
                (3, 256),
                (4, 257),
                (2, 4099),
                (3, 65539),
                (3, 1),
                (3, 0),
                (0, 257),
                (3, 257),
            ]
            rows = []
            for b, n in shapes:
                x = torch.randn(b, n, device="cuda", dtype=dtype, requires_grad=True)
                c, l = runtime.COMPILER_INVOCATIONS, runtime.KERNEL_LAUNCHES
                y = f(x)
                close(y, row(x))
                upstream = torch.randn_like(y)
                grad = torch.autograd.grad(y, x, upstream)[0]
                ref = x.detach().requires_grad_()
                expected = torch.autograd.grad(row(ref), ref, upstream)[0]
                close(grad, expected)
                rows.append(
                    {
                        "shape": [b, n],
                        "compiles": runtime.COMPILER_INVOCATIONS - c,
                        "launches": runtime.KERNEL_LAUNCHES - l,
                    }
                )
                if b and n > 1:
                    event = next(
                        e for e in reversed(runtime.EVENTS) if e["event"] == "launch"
                    )
                    assert event["pointers"][0] == x.data_ptr()
                    assert event["grid"] == [b, 1, 1] and event["shared_bytes"] == 1024
            assert all(v["compiles"] == 0 for v in rows[1:8])
            return rows

        run("row_dynamic_" + str(dtype), row_shapes)

        def layout_suite(dtype=dtype):
            entries = []
            f = compile(column)
            for b, n in ((31, 3), (32, 5), (33, 2), (257, 4), (4099, 3)):
                x = torch.randn(b, n, device="cuda", dtype=dtype)
                before = runtime.KERNEL_LAUNCHES
                y = f(x)
                close(y, column(x))
                assert runtime.KERNEL_LAUNCHES - before == 1
                assert runtime.EVENTS[-1]["pointers"] == [x.data_ptr(), y.data_ptr()]
                entries.append({"shape": [b, n], "kind": "column"})
            for kind, x in (
                ("transpose", torch.randn(257, 3, device="cuda", dtype=dtype).T),
                ("slice", torch.randn(3, 519, device="cuda", dtype=dtype)[:, 3::2]),
                (
                    "stride_zero",
                    torch.randn(3, 1, device="cuda", dtype=dtype).expand(3, 257),
                ),
            ):
                event_start = len(runtime.EVENTS)
                f2 = compile(row)
                y = f2(x)
                close(y, row(x))
                launches = [
                    e for e in runtime.EVENTS[event_start:] if e["event"] == "launch"
                ]
                # Inductor naturally splits the transposed reduction. Its
                # partial tensor belongs to the normal wrapper/allocator.
                assert launches[0]["pointers"][0] == x.data_ptr()
                assert launches[-1]["pointers"][-1] == y.data_ptr()
                assert len(launches) in (1, 2)
                entries.append(
                    {
                        "kind": kind,
                        "shape": list(x.shape),
                        "stride": list(x.stride()),
                        "offset": x.storage_offset(),
                        "launches": launches,
                    }
                )
            return entries

        run("column_strides_" + str(dtype), layout_suite)

        def fusion_suite(dtype=dtype):
            f = compile(mean)
            rows = []
            for b, n in ((3, 255), (5, 256), (2, 257), (4, 4099)):
                x = torch.randn(b, n, device="cuda", dtype=dtype)
                c, l = runtime.COMPILER_INVOCATIONS, runtime.KERNEL_LAUNCHES
                y = f(x)
                close(y, mean(x))
                assert runtime.KERNEL_LAUNCHES - l == 1
                rows.append(
                    {"shape": [b, n], "compiles": runtime.COMPILER_INVOCATIONS - c}
                )
            assert all(r["compiles"] == 0 for r in rows[1:])
            return {
                "rows": rows,
                "profiler": profile("mean-" + str(dtype), lambda: f(x), 1),
            }

        run("fused_mean_" + str(dtype), fusion_suite)

        def norm_suite(dtype=dtype):
            f = compile(rms)
            x = torch.randn(3, 257, device="cuda", dtype=dtype)
            begin = runtime.KERNEL_LAUNCHES
            out = f(x)
            close(out, rms(x))
            assert runtime.KERNEL_LAUNCHES - begin == 1
            inference = profile("rms-inference-" + str(dtype), lambda: f(x), 1)
            f2 = compile(rms)
            x = x.detach().requires_grad_()
            begin = runtime.KERNEL_LAUNCHES
            out = f2(x)
            close(out, rms(x))
            forward = runtime.KERNEL_LAUNCHES - begin
            upstream = torch.randn_like(out)
            begin = runtime.KERNEL_LAUNCHES
            grad = torch.autograd.grad(out, x, upstream)[0]
            backward = runtime.KERNEL_LAUNCHES - begin
            ref = x.detach().requires_grad_()
            close(grad, torch.autograd.grad(rms(ref), ref, upstream)[0])
            # Fresh saved tensors for the warmed backward profile preserve
            # AOTAutograd's normal donated-buffer contract.
            out = f2(x)
            p = profile(
                "rms-backward-" + str(dtype),
                lambda: torch.autograd.grad(out, x, upstream),
                backward,
            )
            assert any(e["name"].startswith("cubecl3_") for e in p)
            return {
                "inference": inference,
                "training_forward_kernels": forward,
                "backward_kernels": backward,
                "backward_profiler": p,
            }

        run("normalization_" + str(dtype), norm_suite)

        def special_suite(dtype=dtype):
            f = compile(row)
            cases = []
            for n in (2, 31, 32, 33, 255, 256, 257, 4099):
                x = torch.zeros(8, n, device="cuda", dtype=dtype)
                x[1].fill_(-0.0)
                x[2, 0] = float("nan")
                x[3, 0] = float("inf")
                x[4, 0] = -float("inf")
                x[5, 0] = float("inf")
                x[5, 1] = -float("inf")
                x[6, ::2] = 100
                x[6, 1::2] = -100
                x[7, ::4] = 10000
                x[7, 1::4] = 1
                x[7, 2::4] = -10000
                y = f(x)
                expected = row(x)
                close(y, expected)
                assert not torch.signbit(y[:2]).any()
                cases.append(
                    {
                        "n": n,
                        "output": y.tolist(),
                        "zero_sign": torch.signbit(y[:2]).tolist(),
                    }
                )
            return cases

        run("specials_cancellation_" + str(dtype), special_suite)

    def streams_pressure():
        f = compile(rms)
        f(torch.randn(3, 257, device="cuda"))
        records = []
        for trial in range(3):
            side = torch.cuda.Stream()
            x = torch.randn(7, 4099, device="cuda")
            expected = rms(x)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                torch.cuda._sleep(10_000_000)
                out = f(x)
                consumer = out.clone()
                event = runtime.EVENTS[-1]
                assert event["stream"] == side.cuda_stream
                assert event["pointers"][0] == x.data_ptr()
                completion = torch.cuda.Event()
                completion.record()
            del x, out
            pressure = [
                torch.empty((7, 4099), device="cuda").fill_(99) for _ in range(32)
            ]
            completion.synchronize()
            close(consumer, expected)
            records.append(
                {"trial": trial, "stream": side.cuda_stream, "launch": event}
            )
            del pressure
        return records

    run("nondefault_stream_allocator_pressure", streams_pressure)

    def accumulation_and_empty_mean():
        rows = []
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            f = compile(mean)
            for n in (0, 1, 257):
                x = torch.full((3, n), 1000, device="cuda", dtype=dtype)
                y = f(x)
                close(y, mean(x))
                rows.append({"dtype": str(dtype), "n": n, "mean_plus_one": y.tolist()})
            if dtype == torch.float16:
                x = torch.full((3, 257), 1000, device="cuda", dtype=dtype)
                summed = compile(row)(x)
                close(summed, row(x))
                assert torch.isposinf(summed).all()
                assert torch.isfinite(f(x)).all()
        return rows

    run("accumulator_precision_and_empty_mean", accumulation_and_empty_mean)

    def repeated_shared_scratch():
        def f(x):
            return x.sum(-1) + (x * x).sum(-1)

        c = compile(f)
        x = torch.randn(3, 257, device="cuda")
        start = len(backend.TRACE)
        y = c(x)
        close(y, f(x))
        kernels = [e for e in backend.TRACE[start:] if e["event"] == "kernel"]
        assert len(kernels) == 1
        assert (
            sum(op["op"] == "reduction" for op in kernels[0]["descriptor"]["ops"]) == 2
        )
        return {
            "nodes": kernels[0]["nodes"],
            "profiler": profile("two-sums-shared-scratch", lambda: c(x), 1),
        }

    run("two_sums_normal_fusion_shared_reuse", repeated_shared_scratch)

    def semantic_mask():
        def f(x):
            return torch.cat((x, x[:, :1]), dim=1).sum(-1)

        c = compile(f)
        x = torch.randn(3, 257, device="cuda")
        y = c(x)
        close(y, f(x))
        kernels = [e for e in backend.TRACE if e["event"] == "kernel"]
        assert any(o["op"] == "masked" for o in kernels[-1]["descriptor"]["ops"])
        return profile("masked-reduction", lambda: c(x), 1)

    run("lazy_semantic_mask_in_reduction", semantic_mask)

    def veto():
        def f(x):
            return x.sum(-1), x.mean(-1) + 1

        c = compile(f)
        x = torch.randn(3, 257, device="cuda")
        start = len(backend.TRACE)
        out = c(x)
        for a, b in zip(out, f(x)):
            close(a, b)
        rejected = [
            e
            for e in backend.TRACE[start:]
            if e["event"] == "capability"
            and e["boundary"] == "fusion"
            and e["decision"] == "reject"
        ]
        assert rejected
        return {"vetoes": rejected, "profiler": profile("fusion-veto", lambda: c(x), 2)}

    run("unsupported_multioutput_fusion_veto", veto)

    def negatives():
        results = []
        cases = [
            ("max", lambda x: x.amax(-1), torch.randn(3, 257, device="cuda")),
            ("prod", lambda x: x.prod(-1), torch.randn(3, 257, device="cuda")),
            ("welford", lambda x: torch.var(x, -1), torch.randn(3, 257, device="cuda")),
            ("float64", row, torch.randn(3, 257, device="cuda", dtype=torch.float64)),
            (
                "reduction_rank",
                lambda x: x.sum((1, 2)),
                torch.randn(3, 5, 257, device="cuda"),
            ),
            (
                "split_with_unsupported_index_dtype",
                row,
                torch.randn(2, 1048577, device="cuda"),
            ),
        ]
        for name, f, x in cases:
            before = runtime.COMPILER_INVOCATIONS
            start = len(backend.TRACE)
            try:
                compile(f, dynamic=False)(x)
            except Exception as error:
                assert "unsupported" in str(error)
                assert runtime.COMPILER_INVOCATIONS == before
                assert not any(
                    e["event"] in ("kernel", "schedule") for e in backend.TRACE[start:]
                )
                results.append(
                    {
                        "name": name,
                        "error": str(error)[-900:],
                        "trace": backend.TRACE[start:],
                    }
                )
            else:
                raise AssertionError("unsupported accepted " + name)
        for field, value in [
            ("barriers", "under_lane_predicate"),
            ("phases", 2),
            ("workgroup", [32, 1, 1]),
        ]:
            plan = ReductionPlan().to_dict()
            plan[field] = value
            try:
                ReductionPlan.from_dict(plan)
            except Exception as error:
                results.append({"invalid_plan": field, "error": str(error)})
            else:
                raise AssertionError("invalid physical plan accepted")
        from torch._inductor import config

        with config.patch(numerics="strict"):
            before = runtime.COMPILER_INVOCATIONS
            try:
                compile(row, dynamic=False)(torch.randn(3, 259, device="cuda"))
            except Exception as error:
                assert "exact eager reduction ordering mode" in str(error)
                assert runtime.COMPILER_INVOCATIONS == before
                results.append(
                    {"name": "strict_reduction_order", "error": str(error)[-900:]}
                )
            else:
                raise AssertionError("strict reduction ordering accepted")
        return results

    run("early_capability_and_divergent_plan_rejection", negatives)
finally:
    sys.setprofile(None)
    evidence.state("after_acceptance")
    inventory = evidence.inventory()
    assert not inventory["loaded_modules"] and not inventory["native_mappings"]
    RESULT["independence"] = inventory
    RESULT["compiler_invocations"] = runtime.COMPILER_INVOCATIONS
    RESULT["kernel_launches"] = runtime.KERNEL_LAUNCHES
    RESULT["native_kernel_launches"] = native.stage0_kernel_launches() - native_begin
    RESULT["native_launcher_calls"] = (
        native.stage0_launcher_calls() - native_calls_begin
    )
    assert RESULT["native_kernel_launches"] == runtime.KERNEL_LAUNCHES
    expected_reductions = sum(
        e["event"] == "launch"
        and e["entrypoint"].startswith("cubecl3_")
        and e["grid"][0] > 0
        for e in runtime.EVENTS
    )
    RESULT["native_reduction_launches"] = (
        native.stage3_kernel_launches() - reduction_begin
    )
    RESULT["native_requested_shared_bytes"] = (
        native.stage3_total_shared_bytes() - shared_begin
    )
    assert RESULT["native_reduction_launches"] == expected_reductions
    assert RESULT["native_requested_shared_bytes"] == expected_reductions * 1024
    RESULT["fp_patch_sha256"] = hashlib.sha256(
        (ROOT.parent / "fp_hardening/cubecl-fp.patch").read_bytes()
    ).hexdigest()
    diff = subprocess.check_output(
        ["git", "-C", str(ROOT.parent / ".stage0-sources/cubecl"), "diff"]
    )
    assert diff == (ROOT.parent / "fp_hardening/cubecl-fp.patch").read_bytes()
    RESULT["cubecl_source_diff_sha256"] = hashlib.sha256(diff).hexdigest()
    (ROOT / "artifacts/results.json").write_text(json.dumps(RESULT, indent=2))
    (ROOT / "artifacts/schedule-replay.json").write_text(
        json.dumps(backend.TRACE, indent=2)
    )
    (ROOT / "artifacts/runtime-events.json").write_text(
        json.dumps(runtime.EVENTS, indent=2)
    )
    (ROOT / "artifacts/compiler-path.json").write_text(
        json.dumps(evidence.CALLS, indent=2)
    )
if RESULT["failures"]:
    raise SystemExit(f"{len(RESULT['failures'])} Stage 3 gates failed")
print("Stage 3 matrix PASS", flush=True)
