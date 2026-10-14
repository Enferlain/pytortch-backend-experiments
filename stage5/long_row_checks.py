"""Guarded default, numerical-boundary and ownership checks on the real backend."""

import ctypes as C
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/long-row"


def run():
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache/long-row/checks")
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage1.overlay import install

    install()
    from torch._inductor.choices import InductorChoices
    from torch._inductor.virtualized import V, threadlocal
    from stage5 import artifact, runtime
    from stage5.backend import compile_backend, TRACE
    from stage5.choices import EVENTS
    from stage5.index_masks import EVENTS as index_events
    from stage5.index_masks import widen_mask_indices
    from stage5.corpus import rms
    from stage5.reduction_policy import qualify_case, qualified_for_runtime
    from stage5.plan import PhysicalCandidate
    from stage5.legality import structural_reason, numerical_reason
    from stage5.fixtures import captured
    from stage5.diagnostics import detailed_trace
    from stage5.attribution import inventory

    OUTPUT.mkdir(parents=True, exist_ok=True)
    native = C.CDLL(str(ROOT.parent / "stage0/libstage0_launcher.so"))
    native.stage0_kernel_launches.restype = C.c_uint64
    rows = []

    # A previously uninitialized handler must not become the temporary CubeCL
    # factory's cached default when compile_fx restores its config patches.
    key = "__torchinductor_choices"
    if hasattr(threadlocal, key):
        delattr(threadlocal, key)
    probe = torch.compile(lambda x: x + 1, backend=compile_backend, fullgraph=True)
    probe(torch.zeros(17, device="cuda"))
    assert type(V.choices) is InductorChoices
    original_choices = V.choices
    rows.append({"case": "initial_choices_factory_does_not_leak"})

    class ExistingOverride(InductorChoices):
        def reduction_split_factor(self, *args):
            assert args[1:] == (65539, 8, True)
            return 13

    assert (
        ExistingOverride().reduction_split_factor_with_extents(
            torch.device("cuda"), 65539, 8, True, reduction_numel=65539, numel=8
        )
        == 13
    )
    rows.append({"case": "generic_companion_preserves_override"})

    from torch._inductor.scheduler import SchedulerNode
    from torch.utils._sympy.value_ranges import ValueRanges
    from types import SimpleNamespace
    import sympy

    class ComponentNode(SchedulerNode):
        def get_name(self):
            return "mask-component"

    for lower, upper, user, should_widen in (
        (0, 31, "lt", True),
        (0, 2**31, "lt", False),
        (0, sympy.oo, "lt", False),
        (0, 31, "add", False),
        (-(2**31), 31, "lt", True),
        (-(2**31) - 1, 31, "lt", False),
        (-sympy.oo, 31, "lt", False),
    ):
        graph = torch.fx.Graph()
        ops = graph.placeholder("ops")
        index = graph.call_module("get_index", ("index0",))
        value = graph.call_method("index_expr", (ops, index, torch.int32))
        graph.call_method(user, (ops, value, value))
        body = SimpleNamespace(
            get_nodes=lambda: list(graph.nodes),
            bounds=lambda: SimpleNamespace(
                get_bounds=lambda: {index: ValueRanges(lower, upper)}
            ),
        )
        node = object.__new__(ComponentNode)
        node._body = body
        widen_mask_indices([node])
        assert (value.args[2] == torch.int64) == should_widen
    rows.append({"case": "mask_widening_bounds_and_use_controls"})

    ref = {
        "case": "controlled",
        "dtype": "torch.float32",
        "shape": [2],
        "input_sha256": "identical",
        "aten": [1, 0],
        "fp64": [1, 0],
        "values": [1, 0],
    }
    bad = qualify_case({**ref, "values": [1, 0.1]}, ref)
    assert (
        not bad["passes"]
        and bad["elements"][0]["passes"]
        and not bad["elements"][1]["passes"]
    )
    better_ref = {**ref, "values": [5, 0]}
    assert (
        qualify_case(ref, better_ref)["elements"][0]["clause"]
        == "no_greater_fp64_error"
    )
    try:
        qualify_case({**ref, "input_sha256": "different"}, ref)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched inputs accepted")
    rows.append({"case": "per_element_reference_or_better_controls"})
    compensated = PhysicalCandidate("shared_tree_compensated")
    runtime.compare(
        torch.tensor([1.0], device="cuda"),
        torch.tensor([0.0], device="cuda"),
        {"schema": 3},
        compensated,
    )
    try:
        runtime.compare(
            torch.tensor([100.0], device="cuda"),
            torch.tensor([1.0], device="cuda"),
            {"schema": 3},
            compensated,
        )
    except AssertionError as error:
        assert "corruption guard" in str(error)
    else:
        raise AssertionError("gross finite corruption accepted")
    rows.append({"case": "qualified_finite_corruption_guard"})

    combined = PhysicalCandidate("plane_shared", plane_width=32, private_accumulators=4)
    assert PhysicalCandidate.from_dict(combined.to_dict()) == combined
    for change in (
        {"workgroup": [128, 1, 1]},
        {"private_accumulators": 2},
        {"vector_width": 4},
    ):
        try:
            PhysicalCandidate.from_dict({**combined.to_dict(), **change})
        except ValueError:
            pass
        else:
            raise AssertionError("uncatalogued combined mapping accepted")
    sem = captured("row_sum", "float32")
    assert not qualified_for_runtime(sem, combined)
    assert numerical_reason(combined, sem, (65539, 8))
    rows.append({"case": "combined_catalog_and_qualification_veto"})

    def profile(label, callback):
        before = native.stage0_kernel_launches()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            callback()
        kernels = [
            e
            for e in prof.profiler.kineto_results.events()
            if e.device_type() == torch.autograd.DeviceType.CUDA
            and e.activity_type() == "kernel"
        ]
        names = [e.name() for e in kernels]
        assert names and all(n.startswith("cubecl5_") for n in names), names
        assert len(names) == native.stage0_kernel_launches() - before
        raw = ROOT / "cache/long-row" / (label + "-profiler.json")
        prof.export_chrome_trace(str(raw))
        return {"names": names, "raw_trace": str(raw)}

    for history in (
        (31, 65539, 31, 131071, 1048576, 65539),
        (65539, 31, 1048576, 65539, 31),
        (1048576, 31, 65539, 131071, 1048576),
    ):
        torch._dynamo.reset()
        graphs = []

        def backend(gm, inputs):
            graphs.append(str(gm.graph))
            return compile_backend(gm, inputs)

        compiled = torch.compile(
            lambda x: x.sum(-1), backend=backend, fullgraph=True, dynamic=True
        )
        steps = []
        for n in history:
            torch.manual_seed(11)
            x = torch.randn((8, n), device="cuda")
            out = compiled(x)
            torch.testing.assert_close(out, x.sum(-1), rtol=2e-5, atol=2e-5)
            steps.append({"n": n, "graphs": len(graphs)})
        assert len(graphs) == len(
            set(
                "short"
                if n <= 8192
                else "middle"
                if n <= 73728
                else "medium_long"
                if n <= 131072
                else "long"
                for n in history
            )
        ), steps
        for n, count in ((31, 1), (65539, 2)):
            x = torch.randn((8, n), device="cuda")
            compiled(x)
            p = profile(f"history-{history[0]}-{n}", lambda inp=x: compiled(inp))
            assert len(p["names"]) == count
            rows.append(
                {
                    "case": "history_kernel_census",
                    "initial": history[0],
                    "n": n,
                    "profile": p,
                }
            )
        rows.append(
            {
                "case": "guarded_history_reuse",
                "history": list(history),
                "steps": steps,
                "graphs": graphs,
            }
        )

    # Exact current target boundary, including the neighboring outer-output regime.
    previous_recompile_limit = torch._dynamo.config.recompile_limit
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        torch._dynamo.reset()
        # This exhaustive sweep combines empty/singleton, five extent regimes
        # and outer regimes in one function. Bound the test explicitly instead
        # of hitting Dynamo's default eight-entry cache or permitting fallback.
        torch._dynamo.config.recompile_limit = 16
        boundary_graphs = []

        def boundary_backend(gm, inputs):
            boundary_graphs.append(str(gm.graph))
            return compile_backend(gm, inputs)

        f = torch.compile(
            lambda x: x.sum(-1), backend=boundary_backend, fullgraph=True, dynamic=True
        )
        for n in (
            0,
            1,
            255,
            256,
            257,
            8191,
            8192,
            8193,
            65536,
            65537,
            73727,
            73728,
            73729,
            131071,
            131072,
            131073,
            1048575,
            1048576,
            1048577,
        ):
            x = torch.randn((8, n), device="cuda", dtype=dtype)
            runtime.compare(f(x), x.sum(-1), {"schema": 3})
        for b in (163, 164, 165):
            x = torch.randn((b, 8193), device="cuda", dtype=dtype)
            runtime.compare(f(x), x.sum(-1), {"schema": 3})
        assert len(boundary_graphs) <= 10, len(boundary_graphs)
        rows.append(
            {
                "case": "tails_and_guard_neighbors",
                "dtype": str(dtype),
                "graph_variants": len(boundary_graphs),
                "test_recompile_limit": 16,
            }
        )
    torch._dynamo.config.recompile_limit = previous_recompile_limit

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        torch._dynamo.reset()
        f = torch.compile(rms, backend=compile_backend, fullgraph=True, dynamic=True)
        for n in (31, 65539):
            x = torch.randn((3, n), device="cuda", dtype=dtype, requires_grad=True)
            refx = x.detach().clone().requires_grad_()
            upstream = torch.randn_like(x)
            out, ref = f(x), rms(refx)
            runtime.compare(out, ref, {"schema": 3})
            runtime.compare(
                torch.autograd.grad(out, x, upstream)[0],
                torch.autograd.grad(ref, refx, upstream)[0],
                {"schema": 3},
            )
        fp = profile(f"rms-{dtype}-forward", lambda inp=x: f(inp))
        out = f(x)
        bp = profile(
            f"rms-{dtype}-backward",
            lambda inp=x: torch.autograd.grad(out, inp, upstream),
        )
        rows.append(
            {
                "case": "normalization_arbitrary_upstream",
                "dtype": str(dtype),
                "forward": fp,
                "backward": bp,
            }
        )

    torch._dynamo.reset()
    f = torch.compile(
        lambda x: x.sum(-1), backend=compile_backend, fullgraph=True, dynamic=True
    )
    f(torch.randn((8, 31), device="cuda"))
    f(torch.randn((8, 65539), device="cuda"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = torch.randn((8, 65539), device="cuda")
        expected = x.sum(-1)
        start = len(artifact.EVENTS)
        before = native.stage0_kernel_launches()
        with detailed_trace(True):
            out = f(x)
        launches = [e for e in artifact.EVENTS[start:] if e["event"] == "launch"]
        assert len(launches) == native.stage0_kernel_launches() - before == 2
        assert launches[0]["pointers"][0] == x.data_ptr()
        assert launches[-1]["pointers"][-1] == out.data_ptr()
        assert all(e["stream"] == stream.cuda_stream for e in launches)
        del x
        pressure = [torch.empty((8, 65539), device="cuda") for _ in range(16)]
        done = torch.cuda.Event()
        done.record()
    torch.cuda.current_stream().wait_event(done)
    runtime.compare(out, expected, {"schema": 3})
    rows.append(
        {
            "case": "split_stream_pressure",
            "allocations": len(pressure),
            "launches": launches,
        }
    )

    target = runtime.CACHE[next(iter(runtime.CACHE))].last.kernel.resources
    assert structural_reason(
        combined, sem, (65539, 8), (out, out), {**target, "device_max_threads": 128}
    )
    for label, operation, dtype in (
        ("float64", lambda x: x.sum(-1), torch.float64),
        ("prod", lambda x: x.prod(-1), torch.float32),
    ):
        try:
            torch.compile(operation, backend=compile_backend, fullgraph=True)(
                torch.ones((2, 257), device="cuda", dtype=dtype)
            )
        except Exception as error:
            assert "unsupported" in str(error).lower()
            rows.append(
                {"case": "neighbor_rejection", "label": label, "reason": str(error)}
            )
        else:
            raise AssertionError("unsupported neighbor accepted")
    evidence.state("after_long_row_checks")
    assert V.choices is original_choices
    short_events = [
        e
        for e in EVENTS
        if e["event"] == "split_regime_guard"
        and e["guards"]
        and e["reduction_hint"] <= 8192
        and e["planning_anchor"] is not None
    ]
    assert len(short_events) >= 3 and all(
        e["planning_anchor"] == 8192 for e in short_events
    )
    rows.append(
        {
            "case": "tested_short_anchor_stability",
            "anchor": 8192,
            "events": len(short_events),
        }
    )
    result = {
        "checks": rows,
        "trace": TRACE,
        "regimes": EVENTS,
        "environment": inventory(torch, "cubecl"),
        "complete": True,
        "index_mask_widening": index_events,
    }
    (OUTPUT / "checks.json").write_text(json.dumps(result, indent=2))
    print("Long-row checks PASS", len(rows), flush=True)


if __name__ == "__main__":
    run()
