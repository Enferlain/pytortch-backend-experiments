"""Bounded Stage5 diagnostics. No production policy changes; fresh CLI processes."""

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    median = statistics.median(values)
    return {
        "median_us": median,
        "mad_us": statistics.median(abs(v - median) for v in values),
        "samples_us": values,
    }


def setup(args):
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(
        ROOT / "cache/attribution" / args.output.stem
    )
    os.environ["CUBECL_STAGE5_TRACE"] = "0"
    cube = args.impl == "cubecl"
    if cube:
        from stage1 import evidence

        evidence.guard()
    import torch

    if cube or args.precision_probe:
        from stage1.overlay import install

        # Reproduce the frozen, pre-fix attribution baseline. The optional probe
        # then applies the correction; ordinary Stage1-5 overlays enable it.
        install(precision_casts=False)
    if args.precision_probe:
        from stage5.precision_probe import install as precision_install

        probe = precision_install(ROOT.parent / "stage1/overlay/torch/_inductor")
        args.output.with_suffix(".patch").write_text(probe["patch"])
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx

    config.compile_threads = 1
    config.fx_graph_cache = False
    torch._dynamo.config.recompile_limit = 128
    torch._dynamo.config.suppress_errors = False
    torch.manual_seed(6142)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graphs = []

    def backend(gm, inputs):
        graphs.append(
            {
                "graph": str(gm.graph),
                "inputs": [str(getattr(x, "shape", x)) for x in inputs],
            }
        )
        if cube:
            from stage5.backend import compile_backend

            return compile_backend(gm, inputs)
        patches = {
            "triton.cudagraphs": False,
            "compile_threads": 1,
            "fx_graph_cache": False,
            "emulate_precision_casts": True,
            "joint_graph_constant_folding": False,
            "eager_numerics.division_rounding": True,
            "eager_numerics.disable_ftz": True,
            "eager_numerics.use_pytorch_libdevice": True,
        }
        with config.patch(patches):
            return compile_fx(gm, inputs, config_patches=patches)

    native = C.CDLL(str(ROOT.parent / "stage0/libstage0_launcher.so")) if cube else None
    if native:
        native.stage0_kernel_launches.restype = C.c_uint64
    return torch, backend, graphs, native


def inventory(torch, impl):
    data = {
        "python": sys.version,
        "gil": sys._is_gil_enabled(),
        "PYTHON_GIL": os.environ.get("PYTHON_GIL"),
        "torch": torch.__version__,
        "torch_commit": torch.version.git_version,
        "cuda": torch.version.cuda,
        "device": str(torch.cuda.get_device_properties(0)),
        "harness_sha256": sha(__file__),
        "fp_patch_sha256": sha(ROOT.parent / "fp_hardening/cubecl-fp.patch"),
    }
    if impl == "cubecl":
        from stage1 import evidence

        evidence.state("after_attribution")
        data["independence"] = evidence.inventory()
    return data


def artifacts():
    from stage5 import runtime

    return [
        {
            "semantic": k.semantic,
            "directory": str(k.last.directory),
            "plan": k.last.kernel.plan.to_dict(),
            "ptx_sha256": sha(k.last.directory / "kernel.ptx"),
        }
        for k in runtime.CACHE.values()
        if k.last is not None
    ]


def profile(torch, run, native, output):
    before = native.stage0_kernel_launches() if native else None
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        run()
    raw = ROOT / "cache/attribution" / (output.stem + "-profiler.json")
    prof.export_chrome_trace(str(raw))
    kernels = [
        e
        for e in prof.profiler.kineto_results.events()
        if e.device_type() == torch.autograd.DeviceType.CUDA
        and e.activity_type() == "kernel"
    ]
    names = [e.name() for e in kernels]
    if native:
        assert (
            sum(n.startswith("cubecl5_") for n in names)
            == native.stage0_kernel_launches() - before
        )
    assert names, "Incomplete GPU capture"
    return {
        "kernels": [
            {"name": e.name(), "duration_us": (e.end_ns() - e.start_ns()) / 1000}
            for e in kernels
        ],
        "raw_trace": str(raw),
        "sha256": sha(raw),
    }


def history(args):
    torch, backend, graphs, native = setup(args)
    from torch._inductor.choices import InductorChoices

    splits = []
    previous = InductorChoices.reduction_split_factor

    def traced(self, device, reduction_numel_hint, numel_hint, inner_reduction):
        result = previous(
            self, device, reduction_numel_hint, numel_hint, inner_reduction
        )
        splits.append(
            {
                "reduction_hint": reduction_numel_hint,
                "outer_hint": numel_hint,
                "inner": inner_reduction,
                "split": result,
            }
        )
        return result

    InductorChoices.reduction_split_factor = traced
    try:

        def row(x):
            return x.sum(-1)

        compiled = torch.compile(row, backend=backend, fullgraph=True, dynamic=True)
        records = []
        shapes = (
            [(64, 31), (64, 257), (8, 257)] if args.history == "short_first" else []
        )
        shapes.append((args.rows, 65539))
        for shape in shapes:
            # Compile history is the independent variable, not input values.
            torch.manual_seed(6142)
            x = torch.randn(shape, device="cuda")
            begin = len(graphs)
            count = len(splits)
            out = compiled(x)
            torch.testing.assert_close(out, row(x), rtol=2e-5, atol=2e-5)
            for _ in range(10):
                compiled(x)
            torch.cuda.synchronize()
            host, intervals = [], []
            for _ in range(args.samples):
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                began = time.perf_counter_ns()
                compiled(x)
                host.append((time.perf_counter_ns() - began) / 1000)
                end.record()
                end.synchronize()
                intervals.append(start.elapsed_time(end) * 1000)
            profiles = [
                profile(
                    torch,
                    lambda: compiled(x),
                    native,
                    args.output.with_stem(
                        args.output.stem + "-" + str(len(records)) + "-" + str(i)
                    ),
                )
                for i in range(5)
            ]
            records.append(
                {
                    "shape": shape,
                    "new_graphs": len(graphs) - begin,
                    "split_decisions": splits[count:],
                    "host_enqueue": summary(host),
                    "queue_inclusive_event": summary(intervals),
                    "profiles": profiles,
                }
            )
        result = {
            "history": args.history,
            "backend_identity": "one stable callable and compiled function per fresh process",
            "records": records,
            "graphs": graphs,
            "splits": splits,
            "environment": inventory(torch, args.impl),
        }
        if args.impl == "cubecl":
            from stage5.backend import TRACE

            result.update(schedule_replay=TRACE, artifacts=artifacts())
        result["wrapper_files"] = [
            {"path": str(p), "sha256": sha(p)}
            for p in Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).rglob("*.py")
        ]
        return result
    finally:
        InductorChoices.reduction_split_factor = previous


def constants(args):
    torch, backend, graphs, native = setup(args)
    from torch._inductor import lowering
    from torch._inductor import config
    from torch._inductor.virtualized import V
    from torch._inductor.index_propagation import SymPyOps

    calls = []
    previous = lowering._full
    previous_cast = lowering.to_dtype
    previous_pre = config.post_grad_custom_pre_pass
    previous_post = config.post_grad_custom_post_pass
    previous_symbolic_cast = SymPyOps.to_dtype
    stages = []
    symbolic_casts = []

    def symbolic_cast(value, dtype, src_dtype=None, use_compute_types=True):
        result = previous_symbolic_cast(value, dtype, src_dtype, use_compute_types)
        reject_fold = result is NotImplemented
        symbolic_casts.append(
            {
                "value": str(value.expr),
                "source": str(value.dtype),
                "dtype": str(dtype),
                "use_compute_types": use_compute_types,
                "result": str(result),
                "fold_vetoed": reject_fold,
            }
        )
        return result

    def capture(graph, phase):
        stages.append(
            {
                "phase": phase,
                "graph": str(graph),
                "nodes": [
                    {
                        "name": n.name,
                        "target": str(n.target),
                        "dtype": str(getattr(n.meta.get("val"), "dtype", None)),
                        "barrier": n.meta.get("low_precision_pointwise_barrier", False),
                    }
                    for n in graph.nodes
                ],
            }
        )

    def pre(graph):
        capture(graph, "pre_post_grad")
        if previous_pre:
            previous_pre(graph)

    def post(graph):
        if previous_post:
            previous_post(graph)
        capture(graph, "post_post_grad")

    def traced(fill_value, device, dtype, size):
        node = V.graph.current_node
        calls.append(
            {
                "kind": "full",
                "value": str(fill_value),
                "dtype": str(dtype),
                "size": str(size),
                "node": str(node),
                "barrier": node.meta.get("low_precision_pointwise_barrier", False),
            }
        )
        result = previous(fill_value, device, dtype, size)
        return result

    def cast(x, dtype, copy=False, use_compute_types=True):
        calls.append(
            {
                "kind": "to_dtype",
                "source": str(x.get_dtype()),
                "dtype": str(dtype),
                "use_compute_types": use_compute_types,
                "node": str(V.graph.current_node),
            }
        )
        return previous_cast(x, dtype, copy=copy, use_compute_types=use_compute_types)

    lowering._full = traced
    lowering.to_dtype = cast
    config.post_grad_custom_pre_pass = pre
    config.post_grad_custom_post_pass = post
    SymPyOps.to_dtype = staticmethod(symbolic_cast)
    functions = {
        "scalar_literal": lambda x, c: x * 0.333333333333,
        "fused_constructor": lambda x, c: torch.full_like(x, 0.333333333333) * x,
        "returned_constructor": lambda x, c: (
            torch.full_like(x, 0.333333333333),
            torch.full_like(x, 0.333333333333) * x,
        ),
        "prematerialized": lambda x, c: c * x,
        "explicit_roundtrip": lambda x, c: (
            torch.full_like(x, 0.333333333333, dtype=torch.float32).to(x.dtype) * x
        ),
        "constructor_only": lambda x, c: torch.full_like(x, 0.333333333333),
    }
    rows = []
    try:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.cat(
                [
                    torch.tensor(
                        [
                            65504.0,
                            3.0,
                            1.0,
                            -1.0,
                            0.0,
                            -0.0,
                            float("inf"),
                            -float("inf"),
                            float("nan"),
                        ],
                        device="cuda",
                        dtype=dtype,
                    ),
                    torch.randn(257, device="cuda", dtype=dtype),
                ]
            )
            c = torch.full_like(x, 0.333333333333)
            for name, fn in functions.items():
                compiled = torch.compile(fn, backend=backend, fullgraph=True)
                first = len(calls)
                first_stage = len(stages)
                first_symbolic = len(symbolic_casts)
                out = compiled(x, c)
                ref = fn(x, c)
                values = out if isinstance(out, tuple) else (out,)
                refs = ref if isinstance(ref, tuple) else (ref,)
                torch.cuda.synchronize()
                rows.append(
                    {
                        "variant": name,
                        "dtype": str(dtype),
                        "lowering_full": calls[first:],
                        "compiler_graphs": stages[first_stage:],
                        "symbolic_casts": symbolic_casts[first_symbolic:],
                        "outputs": [
                            {
                                "numeric_equal": bool(
                                    torch.all(
                                        (a == b) | (torch.isnan(a) & torch.isnan(b))
                                    )
                                ),
                                "classification_equal": torch.equal(
                                    torch.isnan(a), torch.isnan(b)
                                )
                                and torch.equal(torch.isinf(a), torch.isinf(b)),
                                "dtype_equal": a.dtype == b.dtype,
                                "zero_sign_equal": torch.equal(
                                    torch.signbit(a[(a == 0) & (b == 0)]),
                                    torch.signbit(b[(a == 0) & (b == 0)]),
                                ),
                                "compiled_values": a[:9].float().cpu().tolist(),
                                "eager_values": b[:9].float().cpu().tolist(),
                                "compiled_bytes": a.view(torch.uint8)
                                .cpu()
                                .numpy()
                                .tobytes()
                                .hex(),
                                "eager_bytes": b.view(torch.uint8)
                                .cpu()
                                .numpy()
                                .tobytes()
                                .hex(),
                            }
                            for a, b in zip(values, refs)
                        ],
                        "profile": profile(
                            torch,
                            lambda: compiled(x, c),
                            native,
                            args.output.with_stem(
                                args.output.stem + "-" + str(len(rows))
                            ),
                        ),
                    }
                )
                if args.precision_probe:
                    try:
                        for a, b in zip(values, refs):
                            torch.testing.assert_close(
                                a, b, rtol=0, atol=0, equal_nan=True
                            )
                            assert torch.equal(
                                torch.signbit(a[(a == 0) & (b == 0)]),
                                torch.signbit(b[(a == 0) & (b == 0)]),
                            )
                    except AssertionError:
                        failed = {
                            "accepted": False,
                            "records": rows,
                            "symbolic_casts": symbolic_casts,
                            "artifacts": artifacts() if args.impl == "cubecl" else [],
                            "environment": inventory(torch, args.impl),
                        }
                        args.output.with_suffix(".failed.json").write_text(
                            json.dumps(failed, indent=2)
                        )
                        raise
        result = {
            "counterfactual_precision_probe": args.precision_probe,
            "records": rows,
            "graphs": graphs,
            "environment": inventory(torch, args.impl),
            "lowering_calls": calls,
        }
        if args.impl == "cubecl":
            from stage5.backend import TRACE

            result.update(schedule_replay=TRACE, artifacts=artifacts())
        return result
    finally:
        lowering._full = previous
        lowering.to_dtype = previous_cast
        config.post_grad_custom_pre_pass = previous_pre
        config.post_grad_custom_post_pass = previous_post
        SymPyOps.to_dtype = staticmethod(previous_symbolic_cast)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("history", "constants"))
    parser.add_argument("--impl", choices=("cubecl", "triton"), required=True)
    parser.add_argument(
        "--history", choices=("large_first", "short_first"), default="large_first"
    )
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--precision-probe",
        action="store_true",
        help="Apply current precision corrections to the scoped legacy baseline",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = history(args) if args.task == "history" else constants(args)
    args.output.write_text(json.dumps(result, indent=2))
    print(args.task, args.impl, args.history, "PASS", args.output, flush=True)


if __name__ == "__main__":
    main()
