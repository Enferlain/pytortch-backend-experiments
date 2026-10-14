"""Regression matrix for the normal, generic low-precision overlay correction."""

import argparse
import ctypes as C
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("cubecl", "triton", "cpu"), default="cubecl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache/precision" / args.impl)
    os.environ["CUBECL_STAGE5_TRACE"] = "1"
    cube = args.impl == "cubecl"
    if cube:
        from stage1 import evidence

        evidence.guard()
    import torch
    from stage1.overlay import install

    install()
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor.index_propagation import SymPyOps, TypedExpr

    config.compile_threads = 1
    config.fx_graph_cache = False
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.suppress_errors = False
    device = "cpu" if args.impl == "cpu" else "cuda"
    graphs = []
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
    if cube:
        from stage5.backend import compile_backend, TRACE
        from stage5 import runtime

        compiler = partial(compile_backend, tuning="baseline")
        native = C.CDLL(str(PROJECT / "stage0/libstage0_launcher.so"))
        native.stage0_kernel_launches.restype = C.c_uint64
    else:

        def compiler(gm, inputs):
            with config.patch(patches):
                return compile_fx(gm, inputs, config_patches=patches)

        native = None

    def backend(gm, inputs):
        graphs.append(str(gm.graph))
        return compiler(gm, inputs)

    # Both the no-keyword ops protocol and the explicit storage request matter.
    component = []
    for dtype in (torch.float16, torch.bfloat16):
        v = TypedExpr(1 / 3, torch.float32)
        assert SymPyOps.to_dtype(v, dtype).expr == v.expr
        assert SymPyOps.to_dtype(v, dtype, use_compute_types=False) is NotImplemented
        component.append(
            {
                "dtype": str(dtype),
                "default_keeps_opmath": True,
                "storage_cast_vetoes_symbolic_fold": True,
            }
        )
    assert (
        SymPyOps.to_dtype(
            TypedExpr(9, torch.int64), torch.int32, use_compute_types=False
        ).expr
        == 9
    )

    functions = {
        "scalar_right_mul": lambda x, y: x * (1 / 3),
        "scalar_left_mul": lambda x, y: (1 / 3) * x,
        "scalar_add": lambda x, y: x + (1 / 3),
        "scalar_sub": lambda x, y: (1 / 3) - x,
        "scalar_compare": lambda x, y: x > (1 / 3),
        "tensor_compare": lambda x, y: x > torch.full_like(x, 1 / 3),
        "fused_constructor": lambda x, y: torch.full_like(x, 1 / 3) * x,
        "returned_constructor": lambda x, y: (
            torch.full_like(x, 1 / 3),
            torch.full_like(x, 1 / 3) * x,
        ),
        "prematerialized": lambda x, y: y * x,
        "explicit_roundtrip": lambda x, y: (
            torch.full_like(x, 1 / 3, dtype=torch.float32).to(x.dtype) * x
        ),
        "alpha_one": lambda x, y: torch.add(x, y, alpha=1),
        "sub_alpha": lambda x, y: torch.sub(x, y, alpha=1 / 3),
    }
    fills = (0.5, -0.0, 1 / 3, 0.00006105515625, 65504.0, float("inf"), float("nan"))
    for i, value in enumerate(fills):
        # Literal fixtures, not default-argument runtime float inputs. Dynamo
        # otherwise lifts -0.0/NaN scalar arguments into CPU tensor placeholders,
        # outside this CUDA experiment's input contract.
        literal = (
            repr(value)
            if value == value and abs(value) != float("inf")
            else ("float('nan')" if value != value else "float('inf')")
        )
        expression = "torch.full_like(x, " + literal + ")"
        functions["fill_only_" + str(i)] = eval(
            "lambda x, y: " + expression, {"torch": torch}
        )
        functions["fill_fused_" + str(i)] = eval(
            "lambda x, y: " + expression + " * x", {"torch": torch}
        )

    def exact(a, b):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
        assert a.dtype == b.dtype
        if a.is_floating_point():
            assert torch.equal(torch.isnan(a), torch.isnan(b))
            assert torch.equal(torch.isinf(a), torch.isinf(b))
            zeros = (a == 0) & (b == 0)
            assert torch.equal(torch.signbit(a[zeros]), torch.signbit(b[zeros]))

    rows, profiles, backward, negative = [], [], [], []
    result = {
        "accepted": False,
        "implementation": args.impl,
        "checks": rows,
        "component_checks": component,
        "backward": backward,
        "profiles": profiles,
        "negative": negative,
        "graphs": graphs,
    }
    try:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            torch.manual_seed(9241)
            x = torch.cat(
                (
                    torch.tensor(
                        [
                            65504,
                            3,
                            1,
                            -1,
                            0.0,
                            -0.0,
                            float("inf"),
                            -float("inf"),
                            float("nan"),
                            1 / 3,
                            0.333251953125,
                            0.333984375,
                        ],
                        device=device,
                        dtype=dtype,
                    ),
                    torch.randn(257, device=device, dtype=dtype),
                )
            )
            y = torch.full_like(x, 1 / 3)
            for name, fn in functions.items():
                compiled = torch.compile(fn, backend=backend, fullgraph=True)
                before = native.stage0_kernel_launches() if cube else None
                a, b = compiled(x, y), fn(x, y)
                aa = a if isinstance(a, tuple) else (a,)
                bb = b if isinstance(b, tuple) else (b,)
                for av, bv in zip(aa, bb):
                    if name == "sub_alpha":
                        # Eager CUDA uses fused alpha arithmetic; this backend
                        # deliberately retains separate arithmetic. This is an
                        # existing numerical-policy limit, not storage rounding.
                        tol = {
                            torch.float32: (2e-6, 2e-6),
                            torch.float16: (0.002, 0.001),
                            torch.bfloat16: (0.016, 0.008),
                        }[dtype]
                        torch.testing.assert_close(
                            av, bv, rtol=tol[0], atol=tol[1], equal_nan=True
                        )
                    else:
                        exact(av, bv)
                launches = native.stage0_kernel_launches() - before if cube else None
                if cube:
                    assert launches >= 1, "no CubeCL generated execution"
                rows.append(
                    {
                        "case": name,
                        "dtype": str(dtype),
                        "exact": name != "sub_alpha",
                        "native_launches": launches,
                    }
                )
                if (
                    name in ("fused_constructor", "scalar_right_mul")
                    and device == "cuda"
                ):
                    from stage5.attribution import profile

                    p = profile(
                        torch,
                        lambda: compiled(x, y),
                        native,
                        args.output.with_stem(args.output.stem + name + str(dtype)),
                    )
                    if cube:
                        assert len(p["kernels"]) == 1
                        assert all(
                            k["name"].startswith("cubecl5_") for k in p["kernels"]
                        )
                        kernel = next(
                            k
                            for k in runtime.CACHE.values()
                            if k.last is not None
                            and k.last.kernel.manifest["entrypoint"]
                            == p["kernels"][0]["name"]
                        )
                        p["artifact"] = {
                            "directory": str(kernel.last.directory),
                            "semantic": kernel.semantic,
                            "plan": kernel.last.kernel.plan.to_dict(),
                        }
                    profiles.append({"case": name, "dtype": str(dtype), **p})

            # Alpha !=1 is explicit FMA on CUDA: keep CubeCL's capability veto.
            def add_alpha(a, b):
                return torch.add(a, b, alpha=1 / 3)

            compiled = torch.compile(add_alpha, backend=backend, fullgraph=True)
            if cube:
                from stage5 import artifact

                begin = len(TRACE)
                before_compiles = artifact.COMPILER_INVOCATIONS
                before_launches = native.stage0_kernel_launches()
                try:
                    compiled(x, y)
                except Exception as error:
                    assert "unsupported" in str(error) and "fma" in str(error)
                    assert not any(e["event"] == "kernel" for e in TRACE[begin:])
                    assert artifact.COMPILER_INVOCATIONS == before_compiles
                    assert native.stage0_kernel_launches() == before_launches
                    negative.append(
                        {
                            "case": "add_alpha",
                            "dtype": str(dtype),
                            "boundary": "pre_fusion",
                            "error": str(error),
                        }
                    )
                else:
                    raise AssertionError("CubeCL silently accepted explicit FMA")
            else:
                if args.impl == "cpu":
                    tol = {
                        torch.float32: (2e-6, 2e-6),
                        torch.float16: (0.002, 0.001),
                        torch.bfloat16: (0.016, 0.008),
                    }[dtype]
                    torch.testing.assert_close(
                        compiled(x, y),
                        add_alpha(x, y),
                        rtol=tol[0],
                        atol=tol[1],
                        equal_nan=True,
                    )
                else:
                    exact(compiled(x, y), add_alpha(x, y))
                rows.append(
                    {
                        "case": "add_alpha",
                        "dtype": str(dtype),
                        "exact": args.impl != "cpu",
                    }
                )

            # No unit-gradient simplification: finite arbitrary tangents and inputs.
            for name in ("scalar_right_mul", "fused_constructor", "explicit_roundtrip"):
                fn = functions[name]
                a = torch.randn(257, device=device, dtype=dtype, requires_grad=True)
                b = a.detach().clone().requires_grad_(True)
                upstream = torch.randn_like(a)
                compiled = torch.compile(
                    fn, backend=backend, fullgraph=True, dynamic=True
                )
                ca, rb = (
                    compiled(a, torch.full_like(a, 1 / 3)),
                    fn(b, torch.full_like(b, 1 / 3)),
                )
                ca.backward(upstream)
                rb.backward(upstream)
                exact(ca, rb)
                exact(a.grad, b.grad)
                backward.append({"case": name, "dtype": str(dtype), "exact": True})

        if cube:
            # Force just the six representative descriptors through the actual
            # compiler again. The full corpus can legitimately reuse validated
            # persistent PTX; that alone is not fresh compiler-route evidence.
            from stage5 import artifact

            cold_identity = {
                **artifact.compiler_identity(0),
                "precision_validation_cache_partition": str(uuid.uuid4()),
            }
            cold_compiles = artifact.COMPILER_INVOCATIONS
            cold_launches = native.stage0_kernel_launches()
            cold = []
            for p in profiles:
                packet = p["artifact"]
                kernel = artifact.CompiledKernel(
                    packet["semantic"], 0, packet["plan"], cold_identity
                )
                assert not kernel.cache_hit
                dtype = getattr(torch, p["dtype"].removeprefix("torch."))
                x = torch.linspace(-5, 5, 269, device="cuda", dtype=dtype)
                x[:7] = torch.tensor(
                    [
                        65504.0,
                        0.0,
                        -0.0,
                        float("inf"),
                        -float("inf"),
                        float("nan"),
                        3.0,
                    ],
                    device="cuda",
                    dtype=dtype,
                )
                y = torch.empty_like(x)
                semantic = packet["semantic"]
                scalars = ()
                if semantic["scalars"]:
                    # These captured 1D fixtures declare just their extent.
                    # Validate the actual symbolic ABI rather than guessing it.
                    assert len(semantic["scalars"]) == 1
                    extent = {"op": "scalar", "slot": 0}
                    assert semantic["numel"] == extent
                    assert all(b["size"] == [extent] for b in semantic["buffers"])
                    scalars = (x.numel(),)
                kernel(scalars, x, y)
                exact(y, functions[p["case"]](x, x))
                assert sha(kernel.directory / "kernel.ptx") == sha(
                    Path(packet["directory"]) / "kernel.ptx"
                )
                cold.append(
                    {
                        "case": p["case"],
                        "dtype": p["dtype"],
                        "directory": str(kernel.directory),
                        "exact": True,
                        "runtime_scalars": scalars,
                        "ptx_identical_to_cached_representative": True,
                    }
                )
            assert artifact.COMPILER_INVOCATIONS - cold_compiles == 6
            assert native.stage0_kernel_launches() - cold_launches == 6
            result["cold_compiler_checks"] = cold
            # Dynamic reuse, physical tails, side stream and storage lifetime.
            compiled = torch.compile(
                functions["fused_constructor"],
                backend=backend,
                fullgraph=True,
                dynamic=True,
            )
            count = len(graphs)
            tail_calls = []
            for n in (255, 256, 257, 1048583, 0, 1, 257):
                a = torch.randn(n, device=device, dtype=torch.float16)
                before_graphs = len(graphs)
                before_launches = native.stage0_kernel_launches()
                exact(compiled(a, a), functions["fused_constructor"](a, a))
                launches = native.stage0_kernel_launches() - before_launches
                assert launches == (0 if n == 0 else 1)
                if n > 1 and n != 255:
                    assert len(graphs) == before_graphs
                tail_calls.append(
                    {
                        "size": n,
                        "new_graphs": len(graphs) - before_graphs,
                        "native_launches": launches,
                    }
                )
            tail_graphs = len(graphs) - count
            for _ in range(3):
                stream = torch.cuda.Stream()
                a = torch.randn(65539, device=device, dtype=torch.bfloat16)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    out = compiled(a, a)
                    ref = functions["fused_constructor"](a, a)
                    a.record_stream(stream)
                    del a
                    pressure = [torch.empty(65539, device=device) for _ in range(6)]
                stream.synchronize()
                exact(out, ref)
                del pressure
            result.update(
                tail_sizes=[255, 256, 257, 1048583, 0, 1, 257],
                tail_new_graphs=tail_graphs,
                tail_calls=tail_calls,
                side_stream_pressure_trials=3,
            )
            from stage5.attribution import artifacts

            result.update(
                schedule_replay=TRACE,
                artifacts=artifacts(),
                runtime_events=runtime.EVENTS,
            )
        result["accepted"] = True
    except Exception as error:
        result["failure"] = repr(error)
        raise
    finally:
        result.update(
            environment={
                "python": sys.version,
                "gil": sys._is_gil_enabled(),
                "PYTHON_GIL": os.environ.get("PYTHON_GIL"),
                "torch": torch.__version__,
                "device": device,
            },
            overlay=json.loads((PROJECT / "stage1/artifacts/overlay.json").read_text()),
            source_sha256=sha(__file__),
        )
        if cube:
            evidence.state("after_precision_checks")
            result["independence"] = evidence.inventory()
        args.output.write_text(json.dumps(result, indent=2))
    print(
        args.impl,
        len(rows),
        "forward checks /",
        len(backward),
        "backward checks PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
