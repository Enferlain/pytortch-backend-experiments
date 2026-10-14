"""Focused NVTX/System profiling and separate unprofiled warm measurements.

Instrumentation is scoped and restored. Production source/FP/launch policy stays
unchanged. Direct generated replay is a dispatch diagnostic, not a training graph.
"""

import argparse
from contextlib import contextmanager, nullcontext
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import time

from stage5.attribution import inventory, sha, summary

ROOT = Path(__file__).resolve().parent


@contextmanager
def annotations(torch, modules, kernels):
    from stage0.bridge import Kernel as Base
    from stage2.runtime import Kernel as Validation

    undo = []

    def patch(owner, name, label):
        original = getattr(owner, name)

        def measured(*args, **kwargs):
            with torch.cuda.nvtx.range(label):
                return original(*args, **kwargs)

        setattr(owner, name, measured)
        undo.append((owner, name, original))

    class TracedLock:
        def __init__(self, lock):
            self.lock = lock

        def __enter__(self):
            with torch.cuda.nvtx.range("lock_wait"):
                self.lock.acquire()
            return self

        def __exit__(self, *unused):
            self.lock.release()

    try:
        for runtime, artifact in modules:
            original_lock = runtime.LOCK
            runtime.LOCK = TracedLock(original_lock)
            undo.append((runtime, "LOCK", original_lock))
            patch(runtime.AutotunedKernel, "__call__", "selection_dispatch")
            patch(runtime, "signature", "proof_signature")
            patch(artifact.CompiledKernel, "__call__", "compiled_dispatch")
            patch(artifact.Kernel, "__call__", "kernel_dispatch")
            if hasattr(artifact.Kernel, "_validate"):
                patch(artifact.Kernel, "_validate", "validation")
            patch(artifact.Kernel, "_launch_native", "argument_binding")
        patch(Validation, "__call__", "validation")
        patch(Validation, "_prepare", "metadata")
        patch(Base, "_enqueue", "stream_readiness_lifetime")
        patch(torch.Tensor, "record_stream", "record_stream")
        patch(torch.cuda, "current_stream", "current_stream")
        patch(torch.cuda, "is_current_stream_capturing", "capture_query")
        patch(torch.cuda.Event, "query", "event_readiness")
        for kernel in kernels:
            patch(kernel.native, "stage5_launch", "native_enqueue")
        yield
    finally:
        for owner, name, original in reversed(undo):
            setattr(owner, name, original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impl", choices=("cubecl", "reference", "triton", "eager"), required=True
    )
    parser.add_argument("--size", choices=("tiny", "larger"), default="tiny")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUBECL_STAGE5_TRACE"] = "0"
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(
        ROOT / "cache/attribution" / args.output.stem
    )
    cube = args.impl in ("cubecl", "reference")
    if cube:
        from stage1 import evidence

        evidence.guard()
    import torch

    if cube:
        from stage1.overlay import install

        install()
        from stage5 import runtime, artifact
        from stage5.backend import compile_backend
        from stage5.launch_profile import reference_modules

        ref_artifact, ref_runtime = reference_modules()
        native = C.CDLL(str(ROOT.parent / "stage0/libstage0_launcher.so"))
        native.stage0_kernel_launches.restype = C.c_uint64
    else:
        from torch._inductor import config
        from torch._inductor.compile_fx import compile_fx

        def compile_backend(gm, inputs):
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

    from stage5.corpus import Case, Fixture, fixture

    torch._dynamo.config.suppress_errors = False
    torch.manual_seed(6142)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.size == "tiny":
        f = fixture(Case("training", (5, 9)))
    else:

        class Larger(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4096, 128))
                self.linear = torch.nn.Linear(128, 256, bias=False)

            def forward(self, idx):
                x = self.linear(torch.nn.functional.embedding(idx, self.weight))
                x = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5)
                return (
                    torch.nn.functional.dropout(x, 0.25, True).flatten() ** 2
                ).mean()

        model = Larger().cuda()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, foreach=False, fused=False
        )
        for p in model.parameters():
            optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p)
        f = Fixture(
            model,
            (torch.arange(256, device="cuda", dtype=torch.int64),),
            model=model,
            optimizer=optimizer,
            stochastic=True,
        )
    mode = "cubecl"
    proxies = {}
    calls = []
    original_compile = runtime.compile_kernel if cube else None

    class Proxy:
        def __init__(self, semantic, device, policy):
            self.current = original_compile(semantic, device, policy=policy)
            self.reference = ref_runtime.compile_kernel(semantic, device, policy=policy)

            def chosen(*unused):
                if self.current.last is None:
                    raise RuntimeError("optimized path must be warm before reference")
                return self.current.last.kernel.plan

            self.reference.select = chosen

        def __call__(self, scalars, *tensors):
            selected = self.reference if mode == "reference" else self.current
            if args.capture:
                from stage5.runtime import canonical

                label = (
                    "generated_dispatch:"
                    + __import__("hashlib")
                    .sha256(canonical(selected.semantic).encode())
                    .hexdigest()[:12]
                )
                with torch.cuda.nvtx.range(label):
                    out = selected(scalars, *tensors)
            else:
                out = selected(scalars, *tensors)
            if record_calls:
                calls.append((selected, tuple(scalars), tensors))
            return out

    def routed(semantic, device, *, policy="first_use"):
        key = (runtime.canonical(semantic), device, policy)
        if key not in proxies:
            proxies[key] = Proxy(semantic, device, policy)
        return proxies[key]

    record_calls = False
    if cube:
        runtime.compile_kernel = routed
    try:
        compiled = (
            f.fn
            if args.impl == "eager"
            else torch.compile(
                f.fn, backend=compile_backend, fullgraph=True, dynamic=True
            )
        )
        params = list(f.model.parameters())
        originals = [p.detach().clone() for p in params]
        rng = torch.cuda.get_rng_state().clone()

        def reset():
            torch.cuda.set_rng_state(rng)
            with torch.no_grad():
                for p, original in zip(params, originals):
                    p.copy_(original)
                    p.grad = None
                    f.optimizer.state[p]["momentum_buffer"].zero_()

        def scope(label):
            return torch.cuda.nvtx.range(label) if args.capture else nullcontext()

        def run():
            with scope("forward"):
                out = compiled(*f.args)
            with scope("backward"):
                out.backward()
            with scope("optimizer"):
                f.optimizer.step()
            return out

        reset()
        expected = run().detach().clone()
        grads = [p.grad.clone() for p in params]
        updated = [p.detach().clone() for p in params]
        after_rng = torch.cuda.get_rng_state().clone()
        if cube:
            mode = "reference"
            reset()
            reference_out = run()
            torch.testing.assert_close(reference_out, expected, rtol=2e-5, atol=2e-5)
            for p, grad, update in zip(params, grads, updated):
                torch.testing.assert_close(p.grad, grad, rtol=2e-5, atol=2e-5)
                torch.testing.assert_close(p, update, rtol=2e-5, atol=2e-5)
            assert torch.equal(after_rng, torch.cuda.get_rng_state())
        mode = args.impl
        reset()
        out = run()
        torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-5)
        for p, grad, update in zip(params, grads, updated):
            torch.testing.assert_close(p.grad, grad, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(p, update, rtol=2e-5, atol=2e-5)
        assert torch.equal(after_rng, torch.cuda.get_rng_state())
        for _ in range(20):
            reset()
            run()
        torch.cuda.synchronize()
        kernels = (
            [
                c.kernel
                for p in proxies.values()
                for c in (
                    p.reference.compiled if mode == "reference" else p.current.compiled
                ).values()
            ]
            if cube
            else []
        )
        modules = [(runtime, artifact), (ref_runtime, ref_artifact)] if cube else []
        compile_count = (
            artifact.COMPILER_INVOCATIONS + ref_artifact.COMPILER_INVOCATIONS
            if cube
            else 0
        )
        tunes = (
            sum(e["event"] == "tune" for e in runtime.EVENTS + ref_runtime.EVENTS)
            if cube
            else 0
        )
        count = native.stage0_kernel_launches() if cube else 0
        result = {
            "implementation": args.impl,
            "size": args.size,
            "capture": args.capture,
            "deep": args.deep,
            "reset_rng_correctness": True,
            "graph": "embedding -> linear -> RMS -> dropout -> squared mean; SGD outside compile",
            "input_shape": list(f.args[0].shape),
            "parameters": [list(p.shape) for p in params],
        }
        if cube:
            pairs = []
            for proxy in proxies.values():
                assert proxy.reference.kernel.plan == proxy.current.kernel.plan
                assert proxy.reference.directory != proxy.current.directory
                assert (proxy.reference.directory / "kernel.ptx").read_bytes() == (
                    proxy.current.directory / "kernel.ptx"
                ).read_bytes()
                pairs.append(
                    {
                        "plan": proxy.current.kernel.plan.to_dict(),
                        "semantic": proxy.current.semantic,
                        "current_directory": str(proxy.current.directory),
                        "reference_directory": str(proxy.reference.directory),
                        "ptx_sha256": sha(proxy.current.directory / "kernel.ptx"),
                    }
                )
            result["artifacts"] = pairs
        if args.capture:
            with (
                annotations(torch, modules, kernels)
                if args.deep and cube
                else nullcontext()
            ):
                with torch.cuda.nvtx.range("stage5_capture"):
                    for step in range(args.steps):
                        with scope("reset"):
                            reset()
                        with scope("completion"):
                            torch.cuda.synchronize()
                        with scope("step:" + str(step)):
                            run()
                        with scope("completion"):
                            torch.cuda.synchronize()
            result["steps"] = args.steps
        else:
            rounds = []
            for _ in range(3):
                samples = []
                for _ in range(args.samples):
                    reset()
                    torch.cuda.synchronize()
                    began = time.perf_counter_ns()
                    run()
                    samples.append((time.perf_counter_ns() - began) / 1000)
                rounds.append(summary(samples))
            result["host_enqueue"] = summary(
                [v for r in rounds for v in r["samples_us"]]
            )
            result["rounds"] = rounds
            if cube:
                # Same live buffers/current stream and ordinary validator. This
                # omits vendor calls/optimizer/graph plumbing, so is not a graph
                # substitute or an additive estimate of removable overhead.
                reset()
                record_calls = True
                run()
                record_calls = False
                values = []
                for _ in range(100):
                    torch.cuda.synchronize()
                    began = time.perf_counter_ns()
                    for selected, scalars, tensors in calls:
                        selected(scalars, *tensors)
                    values.append((time.perf_counter_ns() - began) / 1000)
                result["direct_generated_dispatch"] = {
                    **summary(values),
                    "calls": len(calls),
                    "qualification": "live validation and stream/lifetime handling; intermediate buffers reused, not equivalent full training semantics",
                }
        if cube:
            assert (
                compile_count
                == artifact.COMPILER_INVOCATIONS + ref_artifact.COMPILER_INVOCATIONS
            )
            assert tunes == sum(
                e["event"] == "tune" for e in runtime.EVENTS + ref_runtime.EVENTS
            )
            result["ordinary_native_launches"] = native.stage0_kernel_launches() - count
            result["warm_compiles"] = result["warm_tunes"] = 0
        result["environment"] = inventory(torch, "cubecl" if cube else args.impl)
        result["harness_sha256"] = sha(__file__)
        result["nsys_version"] = subprocess.check_output(
            ["nsys", "--version"], text=True
        ).strip()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(args.impl, args.size, "PASS", args.output, flush=True)
    finally:
        calls.clear()
        if cube:
            runtime.compile_kernel = original_compile


if __name__ == "__main__":
    main()
