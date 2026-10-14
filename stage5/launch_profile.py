"""Paired warm dispatch measurements; reference source is pinned Git data.

CPU instrumentation is installed only in diagnostic regions. Production does
not pay for its timers, wrappers, or cProfile. GPU profiling is separate.
"""

import argparse
from contextlib import contextmanager, nullcontext
import cProfile
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

REFERENCE = "df4320d093a76a49b7ab33d52cf10d22a9498bd0"
ROOT = Path(__file__).resolve().parent


def summary(values):
    mid = statistics.median(values)
    return {
        "median_us": mid,
        "mad_us": statistics.median(abs(x - mid) for x in values),
        "samples_us": values,
    }


def reference_modules():
    def route(source, old, new):
        if source.count(old) != 1:
            raise RuntimeError("pinned reference import cannot be routed: " + old)
        return source.replace(old, new, 1)

    modules = {}
    directory = ROOT / "cache/launch-reference" / REFERENCE
    directory.mkdir(parents=True, exist_ok=True)
    for part in ("legality", "artifact", "runtime"):
        source = subprocess.check_output(
            ["git", "show", f"{REFERENCE}:stage5/{part}.py"], cwd=ROOT.parent
        ).decode()
        path = directory / (part + ".py")
        path.write_text(source)
        # Frozen files provide honest adapter provenance and separate artifact
        # identities; only imports/cache location are routed for this probe.
        if part == "runtime":
            source = route(
                source,
                "from stage5 import artifact",
                "from stage5 import _launch_reference_artifact as artifact",
            )
        if part != "legality":
            source = route(
                source,
                "from stage5.legality import",
                "from stage5._launch_reference_legality import",
            )
        module = types.ModuleType("stage5._launch_reference_" + part)
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        module.ROOT = ROOT
        modules[part] = module
    return modules["artifact"], modules["runtime"]


@contextmanager
def phases(modules, kernels):
    """Nested exclusive wall costs; every child interval is subtracted once."""
    from stage0.bridge import Kernel as Base
    from stage2.runtime import Kernel as Validation

    totals, counts, stack, undo = {}, {}, [], []

    def patch(owner, name, label):
        original = getattr(owner, name)

        def measured(*args, **kwargs):
            frame = [time.perf_counter_ns(), 0]
            stack.append(frame)
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter_ns() - frame[0]
                stack.pop()
                if stack:
                    stack[-1][1] += elapsed
                totals[label] = totals.get(label, 0) + elapsed - frame[1]
                counts[label] = counts.get(label, 0) + 1

        setattr(owner, name, measured)
        undo.append((owner, name, original))

    for runtime, artifact in modules:
        patch(runtime.AutotunedKernel, "__call__", "selection_dispatch")
        patch(runtime, "signature", "proof_signature")
        patch(artifact.CompiledKernel, "__call__", "compiled_dispatch_and_trace")
        patch(artifact.Kernel, "__call__", "kernel_dispatch")
        if hasattr(artifact.Kernel, "_validate"):
            patch(artifact.Kernel, "_validate", "tensor_scalar_validation")
        patch(artifact.Kernel, "_launch_native", "native_argument_binding")
    patch(Validation, "__call__", "tensor_scalar_validation")
    patch(Base, "_enqueue", "current_stream_readiness_lifetime")
    patch(Validation, "_prepare", "metadata_lookup_setup")
    for kernel in kernels:
        patch(kernel.native, "stage5_launch", "driver_enqueue")
    try:
        yield {"exclusive_ns": totals, "calls": counts}
    finally:
        for owner, name, original in reversed(undo):
            setattr(owner, name, original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Attach CPU-clock/DWARF sampling only to warmed training calls",
    )
    parser.add_argument(
        "--output", default=str(ROOT / "artifacts/launch/performance.json")
    )
    args = parser.parse_args()
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / "cache/launch"))
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage1.overlay import install

    install()
    from stage5 import runtime, artifact
    from stage5.backend import compile_backend
    from stage5.corpus import Case, fixture

    ref_artifact, ref_runtime = reference_modules()
    mode = "optimized"
    proxies = {}
    original_compile = runtime.compile_kernel

    class Proxy:
        def __init__(self, semantic, device, policy):
            self.current = original_compile(semantic, device, policy=policy)
            self.reference = ref_runtime.compile_kernel(semantic, device, policy=policy)

            # Same selected physical plan in both paths. Reference dispatch is
            # measured, not reference tuning or a new search for a GPU winner.
            def reference_choice(*unused):
                if self.current.last is None:
                    raise RuntimeError(
                        "warm optimized dispatch before reference selection"
                    )
                return self.current.last.kernel.plan

            self.reference.select = reference_choice

        def __call__(self, scalars, *tensors):
            self.last_scalars = tuple(scalars)
            return (self.reference if mode == "reference" else self.current)(
                scalars, *tensors
            )

    def compile_proxy(semantic, device, *, policy="first_use"):
        key = (runtime.canonical(semantic), device, policy)
        if key not in proxies:
            proxies[key] = Proxy(semantic, device, policy)
        return proxies[key]

    try:
        runtime.compile_kernel = compile_proxy
        native = C.CDLL(str(ROOT.parent / "stage0/libstage0_launcher.so"))
        native.stage0_kernel_launches.restype = C.c_uint64
        torch.manual_seed(6142)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        output = {
            "reference_ref": REFERENCE,
            "reference_source_sha256": {
                p: hashlib.sha256(
                    subprocess.check_output(
                        ["git", "show", f"{REFERENCE}:stage5/{p}.py"], cwd=ROOT.parent
                    )
                ).hexdigest()
                for p in ("artifact", "runtime", "legality")
            },
            "source_sha256": {
                p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                for p in (
                    "artifact.py",
                    "runtime.py",
                    "legality.py",
                    "diagnostics.py",
                    "launch_profile.py",
                )
            },
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
                "PYTHON_GIL": os.environ.get("PYTHON_GIL"),
                "fp_patch_sha256": hashlib.sha256(
                    (ROOT.parent / "fp_hardening/cubecl-fp.patch").read_bytes()
                ).hexdigest(),
            },
            "records": [],
            "method": "Three rotated rounds; CPU host excludes completion/reset; exclusive phase timings and CUPTI GPU profiles separately instrumented",
        }
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)

        for case in (
            Case("add", (257,)),
            Case("add", (1048583,), "float16"),
            Case("row_sum", (64, 257)),
            Case("training", (5, 9)),
        ):
            proxies.clear()
            f = fixture(case)
            compiled = torch.compile(
                f.fn, backend=compile_backend, fullgraph=True, dynamic=True
            )
            state = torch.cuda.get_rng_state().clone()
            parameters = list(f.model.parameters()) if f.model else []
            originals = [p.detach().clone() for p in parameters]

            def reset():
                if f.stochastic:
                    torch.cuda.set_rng_state(state)
                with torch.no_grad():
                    for p, value in zip(parameters, originals):
                        p.copy_(value)
                        f.optimizer.state[p]["momentum_buffer"].zero_()
                        p.grad = None

            def run(phase_times=None):
                start = time.perf_counter_ns() if phase_times is not None else 0
                out = compiled(*f.args)
                if phase_times is not None:
                    phase_times["forward"].append(
                        (time.perf_counter_ns() - start) / 1000
                    )
                    start = time.perf_counter_ns()
                if f.optimizer:
                    out.backward()
                    if phase_times is not None:
                        phase_times["backward"].append(
                            (time.perf_counter_ns() - start) / 1000
                        )
                        start = time.perf_counter_ns()
                    f.optimizer.step()
                    if phase_times is not None:
                        phase_times["optimizer"].append(
                            (time.perf_counter_ns() - start) / 1000
                        )
                return out

            contexts = getattr(
                artifact, "detailed_trace", lambda enabled: nullcontext()
            )

            def legacy_validation(kernel, scalars, *tensors):
                # Exact legacy delegation, with the live class for super(): a
                # copied method's implicit __class__ still names the old class.
                with kernel._batch_lock:
                    return super(artifact.Kernel, kernel).__call__(scalars, *tensors)

            @contextmanager
            def trace_for(label):
                undo = []
                if label == "optimized_legacy_selection":
                    undo.append(
                        (
                            runtime.AutotunedKernel,
                            "__call__",
                            runtime.AutotunedKernel.__call__,
                        )
                    )
                    runtime.AutotunedKernel.__call__ = (
                        ref_runtime.AutotunedKernel.__call__
                    )
                if label == "optimized_legacy_validation_binding":
                    for method in ("__call__", "_launch_native"):
                        undo.append(
                            (artifact.Kernel, method, getattr(artifact.Kernel, method))
                        )
                        setattr(
                            artifact.Kernel,
                            method,
                            legacy_validation
                            if method == "__call__"
                            else getattr(ref_artifact.Kernel, method),
                        )
                try:
                    with contexts(label == "optimized_detail"):
                        yield
                finally:
                    for owner, method, previous in reversed(undo):
                        setattr(owner, method, previous)

            # Compile/tune optimized first, then load reference with identical plans.
            mode = "optimized"
            reset()
            expected = run().detach().clone()
            expected_grads = [p.grad.detach().clone() for p in parameters]
            expected_parameters = [p.detach().clone() for p in parameters]
            after_rng = torch.cuda.get_rng_state().clone()
            mode = "reference"
            reset()
            actual = run()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            for p, grad in zip(parameters, expected_grads):
                torch.testing.assert_close(p.grad, grad, rtol=0, atol=0, equal_nan=True)
            for p, value in zip(parameters, expected_parameters):
                torch.testing.assert_close(p, value, rtol=0, atol=0, equal_nan=True)
            assert torch.equal(after_rng, torch.cuda.get_rng_state())
            hashes = []
            for proxy in proxies.values():
                assert proxy.current.kernel.plan == proxy.reference.kernel.plan
                assert proxy.current.directory != proxy.reference.directory
                current_ptx = (proxy.current.directory / "kernel.ptx").read_bytes()
                assert (
                    current_ptx
                    == (proxy.reference.directory / "kernel.ptx").read_bytes()
                )
                hashes.append(hashlib.sha256(current_ptx).hexdigest())
            labels = (
                ["reference"]
                if args.baseline_only
                else [
                    "reference",
                    "optimized_detail",
                    "optimized",
                    "optimized_legacy_selection",
                    "optimized_legacy_validation_binding",
                ]
            )
            for label in labels:
                mode = "reference" if label == "reference" else "optimized"
                with trace_for(label):
                    for _ in range(10):
                        reset()
                        run()
            torch.cuda.synchronize()
            cold_count = (
                artifact.COMPILER_INVOCATIONS + ref_artifact.COMPILER_INVOCATIONS
            )
            tune_count = sum(
                e["event"] == "tune" for e in runtime.EVENTS + ref_runtime.EVENTS
            )
            record = {
                **case.to_dict(),
                "ptx_sha256": hashes,
                "paired_exact_correctness": True,
                "modes": {},
            }
            samples = {label: [] for label in labels}
            rounds = {label: [] for label in labels}
            for r in range(3):
                for label in labels[r:] + labels[:r]:
                    mode = "reference" if label == "reference" else "optimized"
                    round_values = []
                    with trace_for(label):
                        for _ in range(args.samples):
                            reset()
                            torch.cuda.synchronize()
                            start = time.perf_counter_ns()
                            run()
                            round_values.append((time.perf_counter_ns() - start) / 1000)
                    samples[label].extend(round_values)
                    rounds[label].append(summary(round_values))
            assert (
                cold_count
                == artifact.COMPILER_INVOCATIONS + ref_artifact.COMPILER_INVOCATIONS
            )
            assert tune_count == sum(
                e["event"] == "tune" for e in runtime.EVENTS + ref_runtime.EVENTS
            )
            for label in labels:
                mode = "reference" if label == "reference" else "optimized"
                phase_times = {k: [] for k in ("forward", "backward", "optimizer")}
                kernels = [
                    c.kernel
                    for p in proxies.values()
                    for c in (
                        p.reference.compiled
                        if mode == "reference"
                        else p.current.compiled
                    ).values()
                ]
                reset()
                # The selection ablation uses reference globals/lock in this serial
                # probe. Instrument both signature globals to account for it.
                with (
                    trace_for(label),
                    phases(
                        [(runtime, artifact), (ref_runtime, ref_artifact)], kernels
                    ) as attribution,
                ):
                    for _ in range(20):
                        reset()
                        run(phase_times)
                profile = cProfile.Profile()
                reset()
                with trace_for(label):
                    profile.enable()
                    for _ in range(20):
                        run()
                    profile.disable()
                top = sorted(
                    profile.getstats(), key=lambda v: v.inlinetime, reverse=True
                )[:20]
                cpu_top = [
                    {
                        "code": str(v.code)
                        if isinstance(v.code, str)
                        else f"{v.code.co_filename}:{v.code.co_firstlineno}:{v.code.co_name}",
                        "calls": v.callcount,
                        "exclusive_us": v.inlinetime * 1e6,
                    }
                    for v in top
                ]
                gpu = []
                for _ in range(3):
                    reset()
                    torch.cuda._sleep(50_000_000)
                    torch.cuda.synchronize()
                    count = native.stage0_kernel_launches()
                    with (
                        trace_for(label),
                        torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as prof,
                    ):
                        run()
                    events = [
                        e
                        for e in prof.profiler.kineto_results.events()
                        if e.device_type() == torch.autograd.DeviceType.CUDA
                        and e.activity_type() == "kernel"
                    ]
                    generated = [e for e in events if e.name().startswith("cubecl5_")]
                    assert generated, "No CubeCL generated work in profiler region"
                    assert len(generated) == native.stage0_kernel_launches() - count
                    gpu.append(
                        {
                            "kernel_count": len(events),
                            "generated_count": len(generated),
                            "kernel_names": [e.name() for e in events],
                            "kernel_sum_us": sum(
                                (e.end_ns() - e.start_ns()) / 1000 for e in events
                            ),
                        }
                    )
                record["modes"][label] = {
                    **summary(samples[label]),
                    "rounds": rounds[label],
                    "exclusive_phases": attribution,
                    "step_phases_us": {
                        k: summary(v) for k, v in phase_times.items() if v
                    },
                    "cprofile_top": cpu_top,
                    "gpu_profiles": gpu,
                }
            record["warm_compiles"] = 0
            record["warm_tunes"] = 0
            censuses = {
                (v["kernel_count"], v["generated_count"])
                for m in record["modes"].values()
                for v in m["gpu_profiles"]
            }
            if len(censuses) != 1:
                (path.parent / "profile-incomplete.json").write_text(
                    json.dumps(record, indent=2)
                )
                raise AssertionError("Warm kernel census changed across dispatch modes")
            if case.family != "training":
                from stage2.indexing import evaluate

                mode = "optimized"
                proxy = next(iter(proxies.values()))
                kernel = proxy.current.kernel
                scalars = proxy.last_scalars
                out = kernel.semantic["buffers"][-1]
                destination = torch.empty_strided(
                    tuple(evaluate(e, scalars) for e in out["size"]),
                    tuple(evaluate(e, scalars) for e in out["stride"]),
                    dtype=f.args[0].dtype,
                    device=f.args[0].device,
                )
                values = []
                with kernel.prepared(
                    scalars,
                    f.args[0],
                    destination,
                ):
                    for _ in range(100):
                        torch.cuda.synchronize()
                        began = time.perf_counter_ns()
                        kernel.batch(1)
                        values.append((time.perf_counter_ns() - began) / 1000)
                record["prepared_native_enqueue"] = {
                    **summary(values),
                    "classification": "Validated fixed buffers/stream; diagnostic lower bound, not production dispatch",
                }
            if args.perf and case.family == "training":
                mode = "optimized"
                perf_path = ROOT / "cache/launch-perf.data"
                command = [
                    "perf",
                    "record",
                    "-e",
                    "cpu-clock:u",
                    "-F",
                    "199",
                    "--call-graph",
                    "dwarf",
                    "-o",
                    str(perf_path),
                    "-p",
                    str(os.getpid()),
                ]
                with (ROOT / "launch-perf.log").open("w") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=log)
                    time.sleep(0.3)
                    if process.poll() is not None:
                        raise RuntimeError(
                            "perf attach failed; inspect launch-perf.log"
                        )
                    with trace_for("optimized"):
                        for _ in range(300):
                            reset()
                            run()
                    process.send_signal(__import__("signal").SIGINT)
                    process.wait(timeout=30)
                    if process.returncode not in (0, -2):
                        raise RuntimeError("perf record failed")
                report = subprocess.check_output(
                    [
                        "perf",
                        "report",
                        "--stdio",
                        "--no-children",
                        "--percent-limit",
                        "1",
                        "-i",
                        str(perf_path),
                    ],
                    text=True,
                )
                (path.parent / "perf-summary.txt").write_text(report)
                record["perf"] = {
                    "command": command,
                    "version": subprocess.check_output(
                        ["perf", "--version"], text=True
                    ).strip(),
                    "qualification": "CPU-clock samples of warmed optimized training including reset; separate diagnostic, not phase timing",
                }
            output["records"].append(record)
            evidence.state("after_launch_profile")
            output["independence"] = evidence.inventory()
            path.write_text(json.dumps(output, indent=2))
            print(
                case.name,
                {k: round(v["median_us"], 2) for k, v in record["modes"].items()},
                flush=True,
            )
    finally:
        runtime.compile_kernel = original_compile


if __name__ == "__main__":
    main()
