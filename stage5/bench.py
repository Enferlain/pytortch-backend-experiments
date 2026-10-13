"""Sequential fixed-corpus measurements; profiler timing is separate evidence."""

import argparse
import ctypes as C
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--impl",
    choices=(
        "cubecl_baseline",
        "aten",
        "triton_matched",
        "triton_default",
        "cubecl_stage5",
    ),
    required=True,
)
parser.add_argument("--families", default="")
parser.add_argument("--names", default="")
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--samples", type=int, default=20)
parser.add_argument("--kernel-samples", type=int, default=7)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--output")
args = parser.parse_args()
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache" / args.impl)
(ROOT / "artifacts").mkdir(parents=True, exist_ok=True)

gil_before = sys._is_gil_enabled()
cubecl = args.impl.startswith("cubecl")
if cubecl:
    from stage1 import evidence

    evidence.guard()
import torch

if cubecl:
    from stage1.overlay import install

    install()
    if args.impl == "cubecl_baseline":
        from stage4.backend import compile_backend
        from stage4 import runtime
    else:
        from stage5.backend import compile_backend
        from stage5 import runtime
    from stage3 import runtime as bridge

    evidence.state("after_backend")
    native = C.CDLL(str(PROJECT / "stage0/libstage0_launcher.so"))
    native.stage0_kernel_launches.restype = C.c_uint64

from stage5.corpus import cases, fixture

torch._dynamo.config.recompile_limit = 128
torch._dynamo.config.suppress_errors = False
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def compile_fn(fn):
    if args.impl == "aten":
        return fn
    if cubecl:
        return torch.compile(fn, backend=compile_backend, fullgraph=True, dynamic=True)
    from torch._inductor import config

    patches = {
        "triton.cudagraphs": False,
        "compile_threads": 1,
        "fx_graph_cache": False,
    }
    if args.impl == "triton_matched":
        patches.update(
            {
                "emulate_precision_casts": True,
                "joint_graph_constant_folding": False,
                "eager_numerics.division_rounding": True,
                "eager_numerics.disable_ftz": True,
                "eager_numerics.use_pytorch_libdevice": True,
            }
        )

    # Backend call owns the config scope through lazy AOT backward compilation.
    def backend(gm, inputs):
        from torch._inductor.compile_fx import compile_fx

        with config.patch(patches):
            return compile_fx(gm, inputs, config_patches=patches)

    return torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)


def tolerances(case):
    if case.dtype == "float16":
        return {"rtol": 0.002, "atol": 0.001, "equal_nan": True}
    if case.dtype == "bfloat16":
        return {"rtol": 0.016, "atol": 0.008, "equal_nan": True}
    reduce = "sum" in case.family or "rms" in case.family or "mean" in case.family
    return {
        "rtol": 2e-5 if reduce else 2e-6,
        "atol": 2e-5 if reduce else 2e-6,
        "equal_nan": True,
    }


class IncompleteProfile(RuntimeError):
    pass


def _census_once(fn, label):
    begin = len(runtime.EVENTS) if cubecl else 0
    count = native.stage0_kernel_launches() if cubecl else 0
    # Equal GPU activity before each capture avoids crediting an autotuner's
    # preceding workload/power state as a physical-kernel optimization.
    torch.cuda._sleep(50_000_000)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        fn()
    path = ROOT / "artifacts" / (label + "-profiler.json")
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    if not kernels:
        raise IncompleteProfile("no GPU kernel recorded for nonempty workload")
    owners = {}
    for e in events:
        external = e.get("args", {}).get("External id")
        if e.get("cat") == "cpu_op" and external is not None:
            owners[str(external)] = e["name"]
    generated = (
        [
            e["entrypoint"]
            for e in runtime.EVENTS[begin:]
            if e["event"] == "launch" and e["grid"][0]
        ]
        if cubecl
        else []
    )
    details = [
        {
            "name": e["name"],
            "duration_us": e["dur"],
            "owner": owners.get(str(e.get("args", {}).get("External id"))),
            "cubecl_generated": e["name"] in generated,
        }
        for e in kernels
    ]
    phases = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e.get("name", "").startswith("stage5_phase_")
    ]
    correlations = {}
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            for phase in phases:
                if phase["ts"] <= e["ts"] < phase["ts"] + phase["dur"]:
                    correlations[e.get("args", {}).get("correlation")] = phase["name"]
    for e, detail in zip(kernels, details):
        detail["phase"] = correlations.get(e.get("args", {}).get("correlation"))
    if cubecl:
        if not (
            sum(e["cubecl_generated"] for e in details)
            == len(generated)
            == native.stage0_kernel_launches() - count
        ):
            raise IncompleteProfile("profiler/native generated kernel counts differ")
    return {
        "kernels": details,
        "kernel_count": len(details),
        "profiled_kernel_sum_us": sum(e["duration_us"] for e in details),
        "timing_scope": "Profiler-instrumented GPU durations, not directly comparable to unprofiled event intervals",
        "raw_trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_trace": str(path),
    }


def census(fn, label, reset):
    failures = []
    for _ in range(3):
        reset()
        try:
            result = _census_once(fn, label)
            result["incomplete_profile_attempts"] = failures
            return result
        except IncompleteProfile as error:
            failures.append(str(error))
    raise IncompleteProfile(str(failures))


def measure(case):
    torch.manual_seed(6142)
    f = fixture(case)
    c = compile_fn(f.fn)
    state = torch.cuda.get_rng_state().clone()
    parameters = list(f.model.parameters()) if f.model else []
    originals = [p.detach().clone() for p in parameters]
    momenta = [f.optimizer.state[p]["momentum_buffer"] for p in parameters]

    def reset():
        if f.stochastic:
            torch.cuda.set_rng_state(state)
        with torch.no_grad():
            for p, original, momentum in zip(parameters, originals, momenta):
                p.copy_(original)
                momentum.zero_()
                p.grad = None
        for x in f.args:
            if x.requires_grad:
                x.grad = None

    launch_examples = []

    def run(profile_scopes=False):
        begin = len(runtime.EVENTS) if cubecl else 0
        scope = (
            torch.profiler.record_function
            if profile_scopes
            else lambda _: nullcontext()
        )
        with scope("stage5_phase_forward"):
            out = c(*f.args)
        if f.optimizer:
            with scope("stage5_phase_backward"):
                out.backward()
            with scope("stage5_phase_optimizer"):
                f.optimizer.step()
        elif f.upstream is not None:
            with scope("stage5_phase_backward"):
                out.backward(f.upstream)
        if cubecl and not launch_examples:
            launch_examples.extend(
                e for e in runtime.EVENTS[begin:] if e["event"] == "launch"
            )
        return out

    reset()
    cold_start = time.perf_counter_ns()
    out = run()
    torch.cuda.synchronize()
    cold_us = (time.perf_counter_ns() - cold_start) / 1000
    after = torch.cuda.get_rng_state().clone()
    saved_grads = [x.grad.detach().clone() for x in f.args if x.requires_grad]
    reset()
    repeated = run()
    torch.testing.assert_close(out, repeated, **tolerances(case))
    assert torch.equal(after, torch.cuda.get_rng_state())
    if not f.stochastic:
        reference = f.fn(*f.args)
        torch.testing.assert_close(out, reference, **tolerances(case))
        if f.upstream is not None:
            leaves = [x for x in f.args if x.requires_grad]
            gradients = torch.autograd.grad(reference, leaves, f.upstream)
            for got, expected in zip(saved_grads, gradients):
                torch.testing.assert_close(got, expected, **tolerances(case))
    elif case.family == "dropout_backward":
        mask = out != 0
        torch.testing.assert_close(
            out, f.args[0] * mask * (1 / 0.75), **tolerances(case)
        )
        torch.testing.assert_close(
            saved_grads[0], f.upstream * mask * (1 / 0.75), **tolerances(case)
        )

    for _ in range(5):
        reset()
        run()
    torch.cuda.synchronize()
    gpu, wall = [], []
    for _ in range(args.samples):
        reset()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        t0 = time.perf_counter_ns()
        run()
        wall.append((time.perf_counter_ns() - t0) / 1000)
        end.record()
        end.synchronize()
        gpu.append(start.elapsed_time(end) * 1000)
    result = {
        **case.to_dict(),
        "implementation": args.impl,
        "cold_compile_first_run_us": cold_us,
        "event_interval_median_us": statistics.median(gpu),
        "event_interval_samples_us": gpu,
        "host_enqueue_median_us": statistics.median(wall),
        "host_enqueue_samples_us": wall,
        "input_layouts": [
            {
                "shape": list(x.shape),
                "stride": list(x.stride()),
                "storage_offset": x.storage_offset(),
            }
            for x in f.args
        ],
        "correctness": "eager output/arbitrary-gradient comparison"
        if not f.stochastic
        else "per-backend restored-state replay; dropout mask gradient where applicable; full training reference in Stage4 regression",
        "rng_replay_state_equal": True,
        "timing_scope": "Unprofiled device event around Python invocation, includes enqueue gaps; host enqueue excludes completion",
    }
    if cubecl:
        result["physical_launches"] = launch_examples
    if args.profile:
        reset()
        result["census"] = census(lambda: run(True), case.name + "-" + args.impl, reset)
        # Reset model/RNG/grad state outside each profiling region so reset
        # copies and clears are never counted as workload kernels.
        device_samples = []
        counts = []
        attempts = []
        for _ in range(args.kernel_samples):
            observed = []
            for _ in range(3):
                reset()
                torch.cuda._sleep(50_000_000)
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    run()
                kernels = [
                    e
                    for e in profiler.profiler.kineto_results.events()
                    if e.device_type() == torch.autograd.DeviceType.CUDA
                    and e.activity_type() == "kernel"
                ]
                observed.append(len(kernels))
                if len(kernels) == result["census"]["kernel_count"]:
                    break
            else:
                raise IncompleteProfile(
                    f"repeated GPU kernel census changed: {observed}"
                )
            attempts.append(observed)
            counts.append(len(kernels))
            device_samples.append(
                sum((e.end_ns() - e.start_ns()) / 1000 for e in kernels)
            )
        if counts and any(n != result["census"]["kernel_count"] for n in counts):
            raise RuntimeError("repeated GPU kernel census changed")
        if device_samples:
            result["kernel_duration_samples_us"] = device_samples
            result["kernel_duration_median_us"] = statistics.median(device_samples)
            result["kernel_duration_mad_us"] = statistics.median(
                abs(v - statistics.median(device_samples)) for v in device_samples
            )
            result["kernel_profile_attempt_counts"] = attempts
    return result


selected = [
    c for c in cases() if not args.families or c.family in args.families.split(",")
]
if args.names:
    selected = [c for c in selected if c.name in args.names.split(",")]
if args.limit:
    selected = selected[: args.limit]
(ROOT / "artifacts/corpus.json").write_text(
    json.dumps([c.to_dict() for c in cases()], indent=2)
)
result = {
    "records": [],
    "failures": [],
    "implementation": args.impl,
    "environment": {
        "python": sys.version,
        "gil_before": gil_before,
        "torch": torch.__version__,
        "torch_commit": torch.version.git_version,
        "cuda": torch.version.cuda,
        "device": str(torch.cuda.get_device_properties(0)),
        "PYTHON_GIL": os.environ.get("PYTHON_GIL"),
        "TRITON_LIBDEVICE_PATH": os.environ.get("TRITON_LIBDEVICE_PATH"),
        "TRITON_LIBCUDA_PATH": os.environ.get("TRITON_LIBCUDA_PATH"),
    },
    "gpu_snapshot": subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem",
            "--format=csv,noheader",
        ],
        text=True,
    ),
}
output = (
    Path(args.output)
    if args.output
    else ROOT / "artifacts" / ("baseline-" + args.impl + ".json")
)
for case in selected:
    try:
        result["records"].append(measure(case))
        print(
            case.name,
            "PASS",
            round(result["records"][-1]["event_interval_median_us"], 2),
            "us",
            flush=True,
        )
    except Exception as e:
        result["failures"].append(
            {**case.to_dict(), "error": str(e), "type": type(e).__name__}
        )
        print(case.name, "FAIL", str(e), flush=True)
    output.write_text(json.dumps(result, indent=2))
if cubecl:
    evidence.state("after_benchmark")
    result["independence"] = evidence.inventory()
    result["compiler_invocations"] = (
        bridge.COMPILER_INVOCATIONS
        if args.impl == "cubecl_baseline"
        else __import__(
            "stage5.artifact", fromlist=["COMPILER_INVOCATIONS"]
        ).COMPILER_INVOCATIONS
    )
    result["native_launches"] = native.stage0_kernel_launches()
    result["artifacts"] = [
        {
            "semantic": c.semantic,
            "manifest": c.kernel.manifest,
            "attributes": c.kernel.attributes,
            "directory": str(c.directory),
        }
        for c in runtime.CACHE.values()
    ]
result["environment"]["gil_after"] = sys._is_gil_enabled()
output.write_text(json.dumps(result, indent=2))
assert not result["failures"], result["failures"]
