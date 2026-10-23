"""Run one scoreboard workload on one execution path and write a JSON result.

Paths:
  eager   PyTorch eager in the isolated 3.14t environment (GIL must stay off)
  triton  Inductor + Triton with the workload's exact compile settings (GIL is re-enabled)
  cubecl  Inductor + the CubeCL backend (cubecl_torch.inductor), GIL must stay off

CubeCL rejections:
  --on-reject fail    stop at the first rejected region and report it (default)
  --on-reject record  record every rejected region, run it eagerly and continue.
                      Inventory only: rejected regions are reported as unsupported,
                      never as CubeCL coverage, and timings are labelled accordingly.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--path", choices=("eager", "triton", "cubecl"), required=True)
parser.add_argument("--workload", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--on-reject", choices=("fail", "record"), default="fail")
parser.add_argument("--batch", type=int)
parser.add_argument("--resolution", type=int)
parser.add_argument("--blocks", type=int)
args = parser.parse_args()

# Fresh compiler caches so first-use cost is measured, not a cache hit.
# They are removed at exit: nothing reads them after the run.
os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(
    prefix=f"scoreboard-{args.path}-"
)
if args.path == "cubecl":
    os.environ["CUBECL_TORCH_CACHE_DIR"] = tempfile.mkdtemp(
        prefix="scoreboard-cubecl_torch-"
    )
for _cache in ("TORCHINDUCTOR_CACHE_DIR", "CUBECL_TORCH_CACHE_DIR"):
    if _cache in os.environ:
        atexit.register(shutil.rmtree, os.environ[_cache], ignore_errors=True)
result = {
    "path": args.path,
    "workload": args.workload,
    "on_reject": args.on_reject if args.path == "cubecl" else None,
    "status": "running",
    "environment": {"python": sys.version, "platform": platform.platform(), "gil": {}},
    "config_deviations": [],
}


def gil(phase):
    result["environment"]["gil"][phase] = sys._is_gil_enabled()


def write():
    Path(args.out).write_text(json.dumps(result, indent=1, default=str))


gil("start")
if args.path == "cubecl":
    from cubecl_torch.runtime import guard

    guard.install()  # Triton must be absent and unimportable.
import torch

if args.path == "cubecl":
    import cubecl_torch.inductor

    # Must precede any torch._inductor import (model imports pull it in).
    cubecl_torch.inductor.install()
    cubecl_compile = cubecl_torch.inductor.compile_backend
gil("after_backend_import")

from scoreboard.workloads import WORKLOADS, source_revision

workload = WORKLOADS[args.workload]
device = torch.device(args.device)
options = {
    k: v
    for k in ("batch", "resolution", "blocks")
    if (v := getattr(args, k)) is not None
}
result["environment"].update(
    torch=torch.__version__,
    device=torch.cuda.get_device_name(device),
    cubecl_backend="cubecl_torch.inductor (native release build)"
    if args.path == "cubecl"
    else None,
    revisions={
        "experiments": source_revision(ROOT),
        workload.name + "_source": source_revision(workload.source),
    },
)
result["compile_requested"] = dict(workload.compile)

torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.accumulated_recompile_limit = 4096
if args.path != "eager":
    import torch._functorch.config

    # Lower backward inside the backend call so CubeCL rejections are attributable
    # per region; applied to both compiled paths so first-use costs stay comparable.
    torch._functorch.config.force_non_lazy_backward_lowering = True


def clocks():
    try:
        q = "clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw"
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu=name,{q}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
        return out.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None


# ---- CubeCL region accounting -------------------------------------------------
REGIONS = []


def region_ops(gm):
    return dict(
        Counter(
            str(n.target)
            for n in gm.graph.nodes
            if n.op in ("call_function", "call_method")
        )
    )


def cubecl_backend(gm, inputs):
    kinds = Counter(
        f"{type(x).__name__}:{getattr(x, 'device', '')}:{getattr(x, 'dtype', '')}"
        for x in inputs
    )
    region = {
        "index": len(REGIONS),
        "inputs": dict(kinds),
        "ops": region_ops(gm),
        "status": "compiling",
    }
    REGIONS.append(region)
    try:
        compiled = cubecl_compile(gm, inputs)
    except torch._dynamo.exc.RestartAnalysis:
        # Dynamo control flow (e.g. specializing a float the tensorify pass
        # cannot keep symbolic), not a rejection: Dynamo retraces the region.
        region["status"] = "restarted"
        raise
    except Exception as e:  # Every rejection stays visible in the result.
        region.update(status="rejected", reason=f"{type(e).__name__}: {e}"[:2000])
        if args.on_reject == "fail":
            raise
        return gm.forward
    region["status"] = "cubecl"
    return compiled


def compile_kwargs():
    requested = dict(workload.compile)
    if args.path == "triton":
        return {"backend": "inductor", **requested}
    mode = requested.pop("mode", None)
    if mode not in (None, "default"):
        result["config_deviations"].append(
            f"mode={mode!r} dropped: the CubeCL backend does not support it "
            "(the backend forces triton.cudagraphs=False; CUDA graph capture unsupported)"
        )
    return {"backend": cubecl_backend, **requested}


# ---- Measurement helpers --------------------------------------------------------
# Attention is split from other vendor kernels so the SDPA implementation actually used
# (flash, memory-efficient, cuDNN, ...) is visible and comparable across paths.
ATTENTION = re.compile(r"flash|fmha|attention|sdpa|attn", re.IGNORECASE)
VENDOR = re.compile(
    r"gemm|cutlass|cublas|xmma|cudnn|conv|wgrad|dgrad|ampere_|sm\d\d_|winograd|implicit",
    re.IGNORECASE,
)


def kernel_class(name):
    if name.startswith("cubecl"):
        return "cubecl"
    if name.startswith("triton_"):
        return "triton_" + name.split("_")[1]
    if ATTENTION.search(name):
        return "attention"
    if VENDOR.search(name):
        return "vendor_library"
    return "aten_other"  # eager glue outside compiled regions, extern ATen calls, fallbacks


def summarize(values):
    values = sorted(values)
    return {
        "median": statistics.median(values),
        "min": values[0],
        "max": values[-1],
        "p10": values[max(0, int(len(values) * 0.1))],
        "p90": values[min(len(values) - 1, int(len(values) * 0.9))],
        "n": len(values),
    }


def grad_norms(parameters):
    return torch.stack(
        [
            p.grad.float().norm()
            if p.grad is not None
            else torch.zeros((), device=device)
            for p in parameters
        ]
    ).cpu()


def relative(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def timed_steps(step, count):
    gpu, wall = [], []
    for _ in range(count):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        start.record()
        step()
        end.record()
        torch.cuda.synchronize(device)
        wall.append((time.perf_counter() - t0) * 1000)
        gpu.append(start.elapsed_time(end))
    return summarize(gpu), summarize(wall)


def kernel_census(step):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize(device)
    events = prof.profiler.kineto_results.events()
    kernels = [
        (e.name(), (e.end_ns() - e.start_ns()) / 1000)
        for e in events
        if e.device_type() == torch.autograd.DeviceType.CUDA
        and e.activity_type() == "kernel"
    ]
    # Direct evidence that CUDA graph replay happened (mode="reduce-overhead" alone does not prove it).
    graph_launches = sum("cudaGraphLaunch" in e.name() for e in events)
    by_class, by_name = {}, {}
    for name, us in kernels:
        c = by_class.setdefault(kernel_class(name), {"calls": 0, "us": 0.0})
        c["calls"] += 1
        c["us"] += us
        n = by_name.setdefault(
            name, {"class": kernel_class(name), "calls": 0, "us": 0.0}
        )
        n["calls"] += 1
        n["us"] += us
    top = sorted(by_name.items(), key=lambda kv: -kv[1]["us"])[:40]
    return {
        "total_us": sum(us for _, us in kernels),
        "kernels": len(kernels),
        "by_class": by_class,
        "cuda_graph_launches": graph_launches,
        "attention_kernels": sorted(
            {k[:200] for k, v in by_name.items() if v["class"] == "attention"}
        ),
        "top_kernels": [{"name": k[:200], **v} for k, v in top],
    }


# ---- Run --------------------------------------------------------------------------
stage = "build"
try:
    torch.cuda.set_device(device)
    instance = workload.build(device, options)
    result["settings"] = instance.settings
    result["clocks_before"] = clocks()
    gil("after_build")

    reference = None
    if args.path != "eager":
        stage = "eager_reference"
        instance.step()  # warm eager (cuDNN/cuBLAS heuristics) before the reference
        outputs = instance.step()
        reference = {
            "outputs": {k: v.cpu() for k, v in outputs.items()},
            "grad_norms": grad_norms(instance.parameters)
            if instance.parameters
            else None,
        }
        del outputs

        stage = "compile"
        kwargs = compile_kwargs()
        result["compile_applied"] = {
            k: (v if isinstance(v, (str, bool, type(None))) else repr(v))
            for k, v in kwargs.items()
        }
        for module in instance.regions:
            module.compile(**kwargs)
        for name, fn in list(instance.functions.items()):
            instance.functions[name] = torch.compile(fn, **kwargs)

    stage = "first_step"
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    outputs = instance.step()
    torch.cuda.synchronize(device)
    result["first_step_s"] = time.perf_counter() - t0
    gil("after_first_step")

    if reference is not None:
        stage = "correctness"
        errors = {
            k: relative(outputs[k].cpu(), v) for k, v in reference["outputs"].items()
        }
        correctness = {
            "output_rel_l2_max": max(errors.values()),
            "output_rel_l2": errors,
        }
        if reference["grad_norms"] is not None:
            g = grad_norms(instance.parameters)
            rel = (
                (g - reference["grad_norms"]).abs()
                / reference["grad_norms"].clamp_min(1e-30)
            ).tolist()
            correctness.update(
                grad_norm_rel_max=max(rel),
                grad_norm_rel_median=statistics.median(rel),
                nonfinite_grads=int((~torch.isfinite(g)).sum()),
            )
        result["correctness"] = correctness
    del outputs

    stage = "warmup"
    for _ in range(args.warmup):
        instance.step()
    stage = "steady_state"
    result["steady_gpu_ms"], result["steady_wall_ms"] = timed_steps(
        instance.step, args.steps
    )
    stage = "kernel_census"
    result["gpu_kernels"] = kernel_census(instance.step)
    result["peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    result["status"] = "ok"
except Exception as e:  # noqa: BLE001 - any failure is recorded in the result
    result["status"] = "failed"
    result["failure"] = {
        "stage": stage,
        "error": f"{type(e).__name__}: {e}"[:4000],
        "traceback": traceback.format_exc()[-6000:],
    }
finally:
    gil("end")
    result["clocks_after"] = clocks()
    from torch._dynamo.utils import counters

    result["dynamo"] = {
        "unique_graphs": counters["stats"]["unique_graphs"],
        "graph_breaks": {k[:300]: v for k, v in counters["graph_break"].items()},
        "inductor_counters": {
            k: v for k, v in counters["inductor"].items() if "cudagraph" in k
        },
    }
    if args.path == "cubecl":
        statuses = Counter(r["status"] for r in REGIONS)
        reasons = Counter(
            r.get("reason", "").split("\n")[0][:300]
            for r in REGIONS
            if r["status"] == "rejected"
        )
        result["coverage"] = {
            "regions": len(REGIONS),
            "by_status": dict(statuses),
            "rejection_reasons": dict(reasons.most_common()),
            "regions_detail": REGIONS,
        }
        if args.on_reject == "record" and statuses.get("rejected"):
            result["config_deviations"].append(
                "inventory run: rejected regions executed eagerly; timings are not a CubeCL result"
            )
    write()

sys.exit(0 if result["status"] == "ok" else 1)
