"""Retain live scheduling/replay and launch census for existing representatives."""

import ctypes as C
import json
from pathlib import Path
from stage1 import evidence

evidence.guard()
import torch
from stage1.overlay import install

install()
from stage5 import backend, runtime
from stage5.corpus import cases, fixture

ROOT = Path(__file__).resolve().parent
torch.manual_seed(6142)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
native = C.CDLL(str(ROOT.parent / "stage0/libstage0_launcher.so"))
native.stage0_kernel_launches.restype = C.c_uint64
records = []
for name in (
    "fused-float16-1048583",
    "rms_backward-float32-64x1024",
    "training-float32-5x9",
):
    case = next(c for c in cases() if c.name == name)
    f = fixture(case)
    compiled = torch.compile(
        f.fn, backend=backend.compile_backend, fullgraph=True, dynamic=True
    )

    def run():
        for x in f.args:
            if x.requires_grad:
                x.grad = None
        if f.model:
            for p in f.model.parameters():
                p.grad = None
        result = compiled(*f.args)
        if f.optimizer:
            result.backward()
            f.optimizer.step()
        elif f.upstream is not None:
            result.backward(f.upstream)
        return result

    begin_trace = len(backend.TRACE)
    run()
    torch.cuda.synchronize()
    begin_events = len(runtime.EVENTS)
    begin_count = native.stage0_kernel_launches()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        run()
    generated = [
        e["entrypoint"] for e in runtime.EVENTS[begin_events:] if e["event"] == "launch"
    ]
    kernels = [
        e
        for e in profiler.profiler.kineto_results.events()
        if e.device_type() == torch.autograd.DeviceType.CUDA
        and e.activity_type() == "kernel"
    ]
    assert (
        len(generated)
        == native.stage0_kernel_launches() - begin_count
        == sum(e.name() in generated for e in kernels)
    )
    records.append(
        {
            "name": name,
            "schedule_replay": backend.TRACE[begin_trace:],
            "launches": runtime.EVENTS[begin_events:],
            "generated_count": len(generated),
            "gpu_kernels": [e.name() for e in kernels],
        }
    )
evidence.state("after_schedule_trace")
(ROOT / "artifacts/schedule-trace.json").write_text(
    json.dumps({"records": records, "independence": evidence.inventory()}, indent=2)
)
print("Schedule/replay trace PASS", flush=True)
