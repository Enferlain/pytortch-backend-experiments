"""Compare generic event and profiler timers against explicit kernel durations."""

import json
from pathlib import Path
import statistics
from stage1 import evidence

evidence.guard()
import torch
from torch._inductor.runtime.benchmarking import (
    InductorBenchmarker,
    TorchProfilerBenchmarker,
)
from stage5.artifact import CompiledKernel, compiler_identity
from stage5.fixtures import captured
from stage5.plan import PhysicalCandidate

ROOT = Path(__file__).resolve().parent
sem = captured("row_sum", "float32")
identity = compiler_identity(0)
x = torch.randn(64, 1024, device="cuda")
y = torch.empty(64, device="cuda")
rows = []
for p in [
    PhysicalCandidate("shared_tree_reduction_baseline"),
    PhysicalCandidate("plane_shared", (128, 1, 1), plane_width=32),
    PhysicalCandidate(
        "independent_planes", (32, 8, 1), outputs_per_workgroup=8, plane_width=32
    ),
]:
    c = CompiledKernel(sem, 0, p.to_dict(), identity)
    with c.kernel.prepared((1024, 64), x, y):
        b = InductorBenchmarker()
        samples = b.benchmark_gpu(
            lambda: c.kernel.batch(128),
            return_mode="all",
            benchmark_iters=12,
            memory_warmup_iters=3,
            max_benchmark_duration=20,
        )
        event = statistics.median(samples) / 128
        profs = []
        for _ in range(3):
            b = TorchProfilerBenchmarker()
            profs.append(
                b.benchmark_gpu(
                    lambda: c.kernel.batch(128),
                    return_mode="mean",
                    rep=8,
                    estimation_iters=3,
                    memory_warmup_iters=3,
                    max_benchmark_duration=20,
                )
                / 128
            )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            c.kernel.batch(128)
        kernels = [
            e
            for e in prof.profiler.kineto_results.events()
            if e.device_type() == torch.autograd.DeviceType.CUDA
            and e.name() == c.entrypoint
        ]
        explicit = statistics.median(
            [(e.end_ns() - e.start_ns()) / 1e6 for e in kernels]
        )
        assert len(kernels) == 128
        print(
            p.strategy,
            "event",
            event,
            "profiler",
            profs,
            "explicit",
            explicit,
            flush=True,
        )
        rows.append(
            {
                "plan": p.to_dict(),
                "event_ms": event,
                "profiler_ms": profs,
                "explicit_kernel_median_ms": explicit,
                "count": len(kernels),
            }
        )
(ROOT / "artifacts/timer-probe.json").write_text(json.dumps(rows, indent=2))
