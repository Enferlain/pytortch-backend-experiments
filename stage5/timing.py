"""Inductor warmup/events plus actual CUPTI kernel durations, excluding ranges.

TorchProfilerBenchmarker's preferred _CALLABLE GPU annotation spans driver
queue gaps on this WSL path. The narrow adapter instead reads the known physical
kernel's events and checks their count. No graph-capture or Triton API is needed.
"""

import torch
from torch._inductor.runtime.benchmarking import InductorBenchmarker


class KernelBenchmarker(InductorBenchmarker):
    def benchmark_kernel_gpu(self, kernel):
        intervals = self.benchmark_gpu(
            lambda: kernel.batch(16),
            return_mode="all",
            benchmark_iters=6,
            estimation_iters=3,
            memory_warmup_iters=3,
            max_benchmark_duration=12,
            device_type="cuda",
        )
        observed = []
        for _ in range(3):
            torch.cuda.synchronize(kernel.device)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
            ) as profiler:
                for _ in range(3):
                    kernel.batch(128)
            events = [
                e
                for e in profiler.profiler.kineto_results.events()
                if e.device_type() == torch.autograd.DeviceType.CUDA
                and e.name() == kernel.manifest["entrypoint"]
                and e.activity_type() != "gpu_user_annotation"
            ]
            observed.append(len(events))
            if len(events) == 384:
                break
        else:
            raise RuntimeError(
                f"physical benchmark expected384 kernels, observed{observed}"
            )
        samples = [(e.end_ns() - e.start_ns()) / 1e6 for e in events]
        if not all(v > 0 for v in samples):
            raise RuntimeError("invalid GPU kernel timestamp")
        return {
            "kernel_ms": samples,
            "enqueue_interval_ms": [float(v) / 16 for v in intervals],
            "kernel_count": len(events),
            "profile_attempt_counts": observed,
            "method": "Inductor event warmup + exact CUPTI kernel events; GPU annotations excluded",
        }
