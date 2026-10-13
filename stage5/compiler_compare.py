"""One semantic/physical kernel through both audited compiler routes."""

import json
from pathlib import Path
import statistics
import subprocess
import time
from stage1 import evidence

evidence.guard()
import torch
from stage0.bridge import native_path
from stage5.artifact import CompiledKernel, Kernel
from stage5.plan import PhysicalCandidate
from torch._inductor.runtime.benchmarking import InductorBenchmarker
from torch._inductor import config
from stage5.timing import KernelBenchmarker

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT / "artifacts/baseline-cubecl_baseline.json").read_text())[
    "artifacts"
]
semantic = next(
    a["semantic"]
    for a in D
    if a["semantic"]["schema"] == 4
    and a["semantic"]["buffers"][0]["dtype"] == "float32"
    and len(a["semantic"]["scalars"]) == 1
    and any(o["op"] == "sin" for o in a["semantic"]["ops"])
)
plan = PhysicalCandidate("scalar_pointwise_baseline").to_dict()
k = CompiledKernel(semantic, 0, plan)
out = ROOT / "artifacts/compiler-compare-llvm"
out.mkdir(parents=True, exist_ok=True)
source = k.directory / "descriptor.json"
began = time.perf_counter()
p = subprocess.run(
    [str(native_path("cubecl-stage0")), str(out), "llvm", str(source)],
    capture_output=True,
    text=True,
)
(out / "compiler.log").write_text(p.stdout + p.stderr)
if p.returncode:
    raise RuntimeError(p.stderr)
l = Kernel(out, semantic, 0, plan, compile_ptx=False)
x = torch.randn(1048583, device="cuda")
y = torch.empty_like(x)
rows = []
for route, c in [("cpp", k.kernel), ("llvm", l)]:
    c((x.numel(),), x, y)
    torch.testing.assert_close(y, torch.sin(x) + 1, rtol=2e-6, atol=2e-6)
    rounds = []
    kernel_rounds = []
    bm = InductorBenchmarker()
    with (
        config.patch({"autotune_cudagraph_benchmarking": False}),
        c.prepared((x.numel(),), x, y),
    ):
        for _ in range(3):
            samples = bm.benchmark_gpu(
                lambda: c.batch(128),
                return_mode="all",
                benchmark_iters=12,
                memory_warmup_iters=3,
                max_benchmark_duration=20,
            )
            rounds.append([float(v) / 128 for v in samples])
            kernel_rounds.append(KernelBenchmarker().benchmark_kernel_gpu(c))
    rows.append(
        {
            "route": route,
            "median_ms": statistics.median(v for r in rounds for v in r),
            "round_samples_ms": rounds,
            "actual_kernel_rounds": kernel_rounds,
            "actual_kernel_median_ms": statistics.median(
                v for r in kernel_rounds for v in r["kernel_ms"]
            ),
            "manifest": c.manifest,
            "attributes": c.attributes,
            "resources": c.resources,
            "module_load_ms": c.load_ms,
            "directory": str(c.artifact_dir),
        }
    )
# The default C++ route supports vectors. Probe the independent LLVM
# alternative explicitly; record its verifier boundary without admitting it.
vector_plan = PhysicalCandidate("vector_pointwise", (128, 1, 1), 4).to_dict()
v = CompiledKernel(semantic, 0, vector_plan)
probe_out = ROOT / "artifacts/compiler-vector-llvm"
probe_out.mkdir(parents=True, exist_ok=True)
probe = subprocess.run(
    [
        str(native_path("cubecl-stage0")),
        str(probe_out),
        "llvm",
        str(v.directory / "descriptor.json"),
    ],
    capture_output=True,
    text=True,
)
(probe_out / "compiler.log").write_text(probe.stdout + probe.stderr)
vector_probe = {
    "returncode": probe.returncode,
    "stderr": probe.stderr,
    "descriptor": str(v.directory / "descriptor.json"),
    "directory": str(probe_out),
    "admitted_tuning_route": False,
}
evidence.state("after_compiler_compare")
(ROOT / "artifacts/compiler-comparison.json").write_text(
    json.dumps(
        {
            "records": rows,
            "vector_llvm_probe": vector_probe,
            "independence": evidence.inventory(),
        },
        indent=2,
    )
)
print("Compiler routes PASS", flush=True)
