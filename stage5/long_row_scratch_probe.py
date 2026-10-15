"""Isolate final scratch synchronization on real semantic descriptors.

No policy/catalog change: compare the saved compiler with the current compiler
using identical descriptors, physical plans, input bytes and the native bridge.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/long-row-diagnosis"
CACHE = ROOT / "cache/long-row-diagnosis"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.artifact import CompiledKernel, Kernel
    from stage5.fixtures import captured
    from stage5.plan import PhysicalCandidate
    from stage5.reference_reduction_probe import sha, tensor_sha
    from stage5.timing import KernelBenchmarker
    from stage2.indexing import evaluate

    OUTPUT.mkdir(parents=True, exist_ok=True)
    before_compiler = CACHE / "before/cubecl-stage0"
    result = {
        "complete": False,
        "before_compiler_sha256": sha(before_compiler),
        "checks": [],
        "timings": [],
        "kernels": [],
    }
    kernels = {}

    def pair(family, dtype, plan):
        key = (family, dtype, plan.identity)
        if key in kernels:
            return kernels[key]
        if family.startswith("split"):
            source = json.loads(
                (ROOT / "artifacts/long-row/production-guarded-large.json").read_text()
            )
            count = 64 if family == "split4" else 8
            item = next(
                t for t in source["timings"] if t["case"] == f"{dtype}-{count}-65539"
            )
            directory = Path(item["selected_kernels"][0]["directory"])
            semantic = json.loads((directory / "semantic.json").read_text())
        elif family == "two_sums":
            semantic = json.loads(
                (
                    ROOT.parent / "stage3/artifacts/representatives/"
                    "two_sums_normal_fusion_shared_reuse-0/descriptor.json"
                ).read_text()
            )
            semantic.pop("entrypoint")
            semantic.pop("physical_plan")
        else:
            semantic = captured("row_sum", dtype)
        after = CompiledKernel(semantic, 0, plan.to_dict())
        packet = json.loads((after.directory / "descriptor.json").read_text())
        directory = CACHE / "scratch-before" / after.directory.name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "descriptor.json"
        path.write_text(json.dumps(packet, indent=2))
        completed = subprocess.run(
            [str(before_compiler), str(directory), "cpp", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        (directory / "compiler.log").write_text(completed.stdout + completed.stderr)
        before = Kernel(directory, semantic, 0, plan.to_dict())
        old_barriers = (directory / "kernel.ptx").read_text().count("bar.sync")
        new_barriers = (after.directory / "kernel.ptx").read_text().count("bar.sync")
        assert old_barriers - new_barriers == (plan.strategy != "independent_planes")
        # Exact descriptors/plans; only their compiler-produced code differs.
        assert packet == json.loads(path.read_text())
        result["kernels"].append(
            {
                "family": family,
                "dtype": dtype,
                "plan": plan.to_dict(),
                "before": str(directory),
                "after": str(after.directory),
                "before_barriers": old_barriers,
                "after_barriers": new_barriers,
                "descriptor_sha256": sha(path),
                "before_ptx_sha256": sha(directory / "kernel.ptx"),
                "after_ptx_sha256": sha(after.directory / "kernel.ptx"),
                "before_attributes": before.attributes,
                "after_attributes": after.kernel.attributes,
            }
        )
        kernels[key] = (before, after.kernel)
        return kernels[key]

    def check(label, x, family="split", plan=None):
        plan = plan or PhysicalCandidate("shared_tree_compensated")
        if x.shape[1] == 0:
            # Direct component fixtures require their declared (N, 1) strides;
            # PyTorch's default empty allocation has nominal (1, 1) strides.
            # This empty view allocates/copies no storage. Captured split graphs
            # retain their N>0 guard; real empty graphs are covered by the gates.
            x = x.as_strided(x.shape, (0, 1))
        before, after = pair(family, str(x.dtype).removeprefix("torch."), plan)
        scalars = (x.shape[1], x.shape[0])
        shape = tuple(
            evaluate(v, scalars) for v in before.semantic["buffers"][-1]["size"]
        )
        output_dtype = getattr(torch, before.semantic["buffers"][-1]["dtype"])
        a = torch.empty(shape, device=x.device, dtype=output_dtype)
        b = torch.empty_like(a)
        before(scalars, x, a)
        after(scalars, x, b)
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), label
        result["checks"].append(
            {
                "case": label,
                "family": family,
                "dtype": str(x.dtype),
                "shape": list(x.shape),
                "input_sha256": tensor_sha(x),
                "output_sha256": tensor_sha(a),
                "bitwise_identical": True,
            }
        )

    try:
        old = json.loads(
            (
                ROOT / "artifacts/attribution/cancellation-counterexample.json"
            ).read_text()
        )
        torch.manual_seed(old["seed"])
        for shape in old["rng_history"]:
            x = torch.randn(shape, device="cuda")
        check("original_rng_history", x)
        for n in (31, 255, 256, 257, 1023, 1024, 1025, 4099, 65539, 131071, 1048576):
            for seed in (9241, 11, 101, 7301):
                torch.manual_seed(seed)
                check(f"random_{seed}_{n}", torch.randn((8, n), device="cuda"))
        for seed in (*range(10000, 10032), *range(30000, 30032)):
            torch.manual_seed(seed)
            check(f"heldout_{seed}", torch.randn((8, 65539), device="cuda"))
        # Bind the original 109 inputs to the already qualified byte corpus.
        original = json.loads(
            (ROOT / "artifacts/long-row/production-guarded-large.json").read_text()
        )
        assert len(result["checks"]) == 109
        for a, b in zip(result["checks"], original["rows"][:109]):
            assert a["case"] == b["case"] and a["input_sha256"] == b["input_sha256"]
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for n in (0, 1, 255, 256, 257, 8193, 65539, 73728, 131071, 1048576):
                torch.manual_seed(7301)
                check(
                    f"tail_{dtype}_{n}",
                    torch.randn((8, n), device="cuda").to(dtype),
                    "plain" if n == 0 else "split",
                )
            special = torch.zeros((5, 65539), device="cuda", dtype=dtype)
            special[0].fill_(-0.0)
            special[1, 0] = float("nan")
            special[2, 0] = float("inf")
            special[3, 0] = -float("inf")
            special[4, 0], special[4, 256] = float("inf"), -float("inf")
            check(f"special_{dtype}", special)
        plans = [
            PhysicalCandidate("shared_tree_reduction_baseline"),
            PhysicalCandidate("shared_tree_compensated"),
            PhysicalCandidate("plane_shared", plane_width=32),
            PhysicalCandidate(
                "independent_planes",
                workgroup=(32, 4, 1),
                outputs_per_workgroup=4,
                plane_width=32,
            ),
        ]
        for plan in plans:
            for family in ("plain", "two_sums"):
                for n in (0, 1, 257, 65539) if family == "plain" else (1, 257, 65539):
                    torch.manual_seed(7301)
                    check(
                        f"{family}_{plan.strategy}_{n}",
                        torch.randn((8, n), device="cuda"),
                        family,
                        plan,
                    )
        check(
            "actual_four_part_control",
            torch.randn((64, 65539), device="cuda"),
            "split4",
        )
        # Explicit non-default stream with allocator churn; no launch-path sync.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for n in (257, 65539):
                x = torch.randn((8, n), device="cuda")
                check(f"stream_{n}", x)
                del x
                for _ in range(8):
                    torch.empty((8, n), device="cuda").fill_(123)
        stream.synchronize()
        if args.measure:
            for shape, family in (
                ((8, 65539), "split"),
                ((64, 65539), "split"),
                ((64, 65539), "split4"),
            ):
                before, after = pair(
                    family, "float32", PhysicalCandidate("shared_tree_compensated")
                )
                torch.manual_seed(7301)
                x = torch.randn(shape, device="cuda")
                y = torch.empty(
                    (shape[0], 4 if family == "split4" else 9), device="cuda"
                )
                for round_id in range(3):
                    items = [("before", before), ("after", after)]
                    if round_id % 2:
                        items.reverse()
                    for name, kernel in items:
                        with kernel.prepared((shape[1], shape[0]), x, y):
                            measured = KernelBenchmarker().benchmark_kernel_gpu(kernel)
                        result["timings"].append(
                            {
                                "shape": list(shape),
                                "round": round_id,
                                "version": name,
                                "family": family,
                                "input_sha256": tensor_sha(x),
                                "samples_us": [v * 1000 for v in measured["kernel_ms"]],
                                "median_us": statistics.median(measured["kernel_ms"])
                                * 1000,
                                "enqueue_interval_ms": measured["enqueue_interval_ms"],
                                "profile_attempt_counts": measured[
                                    "profile_attempt_counts"
                                ],
                                "scope": "isolated actual split partial; prepared native batches, not graph sum",
                            }
                        )
        evidence.state("after_scratch_probe")
        result.update(
            complete=True,
            independence=evidence.inventory(),
            harness_sha256=sha(__file__),
        )
    finally:
        path = OUTPUT / (
            "scratch-probe.json" if result["complete"] else "scratch-probe.failed.json"
        )
        path.write_text(json.dumps(result, indent=2))
    print(
        len(result["checks"]), "bitwise checks", len(result["timings"]), "timing rounds"
    )


if __name__ == "__main__":
    main()
