"""PM counter derivative of the frozen CUPTI primer harness; diagnostic only.

The small CUDA primers isolate operations ATen's general tensor kernels cannot:
register arithmetic with one terminal store, and equal-count volatile loads over
different address spans. They are never reduction candidates or runtime warmups.
"""

import argparse
from contextlib import ExitStack
import copy
import csv
import gzip
import shutil
import ctypes as C
import json
import os
from pathlib import Path
import random
import statistics
import traceback
import time

from stage5.reduction_repeatability_probe import PREVIOUS, ROOT, sha

SOURCE = r"""
extern "C" __global__ void primer_read(const unsigned *p, unsigned *sink,
                                      unsigned count, unsigned stride) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned v = 0;
    const volatile unsigned *q = p;
    for (unsigned i = t; i < count; i += gridDim.x * blockDim.x)
        v += q[(unsigned long long)i * stride];
    sink[t] = v;
}
extern "C" __global__ void primer_compute(float *sink, unsigned iterations) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    float a = (float)t, b = 0.5f;
    for (unsigned i = 0; i < iterations; ++i) {
        a = __fadd_rn(__fmul_rn(a, 1.00000011920928955078125f), b);
        b = __fadd_rn(__fmul_rn(b, 0.999999940395355224609375f), 0.000001f);
    }
    sink[t] = a + b;
}
extern "C" __global__ void primer_write(unsigned *p, unsigned *sink,
                                       unsigned count, unsigned stride) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    volatile unsigned *q = p;
    for (unsigned i = t; i < count; i += gridDim.x * blockDim.x)
        q[(unsigned long long)i * stride] = 1u;
    sink[t] = t;
}
"""

DEFAULT_MODES = [
    "none",
    "same8",
    "other8",
    "pairs8",
    "triton8",
    "compute",
    "dense256k",
    "dense64m",
    "sparse256m",
    "target_sparse",
    "target_dense",
    "same8_dense64m",
    "same8_sparse256m",
    "same8_compute",
    "same8_spin",
    "write64m",
    "write_sparse256m",
    "same8_write64m",
    "dense16m",
    "dense4m",
    "sparse_repeat",
    "compact_repeat",
    "read_write",
    "write_read",
    "compact_long",
    "target_sparse_long",
    "sparse_write_repeat",
    "same8_dense256k",
    "same8_target_sparse",
]


def run(args):
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import numpy as np
    import torch
    from stage0.bridge import nvrtc_compile
    from stage5.attribution import inventory
    from stage5.reduction_codegen_compare import DriverKernel
    from stage5.reduction_codegen_mechanisms import Component
    from stage5.reference_reduction_probe import tensor_sha

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "harness.py").write_bytes(Path(__file__).read_bytes())
    fixture = (
        ROOT / "artifacts/reduction-structure/repeatability/input-float32-64-65539.npy"
    )
    x = torch.from_numpy(np.load(fixture, allow_pickle=False)).cuda()
    input_hash = tensor_sha(x)
    assert (
        input_hash == "199358562e389b69090676348ef457ed0141f285d0609c9dd4ff6bfe3b44e218"
    )
    b, n = x.shape
    old = {
        impl: next(
            t
            for t in json.loads((PREVIOUS / f"fixed-graph-{impl}-r0.json").read_text())[
                "timings"
            ]
            if t["case"] == "float32-64-65539"
        )
        for impl in ("cubecl", "triton")
    }
    cube = [Component(Path(k["directory"])) for k in old["cubecl"]["selected_kernels"]]
    paths = [
        Path(k["directory"]) / "kernel.ptx" for k in old["cubecl"]["selected_kernels"]
    ]
    tk = old["triton"]["selected_kernels"][0]
    tp = (
        ROOT
        / "cache/reference-reduction/fixed-graph-triton-r0/triton"
        / tk["cache_hash"]
        / (tk["entrypoint"] + ".cubin")
    )
    identities = {str(p): sha(p) for p in (*paths, tp)}
    tmp, y = torch.empty((b, 9), device="cuda"), torch.empty(b, device="cuda")
    other = x.clone()
    other_tmp, ttmp = torch.empty_like(tmp), torch.empty_like(tmp)
    memory = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    reset = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    sink = torch.empty((82 * 256,), dtype=torch.int32, device="cuda")
    directory = args.output / "primers"
    directory.mkdir()
    (directory / "kernel.cu").write_text(SOURCE)
    nvrtc_compile(SOURCE, directory, 0)
    read = DriverKernel(directory / "kernel.ptx", "primer_read", (256, 1, 1))
    write = DriverKernel(directory / "kernel.ptx", "primer_write", (256, 1, 1))
    compute = DriverKernel(directory / "kernel.ptx", "primer_compute", (256, 1, 1))
    compute.prepare(82, (sink,), (C.c_uint32(args.iterations),))

    def bind_read(tensor, count, stride, base=None):
        assert (
            count > 0
            and (count - 1) * stride * 4 + 4 <= tensor.numel() * tensor.element_size()
        )
        k = copy.copy(read if base is None else base)
        k.prepare(82, (tensor, sink), (C.c_uint32(count), C.c_uint32(stride)))
        return k

    reads = {
        "dense256k": bind_read(memory, 65536, 1),
        "dense64m": bind_read(memory, 16 * 1024 * 1024, 1),
        "dense16m": bind_read(memory, 4 * 1024 * 1024, 1),
        "dense4m": bind_read(memory, 1024 * 1024, 1),
        "sparse256m": bind_read(memory, 65536, 1024),
        "write64m": bind_read(memory, 16 * 1024 * 1024, 1, write),
        "write_sparse256m": bind_read(memory, 65536, 1024, write),
        "target_sparse": bind_read(x, (x.numel() + 1023) // 1024, 1024),
        "target_dense": bind_read(x, (x.numel() + 1023) // 1024, 1),
    }
    result = {
        "complete": False,
        "input_sha256": input_hash,
        "shape": [b, n],
        "split": 9,
        "artifact_sha256": identities,
        "harness_sha256": sha(__file__),
        "primer_ptx_sha256": sha(directory / "kernel.ptx"),
        "reset": args.reset,
        "primer_iterations": args.iterations,
        "resources": {
            "read": read.attributes,
            "write": write.attributes,
            "compute": compute.attributes,
        },
        "pointer_spans": {
            "input": [x.data_ptr(), x.numel() * 4],
            "other": [other.data_ptr(), other.numel() * 4],
            "memory": [memory.data_ptr(), memory.numel() * 4],
        },
        "captures": [],
        "protocol": "One isolated CubeCL partial after prequeued reset/primer; final/output checks outside capture; all samples retained",
    }
    try:
        with ExitStack() as stack:
            stack.enter_context(cube[0].kernel.prepared((n, b), x, tmp))
            stack.enter_context(cube[1].kernel.prepared((b,), tmp, y))
            partial_name = cube[0].kernel.manifest["entrypoint"]
            alternate = DriverKernel(paths[0], partial_name, (512, 1, 1), 64)
            alternate.prepare(b * 9, (other, other_tmp, cube[0].kernel._prepared[1]))
            triton = DriverKernel(tp, tk["entrypoint"], (512, 1, 1), 64)
            triton.prepare(
                b * 9,
                (other, ttmp),
                (
                    C.c_int64(n),
                    C.c_int32(b * 9),
                    C.c_int32((n + 8) // 9),
                    C.c_uint64(0),
                    C.c_uint64(0),
                ),
            )
            for _ in range(20):
                cube[0].kernel.batch(1)
                cube[1].kernel.batch(1)
            expected_tmp, expected_y = tmp.clone(), y.clone()
            expected_output_hash = tensor_sha(y)
            alternate.batch(1)
            triton.batch(1)
            assert torch.equal(tmp.view(torch.int32), other_tmp.view(torch.int32))
            assert torch.equal(tmp.view(torch.int32), ttmp.view(torch.int32))

            def primer(mode):
                warm = mode.startswith("same8_") or mode == "same8"
                if warm or mode == "pairs8":
                    for _ in range(8):
                        cube[0].kernel.batch(1)
                        if mode == "pairs8":
                            cube[1].kernel.batch(1)
                elif mode in ("other8", "triton8"):
                    (alternate if mode == "other8" else triton).batch(8)
                tail = (
                    mode.removeprefix("same8_") if mode.startswith("same8_") else mode
                )
                if tail in reads:
                    reads[tail].batch(1)
                elif tail in ("sparse_repeat", "compact_repeat"):
                    reads[
                        "sparse256m" if tail == "sparse_repeat" else "dense256k"
                    ].batch(16)
                elif tail == "compact_long":
                    reads["dense256k"].batch(128)
                elif tail == "target_sparse_long":
                    reads["target_sparse"].batch(64)
                elif tail == "sparse_write_repeat":
                    reads["write_sparse256m"].batch(16)
                elif tail == "read_write":
                    reads["dense64m"].batch(1)
                    reads["write64m"].batch(1)
                elif tail == "write_read":
                    reads["write64m"].batch(1)
                    reads["dense64m"].batch(1)
                elif tail == "compute":
                    compute.batch(1)
                elif tail == "spin":
                    torch.cuda._sleep(300_000)

            sampler = None
            if args.pm_metrics:
                sampler = C.CDLL(str(args.sampler_library.resolve()))
                sampler.state_pm_init.argtypes = [C.c_char_p, C.c_uint64]
                sampler.state_pm_stop.argtypes = [C.c_char_p]
                assert sampler.state_pm_init(args.pm_metrics.encode(), args.interval_ns) == 0
                result["pm_metrics"] = args.pm_metrics.split(",")
                result["pm_interval_ns"] = args.interval_ns
                result["sampler_sha256"] = sha(args.sampler_library)
                for filename in ("reduction_state_sampling.cpp",):
                    (args.output / filename).write_bytes((ROOT / filename).read_bytes())

            clock_library = C.CDLL(str(Path("stage0/python-deps/nvidia/cu13/lib/libcupti.so.13").resolve()))
            clock_library.cuptiGetTimestamp.argtypes = [C.POINTER(C.c_uint64)]
            def clock_anchor():
                stamp = C.c_uint64()
                wall_before = time.time_ns()
                before = torch._C._profiler._get_approximate_time()
                assert clock_library.cuptiGetTimestamp(C.byref(stamp)) == 0
                after = torch._C._profiler._get_approximate_time()
                wall_after = time.time_ns()
                return {"approx_before": before, "approx_after": after, "wall_before_ns": wall_before, "wall_after_ns": wall_after, "cupti_timestamp": stamp.value}

            for round_id in range(args.round_start, args.round_start + args.rounds):
                modes = args.modes.copy()
                random.Random(51000 + round_id).shuffle(modes)
                for mode in modes:
                    # Each capture's host startup is followed by a long GPU gate.
                    # Raw API/GPU timestamps must prove that all work was queued.
                    converter = torch._C._profiler._ApproximateClockToUnixTimeConverter()
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        if sampler:
                            assert sampler.state_pm_start() == 0
                        anchor_before = clock_anchor()
                        torch.cuda._sleep(50_000_000)
                        if args.reset == "dense":
                            reset.zero_()
                        primer(mode)
                        cube[0].kernel.batch(1)
                    anchor_after = clock_anchor()
                    pm_path = args.output / f"r{round_id}-{mode}.pm.csv"
                    if sampler:
                        assert sampler.state_pm_stop(str(pm_path).encode()) == 0
                        raw_counter = Path(str(pm_path) + ".counterdata")
                        with raw_counter.open("rb") as src, gzip.open(str(raw_counter) + ".gz", "wb", compresslevel=1) as dest:
                            shutil.copyfileobj(src, dest)
                        raw_counter.unlink()
                    if sampler:
                        with pm_path.open() as stream:
                            reader = csv.DictReader(stream)
                            fields = reader.fieldnames
                            samples = list(reader)
                        with (args.output / f"r{round_id}-{mode}.pm-unix.csv").open("w") as stream:
                            writer = csv.DictWriter(stream, fieldnames=["start_unix_ns", "end_unix_ns", *fields])
                            writer.writeheader()
                            for sample in samples:
                                sample["start_unix_ns"] = converter.to_unix_ns(int(sample["start_ns"]))
                                sample["end_unix_ns"] = converter.to_unix_ns(int(sample["end_ns"]))
                                writer.writerow(sample)
                    trace = args.output / f"r{round_id}-{mode}.json"
                    prof.export_chrome_trace(str(trace))
                    raw = json.loads(trace.read_text())["traceEvents"]
                    events = sorted(
                        (e for e in raw if e.get("cat") == "kernel"),
                        key=lambda e: e["ts"],
                    )
                    assert events[-1]["name"] == partial_name
                    gate = events[0]
                    assert "spin_kernel" in gate["name"]
                    apis = [
                        e
                        for e in raw
                        if "LaunchKernel" in e.get("name", "")
                        and e.get("cat") in ("cuda_driver", "cuda_runtime")
                    ]
                    lead = (
                        gate["ts"] + gate["dur"] - max(e["ts"] + e["dur"] for e in apis)
                    )
                    assert lead > 0, (mode, lead)
                    cube[1].kernel.batch(1)
                    assert torch.equal(
                        tmp.view(torch.int32), expected_tmp.view(torch.int32)
                    )
                    assert torch.equal(
                        y.view(torch.int32), expected_y.view(torch.int32)
                    )
                    target = events[-1]
                    row = {
                        "clock_anchors": [anchor_before, anchor_after],
                        "round": round_id,
                        "mode": mode,
                        "target_us": target["dur"],
                        "target_gap_us": target["ts"]
                        - (events[-2]["ts"] + events[-2]["dur"]),
                        "prequeue_lead_us": lead,
                        "bitwise_partial_and_output": True,
                        "kernels": [
                            {"name": e["name"], "us": e["dur"]} for e in events
                        ],
                        "trace": str(trace),
                        "trace_sha256": sha(trace),
                    }
                    result["captures"].append(row)
                    print(round_id, mode, row["target_us"], flush=True)
                    (args.output / "result.json").write_text(
                        json.dumps(result, indent=2)
                    )
            if sampler:
                assert sampler.state_pm_close() == 0
            assert tensor_sha(x) == tensor_sha(other) == input_hash
            assert all(sha(p) == digest for p, digest in identities.items())
            result["output_sha256"] = expected_output_hash
            result["partial_sha256"] = tensor_sha(tmp)
            result["environment"] = inventory(torch, "cubecl")
            result["summary"] = {
                mode: {
                    "samples_us": [
                        c["target_us"] for c in result["captures"] if c["mode"] == mode
                    ],
                    "median_us": statistics.median(
                        c["target_us"] for c in result["captures"] if c["mode"] == mode
                    ),
                }
                for mode in args.modes
            }
            result["complete"] = True
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1024)
    parser.add_argument("--reset", choices=["dense", "gate"], default="dense")
    parser.add_argument(
        "--modes", choices=DEFAULT_MODES, nargs="+", default=DEFAULT_MODES
    )
    parser.add_argument("--round-start", type=int, default=0)
    parser.add_argument("--pm-metrics", default="")
    parser.add_argument("--interval-ns", type=int, default=1000)
    parser.add_argument("--sampler-library", type=Path, default=Path("/tmp/stage5-state-sampling.so"))
    args = parser.parse_args()
    if (
        args.round_start < 0
        or args.interval_ns < 1000
        or args.rounds < 1
        or not 1 <= args.iterations <= 0xFFFFFFFF
        or len(args.modes) != len(set(args.modes))
    ):
        parser.error("Positive rounds, uint32 iterations and unique modes required")
    run(args)


if __name__ == "__main__":
    main()
