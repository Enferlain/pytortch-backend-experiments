"""Fixed-artifact timing controls; no compiler, planner or arithmetic changes."""

import argparse
from contextlib import ExitStack
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "artifacts/reduction-structure/codegen"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.attribution import inventory
    from stage5.reduction_codegen_compare import DriverKernel
    from stage5.reduction_codegen_mechanisms import Component
    from stage5.reference_reduction_probe import tensor_sha

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    args.output.mkdir(parents=True, exist_ok=False)
    b, n = 64, 65539
    torch.manual_seed(7301)
    x = torch.randn((b, n), device="cuda")
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
    input_hash = tensor_sha(x)
    assert all(row["input_sha256"] == input_hash for row in old.values())
    tmp = {impl: torch.empty((b, 9), device="cuda") for impl in old}
    y = {impl: torch.empty(b, device="cuda") for impl in old}
    cube = [Component(Path(k["directory"])) for k in old["cubecl"]["selected_kernels"]]
    paths = {
        "cubecl": [
            Path(k["directory"]) / "kernel.ptx"
            for k in old["cubecl"]["selected_kernels"]
        ]
    }
    triton_cache = ROOT / "cache/reference-reduction/fixed-graph-triton-r0/triton"
    paths["triton"] = [
        triton_cache / k["cache_hash"] / (k["entrypoint"] + ".cubin")
        for k in old["triton"]["selected_kernels"]
    ]
    triton = [
        DriverKernel(path, k["entrypoint"], block, shared)
        for path, k, block, shared in zip(
            paths["triton"],
            old["triton"]["selected_kernels"],
            ((512, 1, 1), (64, 1, 1)),
            (64, 0),
            strict=True,
        )
    ]
    triton[0].prepare(
        b * 9,
        (x, tmp["triton"]),
        (
            C.c_int64(n),
            C.c_int32(b * 9),
            C.c_int32((n + 8) // 9),
            C.c_uint64(0),
            C.c_uint64(0),
        ),
    )
    triton[1].prepare(
        b,
        (tmp["triton"], y["triton"]),
        (C.c_int32(b), C.c_int32(9), C.c_uint64(0), C.c_uint64(0)),
    )
    identities = {str(p): sha(p) for group in paths.values() for p in group}
    result = {
        "complete": False,
        "shape": [b, n],
        "split": 9,
        "input_sha256": input_hash,
        "artifact_sha256": identities,
        "cubecl_resources": [c.kernel.attributes for c in cube],
        "triton_resources": [k.attributes for k in triton],
        "protocol": "Alternating implementations; exact retained partial/final; CUPTI kernel durations separate from event queue intervals",
        "captures": [],
        "event_intervals": [],
    }
    monitor = None
    log = (args.output / "device.csv").open("w")
    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    if args.monitor:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                uuid,
                "--query-gpu=timestamp,uuid,pstate,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used",
                "--format=csv",
                "--loop-ms=100",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        with ExitStack() as stack:
            stack.enter_context(cube[0].kernel.prepared((n, b), x, tmp["cubecl"]))
            stack.enter_context(
                cube[1].kernel.prepared((b,), tmp["cubecl"], y["cubecl"])
            )

            def launch(impl, validated=False):
                if impl == "cubecl":
                    if validated:
                        cube[0]((n, b), x, tmp[impl])
                        cube[1]((b,), tmp[impl], y[impl])
                    else:
                        cube[0].kernel.batch(1)
                        cube[1].kernel.batch(1)
                else:
                    triton[0].batch(1)
                    triton[1].batch(1)

            for impl in old:
                for _ in range(20):
                    launch(impl)
            assert torch.equal(
                tmp["cubecl"].view(torch.int32), tmp["triton"].view(torch.int32)
            )
            assert torch.equal(
                y["cubecl"].view(torch.int32), y["triton"].view(torch.int32)
            )
            assert y["cubecl"].tolist() == next(
                row["values"]
                for row in json.loads(
                    (PREVIOUS / "fixed-graph-cubecl-r0.json").read_text()
                )["rows"]
                if row["case"] == "storage_torch.float32_64"
            )
            result["bitwise_partials_and_finals"] = True
            names = {
                "cubecl": [c.kernel.manifest["entrypoint"] for c in cube],
                "triton": [k.manifest["entrypoint"] for k in triton],
            }
            for round_id in range(args.rounds):
                # Rotate mode order as well as backend order across rounds.
                modes = ["sparse", "inside_warmup", "burst", "validated"]
                modes = modes[round_id % 4 :] + modes[: round_id % 4]
                for mode in modes:
                    order = (
                        ["cubecl", "triton"]
                        if round_id % 2 == 0
                        else ["triton", "cubecl"]
                    )
                    for impl in order:
                        torch.cuda._sleep(50_000_000)
                        torch.cuda.synchronize()
                        started = time.time_ns()
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as prof:
                            if mode == "inside_warmup":
                                torch.cuda._sleep(50_000_000)
                            repeats = args.burst if mode == "burst" else 1
                            for _ in range(repeats):
                                launch(impl, mode == "validated")
                        finished = time.time_ns()
                        trace = args.output / f"r{round_id}-{mode}-{impl}.json"
                        prof.export_chrome_trace(str(trace))
                        events = [
                            e
                            for e in prof.profiler.kineto_results.events()
                            if e.device_type() == torch.autograd.DeviceType.CUDA
                            and e.activity_type() == "kernel"
                        ]
                        measured = [e for e in events if e.name() in names[impl]]
                        assert [e.name() for e in measured] == names[impl] * repeats
                        assert len(events) == 2 * repeats + int(mode == "inside_warmup")
                        durations = [
                            (e.end_ns() - e.start_ns()) / 1000 for e in measured
                        ]
                        row = {
                            "round": round_id,
                            "mode": mode,
                            "impl": impl,
                            "start_unix_ns": started,
                            "end_unix_ns": finished,
                            "partial_us": durations[::2],
                            "final_us": durations[1::2],
                            "raw_trace": str(trace),
                            "trace_sha256": sha(trace),
                        }
                        result["captures"].append(row)
                        print(
                            round_id,
                            mode,
                            impl,
                            "partial",
                            round(statistics.median(row["partial_us"]), 3),
                            "final",
                            round(statistics.median(row["final_us"]), 3),
                            flush=True,
                        )
                # Queue-inclusive events are an independent, unprofiled control.
                for impl in order:
                    torch.cuda._sleep(50_000_000)
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(args.burst):
                        launch(impl)
                    end.record()
                    end.synchronize()
                    result["event_intervals"].append(
                        {
                            "round": round_id,
                            "impl": impl,
                            "queue_interval_us_per_pair": start.elapsed_time(end)
                            * 1000
                            / args.burst,
                        }
                    )
                (args.output / "result.json").write_text(json.dumps(result, indent=2))
            assert all(sha(p) == value for p, value in identities.items())
            assert tensor_sha(x) == input_hash
            assert torch.equal(
                tmp["cubecl"].view(torch.int32), tmp["triton"].view(torch.int32)
            )
            assert torch.equal(
                y["cubecl"].view(torch.int32), y["triton"].view(torch.int32)
            )
            result["environment"] = inventory(torch, "cubecl")
            result["harness_sha256"] = sha(__file__)
            result["complete"] = True
    finally:
        if monitor:
            monitor.terminate()
            monitor.wait(timeout=10)
        log.close()
        (args.output / "result.json").write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--burst", type=int, default=16)
    parser.add_argument("--monitor", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
