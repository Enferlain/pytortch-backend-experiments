"""Fixed-artifact timing controls; no compiler, planner or arithmetic changes."""

import argparse
from contextlib import ExitStack
import copy
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
    if args.ncu:
        # Application replay reruns the process; retain every pass separately.
        args.output = args.output / f"app-{os.getpid()}"
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "harness.py").write_bytes(Path(__file__).read_bytes())
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
    flush_buffer = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    copy_buffer = torch.empty_like(x)
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
        "harness_sha256": sha(__file__),
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
            if args.address_control or args.module_control:
                # Reuse the existing driver launcher. Shallow copies share the
                # exact module/function, but retain separate PyTorch bindings.
                kind = "modules" if args.module_control else "addresses"
                inputs = (
                    [x] * 16
                    if args.module_control
                    else [x] + [x.clone() for _ in range(15)]
                )
                assert all(tensor_sha(v) == input_hash for v in inputs)
                partial = DriverKernel(
                    paths["cubecl"][0],
                    names["cubecl"][0],
                    (512, 1, 1),
                    cube[0].kernel.manifest["dynamic_shared_bytes"],
                )
                info = cube[0].kernel._prepared[1]
                bound = {impl: [] for impl in old}
                for i, v in enumerate(inputs):
                    c = (
                        DriverKernel(
                            paths["cubecl"][0],
                            names["cubecl"][0],
                            (512, 1, 1),
                            partial.shared,
                        )
                        if args.module_control and i
                        else copy.copy(partial)
                    )
                    c.prepare(b * 9, (v, tmp["cubecl"], info))
                    t = (
                        DriverKernel(
                            paths["triton"][0],
                            names["triton"][0],
                            (512, 1, 1),
                            triton[0].shared,
                        )
                        if args.module_control and i
                        else copy.copy(triton[0])
                    )
                    t.prepare(b * 9, (v, tmp["triton"]), tuple(triton[0].values[2:]))
                    assert c.attributes == partial.attributes
                    assert t.attributes == triton[0].attributes
                    bound["cubecl"].append(c)
                    bound["triton"].append(t)
                control = result[kind + "_control"] = {
                    "input_data_ptrs": [v.data_ptr() for v in inputs],
                    "identical_input_hashes": True,
                    "cubecl_partial_driver_attributes": partial.attributes,
                    "scope": "One versus 16 identical module instances, same input"
                    if args.module_control
                    else "Same function/launch pattern; one allocation versus 16 identical allocations; not production host timing",
                    "function_handles": {
                        impl: [k.function.value for k in ks]
                        for impl, ks in bound.items()
                    },
                }
                for round_id in range(args.rounds):
                    for count in (1, 16) if round_id % 2 == 0 else (16, 1):
                        for impl in (
                            list(old) if round_id % 2 == 0 else list(reversed(old))
                        ):
                            torch.cuda._sleep(50_000_000)
                            torch.cuda.synchronize()
                            with torch.profiler.profile(
                                activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA,
                                ]
                            ) as prof:
                                for i in range(args.burst):
                                    bound[impl][i % count].batch(1)
                                    if impl == "cubecl":
                                        cube[1].kernel.batch(1)
                                    else:
                                        triton[1].batch(1)
                            trace = (
                                args.output / f"r{round_id}-{kind}{count}-{impl}.json"
                            )
                            prof.export_chrome_trace(str(trace))
                            events = [
                                e
                                for e in prof.profiler.kineto_results.events()
                                if e.device_type() == torch.autograd.DeviceType.CUDA
                                and e.activity_type() == "kernel"
                            ]
                            assert [e.name() for e in events] == names[
                                impl
                            ] * args.burst
                            durations = [
                                (e.end_ns() - e.start_ns()) / 1000 for e in events
                            ]
                            row = {
                                "round": round_id,
                                "mode": f"{kind}_{count}",
                                "impl": impl,
                                "partial_us": durations[::2],
                                "final_us": durations[1::2],
                                "raw_trace": str(trace),
                                "trace_sha256": sha(trace),
                            }
                            result["captures"].append(row)
                            print(
                                round_id,
                                row["mode"],
                                impl,
                                row["partial_us"],
                                flush=True,
                            )
                control["input_hashes_preserved"] = all(
                    tensor_sha(v) == input_hash for v in inputs
                )
                assert control["input_hashes_preserved"]
            if args.ncu:
                torch.cuda._sleep(50_000_000)
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                for _ in range(4):
                    launch(args.ncu_impl)
                if args.ncu_control == "prime_flush":
                    flush_buffer.zero_()
                launch(args.ncu_impl)
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                result["ncu_control"] = args.ncu_control
                result["ncu_impl"] = args.ncu_impl
            for round_id in range(
                0
                if args.ncu or args.address_control or args.module_control
                else args.rounds
            ):
                # Rotate mode order as well as backend order across rounds.
                modes = args.modes
                rotate = round_id % len(modes)
                modes = modes[rotate:] + modes[:rotate]
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
                            priming = 4 if mode.startswith("prime") else 0
                            for _ in range(priming):
                                launch(impl)
                            if mode == "prime_idle":
                                torch.cuda.synchronize()
                                time.sleep(0.01)
                            elif mode == "prime_sleep":
                                torch.cuda._sleep(50_000_000)
                            if mode in ("flush", "prime_flush"):
                                flush_buffer.zero_()
                            if mode == "copy":
                                copy_buffer.copy_(x)
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
                        assert [e.name() for e in measured] == names[impl] * (
                            repeats + priming
                        )
                        extras = [
                            e.name() for e in events if e.name() not in names[impl]
                        ]
                        if mode in ("inside_warmup", "prime_sleep"):
                            assert len(extras) == 1 and "spin_kernel" in extras[0]
                        elif mode in ("flush", "prime_flush"):
                            assert len(extras) == 1 and "elementwise" in extras[0]
                        else:
                            assert not extras, extras
                        durations = [
                            (e.end_ns() - e.start_ns()) / 1000 for e in measured
                        ]
                        row = {
                            "round": round_id,
                            "mode": mode,
                            "impl": impl,
                            "start_unix_ns": started,
                            "end_unix_ns": finished,
                            "priming_partial_us": durations[: 2 * priming : 2],
                            "priming_final_us": durations[1 : 2 * priming : 2],
                            "extra_kernel_names": extras,
                            "partial_us": durations[2 * priming :: 2],
                            "final_us": durations[2 * priming + 1 :: 2],
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
    parser.add_argument("--address-control", action="store_true")
    parser.add_argument("--module-control", action="store_true")
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument("--ncu-impl", choices=["cubecl", "triton"], default="cubecl")
    parser.add_argument(
        "--ncu-control", choices=["prime", "prime_flush"], default="prime"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["sparse", "inside_warmup", "burst", "validated"],
        choices=[
            "sparse",
            "inside_warmup",
            "burst",
            "validated",
            "prime",
            "prime_idle",
            "prime_sleep",
            "prime_flush",
            "copy",
            "flush",
        ],
    )
    args = parser.parse_args()
    if sum((args.address_control, args.module_control, args.ncu)) > 1:
        parser.error("Choose only one of address control, module control or NCU")
    run(args)


if __name__ == "__main__":
    main()
