"""Fixed column graph comparison; actual wrapper provenance and separate SASS."""

import argparse
from functools import partial
import gzip
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

from stage5.column_layout_trial import (
    OUT,
    ROOT,
    FROZEN,
    read,
    write,
    sha,
    tensor_sha,
    check,
    initialize,
)


def activity():
    p = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,display_active,utilization.gpu,clocks.current.sm,clocks.current.memory,power.draw",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
    )
    return dict(
        time_ns=time.time_ns(),
        returncode=p.returncode,
        stdout=p.stdout,
        stderr=p.stderr,
    )


def graphs():
    torch = initialize()
    from torch._inductor import config
    from stage5.backend import compile_backend as ordinary_backend
    from stage5.column_layout_backend import compile_backend as component_backend
    from stage5 import runtime, column_layout_runtime as component
    from stage5.layout_qualify import fixture

    assert read(OUT / "gpu.json")["complete"]
    assert not (OUT / "timing.json").exists()
    plan = read(OUT / "timing-plan.json")
    cases = read(OUT / "qualification-plan.json")["cases"]
    # Preserve original symbolic descriptor construction history. Non-column
    # calls use only frozen generic PTX and are setup, never timed or optimized.
    ordered = [
        next(c for c in cases if c["label"] == r["case"]["name"])
        for r in read(FROZEN / "capture.json")["rows"]
    ]
    rows, functions = [], {}
    (OUT / "traces").mkdir(exist_ok=True)
    with config.patch({"fx_graph_cache": False}):
        for case in ordered:
            key = (case["family"], case["dtype"])
            fn, args, _ = fixture(torch, case)
            if key not in functions:
                functions[key] = {
                    name: torch.compile(fn, backend=b, fullgraph=True, dynamic=True)
                    for name, b in (
                        ("generic", partial(component_backend, frozen_generic=True)),
                        ("component", component_backend),
                        ("ordinary", ordinary_backend),
                    )
                }
            funcs = functions[key]
            observed = {}
            active = [None]

            def observe(original):
                def call(self, *a, **kw):
                    path = Path(sys._getframe(1).f_code.co_filename)
                    if path.is_relative_to(ROOT / "stage5/cache/column-layout"):
                        observed.setdefault(active[0], set()).add(path)
                    return original(self, *a, **kw)

                return call

            outputs = {}
            # Observe existing setup calls only; timed calls are unpatched.
            with (
                patch.object(
                    component.Dispatch, "__call__", observe(component.Dispatch.__call__)
                ),
                patch.object(
                    runtime.AutotunedKernel,
                    "__call__",
                    observe(runtime.AutotunedKernel.__call__),
                ),
            ):
                for name, f in funcs.items():
                    active[0] = name
                    outputs[name] = f(*args)
            for y in outputs.values():
                check(torch, y, fn(*args))
            assert len({tensor_sha(y) for y in outputs.values()}) == 1
            if case["family"] != "bias_column":
                continue
            comp = next(
                k
                for k in component.CACHE.values()
                if component.admit(k.semantic) == key
            )
            gen = component.generic(comp.semantic, 0)
            normal = next(
                k.last
                for k in runtime.CACHE.values()
                if k.semantic == comp.semantic
                and k.policy == "first_use"
                and k.last is not None
            )
            paths = {"generic": gen, "component": comp, "ordinary": normal}
            names = {n: k.kernel.manifest["entrypoint"] for n, k in paths.items()}
            wrappers = {}
            for name in names:
                assert len(observed.get(name, ())) == 1, (case, name, observed)
                source = next(iter(observed[name]))
                dest = (
                    OUT
                    / "wrappers"
                    / source.relative_to(ROOT / "stage5/cache/column-layout")
                )
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    assert dest.read_bytes() == source.read_bytes()
                else:
                    dest.write_bytes(source.read_bytes())
                wrappers[name] = dict(
                    path=str(dest.relative_to(OUT)),
                    sha256=sha(dest),
                    observed_cache_path=str(source),
                )
            row = dict(
                label=case["label"],
                case=case,
                input_sha256=[tensor_sha(t) for t in args],
                output_sha256=tensor_sha(outputs["component"]),
                kernels=names,
                wrappers=wrappers,
                artifacts={
                    n: dict(
                        directory=str(k.directory),
                        plan=k.kernel.plan.to_dict(),
                        ptx_sha256=sha(k.directory / "kernel.ptx"),
                    )
                    for n, k in paths.items()
                },
                rounds=[],
            )
            for r, order in enumerate(plan["order"]):
                rr = dict(round=r, order=order, activity_before=activity(), paths={})
                for name in order:
                    torch.cuda.synchronize()
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profiler:
                        for _ in range(plan["calls"]):
                            funcs[name](*args)
                    events = [
                        e
                        for e in profiler.profiler.kineto_results.events()
                        if e.device_type() == torch.autograd.DeviceType.CUDA
                        and e.activity_type() == "kernel"
                    ]
                    assert len(events) == 16 and all(
                        e.name() == names[name] for e in events
                    ), (case, name, [e.name() for e in events])
                    samples = [(e.end_ns() - e.start_ns()) / 1000 for e in events]
                    trace = OUT / "traces" / f"{case['label']}-{r}-{name}.json"
                    profiler.export_chrome_trace(str(trace))
                    packed = trace.with_suffix(".json.gz")
                    packed.write_bytes(gzip.compress(trace.read_bytes(), mtime=0))
                    trace.unlink()
                    rr["paths"][name] = dict(
                        samples_us=samples,
                        median_us=statistics.median(samples),
                        trace=str(packed.relative_to(OUT)),
                        trace_sha256=sha(packed),
                    )
                rr["activity_after"] = activity()
                row["rounds"].append(rr)
                print(
                    case["label"],
                    r,
                    {n: p["median_us"] for n, p in rr["paths"].items()},
                    flush=True,
                )
            assert [tensor_sha(t) for t in args] == row["input_sha256"]
            for f in funcs.values():
                assert tensor_sha(f(*args)) == row["output_sha256"]
            rows.append(row)
            write("timing.json", dict(complete=False, rows=rows))
    assert len(rows) == len(plan["cases"]) == 12
    write(
        "timing.json",
        dict(
            complete=True,
            rows=rows,
            source_sha256=sha(__file__),
            backend_sha256=sha(ROOT / "stage5/column_layout_backend.py"),
            plan_sha256=sha(OUT / "timing-plan.json"),
            observer="Exact generated wrapper caller filename recorded during existing setup; no observer in timed region.",
        ),
    )


def counter():
    torch = initialize()
    from stage5.column_layout_runtime import compile_kernel, generic
    from stage5.artifact import Kernel
    from stage5.layout_qualify import fixture

    timing = read(OUT / "timing.json")
    assert timing["complete"]
    targets = []
    for shape in read(OUT / "timing-plan.json")["counter_shapes"]:
        row = next(
            r
            for r in timing["rows"]
            if r["case"]["shape"] == shape and r["case"]["dtype"] == "float32"
        )
        sem = read(Path(row["artifacts"]["component"]["directory"]) / "semantic.json")
        fn, args, _ = fixture(torch, row["case"])
        for name in ("generic", "component", "ordinary"):
            a = row["artifacts"][name]
            assert sha(Path(a["directory"]) / "kernel.ptx") == a["ptx_sha256"]
            k = (
                compile_kernel(sem, 0)
                if name == "component"
                else generic(sem, 0)
                if name == "generic"
                else Kernel(Path(a["directory"]), sem, 0, a["plan"], compile_ptx=False)
            )
            y = torch.empty_like(fn(*args))
            k(row["case"]["scalars"], *args, y)
            check(torch, y, fn(*args))
            targets.append((k, row["case"]["scalars"], args, y))
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for k, scalars, args, y in targets:
        k(scalars, *args, y)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("graphs", "counter"))
    args = parser.parse_args()
    {"graphs": graphs, "counter": counter}[args.phase]()
