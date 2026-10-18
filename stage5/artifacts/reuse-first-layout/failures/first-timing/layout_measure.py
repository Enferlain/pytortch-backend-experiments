"""Predeclared generated-wrapper CUPTI comparisons and separate SASS capture."""

import argparse
from functools import partial
import gzip
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import time

from stage5.layout_trial import OUT, ROOT, check, sha, tensor_sha, write


def initialize():
    from stage1 import evidence

    evidence.guard()
    from stage1.overlay import install

    install()
    import torch

    torch.set_num_threads(1)
    torch._dynamo.config.recompile_limit = 128
    assert json.loads((OUT / "gpu.json").read_text())["complete"]
    return torch


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
    from stage5.layout_backend import compile_backend as component_backend
    from stage5 import runtime, layout_runtime
    from stage5.layout_qualify import fixture

    assert not (OUT / "timing-plan.json").exists()
    orders = list(itertools.permutations(("baseline", "component", "ordinary")))
    selected = [
        r
        for r in json.loads((OUT / "qualification-plan.json").read_text())["cases"]
        if r["original"]
    ]
    assert len(selected) == 33
    write(
        "timing-plan.json",
        dict(
            rounds=6,
            calls=16,
            order=orders,
            cases=selected,
            measurement="CUPTI activity durations, complete actual generated wrappers; no synthetic primer or replay timing",
        ),
    )
    rows, functions = [], {}
    (OUT / "traces").mkdir(exist_ok=True)
    with config.patch({"fx_graph_cache": False}):
        for case in selected:
            key = (case["family"], case["dtype"])
            fn, args, _ = fixture(torch, case)
            if key not in functions:
                torch._dynamo.reset()
                functions[key] = {
                    name: torch.compile(
                        fn, backend=backend, fullgraph=True, dynamic=True
                    )
                    for name, backend in (
                        ("baseline", partial(ordinary_backend, tuning="baseline")),
                        ("component", component_backend),
                        ("ordinary", ordinary_backend),
                    )
                }
            funcs = functions[key]
            outputs = {n: f(*args) for n, f in funcs.items()}
            for y in outputs.values():
                check(torch, y, fn(*args))
            assert len({tensor_sha(y) for y in outputs.values()}) == 1
            comp = next(
                k
                for k in layout_runtime.CACHE.values()
                if layout_runtime.admit(k.semantic) == key
            )
            normals = {
                policy: next(
                    k.last
                    for k in runtime.CACHE.values()
                    if k.semantic == comp.semantic
                    and k.policy == policy
                    and k.last is not None
                )
                for policy in ("baseline", "first_use")
            }
            paths = {
                "component": comp,
                "baseline": normals["baseline"],
                "ordinary": normals["first_use"],
            }
            # The matched-work-assignment baseline must be the captured binary.
            captured = next(
                r
                for r in json.loads((OUT / "capture.json").read_text())["rows"]
                if r["case"]["family"] == key[0] and r["case"]["dtype"] == key[1]
            )
            assert sha(paths["baseline"].directory / "kernel.ptx") == sha(
                OUT / captured["baseline"] / "kernel.ptx"
            )
            names = {n: p.kernel.manifest["entrypoint"] for n, p in paths.items()}
            row = dict(
                label=case["label"],
                case=case,
                input_sha256=[tensor_sha(t) for t in args],
                output_sha256=tensor_sha(outputs["component"]),
                kernels=names,
                artifacts={
                    n: dict(
                        directory=str(p.directory),
                        plan=p.kernel.plan.to_dict(),
                        ptx_sha256=sha(p.directory / "kernel.ptx"),
                    )
                    for n, p in paths.items()
                },
                rounds=[],
            )
            for r, order in enumerate(orders):
                rr = dict(round=r, order=order, activity_before=activity(), paths={})
                for name in order:
                    torch.cuda.synchronize()
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profiler:
                        for _ in range(16):
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
                    {n: v["median_us"] for n, v in rr["paths"].items()},
                    flush=True,
                )
            assert [tensor_sha(t) for t in args] == row["input_sha256"]
            for f in funcs.values():
                assert tensor_sha(f(*args)) == row["output_sha256"]
            rows.append(row)
            write("timing.json", dict(complete=False, rows=rows))
    for p in (ROOT / "stage5/cache/reuse-first-layout").rglob("*.py"):
        if "layout_compile(" in p.read_text() or "stage5.runtime" in p.read_text():
            dest = OUT / "wrappers" / p.name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(p.read_bytes())
    write("timing.json", dict(complete=True, rows=rows, source_sha256=sha(__file__)))


def counter():
    torch = initialize()
    from stage5.artifact import Kernel
    from stage5.layout_qualify import fixture

    timing = json.loads((OUT / "timing.json").read_text())
    assert timing["complete"]
    targets = []
    for family, shape in (
        ("bias_row", [256, 4099]),
        ("bias_column", [256, 4099]),
        ("slice", [1048583]),
        ("stride_zero", [64, 257]),
    ):
        row = next(
            r
            for r in timing["rows"]
            if r["case"]["family"] == family
            and r["case"]["shape"] == shape
            and r["case"]["dtype"] == "float32"
        )
        from stage5.layout_runtime import compile_kernel

        sem = json.loads(
            (
                Path(row["artifacts"]["component"]["directory"]) / "semantic.json"
            ).read_text()
        )
        fn, args, _ = fixture(torch, row["case"])
        for name in ("baseline", "component", "ordinary"):
            a = row["artifacts"][name]
            k = (
                compile_kernel(sem, 0)
                if name == "component"
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
