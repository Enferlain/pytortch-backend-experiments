"""Fixed complete-graph timing and separate replay code-generation evidence."""

import argparse
import gzip
import json
import statistics

from stage5.transpose_trial import (
    OUT,
    ROOT,
    copy_contiguous,
    fused_contiguous,
    sha,
    tensor_sha,
    write,
)


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


def graphs():
    torch = initialize()
    from stage5.backend import compile_backend as ordinary_backend
    from stage5.transpose_backend import compile_backend as component_backend
    from stage5 import runtime as ordinary
    from stage5 import transpose_runtime as component
    from stage5.transpose_qualify import check
    from stage5.rms_vector_trial import activity
    from torch._inductor import config

    assert not (OUT / "timing-plan.json").exists(), "predeclared experiment is one-shot"
    write(
        "timing-plan.json",
        dict(
            rounds=6,
            calls=16,
            order=[
                ["ordinary", "component"] if r % 2 == 0 else ["component", "ordinary"]
                for r in range(6)
            ],
            shapes=[[33, 257], [256, 513], [256, 512]],
            dtypes=["float32", "float16", "bfloat16"],
            modes=["fused", "copy"],
            measurement="CUPTI activity kernel durations of complete generated wrappers; no replay timing or synthetic primer",
        ),
    )
    rows = []
    (OUT / "traces").mkdir(exist_ok=True)
    with config.patch({"fx_graph_cache": False}):
        for dtype in ("float32", "float16", "bfloat16"):
            for mode, fn in (("fused", fused_contiguous), ("copy", copy_contiguous)):
                torch._dynamo.reset()
                functions = {
                    name: torch.compile(
                        fn, backend=backend, fullgraph=True, dynamic=True
                    )
                    for name, backend in (
                        ("ordinary", ordinary_backend),
                        ("component", component_backend),
                    )
                }
                for m, n in ((33, 257), (256, 513), (256, 512)):
                    label = f"{mode}-{dtype}-{m}x{n}"
                    torch.manual_seed(537 + m + n)
                    x = torch.randn(
                        (m, n), device="cuda", dtype=getattr(torch, dtype)
                    ).T
                    outputs = {name: f(x) for name, f in functions.items()}
                    for y in outputs.values():
                        check(torch, y, fn(x))
                        assert y.is_contiguous()
                    assert tensor_sha(outputs["ordinary"]) == tensor_sha(
                        outputs["component"]
                    )
                    candidates = [
                        k
                        for k in component.CACHE.values()
                        if k.semantic["buffers"][0]["dtype"] == dtype
                        and any(o["op"] == "sin" for o in k.semantic["ops"])
                        == (mode == "fused")
                    ]
                    from stage5.transpose_model import width

                    candidate = next(
                        k for k in candidates if k.vector == width(m, n, dtype)
                    )
                    normals = [
                        k
                        for k in ordinary.CACHE.values()
                        if k.semantic == candidate.semantic and k.last is not None
                    ]
                    assert len(normals) == 1
                    normal = normals[0].last
                    names = {
                        "ordinary": normal.kernel.manifest["entrypoint"],
                        "component": candidate.entrypoint,
                    }
                    row = dict(
                        label=label,
                        dtype=dtype,
                        mode=mode,
                        base_shape=[m, n],
                        shape=[n, m],
                        input_sha256=tensor_sha(x),
                        output_sha256=tensor_sha(outputs["component"]),
                        component=str(candidate.directory.relative_to(OUT)),
                        ordinary=str(normal.directory),
                        ordinary_plan=normal.kernel.plan.to_dict(),
                        kernels=names,
                        rounds=[],
                    )
                    for r in range(6):
                        rr = dict(round=r, activity_before=activity(), paths={})
                        for name in (
                            ("ordinary", "component")
                            if r % 2 == 0
                            else ("component", "ordinary")
                        ):
                            torch.cuda.synchronize()
                            with torch.profiler.profile(
                                activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA,
                                ]
                            ) as profiler:
                                for _ in range(16):
                                    functions[name](x)
                            events = [
                                e
                                for e in profiler.profiler.kineto_results.events()
                                if e.device_type() == torch.autograd.DeviceType.CUDA
                                and e.activity_type() == "kernel"
                            ]
                            assert len(events) == 16 and all(
                                e.name() == names[name] for e in events
                            ), (
                                label,
                                name,
                                [(e.name(), e.end_ns() - e.start_ns()) for e in events],
                            )
                            samples = [
                                (e.end_ns() - e.start_ns()) / 1000 for e in events
                            ]
                            assert min(samples) > 0
                            path = OUT / "traces" / f"{label}-{r}-{name}.json"
                            profiler.export_chrome_trace(str(path))
                            packed = path.with_suffix(".json.gz")
                            with packed.open("wb") as f:
                                f.write(gzip.compress(path.read_bytes(), mtime=0))
                            path.unlink()
                            rr["paths"][name] = dict(
                                samples_us=samples,
                                median_us=statistics.median(samples),
                                trace=str(packed.relative_to(OUT)),
                                trace_sha256=sha(packed),
                            )
                        rr["activity_after"] = activity()
                        row["rounds"].append(rr)
                        print(
                            label,
                            r,
                            {k: v["median_us"] for k, v in rr["paths"].items()},
                            flush=True,
                        )
                    for name, f in functions.items():
                        assert tensor_sha(f(x)) == row["output_sha256"]
                    assert tensor_sha(x) == row["input_sha256"]
                    rows.append(row)
                    write("timing.json", dict(complete=False, rows=rows))
    # Preserve the actual Inductor wrappers; the .debug source comments also
    # bind their descriptor substitutions, allocations and stride assertions.
    for p in (ROOT / "stage5/cache/reuse-first-transpose").rglob("*.py"):
        if "transpose_compile(" in p.read_text() or "stage5.runtime" in p.read_text():
            dest = OUT / "wrappers" / p.name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(p.read_bytes())
    write(
        "timing.json",
        dict(
            complete=True,
            rows=rows,
            source_sha256=sha(__file__),
            runtime_sha256=sha(ROOT / "stage5/transpose_runtime.py"),
        ),
    )


def counter():
    torch = initialize()
    from stage5.transpose_runtime import compile_kernel, TEMPLATES
    from stage5.transpose_qualify import baseline

    targets = []
    for sem in json.loads(TEMPLATES.read_text()):
        if sem["buffers"][0]["dtype"] != "float32":
            continue
        normal = baseline(sem)
        dispatch = compile_kernel(sem, 0)
        for m, n in ((256, 512), (256, 513)):
            x = torch.randn((m, n), device="cuda").T
            y = torch.empty((n, m), device="cuda")
            dispatch((n, m), x, y)
            normal_y = torch.empty_like(y)
            normal((n, m), x, normal_y)
            assert tensor_sha(y) == tensor_sha(normal_y)
            targets.extend([(normal, (n, m), x, normal_y), (dispatch, (n, m), x, y)])
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for kernel, scalars, x, y in targets:
        kernel(scalars, x, y)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("graphs", "counter"))
    args = parser.parse_args()
    {"graphs": graphs, "counter": counter}[args.phase]()
