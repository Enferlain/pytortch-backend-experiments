"""Qualification of normal admission, preserving the old diagnostic packets."""

import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/adoption"


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def setup():
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    from stage1.overlay import install

    install()
    import torch

    # Historical helpers set their own cache at import. Load them first.
    from stage5.layout_qualify import fixture as fixture
    from stage5.column_layout_runtime import generic as generic

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache/column-admission")
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    return torch


def controls():
    torch = setup()
    from stage0.bridge import nvrtc_compile, native_path
    from stage5.artifact import CompiledKernel
    from stage5.column_plan import STRATEGY, semantic_reason, runtime_reason
    from stage5.plan import PhysicalCandidate
    from stage5.layout_runtime import templates
    from stage5.column_layout_runtime import generic
    from stage5.layout_qualify import fixture
    from stage5.layout_trial import check, sha, tensor_sha
    from stage5.legality import candidates, signature

    cpu = json.loads((OUT / "column-cpu.json").read_text())
    assert cpu["complete"] and not cpu["gpu_source_existed"] and len(cpu["rows"]) == 12
    old = ROOT / "artifacts/column-layout"
    frozen = json.loads((old / "post-review-gpu.json").read_text())
    cases = json.loads((old / "qualification-plan.json").read_text())["cases"]
    plan = PhysicalCandidate(STRATEGY)
    variants = json.loads((ROOT / "column_templates.json").read_text())
    compiled = {}
    identities = []
    include = [str(Path(torch.__file__).parent.parent / "nvidia/cu13/include")]
    for i, sem in enumerate(variants):
        k = CompiledKernel(sem, 0, plan.to_dict())
        compiled[json.dumps(sem, sort_keys=True)] = k
        record = next(
            r
            for r in frozen["kernels"]
            if r["semantic"]["buffers"][0]["dtype"] == sem["buffers"][0]["dtype"]
        )
        source = Path(record["directory"])
        packet = json.loads((source / "descriptor.json").read_text())
        packet.update(semantic=sem, physical_plan=plan.to_dict())
        dest = OUT / "column-identity" / str(i)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "descriptor.json").write_text(json.dumps(packet))
        run = subprocess.run(
            [
                str(native_path("cubecl-stage0")),
                str(dest),
                "cpp",
                str(dest / "descriptor.json"),
            ],
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, run.stderr
        assert (dest / "metadata-257.bin").stat().st_size == 80
        nvrtc_compile((dest / "kernel.cu").read_text(), dest, 0, include)
        matches = {
            name: sha(dest / name) == sha(source / name)
            for name in ("cubecl.ir", "kernel.cu", "kernel.ptx")
        }
        assert all(matches.values()), matches
        identities.append(dict(variant=i, source=str(source), matches=matches))
    rows, live, streams = [], [], []
    for family, dtype, sem in templates().values():
        baseline = generic(sem, 0)
        tests = [r for r in cases if r["family"] == family and r["dtype"] == dtype]
        selected = next(r for r in tests if r["original"])
        for row, special in [(r, False) for r in tests] + [(selected, True)]:
            fn, args, stores = fixture(torch, row, special)
            y = torch.empty_like(fn(*args))
            ref = torch.empty_like(y)
            admitted = runtime_reason(sem, row["scalars"]) is None
            current = (
                compiled[json.dumps(sem, sort_keys=True)]
                if admitted
                else CompiledKernel(
                    sem, 0, PhysicalCandidate("scalar_pointwise_baseline").to_dict()
                )
            )
            current(row["scalars"], *args, y)
            baseline(row["scalars"], *args, ref)
            check(torch, y, fn(*args))
            assert tensor_sha(y) == tensor_sha(ref), row
            prior = next(
                r
                for r in frozen["rows"]
                if r["label"] == row["label"] + ("-special" if special else "")
            )
            assert [tensor_sha(t) for t in args] == prior["input_sha256"]
            assert tensor_sha(y) == prior["output_sha256"]
            assert all(bool((s[:8] == 42).all() & (s[-8:] == 42).all()) for s in stores)
            rows.append(
                dict(
                    label=prior["label"],
                    admitted=admitted,
                    output_sha256=tensor_sha(y),
                    frozen_bitwise=True,
                )
            )
    for i, sem in enumerate(variants):
        kernel = compiled[json.dumps(sem, sort_keys=True)]
        dtype = sem["buffers"][0]["dtype"]
        for b, n in ((3, 255), (64, 256), (64, 257)):
            row = dict(
                family="bias_column",
                dtype=dtype,
                shape=[b, n],
                scalars=[n, b],
                label=f"alias-{i}-{b}-{n}",
            )
            fn, args, stores = fixture(torch, row, special=True)
            backing = torch.full(
                (b * n + 16,), 42.0, dtype=args[0].dtype, device="cuda"
            )
            y = backing[8:-8].view(b, n)
            kernel((n, b), *args, y)
            check(torch, y, fn(*args))
            assert bool((backing[:8] == 42).all() & (backing[-8:] == 42).all())
        fn, args, _ = fixture(torch, row)
        y = torch.empty_like(args[0])
        attempts = [
            ("shape", (258, b), (*args, y)),
            ("arity", (n,), (*args, y)),
            ("device", (n, b), (args[0].cpu(), args[1], y)),
            ("dtype", (n, b), (args[0].float(), args[1], y)),
            ("layout", (n, b), (args[0].T, args[1], y)),
            ("alias", (n, b), (*args, args[0])),
        ]
        for label, scalars, tensors in attempts:
            before = kernel.kernel.launch_count
            try:
                kernel(scalars, *tensors)
            except (ValueError, TypeError) as error:
                live.append(dict(variant=i, label=label, error=str(error)))
            else:
                raise AssertionError(label)
            assert kernel.kernel.launch_count == before
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            kernel((n, b), *args, y)
            expected = fn(*args)
        stream.synchronize()
        check(torch, y, expected)
        streams.append(dict(variant=i, passes=True))
        # The finite admission fact must distinguish same-bucket neighbors.
        target = kernel.kernel.resources
        good = signature(sem, (n, b), (*args, y))
        bad = signature(sem, (258, b), (*args, y))
        assert good != bad
        accepted, _ = candidates(sem, (n, b), (*args, y), target)
        excluded, _ = candidates(sem, (258, b), (*args, y), target)
        assert plan in accepted and plan not in excluded
        outside = (
            torch.empty((b, 258), device="cuda", dtype=args[0].dtype),
            torch.empty((b, 1), device="cuda", dtype=args[0].dtype),
            torch.empty((b, 258), device="cuda", dtype=args[0].dtype),
        )
        before = kernel.kernel.launch_count
        try:
            kernel((258, b), *outside)
        except ValueError as error:
            assert "outside initial qualified shapes" in str(error)
            live.append(
                dict(variant=i, label="valid_neighbor_layout", error=str(error))
            )
        else:
            raise AssertionError("valid neighboring layout inherited eligibility")
        assert kernel.kernel.launch_count == before
    rejects = []
    base = variants[0]
    for label in ("fp32", "index", "precision", "effect", "boolean", "extra"):
        sem = copy.deepcopy(base)
        if label == "fp32":
            sem["buffers"][0]["dtype"] = "float32"
        elif label == "index":
            sem["ops"][1]["index"] = {"op": "constant", "value": 0}
        elif label == "precision":
            sem["index_dtype"] = "int32"
        elif label == "effect":
            sem["effects"]["mutation"] = True
        elif label == "boolean":
            sem["scalars"][0]["positive"] = 1
        else:
            sem["unknown"] = 1
        assert semantic_reason(sem), label
        dest = OUT / "column-rejections" / label
        dest.mkdir(parents=True, exist_ok=True)
        packet = dict(
            schema=5,
            semantic=sem,
            physical_plan=plan.to_dict(),
            entrypoint="cubecl5_column_reject",
        )
        (dest / "descriptor.json").write_text(json.dumps(packet))
        result = subprocess.run(
            [
                str(native_path("cubecl-stage0")),
                str(dest),
                "cpp",
                str(dest / "descriptor.json"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, label
        rejects.append(dict(label=label, error=result.stderr))
    write(
        "column-controls.json",
        dict(
            complete=True,
            identities=identities,
            rows=rows,
            live=live,
            streams=streams,
            native_rejections=rejects,
            artifacts=[str(k.directory) for k in compiled.values()],
        ),
    )
    print(
        len(rows),
        "full corpus cases,",
        len(live),
        "live rejections,",
        len(streams),
        "streams; identity and native admission PASS",
        flush=True,
    )


def graphs():
    torch = setup()
    from torch._inductor import config
    from stage5 import runtime
    from stage5.backend import compile_backend
    from stage5.column_plan import STRATEGY, runtime_reason
    from stage5.reduction_codegen_mechanisms import Component
    from stage5.layout_trial import sha, tensor_sha

    assert json.loads((OUT / "column-controls.json").read_text())["complete"]
    plan = {
        "shapes": [[3, 255], [64, 256], [64, 257]],
        "dtypes": ["float16", "bfloat16"],
        "rounds": 6,
        "calls": 16,
        "order": [["normal", "retained"], ["retained", "normal"]] * 3,
    }
    write("column-timing-plan.json", plan)
    rows = []
    (OUT / "traces").mkdir(exist_ok=True)
    observed = []
    original = runtime.AutotunedKernel.__call__

    def observe(obj, *args, **kw):
        result = original(obj, *args, **kw)
        observed.append((obj, Path(sys._getframe(1).f_code.co_filename)))
        return result

    with config.patch({"fx_graph_cache": False}):
        for dtype in plan["dtypes"]:
            td = getattr(torch, dtype)
            f = torch.compile(
                lambda x, bias: x + bias,
                backend=compile_backend,
                fullgraph=True,
                dynamic=True,
            )
            for b, n in plan["shapes"]:
                torch.manual_seed(9351 + b + n)
                x, bias = (
                    torch.randn((b, n), device="cuda", dtype=td),
                    torch.randn((b, 1), device="cuda", dtype=td),
                )
                with patch.object(runtime.AutotunedKernel, "__call__", observe):
                    y = f(x, bias)
                obj, wrapper = observed[-1]
                assert wrapper.is_relative_to(ROOT / "cache/column-admission"), wrapper
                assert runtime_reason(obj.semantic, (n, b)) is None, obj.semantic
                saved = [
                    json.loads(
                        (ROOT / "cache/tuning" / key / "selection.json").read_text()
                    )
                    for key in obj.selections
                ]
                assert any(
                    any(c["plan"]["strategy"] == STRATEGY for c in s["candidates"])
                    for s in saved
                )
                torch.testing.assert_close(y, x + bias, rtol=0, atol=0)
                old_candidates = []
                other_extent = copy.deepcopy(obj.semantic)
                other_extent["numel"]["args"].reverse()
                for directory in (ROOT / "artifacts/composition/ordinary").iterdir():
                    sem = json.loads((directory / "semantic.json").read_text())
                    if sem in (obj.semantic, other_extent):
                        old_candidates.append(directory)
                assert len(old_candidates) == 1, old_candidates
                old = Component(old_candidates[0])
                old_y = torch.empty_like(y)

                def retained():
                    old((n, b), x, bias, old_y)

                retained()
                assert tensor_sha(old_y) == tensor_sha(y)
                names = {
                    "normal": obj.last.entrypoint,
                    "retained": old.kernel.manifest["entrypoint"],
                }
                row = dict(
                    dtype=dtype,
                    shape=[b, n],
                    selected=obj.last.kernel.plan.to_dict(),
                    normal_artifact=str(obj.last.directory),
                    retained_artifact=str(old.directory),
                    wrappers={"path": str(wrapper), "sha256": sha(wrapper)},
                    input_sha256=[tensor_sha(x), tensor_sha(bias)],
                    output_sha256=tensor_sha(y),
                    rounds=[],
                )
                for r, order in enumerate(plan["order"]):
                    rr = dict(
                        order=order,
                        paths={},
                        activity_before=subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=uuid,clocks.current.sm,clocks.current.memory",
                                "--format=csv,noheader",
                            ],
                            text=True,
                        ),
                    )
                    for path in order:
                        torch.cuda.synchronize()
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as prof:
                            for _ in range(plan["calls"]):
                                f(x, bias) if path == "normal" else retained()
                        events = [
                            e
                            for e in prof.profiler.kineto_results.events()
                            if e.device_type() == torch.autograd.DeviceType.CUDA
                            and e.activity_type() == "kernel"
                        ]
                        assert len(events) == 16 and all(
                            e.name() == names[path] for e in events
                        )
                        values = [(e.end_ns() - e.start_ns()) / 1000 for e in events]
                        trace = OUT / "traces" / f"{dtype}-{b}-{n}-{r}-{path}.json"
                        prof.export_chrome_trace(str(trace))
                        packed = trace.with_suffix(".json.gz")
                        packed.write_bytes(gzip.compress(trace.read_bytes(), mtime=0))
                        trace.unlink()
                        rr["paths"][path] = dict(
                            samples_us=values,
                            median_us=statistics.median(values),
                            trace=str(packed.relative_to(OUT)),
                            trace_sha256=sha(packed),
                        )
                    row["rounds"].append(rr)
                row["paired_wins"] = sum(
                    rr["paths"]["normal"]["median_us"]
                    < rr["paths"]["retained"]["median_us"]
                    for rr in row["rounds"]
                )
                rows.append(row)
                write("column-graphs.json", dict(complete=False, rows=rows))
                print(
                    dtype,
                    b,
                    n,
                    row["selected"]["strategy"],
                    row["paired_wins"],
                    "/6 wins",
                    flush=True,
                )
            # Same symbolic kernel must exclude a valid same-bucket neighbor.
            x, bias = (
                torch.randn((64, 258), device="cuda", dtype=td),
                torch.randn((64, 1), device="cuda", dtype=td),
            )
            with patch.object(runtime.AutotunedKernel, "__call__", observe):
                y = f(x, bias)
            obj, _ = observed[-1]
            assert obj.last.kernel.plan.strategy != STRATEGY
            torch.testing.assert_close(y, x + bias, rtol=0, atol=0)
    write(
        "column-graphs.json",
        dict(
            complete=True,
            rows=rows,
            neighbor_fallbacks=2,
            scope="normal generated one-kernel wrappers versus retained pre-admission ordinary selected kernels; GPU durations only",
        ),
    )


def cache():
    torch = setup()
    from stage5 import artifact, runtime
    from stage5.column_plan import STRATEGY

    sem = json.loads((ROOT / "column_templates.json").read_text())[0]
    obj = runtime.compile_kernel(sem, 0)
    dtype = getattr(torch, sem["buffers"][0]["dtype"])
    x, bias = (
        torch.randn((64, 257), device="cuda", dtype=dtype),
        torch.randn((64, 1), device="cuda", dtype=dtype),
    )
    y = torch.empty_like(x)
    obj((257, 64), x, bias, y)
    torch.testing.assert_close(y, x + bias, rtol=0, atol=0)
    assert any(
        p.strategy == STRATEGY for plans, _ in obj._legality.values() for p in plans
    )
    if (OUT / "column-cache-init.json").exists():
        assert artifact.COMPILER_INVOCATIONS == 0
        assert any(e["event"] == "tuning_cache_hit" for e in runtime.EVENTS)
        write(
            "column-cache-reload.json",
            dict(complete=True, compiler_invocations=0, events=runtime.EVENTS),
        )
    else:
        write("column-cache-init.json", dict(complete=True, events=runtime.EVENTS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("controls", "graphs", "cache"))
    mode = parser.parse_args().mode
    globals()[mode]()
