"""Qualification and complete-graph CUPTI ablations of two frozen components."""

from functools import partial
import copy
import gzip
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

from stage5.composition_trial import ROOT, OUT, initialize, read, write, sha, parallel


def references(torch):
    return torch.load(ROOT / "stage5/artifacts/reuse-first-rms-vector4/reference.pt", weights_only=True)


def rms_label(case):
    prefix = "corpus" if case["rms_rows"] == 64 else "vector_row_tail"
    return f'{prefix}_{case["dtype"]}_{case["rms_rows"]}_1024'


def column_case(case):
    return next(c for c in read(ROOT / "stage5/artifacts/reuse-first-layout/qualification-plan.json")["cases"]
                if c["family"] == "bias_column" and c["dtype"] == case["dtype"]
                and c["shape"] == case["column_shape"])


def gpu():
    torch = initialize()
    from stage5.composition_runtime import compile_kernel, source, CACHE
    from stage5.rms_packed_trial_runtime import templates as rms_templates
    from stage5.layout_runtime import templates as layout_templates
    from stage5.layout_qualify import fixture
    from stage5.rms_trial import qualify
    from stage5.layout_trial import tensor_sha, check

    assert not (OUT / "gpu.json").exists()
    rows, guards, streams, mutations = [], [], [], []
    data = references(torch)
    rms_semantics = rms_templates()
    old = read(ROOT / "stage5/artifacts/reuse-first-rms-packed-trial/gpu.json")
    assert old["complete"]
    for label, packet in data.items():
        x0 = packet["input"]
        b, n = x0.shape
        dtype = str(x0.dtype).removeprefix("torch.")
        k = compile_kernel(rms_semantics[dtype], 0, "both")
        xs = torch.full((x0.numel() + 8,), 42, dtype=x0.dtype, device="cuda")
        ys = torch.full_like(xs, 42)
        x, y = xs[4:-4].view(b, n), ys[4:-4].view(b, n)
        x.copy_(x0)
        k((n, b), x, y)
        result = qualify(torch, label, x0, y.cpu(), packet)
        prior = next(r for r in old["rows"] if r["case"] == label)
        assert result["passes"] and tensor_sha(y) == prior["output_sha256"]
        assert tensor_sha(x) == tensor_sha(x0)
        assert all(bool((t[:4] == 42).all() & (t[-4:] == 42).all()) for t in (xs, ys))
        rows.append(dict(label=label, kind="rms", input_sha256=tensor_sha(x),
                         output_sha256=tensor_sha(y), policy=True, frozen_bitwise=True, canaries=True))
    # All changed column cases from l1m, including its three special rows.
    prior = read(ROOT / "stage5/artifacts/column-layout/post-review-gpu.json")
    plan = read(ROOT / "stage5/artifacts/reuse-first-layout/qualification-plan.json")["cases"]
    for dtype in ("float32", "float16", "bfloat16"):
        semantic = next(s for family, d, s in layout_templates().values()
                        if (family, d) == ("bias_column", dtype))
        k = compile_kernel(semantic, 0, "both")
        tests = [r for r in plan if r["family"] == "bias_column" and r["dtype"] == dtype]
        for row, special in [(r, False) for r in tests] + [(tests[0], True)]:
            fn, args, stores = fixture(torch, row, special)
            expected = fn(*args)
            ys = torch.full((expected.numel() + 16,), 42, dtype=expected.dtype, device="cuda")
            y = ys[8:-8].view(expected.shape)
            inputs = [tensor_sha(t) for t in args]
            k(row["scalars"], *args, y)
            check(torch, y, expected)
            label = row["label"] + ("-special" if special else "")
            oldrow = next(r for r in prior["rows"] if r["label"] == label)
            assert tensor_sha(y) == oldrow["output_sha256"] and inputs == oldrow["input_sha256"]
            assert inputs == [tensor_sha(t) for t in args]
            assert all(bool((t[:8] == 42).all() & (t[-8:] == 42).all()) for t in stores + [ys])
            rows.append(dict(label=label, kind="column", input_sha256=inputs,
                             output_sha256=tensor_sha(y), policy=True, frozen_bitwise=True, canaries=True))
        for kind, sem in (("column", semantic), ("rms", rms_semantics[dtype])):
            owned = copy.deepcopy(sem)
            dispatch = compile_kernel(owned, 0, "both")
            owned["buffers"].clear()
            owned["scalars"].clear()
            assert dispatch.semantic == sem == dispatch.kernel.semantic
            assert compile_kernel(copy.deepcopy(sem), 0, "both") is dispatch
            mutations.append(dict(kind=kind, dtype=dtype, isolated=True))
            if kind == "column":
                fn, args, _ = fixture(torch, tests[1])
                scalars = tests[1]["scalars"]
                expected = fn(*args)
            else:
                x0 = data[f"corpus_{dtype}_64_1024"]["input"]
                args = (x0.cuda(),)
                scalars = (1024, 64)
                expected = None
            y = torch.empty_like(args[0])
            attempts = [("scalar", (0, *scalars[1:]), (*args, y)),
                        ("arity", scalars[:-1], (*args, y)),
                        ("dtype", scalars, (args[0].double(), *args[1:], y)),
                        ("layout", scalars, (args[0].flatten(), *args[1:], y)),
                        ("alias", scalars, (*args, args[0]))]
            if kind == "rms":
                raw = torch.empty(args[0].numel() + 1, device="cuda", dtype=args[0].dtype)
                attempts.append(("unaligned", scalars, (raw[1:].view_as(args[0]), y)))
            for name, ss, tensors in attempts:
                before = dispatch.kernel.launch_count
                try:
                    dispatch(ss, *tensors)
                except (ValueError, TypeError):
                    pass
                else:
                    raise AssertionError("missing prelaunch rejection " + name)
                assert dispatch.kernel.launch_count == before
                guards.append(dict(kind=kind, dtype=dtype, name=name, prelaunch=True))
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                sargs = []
                for t in args:
                    store = torch.empty(t.numel() + 16, device="cuda", dtype=t.dtype)
                    view = store[8:-8].view_as(t)
                    view.copy_(t)
                    sargs.append(view)
                sy = torch.empty_like(y)
                dispatch(scalars, *sargs, sy)
            stream.synchronize()
            if kind == "rms":
                assert tensor_sha(sy) == next(r["output_sha256"] for r in rows if r["label"] == f"corpus_{dtype}_64_1024")
            else:
                check(torch, sy, expected)
            streams.append(dict(kind=kind, dtype=dtype, passes=True, stream=stream.cuda_stream, offset=8))
    artifacts = []
    for k in CACHE.values():
        origin = source(k.kind, k.semantic["buffers"][0]["dtype"])
        files = {p.name: sha(p) for p in k.directory.iterdir() if p.is_file()}
        assert all(sha(origin / name) == digest for name, digest in files.items())
        artifacts.append(dict(kind=k.kind, directory=str(k.directory), source=str(origin),
                              semantic=k.semantic, files=files, attributes=k.kernel.attributes))
    assert len(rows) == 60 and len(guards) == 33 and len(streams) == len(mutations) == len(artifacts) == 6
    write("gpu.json", dict(complete=True, rows=rows, guards=guards, streams=streams,
                           mutations=mutations, artifacts=artifacts, source_sha256=sha(__file__)))
    print("PASS60 frozen-corpus cases/33 prelaunch guards/6 streams/6 caller-isolation controls", flush=True)


def graphs():
    torch = initialize()
    from torch._inductor import config
    from stage5.composition_backend import compile_backend
    from stage5.composition_runtime import Dispatch
    from stage5 import runtime
    from stage5.layout_qualify import fixture
    from stage5.layout_trial import tensor_sha, check
    from stage5.rms_trial import qualify
    from stage5.column_layout_measure import activity

    assert read(OUT / "gpu.json")["complete"]
    assert not (OUT / "timing.json").exists()
    plan, data = read(OUT / "plan.json"), references(torch)
    modes = plan["modes"]
    functions, cases = {}, []
    active, seen = [None], {}

    def observe(original):
        def call(self, *args, **kw):
            wrapper = Path(sys._getframe(1).f_code.co_filename)
            assert wrapper.is_relative_to(ROOT / "stage5/cache/composition"), wrapper
            result = original(self, *args, **kw)
            target = self if isinstance(self, Dispatch) else self.last
            seen.setdefault(active[0], []).append((target, wrapper))
            return result
        return call

    # All graph qualification/setup completes before any admitted timing capture.
    with config.patch({"fx_graph_cache": False}):
        for case in plan["cases"]:
            dtype = case["dtype"]
            if dtype not in functions:
                functions[dtype] = {mode: torch.compile(parallel, backend=partial(compile_backend, mode=mode),
                                                       fullgraph=True, dynamic=True) for mode in modes}
            fns = functions[dtype]
            packet = data[rms_label(case)]
            r = packet["input"].cuda()
            row = column_case(case)
            _, (x, bias), _ = fixture(torch, row)
            inputs = (r, x, bias)
            label = f'{dtype}-{case["rms_rows"]}-{case["column_shape"][0]}x{case["column_shape"][1]}'
            seen.clear()
            outputs, selected, wrappers = {}, {}, {}
            with patch.object(Dispatch, "__call__", observe(Dispatch.__call__)), patch.object(
                runtime.AutotunedKernel, "__call__", observe(runtime.AutotunedKernel.__call__)
            ):
                for mode in modes:
                    active[0] = mode
                    outputs[mode] = fns[mode](*inputs)
            expected_rms = next(rr["output_sha256"] for rr in read(OUT / "gpu.json")["rows"]
                                if rr["label"] == rms_label(case))
            for mode in modes:
                actual = outputs[mode]
                assert qualify(torch, rms_label(case), packet["input"], actual[0].cpu(), packet)["passes"]
                check(torch, actual[1], x + bias)
                if mode in ("rms", "both"):
                    assert tensor_sha(actual[0]) == expected_rms
                assert tensor_sha(actual[1]) == tensor_sha(outputs["ordinary"][1])
                assert len(seen[mode]) == 2
                selected[mode] = []
                paths = {w for _, w in seen[mode]}
                assert len(paths) == 1
                wrapper = next(iter(paths))
                dest = OUT / "wrappers" / wrapper.name
                dest.parent.mkdir(exist_ok=True)
                if dest.exists():
                    assert dest.read_bytes() == wrapper.read_bytes()
                else:
                    dest.write_bytes(wrapper.read_bytes())
                wrappers[mode] = dict(path=str(dest.relative_to(OUT)), sha256=sha(dest), executed_path=str(wrapper))
                for k, _ in seen[mode]:
                    d = k.directory
                    selected[mode].append(dict(directory=str(d), entrypoint=k.kernel.manifest["entrypoint"],
                                               ptx_sha256=sha(d / "kernel.ptx"), plan=k.kernel.plan.to_dict()))
            # Untouched branches must select the exact same ordinary object/body.
            assert selected["rms"][1] == selected["ordinary"][1]
            assert selected["column"][0] == selected["ordinary"][0]
            assert selected["both"][0] == selected["rms"][0]
            assert selected["both"][1] == selected["column"][1]
            _, (sx, sb), _ = fixture(torch, row, True)
            for mode in modes:
                special_out = fns[mode](r, sx, sb)
                check(torch, special_out[1], sx + sb)
                assert tensor_sha(special_out[0]) == tensor_sha(outputs[mode][0])
            cases.append(dict(label=label, case=case, inputs=inputs, functions=fns,
                              input_sha256=[tensor_sha(t) for t in inputs],
                              output_sha256={m: [tensor_sha(t) for t in outputs[m]] for m in modes},
                              artifacts=selected, wrappers=wrappers, rounds=[]))
            write("graphs-gpu.json", dict(complete=False, labels=[r["label"] for r in cases]))
    write("graphs-gpu.json", dict(complete=True, graphs=len(cases), normal_calls=36,
                                  special_calls=36, source_sha256=sha(__file__)))
    (OUT / "traces").mkdir(exist_ok=True)
    for c in cases:
        before_hashes = c["input_sha256"]
        for round_id, order in enumerate(plan["order"]):
            rr = dict(round=round_id, order=order, activity_before=activity(), paths={})
            for mode in order:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    for _ in range(plan["timing_calls"]):
                        c["functions"][mode](*c["inputs"])
                events = [e for e in profiler.profiler.kineto_results.events()
                          if e.device_type() == torch.autograd.DeviceType.CUDA and e.activity_type() == "kernel"]
                names = [a["entrypoint"] for a in c["artifacts"][mode]]
                assert len(events) == 32 and [e.name() for e in events] == names * 16
                samples = [sum((events[2*i+j].end_ns()-events[2*i+j].start_ns())/1000 for j in (0, 1))
                           for i in range(16)]
                trace = OUT / "traces" / f'{c["label"]}-{round_id}-{mode}.json'
                profiler.export_chrome_trace(str(trace))
                zipped = trace.with_suffix(".json.gz")
                zipped.write_bytes(gzip.compress(trace.read_bytes()))
                trace.unlink()
                rr["paths"][mode] = dict(trace=str(zipped.relative_to(OUT)), sha256=sha(zipped),
                                         samples_us=samples, median_us=statistics.median(samples))
            rr["activity_after"] = activity()
            c["rounds"].append(rr)
        assert [tensor_sha(t) for t in c["inputs"]] == before_hashes
        write("timing-progress.json", dict(complete=False, rows=[
            {k: v for k, v in row.items() if k not in ("inputs", "functions")} for row in cases]))
    write("timing.json", dict(complete=True, source_sha256=sha(__file__), rows=[
        {k: v for k, v in c.items() if k not in ("inputs", "functions")} for c in cases]))
    print("PASS9 mixed graphs/72 qualification calls/288 traces/9216 kernel durations", flush=True)


if __name__ == "__main__":
    {"gpu": gpu, "graphs": graphs}[sys.argv[1]]()
