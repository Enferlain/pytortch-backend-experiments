"""Phased reference/CPU gate for the single reuse-first RMS mechanism."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

from stage5.rms_trial_data import OUT, ROOT, fixtures, model, tensor_sha


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def special(torch, actual, reference):
    assert torch.equal(torch.isnan(actual), torch.isnan(reference))
    assert torch.equal(torch.isinf(actual), torch.isinf(reference))
    signs = (torch.isinf(actual) & torch.isinf(reference)) | (
        (actual == 0) & (reference == 0)
    )
    assert torch.equal(torch.signbit(actual[signs]), torch.signbit(reference[signs]))


def reference():
    import torch
    from stage1.overlay import install

    install()
    from stage5.corpus import rms
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx
    from stage5.attribution import inventory

    torch.set_num_threads(1)
    torch._dynamo.config.recompile_limit = 256
    patches = {
        "triton.cudagraphs": False,
        "compile_threads": 1,
        "fx_graph_cache": False,
        "emulate_precision_casts": True,
        "joint_graph_constant_folding": False,
        "eager_numerics.division_rounding": True,
        "eager_numerics.disable_ftz": True,
        "eager_numerics.use_pytorch_libdevice": True,
    }

    def backend(gm, inputs):
        return compile_fx(gm, inputs, config_patches=patches)

    compiled = torch.compile(rms, backend=backend, fullgraph=True, dynamic=True)
    rows, values = [], {}
    from stage5.reduction_policy import TOLERANCES

    with config.patch(patches):
        for label, x in fixtures(torch):
            actual, aten = compiled(x), rms(x)
            torch.cuda.synchronize()
            rtol, atol = TOLERANCES[str(x.dtype)]
            torch.testing.assert_close(
                actual, aten, rtol=rtol, atol=atol, equal_nan=True
            )
            special(torch, actual, aten)
            cpu = x.cpu()
            d = cpu.double()
            truth = d * torch.rsqrt((d * d).mean(-1, keepdim=True) + 1e-5)
            values[label] = {
                "input": cpu,
                "reference": actual.cpu(),
                "aten": aten.cpu(),
                "fp64": truth,
            }
            rows.append(
                {
                    "case": label,
                    "shape": list(x.shape),
                    "dtype": str(x.dtype),
                    "input_sha256": tensor_sha(x),
                    "reference_sha256": tensor_sha(actual),
                }
            )
            print("reference", label, "PASS", flush=True)
    torch.save(values, OUT / "reference.pt")
    cache = ROOT / "cache/reuse-first-rms/reference"
    retained = OUT / "reference-code"
    for p in cache.rglob("*"):
        if p.is_file() and p.suffix in (
            ".py",
            ".ptx",
            ".cubin",
            ".ttir",
            ".ttgir",
            ".llir",
            ".json",
        ):
            target = retained / p.relative_to(cache)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
    write(
        "reference.json",
        {
            "complete": True,
            "rows": rows,
            "config": patches,
            "data_sha256": sha(OUT / "reference.pt"),
            "source_sha256": {
                "rms_trial.py": sha(__file__),
                "rms_trial_data.py": sha(ROOT / "rms_trial_data.py"),
            },
            "environment": inventory(torch, "triton"),
            "code": {
                str(p.relative_to(OUT)): sha(p)
                for p in retained.rglob("*")
                if p.is_file()
            },
        },
    )


def qualify(torch, label, x, actual, data):
    from stage5.reduction_policy import qualify_case, TOLERANCES

    ref = {
        "case": label,
        "dtype": str(x.dtype),
        "shape": list(x.shape),
        "input_sha256": tensor_sha(x),
        "aten": data["aten"].flatten().tolist(),
        "fp64": data["fp64"].flatten().tolist(),
        "values": data["reference"].flatten().tolist(),
    }
    row = {**ref, "values": actual.flatten().tolist()}
    result = qualify_case(row, ref)
    rtol, atol = TOLERANCES[str(x.dtype)]
    torch.testing.assert_close(
        actual, data["aten"], rtol=rtol, atol=atol, equal_nan=True
    )
    special(torch, actual, data["reference"])
    failures = [e for e in result["elements"] if not e["passes"]]
    return {
        "case": label,
        "passes": result["passes"],
        "elements": len(result["elements"]),
        "policy": result["policy"],
        "failures": failures[:32],
        "input_sha256": ref["input_sha256"],
    }


def cpu():
    from stage1 import evidence

    evidence.guard()
    import torch

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    reference_meta = json.loads((OUT / "reference.json").read_text())
    assert reference_meta["complete"] and reference_meta["data_sha256"] == sha(
        OUT / "reference.pt"
    )
    values = torch.load(OUT / "reference.pt", weights_only=True)
    rows, models = [], {}
    for label, data in values.items():
        x = data["input"]
        sums, output = model(x)
        special(torch, sums, sums[:, :1].expand_as(sums))
        torch.testing.assert_close(
            sums, sums[:, :1].expand_as(sums), rtol=0, atol=0, equal_nan=True
        )
        result = qualify(torch, label, x, output, data)
        rows.append(result)
        models[label] = {"sums": sums, "output": output}
        write("cpu.json", {"complete": False, "rows": rows})
        assert result["passes"], result
        print("cpu", label, "PASS", flush=True)
    torch.save(models, OUT / "cpu-model.pt")
    write(
        "cpu.json",
        {
            "complete": True,
            "rows": rows,
            "reference_sha256": sha(OUT / "reference.json"),
            "model_sha256": sha(OUT / "cpu-model.pt"),
            "order": "FP32 chain by lane, ascending k; XOR1,2,4,8,16; all32 lanes",
            "source_sha256": {
                "rms_trial.py": sha(__file__),
                "rms_trial_data.py": sha(ROOT / "rms_trial_data.py"),
            },
            "independence": evidence.inventory(),
        },
    )


def gpu():
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.rms_trial_runtime import compile_kernel, templates
    from stage5.reduction_policy import TOLERANCES

    torch.set_num_threads(1)
    data = torch.load(OUT / "reference.pt", weights_only=True)
    models = torch.load(OUT / "cpu-model.pt", weights_only=True)
    kernels = {d: compile_kernel(s, 0) for d, s in templates().items()}
    rows, rejected = [], []
    for label, packet in data.items():
        original = packet["input"]
        b, n = original.shape
        dtype = str(original.dtype).removeprefix("torch.")
        store_x = torch.full(
            (original.numel() + 2,), 12345.0, device="cuda", dtype=original.dtype
        )
        x = store_x[1:-1].view(b, n)
        x.copy_(original)
        store_y = torch.full_like(store_x, 12345.0)
        y = store_y[1:-1].view(b, n)
        kernels[dtype]((n, b), x, y)
        actual = y.cpu()
        result = qualify(torch, label, original, actual, packet)
        rtol, atol = TOLERANCES[str(original.dtype)]
        torch.testing.assert_close(
            actual, models[label]["output"], rtol=rtol, atol=atol, equal_nan=True
        )
        special(torch, actual, models[label]["output"])
        guard = torch.tensor([12345.0, 12345.0], dtype=original.dtype)
        assert torch.equal(store_x[[0, -1]].cpu(), guard) and torch.equal(
            store_y[[0, -1]].cpu(), guard
        )
        assert tensor_sha(x) == tensor_sha(original)
        result.update(
            output_sha256=tensor_sha(actual),
            model_bitwise=tensor_sha(actual) == tensor_sha(models[label]["output"]),
            canaries=True,
        )
        rows.append(result)
        write("gpu.json", {"complete": False, "rows": rows})
        assert result["passes"], result
        print("gpu", label, "PASS", flush=True)
    for dtype, c in kernels.items():
        td = getattr(torch, dtype)
        x = torch.randn((10, 33), device="cuda", dtype=td)
        y = torch.empty_like(x)
        c((33, 10), x, y)
        for name, scalars, args in (
            ("extent", (34, 10), (x, y)),
            ("zero", (33, 0), (x, y)),
            ("shape", (33, 10), (x[:9], y)),
            (
                "stride",
                (33, 10),
                (torch.empty_strided((10, 33), (34, 1), device="cuda", dtype=td), y),
            ),
            ("dtype", (33, 10), (x.double(), y)),
            ("device", (33, 10), (x.cpu(), y)),
            ("alias", (33, 10), (x, x)),
        ):
            before = c.kernel.launch_count
            try:
                c(scalars, *args)
            except ValueError as error:
                rejected.append({"dtype": dtype, "control": name, "error": str(error)})
            else:
                raise AssertionError("RMS live ABI guard bypassed: " + name)
            assert c.kernel.launch_count == before
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            backing = torch.randn(10 * 33 + 2, device="cuda", dtype=td)
            x = backing[1:-1].view(10, 33)
            y = torch.empty(10 * 33 + 2, device="cuda", dtype=td)[1:-1].view(10, 33)
            c((33, 10), x, y)
            from stage5.corpus import rms

            expected = rms(x)
        stream.synchronize()
        rtol, atol = TOLERANCES[str(td)]
        torch.testing.assert_close(y, expected, rtol=rtol, atol=atol)
        assert c.kernel._call_pointers is None
    write(
        "gpu.json",
        {
            "complete": True,
            "rows": rows,
            "rejected": rejected,
            "stream_offset_controls": 3,
            "kernels": {
                d: {
                    "directory": str(c.directory),
                    "attributes": c.kernel.attributes,
                    "resources": c.kernel.resources,
                }
                for d, c in kernels.items()
            },
            "independence": evidence.inventory(),
        },
    )


def graphs():
    from stage1 import evidence

    evidence.guard()
    from stage1.overlay import install

    install()
    import torch
    from stage5.corpus import rms
    from stage5.backend import compile_backend as ordinary_backend
    from stage5.rms_trial_backend import compile_backend as component_backend
    from stage5 import runtime as ordinary
    from stage5 import rms_trial_runtime as component
    from torch._inductor import config

    assert json.loads((OUT / "gpu.json").read_text())["complete"]
    torch.set_num_threads(1)
    data = torch.load(OUT / "reference.pt", weights_only=True)
    rows = []

    def capture(fn, destination, expected_name):
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            for _ in range(16):
                fn()
        profiler.export_chrome_trace(str(destination))
        events = [
            e
            for e in profiler.profiler.kineto_results.events()
            if e.device_type() == torch.autograd.DeviceType.CUDA
            and e.activity_type() == "kernel"
        ]
        assert len(events) == 16 and all(e.name() == expected_name for e in events), [
            (e.name(), e.end_ns() - e.start_ns()) for e in events
        ]
        samples = [(e.end_ns() - e.start_ns()) / 1000 for e in events]
        assert all(v > 0 for v in samples)
        return {
            "samples_us": samples,
            "trace": str(destination.relative_to(OUT)),
            "trace_sha256": sha(destination),
            "kernel": expected_name,
        }

    (OUT / "traces").mkdir(exist_ok=True)
    with config.patch({"fx_graph_cache": False}):
        functions = {}
        for label, packet in data.items():
            if not label.startswith("corpus_"):
                continue
            dtype = str(packet["input"].dtype)
            if dtype not in functions:
                functions[dtype] = {
                    "component": torch.compile(
                        rms, backend=component_backend, fullgraph=True, dynamic=True
                    ),
                    "ordinary": torch.compile(
                        rms, backend=ordinary_backend, fullgraph=True, dynamic=True
                    ),
                }
            x = packet["input"].cuda()
            b, n = x.shape
            outputs = {}
            for name, f in functions[dtype].items():
                actual = f(x)
                result = qualify(torch, label, packet["input"], actual.cpu(), packet)
                assert result["passes"], result
                outputs[name] = tensor_sha(actual)
            # Bind the actual selected normal kernel and component to this call.
            d = str(x.dtype).removeprefix("torch.")
            c = next(
                k
                for k in component.CACHE.values()
                if k.semantic == component.templates()[d]
            )
            matching = [k for k in ordinary.CACHE.values() if k.semantic == c.semantic]
            assert len(matching) == 1 and matching[0].last is not None
            normal = matching[0].last
            selection = {
                "ordinary": normal.kernel.manifest["entrypoint"],
                "component": c.entrypoint,
            }
            row = {
                "case": label,
                "shape": list(x.shape),
                "input_sha256": tensor_sha(x),
                "output_sha256": outputs,
                "ordinary_plan": normal.kernel.plan.to_dict(),
                "ordinary_directory": str(normal.directory),
                "component_directory": str(c.directory),
                "rounds": [],
                "isolated": [],
            }
            # Isolated primary measurement precedes complete generated-graph timings.
            if label == "corpus_float32_64_1024":
                y0, y1 = torch.empty_like(x), torch.empty_like(x)
                with (
                    normal.kernel.prepared((n, b), x, y0),
                    c.kernel.prepared((n, b), x, y1),
                ):
                    kernels = {"ordinary": normal.kernel, "component": c.kernel}
                    for round_id in range(6):
                        order = (
                            ["ordinary", "component"]
                            if round_id % 2 == 0
                            else ["component", "ordinary"]
                        )
                        for name in order:
                            path = (
                                OUT
                                / "traces"
                                / f"isolated-{label}-r{round_id}-{name}.json"
                            )
                            row["isolated"].append(
                                {
                                    "round": round_id,
                                    "name": name,
                                    **capture(
                                        lambda: kernels[name].batch(1),
                                        path,
                                        selection[name],
                                    ),
                                }
                            )
            for round_id in range(6):
                order = (
                    ["ordinary", "component"]
                    if round_id % 2 == 0
                    else ["component", "ordinary"]
                )
                for name in order:
                    path = OUT / "traces" / f"graph-{label}-r{round_id}-{name}.json"
                    row["rounds"].append(
                        {
                            "round": round_id,
                            "name": name,
                            **capture(
                                lambda: functions[dtype][name](x), path, selection[name]
                            ),
                        }
                    )
            rows.append(row)
            write("graphs.json", {"complete": False, "rows": rows})
            print("graphs", label, "PASS", flush=True)
    wrappers = OUT / "generated-wrappers"
    cache = ROOT / "cache/reuse-first-rms/graphs/inductor"
    for p in cache.rglob("*.py"):
        target = wrappers / p.relative_to(cache)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    write(
        "graphs.json",
        {
            "complete": True,
            "rows": rows,
            "method": "6 same-process alternating rounds x16 calls; CUPTI kernel duration only; no synthetic activity",
            "wrappers": {
                str(p.relative_to(OUT)): sha(p) for p in wrappers.rglob("*.py")
            },
            "independence": evidence.inventory(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("reference", "cpu", "gpu", "graphs"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "cache/reuse-first-rms" / args.phase
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    try:
        globals()[args.phase]()
    except Exception:
        write(
            args.phase + "-failure.json",
            {"error": traceback.format_exc(), "phase": args.phase},
        )
        raise


if __name__ == "__main__":
    main()
