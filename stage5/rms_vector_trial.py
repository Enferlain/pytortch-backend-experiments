"""Phased wqc trial; CPU qualification precedes any vector4 GPU source."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

from stage5.rms_trial import qualify, special
from stage5.rms_vector_trial_data import (
    OLD,
    OUT,
    ROOT,
    ROWS,
    admitted,
    model,
    tensor_sha,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def reference():
    import torch
    from stage1.overlay import install

    install()
    from stage5.corpus import rms
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx
    from stage5.reduction_policy import TOLERANCES
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

    old_meta = json.loads((OLD / "reference.json").read_text())
    assert old_meta["complete"] and old_meta["data_sha256"] == sha(OLD / "reference.pt")
    old = torch.load(OLD / "reference.pt", weights_only=True)
    values = {label: d for label, d in old.items() if admitted(d["input"].shape)}
    rejected = [label for label, d in old.items() if not admitted(d["input"].shape)]
    assert len(values) == 15 and len(rejected) == 51
    compiled = torch.compile(rms, backend=backend, fullgraph=True, dynamic=True)
    with config.patch(patches):
        for dtype in ("float32", "float16", "bfloat16"):
            td = getattr(torch, dtype)
            for b in ROWS:
                torch.manual_seed(19340 + b)
                cpu_x = torch.randn((b, 1024), dtype=torch.float32).to(td)
                x = cpu_x.cuda()
                actual, aten = compiled(x), rms(x)
                rtol, atol = TOLERANCES[str(td)]
                torch.testing.assert_close(
                    actual, aten, rtol=rtol, atol=atol, equal_nan=True
                )
                special(torch, actual, aten)
                d = cpu_x.double()
                label = f"vector_row_tail_{dtype}_{b}_1024"
                values[label] = {
                    "input": cpu_x,
                    "reference": actual.cpu(),
                    "aten": aten.cpu(),
                    "fp64": d * torch.rsqrt((d * d).mean(-1, keepdim=True) + 1e-5),
                }
                print("reference", label, "PASS", flush=True)
    torch.save(values, OUT / "reference.pt")
    code = OUT / "reference-code"
    for p in (ROOT / "cache/reuse-first-rms-vector4/reference").rglob("*"):
        if p.is_file() and p.suffix in (
            ".py",
            ".ptx",
            ".cubin",
            ".ttir",
            ".ttgir",
            ".llir",
            ".json",
        ):
            target = code / p.relative_to(
                ROOT / "cache/reuse-first-rms-vector4/reference"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
    write(
        "reference.json",
        {
            "complete": True,
            "cases": len(values),
            "data_sha256": sha(OUT / "reference.pt"),
            "parent_reference_sha256": sha(OLD / "reference.json"),
            "parent_data_sha256": sha(OLD / "reference.pt"),
            "rejected_old_cases": rejected,
            "input_sha256": {k: tensor_sha(d["input"]) for k, d in values.items()},
            "source_sha256": {
                "rms_vector_trial.py": sha(__file__),
                "rms_vector_trial_data.py": sha(ROOT / "rms_vector_trial_data.py"),
            },
            "configuration": patches,
            "inventory": inventory(torch, "triton"),
            "code": {
                str(p.relative_to(OUT)): sha(p) for p in code.rglob("*") if p.is_file()
            },
        },
    )


def cpu():
    from stage1 import evidence

    evidence.guard()
    import torch

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    meta = json.loads((OUT / "reference.json").read_text())
    assert meta["complete"] and meta["data_sha256"] == sha(OUT / "reference.pt")
    data = torch.load(OUT / "reference.pt", weights_only=True)
    assert len(data) == 36
    recheck = (ROOT / "rms_vector_probe/src/main.rs").exists()
    if recheck:
        # Preserve the initial chronological gate; reproduction gets a new record.
        original_gate = json.loads((OUT / "cpu.json").read_text())
        assert (
            original_gate["complete"]
            and not original_gate["gpu_candidate_source_existed"]
        )
    record_name = "cpu-recheck.json" if recheck else "cpu.json"
    model_name = "cpu-recheck-model.pt" if recheck else "cpu-model.pt"
    rows, models = [], {}
    for label, d in data.items():
        sums, output = model(d["input"])
        special(torch, sums, sums[:, :1].expand_as(sums))
        torch.testing.assert_close(
            sums, sums[:, :1].expand_as(sums), rtol=0, atol=0, equal_nan=True
        )
        result = qualify(torch, label, d["input"], output, d)
        rows.append(result)
        models[label] = {"sums": sums, "output": output}
        write(record_name, {"complete": False, "rows": rows})
        assert result["passes"], result
        print("cpu", label, "PASS", flush=True)
    torch.save(models, OUT / model_name)
    write(
        record_name,
        {
            "complete": True,
            "rows": rows,
            "reference_sha256": sha(OUT / "reference.json"),
            "model_sha256": sha(OUT / model_name),
            "gpu_candidate_source_existed": recheck,
            "order": "FP32 square; per lane four component chains at128*k+4*lane+c; ascending XOR1,2,4,8,16 per component; +0 then ascending component fold; original epilogue/all-lane output",
            "source_sha256": {
                "rms_vector_trial.py": sha(__file__),
                "rms_vector_trial_data.py": sha(ROOT / "rms_vector_trial_data.py"),
            },
            "independence": evidence.inventory(),
        },
    )


def gpu():
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.rms_vector_trial_runtime import compile_kernel, templates
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
            (original.numel() + 8,), 12345.0, device="cuda", dtype=original.dtype
        )
        x = store_x[4:-4].view(b, n)
        x.copy_(original)
        store_y = torch.full_like(store_x, 12345.0)
        y = store_y[4:-4].view(b, n)
        kernels[dtype]((n, b), x, y)
        actual = y.cpu()
        result = qualify(torch, label, original, actual, packet)
        rtol, atol = TOLERANCES[str(original.dtype)]
        torch.testing.assert_close(
            actual, models[label]["output"], rtol=rtol, atol=atol, equal_nan=True
        )
        special(torch, actual, models[label]["output"])
        guard = torch.tensor([12345.0] * 8, dtype=original.dtype)
        assert torch.equal(
            store_x[[0, 1, 2, 3, -4, -3, -2, -1]].cpu(), guard
        ) and torch.equal(store_y[[0, 1, 2, 3, -4, -3, -2, -1]].cpu(), guard)
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
        x = torch.randn((10, 32), device="cuda", dtype=td)
        y = torch.empty_like(x)
        c((32, 10), x, y)
        for name, scalars, args in (
            ("extent", (34, 10), (x, y)),
            ("zero", (32, 0), (x, y)),
            (
                "odd",
                (33, 10),
                (
                    torch.empty((10, 33), device="cuda", dtype=td),
                    torch.empty((10, 33), device="cuda", dtype=td),
                ),
            ),
            (
                "alignment",
                (32, 10),
                (torch.empty(321, device="cuda", dtype=td)[1:].view(10, 32), y),
            ),
            ("shape", (32, 10), (x[:9], y)),
            (
                "stride",
                (32, 10),
                (torch.empty_strided((10, 32), (34, 1), device="cuda", dtype=td), y),
            ),
            ("dtype", (32, 10), (x.double(), y)),
            ("device", (32, 10), (x.cpu(), y)),
            ("alias", (32, 10), (x, x)),
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
            backing = torch.randn(10 * 32 + 8, device="cuda", dtype=td)
            x = backing[4:-4].view(10, 32)
            y = torch.empty(10 * 32 + 8, device="cuda", dtype=td)[4:-4].view(10, 32)
            c((32, 10), x, y)
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


def activity():
    import subprocess, time

    result = {"time_ns": time.time_ns()}
    for key, args in (
        (
            "gpu",
            [
                "--query-gpu=index,uuid,name,display_active,utilization.gpu,clocks.current.sm,clocks.current.memory,power.draw",
                "--format=csv",
            ],
        ),
        (
            "processes",
            [
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv",
            ],
        ),
    ):
        p = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True)
        result[key] = {
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
        }
    return result


def graphs():
    from stage1 import evidence

    evidence.guard()
    from stage1.overlay import install

    install()
    import torch
    from stage5.corpus import rms
    from stage5.backend import compile_backend as ordinary_backend
    from stage5.rms_vector_trial_backend import (
        compile_backend as component_backend,
        compile_scalar_backend,
    )
    from stage5 import rms_vector_scalar_runtime as scalar_control
    from stage5 import runtime as ordinary
    from stage5 import rms_vector_trial_runtime as component
    from torch._inductor import config

    assert json.loads((OUT / "gpu.json").read_text())["complete"]
    torch.set_num_threads(1)
    torch._dynamo.config.recompile_limit = 256
    data = torch.load(OUT / "reference.pt", weights_only=True)
    rows = []

    def capture(fn, destination, expected_name):
        torch.cuda.synchronize()
        before = activity()
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
            "activity_before": before,
            "activity_after": activity(),
        }

    (OUT / "traces").mkdir(exist_ok=True)
    with config.patch({"fx_graph_cache": False}):
        functions = {}
        for label, packet in data.items():
            if not (
                label.startswith("corpus_")
                or (
                    label.startswith("vector_row_tail_")
                    and packet["input"].shape[0] in (16, 65)
                )
            ):
                continue
            dtype = str(packet["input"].dtype)
            if dtype not in functions:
                functions[dtype] = {
                    "component": torch.compile(
                        rms, backend=component_backend, fullgraph=True, dynamic=True
                    ),
                    "scalar": torch.compile(
                        rms,
                        backend=compile_scalar_backend,
                        fullgraph=True,
                        dynamic=True,
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
            scalar = next(
                k for k in scalar_control.CACHE.values() if k.semantic == c.semantic
            )
            selection = {
                "scalar": scalar.entrypoint,
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
                "scalar_directory": str(scalar.directory),
                "rounds": [],
                "isolated": [],
            }
            # Isolated primary measurement precedes complete generated-graph timings.
            if label == "corpus_float32_64_1024":
                y0, y1, y2 = (
                    torch.empty_like(x),
                    torch.empty_like(x),
                    torch.empty_like(x),
                )
                with (
                    normal.kernel.prepared((n, b), x, y0),
                    c.kernel.prepared((n, b), x, y1),
                    scalar.kernel.prepared((n, b), x, y2),
                ):
                    kernels = {
                        "ordinary": normal.kernel,
                        "component": c.kernel,
                        "scalar": scalar.kernel,
                    }
                    for round_id in range(6):
                        order = (
                            ["ordinary", "scalar", "component"]
                            if round_id % 2 == 0
                            else ["component", "scalar", "ordinary"]
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
                    ["ordinary", "scalar", "component"]
                    if round_id % 2 == 0
                    else ["component", "scalar", "ordinary"]
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
    cache = ROOT / "cache/reuse-first-rms-vector4/graphs/inductor"
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


def counter():
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.artifact import Kernel
    from stage5.rms_vector_trial_runtime import Kernel as ComponentKernel
    from stage5.rms_trial_runtime import Kernel as ScalarKernel

    graphs = json.loads((OUT / "graphs.json").read_text())
    assert graphs["complete"] and json.loads((OUT / "gpu.json").read_text())["complete"]
    row = graphs["rows"][0]
    assert row["case"] == "corpus_float32_64_1024"
    normal_path = Path(row["ordinary_directory"])
    component_path = Path(row["component_directory"])
    semantic = json.loads((component_path / "semantic.json").read_text())
    normal = Kernel(normal_path, semantic, 0, row["ordinary_plan"], compile_ptx=False)
    component = ComponentKernel(component_path, semantic, 0)
    scalar_path = Path(row["scalar_directory"])
    scalar = ScalarKernel(scalar_path, semantic, 0)
    original = torch.load(OUT / "reference.pt", weights_only=True)[row["case"]]["input"]
    x = original.cuda()
    a, b, c = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
    with (
        normal.prepared((1024, 64), x, a),
        component.prepared((1024, 64), x, b),
        scalar.prepared((1024, 64), x, c),
    ):
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        normal.batch(1)
        scalar.batch(1)
        component.batch(1)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    write(
        "counter-kernels.json",
        {
            "scope": "Production driver-loaded code for the same three timed primary kernels; profiling durations are not warm timing",
            "kernels": [
                {
                    "name": name,
                    "entrypoint": k.manifest["entrypoint"],
                    "attributes": k.attributes,
                    "resources": k.resources,
                    "ptx_sha256": sha(p / "kernel.ptx"),
                }
                for name, k, p in [
                    ("ordinary", normal, normal_path),
                    ("scalar", scalar, scalar_path),
                    ("component", component, component_path),
                ]
            ],
        },
    )


def admission():
    from stage1 import evidence

    evidence.guard()
    import subprocess
    from stage5.rms_vector_trial_runtime import (
        BINARY,
        PLAN,
        MECHANISM,
        admit,
        templates,
    )

    original = templates()["float32"]
    mutations = {}

    def change(name, fn):
        sem = copy.deepcopy(original)
        fn(sem)
        mutations[name] = sem

    change("reducer", lambda d: d["ops"][2].update(reduction_type="max"))
    change("square", lambda d: d["ops"][1].update(op="add"))
    change("casts", lambda d: d["ops"][4].update(dtype="float16"))
    change("epsilon", lambda d: d["ops"][6].update(value=0.0))
    change("stride", lambda d: d["buffers"][0]["stride"][1].update(value=2))
    change(
        "extent", lambda d: d.update(reduction_extent={"op": "constant", "value": 1024})
    )
    change("owner", lambda d: d["ops"][-1].update(iteration_scope="outer"))
    change("effect", lambda d: d.update(effects={"rng": True, "mutation": False}))
    change("bool_number", lambda d: d["scalars"][0].update(positive=1))
    cases = []
    for name, sem in mutations.items():
        try:
            admit(sem)
        except ValueError:
            pass
        else:
            raise AssertionError("Python admitted " + name)
        cases.append(
            (
                name,
                {
                    "schema": 5,
                    "semantic": sem,
                    "physical_plan": PLAN,
                    "mechanism": MECHANISM,
                    "entrypoint": "cubecl5_rms_cubek_vec4_rejected",
                },
            )
        )
    packet = {
        "schema": 5,
        "semantic": original,
        "physical_plan": PLAN,
        "mechanism": MECHANISM,
        "entrypoint": "cubecl5_rms_cubek_vec4_rejected",
    }
    for name, fn in [
        ("planes", lambda p: p["physical_plan"].update(workgroup=[32, 4, 1])),
        ("mechanism", lambda p: p.update(mechanism="other")),
        ("entrypoint", lambda p: p.update(entrypoint="invalid;entry")),
    ]:
        p = copy.deepcopy(packet)
        fn(p)
        cases.append((name, p))
    rows = []
    for name, packet in cases:
        directory = OUT / "rejected" / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "descriptor.json"
        path.write_text(json.dumps(packet, indent=2))
        run = subprocess.run(
            [str(BINARY), str(directory), str(path)], capture_output=True, text=True
        )
        assert (
            run.returncode
            and not (directory / "kernel.cu").exists()
            and not (directory / "cubecl.ir").exists()
        ), name
        (directory / "error.txt").write_text(run.stderr)
        rows.append(
            {"case": name, "rejected_before_IR": True, "returncode": run.returncode}
        )
    write(
        "admission.json",
        {
            "complete": True,
            "negative_cases": rows,
            "independence": evidence.inventory(),
        },
    )


def scalar_qualification():
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.rms_vector_scalar_runtime import compile_kernel
    from stage5.rms_trial_runtime import templates

    data = torch.load(OUT / "reference.pt", weights_only=True)
    kernels = {d: compile_kernel(s, 0) for d, s in templates().items()}
    rows = []
    for label, packet in data.items():
        x = packet["input"].cuda()
        b, n = x.shape
        y = torch.empty_like(x)
        kernels[str(x.dtype).removeprefix("torch.")]((n, b), x, y)
        actual = y.cpu()
        result = qualify(torch, label, packet["input"], actual, packet)
        assert result["passes"], result
        result["output_sha256"] = tensor_sha(actual)
        rows.append(result)
    write(
        "scalar-qualification.json",
        {"complete": True, "rows": rows, "independence": evidence.inventory()},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "reference",
            "cpu",
            "gpu",
            "graphs",
            "counter",
            "admission",
            "scalar_qualification",
        ),
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "cache/reuse-first-rms-vector4" / args.phase
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    try:
        globals()[args.phase]()
    except Exception:
        write(args.phase + "-failure.json", {"error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
