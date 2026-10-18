"""Single packed-storage-load trial; immutable wqc references and components."""

import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
import traceback

from stage5.rms_trial import qualify, special
from stage5.rms_vector_trial_data import model, tensor_sha
from stage5.rms_vector_trial import activity

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "artifacts/reuse-first-rms-vector4"
OUT = ROOT / "artifacts/reuse-first-rms-packed-trial"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def cpu():
    from stage1 import evidence

    evidence.guard()
    import torch

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    assert not (OUT / "cpu.json").exists(), "preserve the chronological gate"
    write("cpu.json", {"complete": False})
    assert not (ROOT / "rms_packed_probe/src/main.rs").exists()
    meta = json.loads((OLD / "reference.json").read_text())
    assert meta["complete"] and meta["data_sha256"] == sha(OLD / "reference.pt")
    data = torch.load(OLD / "reference.pt", weights_only=True)
    assert len(data) == 36
    rows = []
    for label, packet in data.items():
        sums, output = model(packet["input"])
        special(torch, sums, sums[:, :1].expand_as(sums))
        torch.testing.assert_close(
            sums, sums[:, :1].expand_as(sums), rtol=0, atol=0, equal_nan=True
        )
        row = qualify(torch, label, packet["input"], output, packet)
        assert row["passes"], row
        rows.append(row)
        print("cpu", label, "PASS", flush=True)
    snapshot = OUT / "phase-sources"
    snapshot.mkdir(exist_ok=True)
    (snapshot / "cpu-rms_packed_trial.py").write_bytes(Path(__file__).read_bytes())
    write(
        "cpu.json",
        {
            "complete": True,
            "rows": rows,
            "gpu_candidate_source_existed": False,
            "reference_sha256": sha(OLD / "reference.json"),
            "data_sha256": sha(OLD / "reference.pt"),
            "model_source_sha256": sha(ROOT / "rms_vector_trial_data.py"),
            "producer_sha256": sha(snapshot / "cpu-rms_packed_trial.py"),
            "independence": evidence.inventory(),
        },
    )


def gpu():
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.rms_packed_trial_runtime import compile_kernel, templates
    from stage5.reduction_policy import TOLERANCES

    torch.set_num_threads(1)
    data = torch.load(OLD / "reference.pt", weights_only=True)
    models = torch.load(OLD / "cpu-model.pt", weights_only=True)
    kernels = {d: compile_kernel(s, 0) for d, s in templates().items()}
    from stage5.rms_packed_frozen_runtime import compile_kernel as compile_frozen

    frozen = {d: compile_frozen(s, 0) for d, s in templates().items()}
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
        frozen_y = torch.empty_like(y)
        frozen[dtype]((n, b), x, frozen_y)
        frozen_result = qualify(torch, label, original, frozen_y.cpu(), packet)
        assert frozen_result["passes"], frozen_result
        assert tensor_sha(actual) == tensor_sha(frozen_y), (
            "packed/frozen final bits differ: " + label
        )
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
            frozen_bitwise=True,
            frozen_policy_passes=True,
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


def parity():
    from stage1 import evidence

    evidence.guard()
    from stage5.rms_packed_observe import run

    run()


def counter():
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.artifact import Kernel as PhysicalKernel
    from stage5.rms_packed_trial_runtime import (
        Kernel as PackedKernel,
        PLAN,
        admit,
        MECHANISM,
    )
    from stage5.rms_vector_trial import activity

    class Loaded(PackedKernel):
        def __init__(self, directory, semantic):
            admit(semantic)
            PhysicalKernel.__init__(
                self, directory, semantic, 0, PLAN, compile_ptx=False
            )
            assert self.manifest["mechanism"] == MECHANISM
            assert self.manifest["storage_vector_width"] == 4

    gpu = json.loads((OUT / "gpu.json").read_text())
    assert gpu["complete"] and json.loads((OUT / "parity.json").read_text())["complete"]
    data = torch.load(OLD / "reference.pt", weights_only=True)
    kernels = []
    for dtype, info in gpu["kernels"].items():
        directory = Path(info["directory"])
        sem = json.loads((directory / "semantic.json").read_text())
        kernel = Loaded(directory, sem)
        x = data[f"corpus_{dtype}_64_1024"]["input"].cuda()
        y = torch.empty_like(x)
        kernel((1024, 64), x, y)
        kernels.append((dtype, kernel, directory, x, y))
    before = activity()
    from contextlib import ExitStack

    with ExitStack() as stack:
        for _, kernel, _, x, y in kernels:
            stack.enter_context(kernel.prepared((1024, 64), x, y))
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for _, kernel, _, _, _ in kernels:
            kernel.batch(1)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    write(
        "counter-kernels.json",
        {
            "scope": "production driver-loaded candidate code, before warm timing; replay durations excluded",
            "activity_before": before,
            "activity_after": activity(),
            "kernels": [
                {
                    "dtype": dtype,
                    "entrypoint": kernel.manifest["entrypoint"],
                    "directory": str(directory),
                    "ptx_sha256": sha(directory / "kernel.ptx"),
                    "attributes": kernel.attributes,
                    "resources": kernel.resources,
                }
                for dtype, kernel, directory, _, _ in kernels
            ],
        },
    )


def admission():
    from stage1 import evidence

    evidence.guard()
    import subprocess
    from stage5.rms_packed_trial_runtime import (
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
                    "entrypoint": "cubecl5_rms_cubek_pack4_rejected",
                },
            )
        )
    packet = {
        "schema": 5,
        "semantic": original,
        "physical_plan": PLAN,
        "mechanism": MECHANISM,
        "entrypoint": "cubecl5_rms_cubek_pack4_rejected",
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


ORDERS = (
    ("ordinary", "wqc", "packed"),
    ("wqc", "packed", "ordinary"),
    ("packed", "ordinary", "wqc"),
    ("packed", "wqc", "ordinary"),
    ("wqc", "ordinary", "packed"),
    ("ordinary", "packed", "wqc"),
)


def graphs():
    from stage1 import evidence

    evidence.guard()
    from stage1.overlay import install

    install()
    import torch
    from stage5.corpus import rms
    from stage5.backend import compile_backend as ordinary_backend
    from stage5.rms_packed_trial_backend import (
        compile_backend as packed_backend,
        compile_wqc_backend,
    )
    from stage5 import rms_packed_frozen_runtime as wqc_control
    from stage5 import runtime as ordinary
    from stage5 import rms_packed_trial_runtime as packed
    from torch._inductor import config

    assert json.loads((OUT / "gpu.json").read_text())["complete"]
    assert json.loads((OUT / "packing.json").read_text())["complete"]
    assert json.loads((OUT / "parity.json").read_text())["complete"]
    torch.set_num_threads(1)
    torch._dynamo.config.recompile_limit = 256
    data = torch.load(OLD / "reference.pt", weights_only=True)
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
                    "packed": torch.compile(
                        rms, backend=packed_backend, fullgraph=True, dynamic=True
                    ),
                    "wqc": torch.compile(
                        rms,
                        backend=compile_wqc_backend,
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
            assert outputs["packed"] == outputs["wqc"]
            # Bind the actual selected normal kernel and packed to this call.
            d = str(x.dtype).removeprefix("torch.")
            c = next(
                k for k in packed.CACHE.values() if k.semantic == packed.templates()[d]
            )
            matching = [k for k in ordinary.CACHE.values() if k.semantic == c.semantic]
            assert len(matching) == 1 and matching[0].last is not None
            normal = matching[0].last
            wqc = next(
                k for k in wqc_control.CACHE.values() if k.semantic == c.semantic
            )
            selection = {
                "wqc": wqc.entrypoint,
                "ordinary": normal.kernel.manifest["entrypoint"],
                "packed": c.entrypoint,
            }
            row = {
                "case": label,
                "shape": list(x.shape),
                "input_sha256": tensor_sha(x),
                "output_sha256": outputs,
                "ordinary_plan": normal.kernel.plan.to_dict(),
                "ordinary_directory": str(normal.directory),
                "packed_directory": str(c.directory),
                "wqc_directory": str(wqc.directory),
                "rounds": [],
                "isolated": [],
            }
            # Isolated primary measurement precedes complete generated-graph timings.
            if label.startswith("corpus_"):
                y0, y1, y2 = (
                    torch.empty_like(x),
                    torch.empty_like(x),
                    torch.empty_like(x),
                )
                with (
                    normal.kernel.prepared((n, b), x, y0),
                    c.kernel.prepared((n, b), x, y1),
                    wqc.kernel.prepared((n, b), x, y2),
                ):
                    kernels = {
                        "ordinary": normal.kernel,
                        "packed": c.kernel,
                        "wqc": wqc.kernel,
                    }
                    for round_id in range(6):
                        order = ORDERS[round_id]
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
                order = ORDERS[round_id]
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
    cache = ROOT / "cache/reuse-first-rms-packed-trial/graphs/inductor"
    for p in cache.rglob("*.py"):
        target = wrappers / p.relative_to(cache)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    write(
        "graphs.json",
        {
            "complete": True,
            "rows": rows,
            "method": "6 same-process fully permuted rounds x16 calls; CUPTI kernel duration only; no synthetic activity",
            "wrappers": {
                str(p.relative_to(OUT)): sha(p) for p in wrappers.rglob("*.py")
            },
            "independence": evidence.inventory(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("cpu", "gpu", "parity", "counter", "admission", "graphs")
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "cache/reuse-first-rms-packed-trial" / args.phase
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
