"""Numerical, bounds and runtime qualification for the single transpose mechanism."""

import argparse
import copy
import json
import subprocess

from stage5.transpose_trial import (
    OUT,
    copy_contiguous,
    fused_contiguous,
    sha,
    tensor_sha,
    write,
)
from stage5.transpose_model import SHAPES


def baseline(semantic, device=0):
    from stage5.artifact import Kernel

    for row in json.loads((OUT / "capture.json").read_text())["records"]:
        path = OUT / row["artifact"]
        if json.loads((path / "semantic.json").read_text()) == semantic:
            return Kernel(
                path,
                semantic,
                device,
                json.loads((path / "physical-plan.json").read_text()),
                compile_ptx=False,
            )
    raise ValueError("no captured baseline")


def check(torch, actual, expected):
    dtype = str(actual.dtype).removeprefix("torch.")
    tol = {
        "float32": (2e-6, 2e-6),
        "float16": (0.002, 0.001),
        "bfloat16": (0.016, 0.008),
    }[dtype]
    torch.testing.assert_close(
        actual, expected, rtol=tol[0], atol=tol[1], equal_nan=True
    )
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    zeros = (actual == 0) & (expected == 0)
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


def gpu():
    from stage1 import evidence

    evidence.guard()
    from stage1.overlay import install

    install()
    import torch
    from stage5.transpose_runtime import TEMPLATES, compile_kernel, CACHE, BINARY

    torch.set_num_threads(1)
    assert not (OUT / "gpu.json").exists(), "preserve the first GPU gate"
    rows, controls, stream_rows = [], [], []
    values = {}
    for sem in json.loads(TEMPLATES.read_text()):
        d = sem["buffers"][0]["dtype"]
        td = getattr(torch, d)
        mode = "fused" if any(o["op"] == "sin" for o in sem["ops"]) else "copy"
        fn = fused_contiguous if mode == "fused" else copy_contiguous
        normal, dispatch = baseline(sem), compile_kernel(sem, 0)
        for m, n, special in [(m, n, False) for m, n in SHAPES] + [(256, 512, True)]:
            label = f"{mode}-{d}-{m}x{n}" + ("-special" if special else "")
            torch.manual_seed(537 + m + n)
            store = torch.full((m * n + 16,), 42.0, dtype=td, device="cuda")
            base = store[8:-8].view(m, n)
            base.copy_(torch.randn((m, n), device="cuda", dtype=td))
            if special:
                pattern = torch.tensor(
                    [
                        0.0,
                        -0.0,
                        1e8,
                        1.0,
                        -1e8,
                        1e-30,
                        -1e-30,
                        float("nan"),
                        float("inf"),
                        -float("inf"),
                        torch.finfo(td).max,
                        torch.finfo(td).tiny / 2,
                    ],
                    device="cuda",
                    dtype=td,
                )
                base.flatten().copy_(
                    pattern.repeat((m * n + len(pattern) - 1) // len(pattern))[: m * n]
                )
            x = base.T
            original = tensor_sha(x)
            output_store = torch.full_like(store, 42.0)
            y = output_store[8:-8].view(n, m)
            reference = torch.empty_like(y)
            normal((n, m), x, reference)
            dispatch((n, m), x, y)
            torch.cuda.synchronize()
            check(torch, reference, fn(x))
            check(torch, y, fn(x))
            assert tensor_sha(y) == tensor_sha(reference), label
            assert tensor_sha(x) == original
            assert bool(
                (store[:8] == 42).all()
                & (store[-8:] == 42).all()
                & (output_store[:8] == 42).all()
                & (output_store[-8:] == 42).all()
            )
            rows.append(
                dict(
                    label=label,
                    dtype=d,
                    mode=mode,
                    shape=[n, m],
                    special=special,
                    bitwise=True,
                    policy=True,
                    canaries=True,
                    input_sha256=original,
                    output_sha256=tensor_sha(y),
                    candidate=str(dispatch.last.directory.relative_to(OUT)),
                    baseline_entry=normal.manifest["entrypoint"],
                    width=dispatch.last.vector,
                )
            )
            if special:
                values[label] = dict(
                    input=x.cpu(),
                    actual=y.cpu(),
                    baseline=reference.cpu(),
                    eager=fn(x).cpu(),
                )
            write("gpu.json", dict(complete=False, rows=rows))
        x = torch.empty((256, 512), device="cuda", dtype=td).T
        y = torch.empty((512, 256), device="cuda", dtype=td)
        for label, scalars, tensors in (
            ("extent", (512, 255), (x, y)),
            ("zero", (512, 0), (x, y)),
            ("layout", (512, 256), (x.contiguous(), y)),
            ("alias", (512, 256), (x, x.T.reshape(512, 256))),
            ("dtype", (512, 256), (x.to(torch.float64), y)),
            ("device", (512, 256), (x.cpu(), y)),
            (
                "alignment",
                (512, 256),
                (
                    torch.empty(256 * 512 + 1, device="cuda", dtype=td)[1:]
                    .view(256, 512)
                    .T,
                    y,
                ),
            ),
        ):
            try:
                dispatch(scalars, *tensors)
            except (ValueError, TypeError) as e:
                controls.append(dict(mode=mode, dtype=d, control=label, error=str(e)))
            else:
                raise AssertionError("missing rejection: " + label)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            storage = torch.empty(256 * 512 + 16, device="cuda", dtype=td)
            sx = storage[8:-8].view(256, 512).T
            sx.copy_(torch.randn_like(sx))
            sy_store = torch.empty_like(storage)
            sy = sy_store[8:-8].view(512, 256)
            dispatch((512, 256), sx, sy)
            expected = fn(sx)
        stream.synchronize()
        check(torch, sy, expected)
        stream_rows.append(
            dict(
                mode=mode,
                dtype=d,
                stream=stream.cuda_stream,
                aligned_offset=True,
                passes=True,
            )
        )
    torch.save(values, OUT / "special-values.pt")
    # The original transpose-view workload must retain its layout and pass the
    # actual pointwise corpus tolerances. The initial descriptor-only capture
    # used looser FP32 checks; this is the strict admission record.
    from stage5.corpus import sin
    from stage5.backend import compile_backend
    from torch._inductor import config

    view_rows = []
    with config.patch({"fx_graph_cache": False}):
        for d in ("float32", "float16", "bfloat16"):
            torch._dynamo.reset()
            f = torch.compile(
                sin, backend=compile_backend, fullgraph=True, dynamic=True
            )
            for m, n in SHAPES[:3]:
                torch.manual_seed(537 + m + n)
                x = torch.randn((m, n), device="cuda", dtype=getattr(torch, d)).T
                y = f(x)
                check(torch, y, sin(x))
                assert y.stride() == x.stride()
                view_rows.append(
                    dict(dtype=d, shape=[n, m], stride=list(y.stride()), passes=True)
                )
    kernel_rows = [
        dict(
            directory=str(k.directory.relative_to(OUT)),
            entrypoint=k.entrypoint,
            vector=k.vector,
            ptx_sha256=sha(k.directory / "kernel.ptx"),
        )
        for k in CACHE.values()
    ]
    write(
        "gpu.json",
        dict(
            complete=True,
            rows=rows,
            live_controls=controls,
            stream_controls=stream_rows,
            view_controls=view_rows,
            kernels=kernel_rows,
            binary_sha256=sha(BINARY),
            special_values_sha256=sha(OUT / "special-values.pt"),
            independence=evidence.inventory(),
        ),
    )
    print(
        "PASS",
        len(rows),
        "GPU cases",
        len(controls),
        "live guards",
        len(stream_rows),
        "streams",
        len(view_rows),
        "views",
        flush=True,
    )


def native():
    from stage5.transpose_runtime import BINARY

    kernel = json.loads((OUT / "gpu.json").read_text())["kernels"][0]
    packet = json.loads((OUT / kernel["directory"] / "descriptor.json").read_text())
    results = []
    for label in (
        "entry",
        "schema",
        "mechanism",
        "plan",
        "width",
        "semantic",
        "effects",
    ):
        p = copy.deepcopy(packet)
        if label == "entry":
            p["entrypoint"] = "bad"
        elif label == "schema":
            p["schema"] = 4
        elif label == "mechanism":
            p["mechanism"] = "unknown"
        elif label == "plan":
            p["physical_plan"]["workgroup"] = [128, 1, 1]
        elif label == "width":
            p["storage_vector_width"] = 3
        elif label == "semantic":
            p["semantic"]["ops"][0]["index"] = {"op": "linear"}
        elif label == "effects":
            p["semantic"]["effects"]["mutation"] = True
        path = OUT / "native-controls" / label
        path.mkdir(parents=True, exist_ok=True)
        desc = path / "descriptor.json"
        desc.write_text(json.dumps(p))
        proc = subprocess.run(
            [str(BINARY), str(path), str(desc)], capture_output=True, text=True
        )
        assert proc.returncode and not (path / "cubecl.ir").exists(), label
        results.append(
            dict(label=label, returncode=proc.returncode, error=proc.stderr.strip())
        )
    write("native-controls.json", dict(complete=True, rows=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("gpu", "native"))
    args = parser.parse_args()
    {"gpu": gpu, "native": native}[args.phase]()
