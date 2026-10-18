"""Full frozen-corpus qualification for the one column specialization."""

import argparse
import copy
import json
import subprocess

from stage5.column_layout_trial import OUT, check, sha, tensor_sha, write
from stage5.layout_qualify import fixture


def baseline(semantic):
    from stage5.column_layout_runtime import generic

    return generic(semantic, 0)


def gpu(output="gpu.json", special_output="special-values.pt"):
    from stage1 import evidence

    evidence.guard()
    from stage1.overlay import install

    install()
    import torch
    from stage5.column_layout_runtime import (
        qualified_dispatch as compile_kernel,
        CACHE,
        BINARY,
    )
    from stage5.layout_runtime import templates

    torch.set_num_threads(1)
    assert not (OUT / output).exists()
    plan = json.loads((OUT / "qualification-plan.json").read_text())["cases"]
    rows = []
    controls = []
    streams = []
    values = {}
    for family, dtype, sem in templates().values():
        normal = baseline(sem)
        candidate = compile_kernel(sem, 0)
        tests = [
            (r, False) for r in plan if r["family"] == family and r["dtype"] == dtype
        ]
        selected = next(r for r, _ in tests if r["original"])
        tests.append((selected, True))
        for row, special in tests:
            label = row["label"] + ("-special" if special else "")
            fn, args, stores = fixture(torch, row, special)
            expected = fn(*args)
            store_y = torch.full(
                (expected.numel() + 16,), 42.0, device="cuda", dtype=expected.dtype
            )
            y = store_y[8:-8].view(expected.shape)
            ref = torch.empty_like(y)
            before = [tensor_sha(t) for t in args]
            normal(row["scalars"], *args, ref)
            candidate(row["scalars"], *args, y)
            check(torch, y, expected)
            check(torch, ref, expected)
            assert tensor_sha(y) == tensor_sha(ref), label
            assert [tensor_sha(t) for t in args] == before
            assert all(
                bool((st[:8] == 42).all() & (st[-8:] == 42).all())
                for st in stores + [store_y]
            )
            rows.append(
                dict(
                    label=label,
                    special=special,
                    scalars=row["scalars"],
                    input_sha256=before,
                    output_sha256=tensor_sha(y),
                    bitwise=True,
                    policy=True,
                    canaries=True,
                    candidate=str(candidate.directory),
                    baseline=str(normal.directory),
                )
            )
            if special:
                values[label] = dict(
                    inputs=[t.cpu() for t in args],
                    actual=y.cpu(),
                    baseline=ref.cpu(),
                    eager=expected.cpu(),
                )
            write(output, dict(complete=False, rows=rows))
        row = selected
        fn, args, _ = fixture(torch, row)
        expected = fn(*args)
        y = torch.empty_like(expected)
        attempts = [
            ("scalar", (0, *row["scalars"][1:]), (*args, y)),
            ("arity", row["scalars"][:-1], (*args, y)),
            ("device", row["scalars"], (args[0].cpu(), *args[1:], y)),
            ("dtype", row["scalars"], (args[0].double(), *args[1:], y)),
            (
                "input_layout",
                row["scalars"],
                (args[0].flatten().reshape(-1, 1), *args[1:], y),
            ),
            ("output_layout", row["scalars"], (*args, y.flatten().reshape(-1, 1))),
        ]
        # Real contiguous output alias over the input storage, with identical
        # descriptor layout, isolates disjointness from a shape/stride failure.
        span = 1 + sum((sz - 1) * st for sz, st in zip(args[0].shape, args[0].stride()))
        alias_storage = torch.empty(
            max(span, expected.numel()) + 16, device="cuda", dtype=args[0].dtype
        )
        alias_input = torch.as_strided(
            alias_storage, args[0].shape, args[0].stride(), storage_offset=8
        )
        alias = torch.as_strided(
            alias_storage, expected.shape, expected.stride(), storage_offset=8
        )
        attempts.append(("alias", row["scalars"], (alias_input, *args[1:], alias)))
        if len(args) == 2:
            attempts.append(
                (
                    "broadcast_layout",
                    row["scalars"],
                    (
                        args[0],
                        args[1].flatten()
                        if args[1].ndim == 2
                        else args[1].reshape(-1, 1),
                        y,
                    ),
                )
            )
        for name, scalars, tensors in attempts:
            before_launch = candidate.kernel.launch_count
            try:
                candidate(scalars, *tensors)
            except (ValueError, TypeError) as e:
                controls.append(
                    dict(family=family, dtype=dtype, name=name, error=str(e))
                )
            else:
                raise AssertionError("missing rejection " + family + " " + name)
            assert candidate.kernel.launch_count == before_launch
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            fn, sargs, _ = fixture(torch, row)
            sy = torch.empty_like(fn(*sargs))
            candidate(row["scalars"], *sargs, sy)
            sx = fn(*sargs)
        stream.synchronize()
        check(torch, sy, sx)
        streams.append(
            dict(
                family=family,
                dtype=dtype,
                stream=stream.cuda_stream,
                offset=8,
                passes=True,
            )
        )
    torch.save(values, OUT / special_output)
    kernels = [
        dict(
            directory=str(k.directory),
            entrypoint=k.entrypoint,
            semantic=k.semantic,
            ptx_sha256=sha(k.directory / "kernel.ptx"),
        )
        for k in CACHE.values()
    ]
    write(
        output,
        dict(
            complete=True,
            rows=rows,
            live_controls=controls,
            stream_controls=streams,
            kernels=kernels,
            special_values_sha256=sha(OUT / special_output),
            binary_sha256=sha(BINARY),
            source_sha256=sha(__file__),
            independence=evidence.inventory(),
        ),
    )
    print(
        "PASS",
        len(rows),
        "GPU cases",
        len(controls),
        "live controls",
        len(streams),
        "streams",
        flush=True,
    )


def native():
    from stage5.column_layout_runtime import BINARY, admit

    first = json.loads((OUT / "gpu.json").read_text())["kernels"][0]
    packet = json.loads((OUT / first["directory"] / "descriptor.json").read_text())
    rows = []
    for label in (
        "schema",
        "entry",
        "mechanism",
        "plan",
        "index",
        "precision",
        "effect",
        "boolean_number",
    ):
        p = copy.deepcopy(packet)
        if label == "schema":
            p["schema"] = 4
        elif label == "entry":
            p["entrypoint"] = "invalid;entry"
        elif label == "mechanism":
            p["mechanism"] = "other"
        elif label == "plan":
            p["physical_plan"]["workgroup"] = [128, 1, 1]
        elif label == "index":
            p["semantic"]["ops"][1]["index"] = {"op": "constant", "value": 0}
        elif label == "precision":
            p["semantic"]["index_dtype"] = "int32"
        elif label == "effect":
            p["semantic"]["effects"]["mutation"] = True
        else:
            p["semantic"]["scalars"][0]["positive"] = 1
        if label in ("index", "precision", "effect", "boolean_number"):
            try:
                admit(p["semantic"])
            except ValueError:
                pass
            else:
                raise AssertionError("Python admitted " + label)
        path = OUT / "rejected" / label
        path.mkdir(parents=True, exist_ok=True)
        desc = path / "descriptor.json"
        desc.write_text(json.dumps(p))
        proc = subprocess.run(
            [str(BINARY), str(path), str(desc)], capture_output=True, text=True
        )
        assert proc.returncode and not (path / "cubecl.ir").exists(), label
        rows.append(dict(label=label, returncode=proc.returncode, error=proc.stderr))
    write("native-controls.json", dict(complete=True, rows=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("gpu", "native"))
    args = parser.parse_args()
    {
        "gpu": gpu,
        "native": native,
    }[args.phase]()
