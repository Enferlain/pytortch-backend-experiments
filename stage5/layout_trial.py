"""Bounded broadcast/strided layout reuse: capture first, immutable evidence."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stage5/artifacts/reuse-first-layout"
os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "stage5/cache/reuse-first-layout")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def tensor_sha(t):
    import torch

    return hashlib.sha256(
        t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def check(torch, actual, expected):
    dtype = str(actual.dtype).removeprefix("torch.")
    rtol, atol = {
        "float32": (2e-6, 2e-6),
        "float16": (0.002, 0.001),
        "bfloat16": (0.016, 0.008),
    }[dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    for predicate in (torch.isnan, torch.isposinf, torch.isneginf):
        assert torch.equal(predicate(actual), predicate(expected))
    zero = (actual == 0) & (expected == 0)
    assert torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))


def cases():
    from stage5.corpus import cases as corpus

    return [
        c
        for c in corpus()
        if c.family in ("bias_row", "bias_column", "slice", "stride_zero")
    ]


def capture():
    from stage1 import evidence

    evidence.guard()
    from stage1.overlay import install

    install()
    import torch
    from torch._inductor import config
    from stage5 import artifact, runtime
    from stage5.backend import compile_backend
    from stage5.corpus import fixture
    from stage5.diagnostics import detailed_trace

    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "capture.json").exists()
    torch.set_num_threads(1)
    original = runtime.AutotunedKernel.__call__
    observed = []

    def observe(self, scalars, *tensors):
        result = original(self, scalars, *tensors)
        observed.append((tuple(int(s) for s in scalars), self.last))
        return result

    rows, functions = [], {}
    with (
        patch.object(runtime.AutotunedKernel, "__call__", observe),
        config.patch({"fx_graph_cache": False}),
    ):
        for i, case in enumerate(cases()):
            torch.manual_seed(7101 + i)
            f = fixture(case)
            key = (case.family, case.dtype)
            if key not in functions:
                functions[key] = torch.compile(
                    f.fn,
                    backend=lambda g, i: compile_backend(g, i, tuning="baseline"),
                    fullgraph=True,
                    dynamic=True,
                )
            start = len(artifact.EVENTS)
            with detailed_trace():
                actual = functions[key](*f.args)
            torch.cuda.synchronize()
            check(torch, actual, f.fn(*f.args))
            scalars, selected = observed[-1]
            source = selected.directory
            target = OUT / "baseline" / source.name
            target.mkdir(parents=True, exist_ok=True)
            for name in (
                "semantic.json",
                "descriptor.json",
                "physical-plan.json",
                "manifest.json",
                "cubecl.ir",
                "kernel.cu",
                "kernel.ptx",
                "nvrtc.json",
                "resources.json",
            ):
                shutil.copyfile(source / name, target / name)
            launches = [e for e in artifact.EVENTS[start:] if e["event"] == "launch"]
            assert len(launches) == 1
            rows.append(
                dict(
                    case=case.to_dict(),
                    scalars=list(scalars),
                    input_layouts=[
                        dict(
                            shape=list(t.shape),
                            stride=list(t.stride()),
                            offset=t.storage_offset(),
                        )
                        for t in f.args
                    ],
                    output_layout=dict(
                        shape=list(actual.shape), stride=list(actual.stride())
                    ),
                    input_sha256=[tensor_sha(t) for t in f.args],
                    output_sha256=tensor_sha(actual),
                    baseline=str(target.relative_to(OUT)),
                    launch=launches[0],
                    policy=True,
                )
            )
            print(case.name, "PASS", scalars, flush=True)
    write(
        "capture.json",
        dict(
            complete=True,
            rows=rows,
            source_sha256=sha(__file__),
            compiler_identity=artifact.compiler_identity(0),
            independence=evidence.inventory(),
            head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        ),
    )


if __name__ == "__main__":
    capture()
