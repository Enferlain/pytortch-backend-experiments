"""Bounded composition of previously qualified kernels, with no new GPU body."""

import copy
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stage5/artifacts/composition"
os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def initialize():
    from stage1 import evidence

    evidence.guard()
    from stage1.overlay import install

    install()
    import torch

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "stage5/cache/composition")
    torch.set_num_threads(1)
    torch._dynamo.config.recompile_limit = 256
    return torch


def role(semantic):
    from stage5.rms_packed_trial_runtime import admit as rms_admit
    from stage5.column_layout_runtime import admit as column_admit

    try:
        rms_admit(semantic)
        return "rms"
    except ValueError:
        pass
    try:
        column_admit(semantic)
        return "column"
    except ValueError:
        return "unadmitted"


def parallel(r, x, bias):
    from stage5.corpus import rms

    return rms(r), x + bias


def serial(x, bias):
    from stage5.corpus import rms

    return rms(x + bias)


def prepare():
    assert not OUT.exists(), "preserve completed or failed phases"
    OUT.mkdir(parents=True)
    preserved = {}
    names = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"]
    ).decode().split("\0")
    for name in sorted(set(names) - {""}):
        if name.startswith((".beads/", "stage5/composition", "stage5/artifacts/composition/")):
            continue
        if name in ("STAGE5_REPORT.md", "stage5/.gitignore"):
            (OUT / (Path(name).name + ".before")).write_bytes((ROOT / name).read_bytes())
        elif (ROOT / name).is_file():
            preserved[name] = sha(ROOT / name)
    write("preservation.json", preserved)
    write("plan.json", dict(
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        fit_checks=["natural serial bias then RMS", "independent two-output RMS and column"],
        modes=["ordinary", "rms", "column", "both"],
        cases=[dict(dtype=d, rms_rows=b, column_shape=s)
               for d in ("float32", "float16", "bfloat16")
               for b, s in ((16, [3, 255]), (64, [64, 256]), (65, [64, 257]))],
        rms_width=1024,
        timing_calls=16,
        order=[["ordinary", "rms", "both", "column"],
               ["rms", "column", "ordinary", "both"],
               ["column", "both", "rms", "ordinary"],
               ["both", "ordinary", "column", "rms"],
               ["column", "both", "rms", "ordinary"],
               ["both", "ordinary", "column", "rms"],
               ["ordinary", "rms", "both", "column"],
               ["rms", "column", "ordinary", "both"]],
        stop="No forced graph breaks, barriers, altered fusion or broader descriptor mapping",
        no_new_gpu_body=True,
    ))


def preflight():
    torch = initialize()
    from torch._inductor import config
    from stage4 import backend as semantic
    from stage5.backend import CubeCLScheduling
    from stage5.choices import CubeCLChoices
    from stage5 import runtime
    from stage5.layout_trial import check, tensor_sha

    assert not (OUT / "preflight.json").exists()
    records = []
    active = []

    class Capture(CubeCLScheduling):
        def define_kernel(self, src_code, node_schedule, kernel):
            active.append(copy.deepcopy(kernel.descriptor))
            return super().define_kernel(src_code, node_schedule, kernel)

    def backend(gm, args):
        return semantic.compile_backend(
            gm, args, scheduling=partial(Capture, tuning="first_use"),
            choices_class=CubeCLChoices, backend_name="cubecl_stage5_composition_fit",
        )

    original = runtime.AutotunedKernel.__call__
    selected = []

    def observe(self, *args, **kw):
        value = original(self, *args, **kw)
        selected.append(self.last)
        return value

    torch.manual_seed(68001)
    x = torch.randn(64, 256, device="cuda")
    b = torch.randn(64, 1, device="cuda")
    r = torch.randn(64, 1024, device="cuda")
    with config.patch({"fx_graph_cache": False}), patch.object(runtime.AutotunedKernel, "__call__", observe):
        for name, fn, args in (("serial", serial, (x, b)), ("parallel", parallel, (r, x, b))):
            active.clear()
            selected.clear()
            compiled = torch.compile(fn, backend=backend, dynamic=True, fullgraph=True)
            actual, expected = compiled(*args), fn(*args)
            aa = actual if isinstance(actual, tuple) else (actual,)
            ee = expected if isinstance(expected, tuple) else (expected,)
            for a, e in zip(aa, ee):
                check(torch, a, e)
            records.append(dict(
                graph=name, descriptors=[dict(semantic=s, role=role(s)) for s in active],
                outputs=[tensor_sha(a) for a in aa],
                selected=[dict(directory=str(k.directory), plan=k.kernel.plan.to_dict(),
                               ptx_sha256=sha(k.directory / "kernel.ptx")) for k in selected],
            ))
            write("preflight-progress.json", records)
    # Historical imports set their own caches; restore the new packet cache.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "stage5/cache/composition")
    write("preflight.json", dict(complete=True, records=records))
    print([(r["graph"], [s["role"] for s in r["descriptors"]]) for r in records], flush=True)


if __name__ == "__main__":
    import sys

    {"prepare": prepare, "preflight": preflight}[sys.argv[1]]()
