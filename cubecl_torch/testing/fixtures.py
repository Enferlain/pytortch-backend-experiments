"""Shared test fixtures that use only the runtime layer."""

from pathlib import Path

from cubecl_torch.runtime import rules
from cubecl_torch.runtime.artifact import (
    Guard,
    KernelArtifact,
    LaunchShape,
    ScalarParam,
    TensorParam,
    audited_provenance,
)

REPO = Path(__file__).resolve().parents[2]
STAGE0_PTX = REPO / "stage0/artifacts/cpp/kernel.ptx"
STAGE0_CUDA = REPO / "stage0/artifacts/cpp/kernel.cu"
EXTENT_LIMIT = 1 << 24


def stage0_sin_plus_one(**overrides):
    """Hand-written contract for the Stage 0 programmatic ``y = sin(x) + 1`` PTX.

    The PTX was not produced by the descriptor compiler. Its CUDA source is
    byte-identical to a rebuild from the audited CubeCL source (checked by
    ``test_stage0_source_matches_audited_build``), so it carries the audited
    provenance.
    """
    n = rules.scalar(0)
    tensor = {"size": (n,), "stride": (rules.constant(1),), "offset": rules.constant(0)}
    fields = {
        "ptx": STAGE0_PTX.read_bytes(),
        "entry": "cubecl_sin_add",
        "tensors": (
            TensorParam("float32", **tensor),
            TensorParam("float32", **tensor, access="write"),
        ),
        "scalars": (ScalarParam(),),
        "extent": n,
        "info_scalars": (),
        "metadata_words": 2,
        "launch": LaunchShape(rules.ceil_div(n, 256), (256, 1, 1)),
        "guards": (
            Guard("output_numel", (n,), "sin_plus_one output extent mismatch"),
            Guard(
                "range",
                (n,),
                "sin_plus_one extent outside the declared range",
                minimum=0,
                maximum=EXTENT_LIMIT,
            ),
        ),
        "provenance": {**audited_provenance(), "producer": "stage0-programmatic"},
    }
    return KernelArtifact(**{**fields, **overrides})


DATA = Path(__file__).resolve().parent / "data"
STORAGE_DTYPES = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "bool": "bool",
    "int64": "int64",
}


def descriptor(name):
    """A real Stage 5 semantic descriptor stored under ``testing/data``."""
    import json

    return json.loads((DATA / f"{name}.json").read_text())


def tensors_for(semantic, scalars, device="cuda", seed=0):
    """Inputs (random) and output (empty) matching the descriptor's layouts."""
    import torch

    from cubecl_torch.runtime.rules import evaluate

    generator = torch.Generator(device=device).manual_seed(seed)
    result = []
    for b in semantic["buffers"]:
        size = [evaluate(v, scalars) for v in b["size"]]
        stride = [evaluate(v, scalars) for v in b["stride"]]
        dtype = getattr(torch, STORAGE_DTYPES[b["dtype"]])
        t = torch.empty_strided(size, stride, dtype=dtype, device=device)
        if b["access"] == "read" and dtype.is_floating_point:
            t.copy_(torch.randn(size, generator=generator, device=device))
        result.append(t)
    return result


def sin_plus_one_descriptor():
    """Hand-authored dynamic schema-4 descriptor for ``y = sin(x) + 1``."""
    n = rules.scalar(0)
    buffer = {
        "dtype": "float32",
        "offset": rules.constant(0),
        "size": [n],
        "stride": [rules.constant(1)],
    }
    linear = {"op": "linear"}
    return {
        "schema": 4,
        "index_dtype": "int64",
        "scalars": [{"dtype": "int64", "id": 0, "positive": True}],
        "numel": n,
        "effects": {"mutation": False, "rng": None},
        "buffers": [{**buffer, "access": "read"}, {**buffer, "access": "write"}],
        "ops": [
            {
                "op": "load",
                "id": "tmp0",
                "args": [],
                "buffer": 0,
                "index": linear,
                "dtype": "float32",
                "semantic_dtype": "float32",
            },
            {
                "op": "sin",
                "id": "tmp1",
                "args": ["tmp0"],
                "dtype": "float32",
                "semantic_dtype": "float32",
            },
            {
                "op": "constant",
                "id": "tmp2",
                "args": [],
                "value": 1.0,
                "dtype": "float32",
                "semantic_dtype": "float32",
            },
            {
                "op": "add",
                "id": "tmp3",
                "args": ["tmp1", "tmp2"],
                "dtype": "float32",
                "semantic_dtype": "float32",
            },
            {
                "op": "store",
                "args": ["tmp3"],
                "buffer": 1,
                "index": linear,
                "dtype": "float32",
                "semantic_dtype": "float32",
            },
        ],
    }


BACKEND_PRELUDE = """
import json, os, sys
from cubecl_torch.runtime import guard
guard.install()
import torch
import cubecl_torch.inductor as ci
ci.install()
"""


def run_python(code, cache_dir, *, prelude=BACKEND_PRELUDE, env=None, check=True):
    """Run ``prelude + code`` in a fresh interpreter; parse the last stdout line as JSON."""
    import json
    import os
    import subprocess
    import sys

    environment = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHON_GIL", "CUBECL_TORCH_TRACE")
    }
    environment.update({"CUBECL_TORCH_CACHE_DIR": str(cache_dir), **(env or {})})
    proc = subprocess.run(
        [sys.executable, "-c", prelude + code],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode:
        raise AssertionError(
            f"subprocess failed ({proc.returncode}):\n{proc.stderr[-4000:]}"
        )
    lines = proc.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines else None


def captured(name):
    """Descriptors captured by the Stage 5 baseline run (``stage5/fixtures.py``)."""
    import copy
    import json

    return copy.deepcopy(json.loads((DATA / "captured.json").read_text())[name])


def native_launches():
    """Process-wide kernel launches counted by the native launcher."""
    import ctypes

    from cubecl_torch.runtime import native

    lib = native.launcher()
    lib.stage0_kernel_launches.restype = ctypes.c_uint64
    return lib.stage0_kernel_launches()


def physical(semantic, strategy, *args, **fields):
    """A loaded package kernel for ``semantic`` under one catalog plan."""
    from cubecl_torch.compiler.plan import PhysicalCandidate
    from cubecl_torch.inductor.kernels import CompiledKernel

    return CompiledKernel(
        semantic, 0, PhysicalCandidate(strategy, *args, **fields).to_dict()
    )
