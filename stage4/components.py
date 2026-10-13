"""Typed descriptor/IR controls complement real Dynamo acceptance."""

import copy
import ctypes
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from stage1 import evidence

evidence.guard()
import torch
from stage4 import runtime
from stage3 import runtime as bridge
from stage1.plan import PhysicalKernelPlan
from stage0.bridge import native_path
from stage4.reference import uniform

native = ctypes.CDLL(str(ROOT.parent / "stage0/libstage0_launcher.so"))
native.stage4_kernel_launches.restype = ctypes.c_uint64
native_begin = native.stage4_kernel_launches()


def const(n):
    return {"op": "constant", "value": n}


def scalar(n):
    return {"op": "scalar", "slot": n}


def descriptor():
    return {
        "schema": 4,
        "index_dtype": "int64",
        "numel": scalar(0),
        "scalars": [{"id": 0, "dtype": "int64", "positive": False}],
        "buffers": [
            {
                "access": "read",
                "dtype": "int64",
                "size": [const(1)],
                "stride": [const(1)],
                "offset": const(0),
            },
            {
                "access": "write",
                "dtype": "float32",
                "size": [scalar(0)],
                "stride": [const(1)],
                "offset": const(0),
            },
        ],
        "effects": {"rng": "explicit_seed_counter", "mutation": False},
        "ops": [
            {
                "id": "seed",
                "op": "load",
                "args": [],
                "dtype": "int64",
                "semantic_dtype": "int64",
                "buffer": 0,
                "index": const(0),
            },
            {
                "id": "wide",
                "op": "index",
                "args": [],
                "dtype": "int64",
                "semantic_dtype": "int64",
                "index": {"op": "linear"},
            },
            {
                "id": "counter",
                "op": "cast",
                "args": ["wide"],
                "dtype": "int32",
                "semantic_dtype": "int32",
            },
            {
                "id": "rng",
                "op": "rand",
                "args": ["seed", "counter"],
                "dtype": "float32",
                "semantic_dtype": "float32",
                "algorithm": "philox4x32_10_top24",
            },
            {
                "op": "store",
                "args": ["rng"],
                "dtype": "float32",
                "semantic_dtype": "float32",
                "buffer": 1,
                "index": {"op": "linear"},
            },
        ],
    }


base = descriptor()
kernel = runtime.compile_kernel(base, 0, PhysicalKernelPlan().to_dict())
rows = []
for seed in (0, 1, -1, -(1 << 63), (1 << 63) - 1, 0x12345678ABCDEF):
    for n in (0, 1, 255, 256, 257):
        s = torch.tensor([seed], device="cuda", dtype=torch.int64)
        backing = torch.full((n + 2,), 123.0, device="cuda")
        out = backing[1 : n + 1]
        kernel((n,), s, out)
        expected = torch.tensor(
            [uniform(seed, i) for i in range(n)], device="cuda", dtype=torch.float32
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert backing[0].item() == backing[-1].item() == 123
        rows.append({"seed": seed, "n": n, "exact": True, "canaries": True})
counter_bits = []
for counter in (0, -1, -(1 << 31), (1 << 31) - 1):
    d = descriptor()
    d["ops"][2] = {
        "id": "counter",
        "op": "constant",
        "args": [],
        "dtype": "int32",
        "semantic_dtype": "int32",
        "value": counter,
    }
    k = runtime.compile_kernel(d, 0, PhysicalKernelPlan().to_dict())
    seed = torch.tensor([0], device="cuda", dtype=torch.int64)
    y = torch.empty(1, device="cuda")
    k((1,), seed, y)
    torch.testing.assert_close(
        y,
        torch.tensor([uniform(0, counter & ((1 << 32) - 1))], device="cuda"),
        rtol=0,
        atol=0,
    )
    counter_bits.append(
        {
            "counter_i32": counter,
            "unsigned_bits": counter & ((1 << 32) - 1),
            "artifact": str(k.directory),
            "exact": True,
        }
    )
# Runtime-dependent lazy regions protect genuinely invalid tensor indices.
masked = descriptor()
masked["effects"]["rng"] = None
masked["buffers"] = [
    {
        "access": "read",
        "dtype": "float32",
        "size": [const(7)],
        "stride": [const(1)],
        "offset": const(0),
    },
    {
        "access": "read",
        "dtype": "int64",
        "size": [scalar(0)],
        "stride": [const(1)],
        "offset": const(0),
    },
    {
        "access": "write",
        "dtype": "float32",
        "size": [scalar(0)],
        "stride": [const(1)],
        "offset": const(0),
    },
]
masked["ops"] = [
    {
        "id": "position",
        "op": "index",
        "args": [],
        "dtype": "int64",
        "semantic_dtype": "int64",
        "index": {"op": "linear"},
    },
    {
        "id": "bound",
        "op": "constant",
        "args": [],
        "dtype": "int64",
        "semantic_dtype": "int64",
        "value": 128,
    },
    {
        "id": "pred",
        "op": "lt",
        "args": ["position", "bound"],
        "dtype": "bool",
        "semantic_dtype": "bool",
    },
    {
        "id": "mask",
        "op": "masked",
        "args": ["pred"],
        "dtype": "float32",
        "semantic_dtype": "float32",
        "body": [
            {
                "id": "index",
                "op": "load",
                "args": [],
                "dtype": "int64",
                "semantic_dtype": "int64",
                "buffer": 1,
                "index": {"op": "linear"},
            },
            {
                "id": "checked",
                "op": "indirect",
                "args": ["index"],
                "dtype": "int64",
                "semantic_dtype": "int64",
                "size": const(7),
                "bounds": "require_in_range",
            },
            {
                "id": "loaded",
                "op": "load",
                "args": [],
                "dtype": "float32",
                "semantic_dtype": "float32",
                "buffer": 0,
                "index": {"op": "value", "id": "checked"},
            },
        ],
        "result": "loaded",
        "other": 7.0,
    },
    {
        "op": "store",
        "args": ["mask"],
        "dtype": "float32",
        "semantic_dtype": "float32",
        "buffer": 2,
        "index": {"op": "linear"},
    },
]
masks = []
for nested in (False, True):
    d = copy.deepcopy(masked)
    limit = 64 if nested else 128
    if nested:
        inner = copy.deepcopy(d["ops"][3])
        inner["id"] = "inner"
        inner["args"] = ["inner_pred"]
        d["ops"][3]["body"] = [
            {
                "id": "inner_bound",
                "op": "constant",
                "args": [],
                "dtype": "int64",
                "semantic_dtype": "int64",
                "value": 64,
            },
            {
                "id": "inner_pred",
                "op": "lt",
                "args": ["position", "inner_bound"],
                "dtype": "bool",
                "semantic_dtype": "bool",
            },
            inner,
        ]
        d["ops"][3]["result"] = "inner"
    k = runtime.compile_kernel(d, 0, PhysicalKernelPlan().to_dict())
    x = torch.arange(7, device="cuda", dtype=torch.float32)
    indices = torch.arange(257, device="cuda") % 7
    indices[limit:] = 1 << 32
    backing = torch.full((259,), 123.0, device="cuda")
    y = backing[1:258]
    k((257,), x, indices, y)
    expected = torch.full_like(y, 7)
    expected[:limit] = x[indices[:limit]]
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    assert backing[0].item() == backing[-1].item() == 123
    masks.append(
        {
            "nested": nested,
            "artifact": str(k.directory),
            "runtime_predicate": True,
            "invalid_indices_not_accessed": True,
            "canaries": True,
        }
    )
invalid = []
for name in (
    "counter_width",
    "rng_algorithm",
    "missing_effect",
    "bad_plan",
    "index_storage",
    "untyped_counter",
    "unchecked_address",
    "escaped_checked_address",
    "oversized_rng_extent",
):
    d = copy.deepcopy(base)
    d["entrypoint"] = "cubecl4_invalid"
    d["physical_plan"] = PhysicalKernelPlan().to_dict()
    if name == "counter_width":
        d["ops"][2] = {
            "id": "counter",
            "op": "constant",
            "args": [],
            "dtype": "int32",
            "semantic_dtype": "int32",
            "value": 1 << 31,
        }
    elif name == "rng_algorithm":
        d["ops"][3]["algorithm"] = "host_random"
    elif name == "missing_effect":
        d["effects"]["rng"] = None
    elif name == "bad_plan":
        d["physical_plan"]["local_vector_width"] = 4
    elif name == "index_storage":
        d["buffers"][0]["dtype"] = "int32"
        d["ops"][0]["dtype"] = "int32"
        d["ops"][0]["semantic_dtype"] = "int32"
    elif name == "oversized_rng_extent":
        d["numel"] = const(1 << 31)
    elif name == "untyped_counter":
        d["ops"][2]["args"] = ["missing"]
    else:
        d = copy.deepcopy(masked)
        d["entrypoint"] = "cubecl4_invalid"
        d["physical_plan"] = PhysicalKernelPlan().to_dict()
        if name == "unchecked_address":
            d["ops"][3]["body"][2]["index"]["id"] = "index"
        else:
            d["ops"][-1]["index"] = {"op": "value", "id": "checked"}
    dest = ROOT / "artifacts/invalid-descriptor" / name
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "descriptor.json"
    path.write_text(json.dumps(d))
    proc = subprocess.run(
        [str(native_path("cubecl-stage0")), str(dest), "cpp", str(path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode and not (dest / "cubecl.ir").exists(), name
    invalid.append({"case": name, "error": proc.stderr.strip()})
# Missing native exports fail at selection, before codegen or execution.
from unittest.mock import patch
from types import SimpleNamespace
from stage4.backend import CubeCLScheduling, PreviousScheduling
from stage1.runtime import CubeCLBackendUnavailable

availability_rejections = []
for missing in ("metadata", "launcher"):
    libs = [
        SimpleNamespace(**({} if missing == "metadata" else {"stage4_metadata": None})),
        SimpleNamespace(**({} if missing == "launcher" else {"stage4_launch": None})),
    ]
    with (
        patch.object(PreviousScheduling, "check_backend_available"),
        patch("ctypes.CDLL", side_effect=libs),
    ):
        try:
            CubeCLScheduling.__new__(CubeCLScheduling).check_backend_available(
                torch.device("cuda")
            )
        except CubeCLBackendUnavailable as error:
            availability_rejections.append({"missing": missing, "error": str(error)})
        else:
            raise AssertionError("missing ABI accepted")
runtime_guard = runtime.Kernel.__new__(runtime.Kernel)
runtime_guard.semantic = descriptor()


class FakeOutput:
    def numel(self):
        return 1 << 31


try:
    runtime_guard.launch_extent((1 << 31,), (FakeOutput(),))
except ValueError as error:
    assert "unsupported RNG physical extent" in str(error)
else:
    raise AssertionError("runtime counter domain accepted")
assert bridge.KERNEL_LAUNCHES == native.stage4_kernel_launches() - native_begin
evidence.state("after_components")
(ROOT / "artifacts/components.json").write_text(
    json.dumps(
        {
            "rng_checks": rows,
            "counter_bit_checks": counter_bits,
            "lazy_invalid_address_guards": masks,
            "invalid": invalid,
            "availability_rejections": availability_rejections,
            "native_launches": native.stage4_kernel_launches() - native_begin,
            "compiles": bridge.COMPILER_INVOCATIONS,
            "launches": bridge.KERNEL_LAUNCHES,
            "independence": evidence.inventory(),
        },
        indent=2,
    )
)
print(
    "Stage4 components PASS",
    len(rows),
    "RNG cases + two runtime lazy masks +",
    len(invalid),
    "rejects",
)
