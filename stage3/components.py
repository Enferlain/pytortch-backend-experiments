"""Physical-plan boundary tests complement real lowering's 0/1 specialization."""

import copy
import ctypes as C
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from stage1 import evidence

evidence.guard()
import torch
from stage3 import runtime
from stage3.plan import ReductionPlan
from stage0.bridge import native_path

native = C.CDLL(str(ROOT.parent / "stage0/libstage0_launcher.so"))
native.stage3_kernel_launches.restype = C.c_uint64
native.stage3_total_shared_bytes.restype = C.c_uint64
reduction_begin = native.stage3_kernel_launches()
shared_begin = native.stage3_total_shared_bytes()


def constant(v):
    return {"op": "constant", "value": v}


def scalar(i):
    return {"op": "scalar", "slot": i}


def descriptor(dtype):
    return {
        "schema": 3,
        "index_dtype": "int64",
        "numel": scalar(0),
        "outer": scalar(0),
        "reduction_extent": scalar(1),
        "scalars": [
            {"dtype": "int64", "id": 0, "positive": True},
            {"dtype": "int64", "id": 1, "positive": False},
        ],
        "buffers": [
            {
                "access": "read",
                "dtype": dtype,
                "size": [{"op": "mul", "args": [scalar(0), scalar(1)]}],
                "stride": [constant(1)],
                "offset": constant(0),
            },
            {
                "access": "write",
                "dtype": dtype,
                "size": [scalar(0)],
                "stride": [constant(1)],
                "offset": constant(0),
            },
        ],
        "ops": [
            {
                "id": "x",
                "op": "load",
                "args": [],
                "dtype": dtype,
                "semantic_dtype": dtype,
                "buffer": 0,
                "index": {
                    "op": "add",
                    "args": [
                        {"op": "mul", "args": [{"op": "linear"}, scalar(1)]},
                        {"op": "reduction"},
                    ],
                },
            },
            {
                "id": "wide",
                "op": "cast",
                "args": ["x"],
                "dtype": "float32",
                "semantic_dtype": dtype,
            },
            {
                "id": "sum",
                "op": "reduction",
                "args": ["wide"],
                "dtype": "float32",
                "semantic_dtype": dtype,
                "source_dtype": dtype,
                "reduction_type": "sum",
                "accumulator_dtype": "float32",
            },
            {
                "id": "out",
                "op": "cast",
                "args": ["sum"],
                "dtype": dtype,
                "semantic_dtype": dtype,
            },
            {
                "op": "store",
                "args": ["out"],
                "dtype": dtype,
                "semantic_dtype": dtype,
                "buffer": 1,
                "index": {"op": "linear"},
                "iteration_scope": "outer",
            },
        ],
    }


rows = []
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    semantic = descriptor(str(dtype).removeprefix("torch."))
    k = runtime.compile_kernel(semantic, 0, ReductionPlan().to_dict())
    before = runtime.COMPILER_INVOCATIONS
    for n in (0, 1, 31, 32, 33, 255, 256, 257, 4099, 65539):
        x = torch.randn(3 * n, device="cuda", dtype=dtype)
        backing = torch.full((5,), 123, device="cuda", dtype=dtype)
        y = backing[1:4]
        k((3, n), x, y)
        expected = x.view(3, n).sum(-1)
        torch.testing.assert_close(
            y,
            expected,
            rtol=0.016
            if dtype == torch.bfloat16
            else (0.002 if dtype == torch.float16 else 2e-5),
            atol=0.008
            if dtype == torch.bfloat16
            else (0.001 if dtype == torch.float16 else 2e-5),
        )
        assert torch.equal(
            backing[[0, 4]], torch.full((2,), 123, device="cuda", dtype=dtype)
        )
        rows.append(
            {
                "dtype": str(dtype),
                "n": n,
                "artifact": str(k.directory),
                "canaries": True,
            }
        )
    assert runtime.COMPILER_INVOCATIONS == before
    try:
        k((3, 1048577), x, y)
    except ValueError as error:
        assert "unsupported physical reduction extent" in str(error)
    else:
        raise AssertionError("unsupported physical reduction extent accepted")

masked_checks = []
# Scoped semantic CSE can return an enclosing value without emitting any local
# operations. Its result edge remains a dependency, including in nested masks.
for nested, outer_true, inner_true in (
    (False, True, True),
    (False, False, True),
    (True, True, True),
    (True, True, False),
    (True, False, True),
):
    semantic = descriptor("float32")
    prefix = [
        {
            "id": name,
            "op": "constant",
            "args": [],
            "dtype": dtype,
            "semantic_dtype": dtype,
            "value": value,
        }
        for name, dtype, value in (
            ("uniform", "float32", 7.0),
            ("outer_predicate", "bool", outer_true),
            ("inner_predicate", "bool", inner_true),
        )
    ]
    inner = {
        "id": "inner",
        "op": "masked",
        "args": ["inner_predicate"],
        "dtype": "float32",
        "semantic_dtype": "float32",
        "body": [],
        "result": "uniform",
        "other": 0.0,
    }
    mask = {
        "id": "masked",
        "op": "masked",
        "args": ["outer_predicate"],
        "dtype": "float32",
        "semantic_dtype": "float32",
        "body": [inner] if nested else [],
        "result": "inner" if nested else "uniform",
        "other": 0.0,
    }
    suffix = copy.deepcopy(semantic["ops"][2:])
    suffix[0]["args"] = ["masked"]
    semantic["ops"] = prefix + [mask] + suffix
    k = runtime.compile_kernel(semantic, 0, ReductionPlan().to_dict())
    x = torch.empty(3 * 257, device="cuda")
    backing = torch.full((5,), 123.0, device="cuda")
    y = backing[1:4]
    k((3, 257), x, y)
    expected = 7 * 257 if outer_true and (inner_true or not nested) else 0
    torch.testing.assert_close(y, torch.full_like(y, expected), rtol=0, atol=0)
    assert torch.equal(backing[[0, 4]], torch.full((2,), 123.0, device="cuda"))
    masked_checks.append(
        {
            "nested": nested,
            "outer_predicate": outer_true,
            "inner_predicate": inner_true,
            "expected": expected,
            "artifact": str(k.directory),
            "canaries": True,
            "launch": runtime.EVENTS[-1],
        }
    )

invalid = []
base = descriptor("float32")
for name in (
    "divergent_barrier",
    "missing_conversion",
    "unsupported_accumulator",
    "outer_varying_store",
    "masked_reduction",
    "wrong_semantic_dtype",
):
    d = copy.deepcopy(base)
    d["physical_plan"] = ReductionPlan().to_dict()
    d["entrypoint"] = "cubecl3_invalid"
    if name == "divergent_barrier":
        d["physical_plan"]["barriers"] = "under_lane_predicate"
    elif name == "missing_conversion":
        d["ops"][1]["dtype"] = "bool"
    elif name == "unsupported_accumulator":
        d["ops"][2]["accumulator_dtype"] = "float16"
    elif name == "outer_varying_store":
        d["ops"][-1]["index"] = {"op": "reduction"}
    elif name == "wrong_semantic_dtype":
        d["ops"][2]["semantic_dtype"] = "int64"
    else:
        d["ops"][2] = {
            "id": "sum",
            "op": "masked",
            "args": ["wide"],
            "dtype": "float32",
            "semantic_dtype": "float32",
            "body": [d["ops"][2]],
            "result": "sum",
            "other": 0.0,
        }
    out = ROOT / "artifacts/invalid-descriptor" / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "descriptor.json"
    path.write_text(json.dumps(d))
    proc = subprocess.run(
        [str(native_path("cubecl-stage0")), str(out), "cpp", str(path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode and not (out / "cubecl.ir").exists(), name
    invalid.append({"name": name, "error": proc.stderr.strip()})

evidence.state("after_components")
native_reductions = native.stage3_kernel_launches() - reduction_begin
native_shared_bytes = native.stage3_total_shared_bytes() - shared_begin
assert native_reductions == len(rows) + len(masked_checks) == runtime.KERNEL_LAUNCHES
assert native_shared_bytes == native_reductions * 1024
(ROOT / "artifacts/components.json").write_text(
    json.dumps(
        {
            "checks": rows,
            "masked_result_checks": masked_checks,
            "invalid": invalid,
            "compiles": runtime.COMPILER_INVOCATIONS,
            "launches": runtime.KERNEL_LAUNCHES,
            "native_reduction_launches": native_reductions,
            "native_requested_shared_bytes": native_shared_bytes,
            "independence": evidence.inventory(),
        },
        indent=2,
    )
)
print("Stage 3 components PASS")
