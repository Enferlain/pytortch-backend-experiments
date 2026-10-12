"""Execute fixed audit IR through the existing compiler/PyTorch-stream bridge."""

import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from stage1 import evidence

evidence.guard()
import torch
from stage0.bridge import Kernel, native_path

evidence.state("audit_after_torch_bridge")
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
CASES = (
    "add_positive_zero_left",
    "add_positive_zero_right",
    "add_negative_zero_left",
    "add_negative_zero_right",
    "sub_self",
    "mul_positive_zero_left",
    "mul_positive_zero_right",
    "mul_negative_zero_left",
    "mul_negative_zero_right",
    "div_positive_zero",
    "div_negative_zero",
    "rem_by_one",
    "rem_positive_zero",
    "rem_negative_zero",
    "floor_positive_zero",
    "floor_negative_zero",
    "loaded_add",
    "loaded_mul",
    "loaded_div",
    "loaded_rem",
    "loaded_floor",
    "contract_add",
    "contract_sub",
    "contract_rsub",
)
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "uint64": torch.uint64,
}
lib = C.CDLL(str(ROOT.parent / "stage0/libstage0_launcher.so"))
lib.stage0_kernel_launches.restype = C.c_uint64
lib.stage0_launcher_calls.restype = C.c_uint64
RESULT = {
    "checks": [],
    "compiler_invocations": 0,
    "launches": 0,
    "policy": {
        "implicit_contraction": False,
        "fp_compute": "float32",
        "fast_math": "no relaxed assumptions",
        "payloads_exceptions_ftz_all_backends": "not claimed",
    },
    "failures": [],
}


def inputs(dtype, case, n=257):
    if dtype in (torch.int64, torch.uint64):
        values = [-7, -1, 0, 1, 7] if dtype == torch.int64 else [0, 1, 7, 2**32 + 257]
        return tuple(
            torch.tensor(
                (values * ((n + len(values) - 1) // len(values)))[:n],
                device="cuda",
                dtype=dtype,
            )
            for _ in range(2)
        )
    xs = [
        -float("inf"),
        -2.5,
        -1.5,
        -1.0,
        -0.0,
        0.0,
        1.0,
        1.5,
        2.5,
        float("inf"),
        float("nan"),
        2**-126,
        2**-149,
        1e10,
        -1e10,
    ]
    ys = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        float("inf"),
        -float("inf"),
        float("nan"),
        0.5,
        -0.5,
        0.125,
        3.0,
        -3.0,
    ]
    pairs = [(x, y) for x in xs for y in ys]
    pairs = (pairs * ((n + len(pairs) - 1) // len(pairs)))[:n]
    x = torch.tensor([p[0] for p in pairs], device="cuda", dtype=dtype)
    y = torch.tensor([p[1] for p in pairs], device="cuda", dtype=dtype)
    if case.startswith("contract"):
        epsilon = (
            2**-23
            if dtype == torch.float32
            else 2**-10
            if dtype == torch.float16
            else 2**-7
        )
        x.fill_(1 + epsilon)
        y.fill_(1 - epsilon)
    return x, y


def reference(case, x, y):
    if case in ("srem_by_one", "urem_by_one"):
        return torch.zeros_like(x)
    x, y = x.float(), y.float()
    zero = -0.0 if "negative_zero" in case else 0.0
    if case.startswith("add_"):
        return zero + x if case.endswith("left") else x + zero
    if case.startswith("mul_"):
        return zero * x if case.endswith("left") else x * zero
    if case == "sub_self":
        return x - x
    if case.startswith("div_"):
        return zero / y
    if case == "rem_by_one":
        return torch.fmod(x, 1)
    if case.startswith("rem_"):
        return torch.fmod(torch.full_like(x, zero), y)
    if case.startswith("floor_"):
        x = torch.full_like(x, zero)
        return x - torch.floor(x / y) * y
    if case == "loaded_add":
        return x + y
    if case == "loaded_mul":
        return x * y
    if case == "loaded_div":
        return x / y
    if case == "loaded_rem":
        return torch.fmod(x, y)
    if case == "loaded_floor":
        return x - torch.floor(x / y) * y
    if case in ("contract_add", "contract_sub"):
        return x * y - 1
    if case == "contract_rsub":
        return 1 - x * y
    raise ValueError(case)


def enqueue(k, x, y, out):
    # This lower-layer probe intentionally bypasses schema-1's float32-only public
    # input contract, after validating its own exact three-buffer ABI here.
    assert x.device == y.device == out.device == k.device
    assert x.dtype == y.dtype == out.dtype and x.shape == y.shape == out.shape
    assert all(t.is_contiguous() and t.storage_offset() == 0 for t in (x, y, out))
    intervals = [
        (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in (x, y, out)
    ]
    assert out.numel() == 0 or all(
        a[1] <= b[0] or b[1] <= a[0]
        for index, a in enumerate(intervals)
        for b in intervals[index + 1 :]
    )
    before = lib.stage0_kernel_launches()
    with torch.profiler.record_function("fp_hardening::CubeCL.launch"):
        k._enqueue((x, y, out), x.numel(), x.numel())
    delta = lib.stage0_kernel_launches() - before
    assert delta == int(x.numel() > 0)
    RESULT["launches"] += delta


def check(actual, expected):
    expected = expected.to(actual.dtype)
    torch.testing.assert_close(actual, expected, equal_nan=True, rtol=2e-6, atol=0)
    zero = expected == 0
    if actual.is_floating_point():
        assert torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))


try:
    for route in ("cpp", "llvm"):
        for dtype_name in ("float32", "float16", "bfloat16", "int64", "uint64"):
            if route == "llvm" and dtype_name in ("float16", "bfloat16"):
                continue  # LLVM f32 + integer audit; lowp acceptance uses established C++ route.
            cases = (
                ("srem_by_one",)
                if dtype_name == "int64"
                else ("urem_by_one",)
                if dtype_name == "uint64"
                else CASES
            )
            for case in cases:
                directory = ART / "kernels" / route / dtype_name / case
                directory.mkdir(parents=True, exist_ok=True)
                proc = subprocess.run(
                    [
                        str(native_path("fp_probe")),
                        str(directory),
                        route,
                        case,
                        dtype_name,
                    ],
                    capture_output=True,
                    text=True,
                )
                (directory / "compiler.log").write_text(proc.stdout + proc.stderr)
                if proc.returncode:
                    raise RuntimeError(proc.stderr)
                RESULT["compiler_invocations"] += 1
                headers = Path(torch.__file__).parent.parent / "nvidia/cu13/include"
                k = Kernel(
                    directory, compile_ptx=route == "cpp", include_dirs=[headers]
                )
                x, y = inputs(DTYPES[dtype_name], case)
                out = torch.empty_like(x)
                enqueue(k, x, y, out)
                check(out, reference(case, x, y))
                if case.startswith("contract"):
                    source = (
                        directory / ("kernel.cu" if route == "cpp" else "kernel.ll")
                    ).read_text()
                    ptx = (directory / "kernel.ptx").read_text()
                    if route == "cpp":
                        assert "fma(" not in source and "__hfma" not in source
                    else:
                        assert (
                            " contract " not in source and "llvm.fmuladd" not in source
                        )
                    assert not any("fma." in line for line in ptx.splitlines())
                    if dtype_name == "float32":
                        assert torch.count_nonzero(out) == 0
                record = {
                    "route": route,
                    "dtype": dtype_name,
                    "case": case,
                    "directory": str(directory),
                    "ptx_sha256": hashlib.sha256(
                        (directory / "kernel.ptx").read_bytes()
                    ).hexdigest(),
                    "pointers": [x.data_ptr(), y.data_ptr(), out.data_ptr()],
                    "stream": torch.cuda.current_stream().cuda_stream,
                }
                if case == "loaded_floor" and dtype_name == "float32":
                    a = torch.full_like(x, 1e10)
                    b = torch.full_like(y, 3.0)
                    z = torch.empty_like(a)
                    enqueue(k, a, b, z)
                    check(z, reference(case, a, b))
                    pytorch = torch.remainder(a, b)
                    assert z[0].item() == 0 and pytorch[0].item() == 1
                    record["floor_formula_is_not_pytorch_remainder"] = {
                        "x": 1e10,
                        "y": 3.0,
                        "cubecl_formula": z[0].item(),
                        "pytorch_remainder": pytorch[0].item(),
                        "backend_support": "rejected before fusion",
                    }
                if case == "rem_by_one" and dtype_name == "float32":
                    for n in (0, 1, 255, 256, 257):
                        a, b = inputs(torch.float32, case, n)
                        z = torch.empty_like(a)
                        enqueue(k, a, b, z)
                        check(z, reference(case, a, b))
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        enqueue(k, x, y, out)
                        done = torch.cuda.Event()
                        done.record()
                    done.synchronize()
                    check(out, reference(case, x, y))
                    record["non_default_stream"] = stream.cuda_stream
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profiler:
                        enqueue(k, x, y, out)
                        torch.cuda.current_stream().synchronize()
                    profiler.export_chrome_trace(str(directory / "profiler.json"))
                    events = json.loads((directory / "profiler.json").read_text())[
                        "traceEvents"
                    ]
                    kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
                    assert len(kernels) == 1 and kernels[0].startswith("cubecl_fp_"), (
                        kernels
                    )
                    record["profiler_kernels"] = kernels
                RESULT["checks"].append(record)
                print(route, dtype_name, case, "PASS", flush=True)
except Exception as error:
    import traceback

    RESULT["failures"].append(
        {"error": str(error), "traceback": traceback.format_exc()}
    )
    raise
finally:
    evidence.state("audit_after_runtime")
    RESULT["independence"] = evidence.inventory()
    RESULT["native_launches"] = lib.stage0_kernel_launches()
    RESULT["native_launcher_calls"] = lib.stage0_launcher_calls()
    assert RESULT["native_launches"] == RESULT["launches"]
    RESULT["compiler_patch_sha256"] = hashlib.sha256(
        (ROOT / "cubecl-fp.patch").read_bytes()
    ).hexdigest()
    (ART / "results.json").write_text(json.dumps(RESULT, indent=2))
