"""Ported live FP-policy contracts (see contract_inventory.md).

Sources: fp_hardening/verify.py (C++ route), fp_hardening/verify_inductor.py,
stage0/tests/fp_rewrites.rs (run on the package's copy of the native crate).
"""

import json
import os
import subprocess

import pytest
import torch

from cubecl_torch.compiler import nvrtc
from cubecl_torch.runtime import native
from cubecl_torch.testing.fixtures import run_python

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CASES = (
    "add_positive_zero_left", "add_positive_zero_right", "add_negative_zero_left",
    "add_negative_zero_right", "sub_self", "mul_positive_zero_left",
    "mul_positive_zero_right", "mul_negative_zero_left", "mul_negative_zero_right",
    "div_positive_zero", "div_negative_zero", "rem_by_one", "rem_positive_zero",
    "rem_negative_zero", "floor_positive_zero", "floor_negative_zero", "loaded_add",
    "loaded_mul", "loaded_div", "loaded_rem", "loaded_floor", "contract_add",
    "contract_sub", "contract_rsub",
)  # fmt: skip


def inputs(dtype, case, n=257):
    if dtype == torch.int64:
        values = [-7, -1, 0, 1, 7]
        data = (values * ((n + 4) // 5))[:n]
        return tuple(torch.tensor(data, device="cuda", dtype=dtype) for _ in range(2))
    inf, nan = float("inf"), float("nan")
    xs = [
        -inf,
        -2.5,
        -1.5,
        -1.0,
        -0.0,
        0.0,
        1.0,
        1.5,
        2.5,
        inf,
        nan,
        2**-126,
        2**-149,
        1e10,
        -1e10,
    ]
    ys = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, inf, -inf, nan, 0.5, -0.5, 0.125, 3.0, -3.0]
    pairs = [(x, y) for x in xs for y in ys]
    pairs = (pairs * ((n + len(pairs) - 1) // len(pairs)))[:n]
    x = torch.tensor([p[0] for p in pairs], device="cuda", dtype=dtype)
    y = torch.tensor([p[1] for p in pairs], device="cuda", dtype=dtype)
    if case.startswith("contract"):
        epsilon = {torch.float32: 2**-23, torch.float16: 2**-10}.get(dtype, 2**-7)
        x.fill_(1 + epsilon)
        y.fill_(1 - epsilon)
    return x, y


def reference(case, x, y):
    if case == "srem_by_one":
        return torch.zeros_like(x)
    x, y = x.float(), y.float()
    zero = -0.0 if "negative_zero" in case else 0.0
    if case.startswith("add_"):
        return zero + x if case.endswith("left") else x + zero
    if case.startswith("mul_"):
        return zero * x if case.endswith("left") else x * zero
    if case.startswith("div_"):
        return zero / y
    if case.startswith("rem_") and case != "rem_by_one":
        return torch.fmod(torch.full_like(x, zero), y)
    if case.startswith("floor_"):
        z = torch.full_like(x, zero)
        return z - torch.floor(z / y) * y
    return {
        "sub_self": lambda: x - x,
        "rem_by_one": lambda: torch.fmod(x, 1),
        "loaded_add": lambda: x + y,
        "loaded_mul": lambda: x * y,
        "loaded_div": lambda: x / y,
        "loaded_rem": lambda: torch.fmod(x, y),
        "loaded_floor": lambda: x - torch.floor(x / y) * y,
        "contract_add": lambda: x * y - 1,
        "contract_sub": lambda: x * y - 1,
        "contract_rsub": lambda: 1 - x * y,
    }[case]()


def audit_kernel(tmp_path, case, dtype, *, numerics=None):
    """fp_probe output (not a descriptor) through NVRTC and a hand-written contract.

    By default Stage 5's numerics (no contraction, correctly rounded division),
    so the PTX shows only CubeCL's own rewrites.
    """
    from cubecl_torch.runtime import rules
    from cubecl_torch.runtime.artifact import (
        Guard,
        KernelArtifact,
        LaunchShape,
        ScalarParam,
        TensorParam,
        audited_provenance,
    )  # fmt: skip
    from cubecl_torch.runtime.launch import LoadedKernel

    numerics = numerics or nvrtc.STAGE5
    out = (
        tmp_path / f"{case}-{dtype}-{numerics.fp_contract:d}{numerics.exact_division:d}"
    )
    out.mkdir()
    subprocess.run(
        [str(native.NATIVE / "target/release/fp_probe"), str(out), "cpp", case, dtype],
        check=True,
        capture_output=True,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    source = (out / "kernel.cu").read_text()
    major, minor = torch.cuda.get_device_capability()
    ptx = nvrtc.compile_ptx(source, out, (major, minor), numerics=numerics)
    n = rules.scalar(0)
    param = {"size": (n,), "stride": (rules.constant(1),), "offset": rules.constant(0)}
    artifact = KernelArtifact(
        ptx=ptx,
        entry=manifest["entrypoint"],
        tensors=(
            TensorParam(dtype, **param),
            TensorParam(dtype, **param),
            TensorParam(dtype, **param, access="write"),
        ),
        scalars=(ScalarParam(),),
        extent=n,
        info_scalars=(),
        metadata_words=manifest["metadata"]["bytes"] // 8,
        launch=LaunchShape(rules.ceil_div(n, 256), (256, 1, 1)),
        guards=(Guard("output_numel", (n,), "audit output extent mismatch"),),
        provenance={**audited_provenance(), "producer": "fp_probe"},
    )
    return LoadedKernel(artifact), source, ptx.decode()


def check(actual, expected):
    expected = expected.to(actual.dtype)
    torch.testing.assert_close(actual, expected, equal_nan=True, rtol=2e-6, atol=0)
    if actual.is_floating_point():
        zero = expected == 0
        assert torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))


@CUDA
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_fp_audit_cases_cpp_route(tmp_path, dtype):
    td = getattr(torch, dtype)
    for case in CASES:
        kernel, source, ptx = audit_kernel(tmp_path, case, dtype)
        x, y = inputs(td, case)
        out = torch.empty_like(x)
        kernel((x.numel(),), x, y, out)
        check(out, reference(case, x, y))
        if case.startswith("contract"):
            assert "fma(" not in source and "__hfma" not in source
            assert not any("fma." in line for line in ptx.splitlines())
            if dtype == "float32":
                assert torch.count_nonzero(out) == 0
        if case == "loaded_floor" and dtype == "float32":
            a, b = torch.full_like(x, 1e10), torch.full_like(y, 3.0)
            z = torch.empty_like(a)
            kernel((a.numel(),), a, b, z)
            check(z, reference(case, a, b))
            assert z[0].item() == 0 and torch.remainder(a, b)[0].item() == 1
        if case == "rem_by_one" and dtype == "float32":
            for n in (0, 1, 255, 256, 257):
                a, b = inputs(td, case, n)
                z = torch.empty_like(a)
                kernel((n,), a, b, z)
                check(z, reference(case, a, b))


@CUDA
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_contraction_is_nvrtc_policy_only(tmp_path, dtype):
    """CubeCL never fuses; NVRTC --fmad does, rounding a*b+c once."""
    td = getattr(torch, dtype)
    # NVVM leaves the raw probe's ``1.0 - x*y`` separate; Inductor's form of it
    # fuses (test_contraction_policy_through_backend).
    for case in ("contract_add", "contract_sub"):
        kernel, source, ptx = audit_kernel(
            tmp_path, case, dtype, numerics=nvrtc.Numerics()
        )
        assert "fma(" not in source and "__hfma" not in source
        assert any("fma." in line for line in ptx.splitlines())
        x, y = inputs(td, case)
        out = torch.empty_like(x)
        kernel((x.numel(),), x, y, out)
        # (1+e)(1-e)-1 = -e*e is exact in FP64 and representable in each dtype.
        fused = (x.double() * y.double() - 1).to(td)
        assert torch.count_nonzero(fused) == fused.numel()
        assert torch.equal(out, fused)


@CUDA
def test_integer_remainder_by_one(tmp_path):
    kernel, _, _ = audit_kernel(tmp_path, "srem_by_one", "int64")
    x, y = inputs(torch.int64, "srem_by_one")
    out = torch.empty_like(x)
    kernel((x.numel(),), x, y, out)
    assert torch.equal(out, torch.zeros_like(x))


@CUDA
def test_contraction_policy_through_backend(tmp_path):
    code = """
import functools, itertools
from cubecl_torch.inductor import kernels
from cubecl_torch.runtime import trace
from cubecl_torch.testing.fixtures import native_launches
from torch._inductor import config
def fma_form(x, y):
    return x * y - 1
def rsub_form(x, y):
    return 1 - x * y
rows = []
# Default numerics contract each form into one rounding (+-e*e); eager precision
# emulation keeps eager's separate roundings (0), as Triton's fp fusion switch.
for emulate, fn, (dtype, epsilon) in itertools.product(
    (False, True), (fma_form, rsub_form), [(torch.float32, 2**-23), (torch.float16, 2**-10), (torch.bfloat16, 2**-7)]
):
    config.emulate_precision_casts = emulate
    torch._dynamo.reset()
    x = torch.full((257,), 1 + epsilon, dtype=dtype, device="cuda", requires_grad=True)
    y = torch.full_like(x, 1 - epsilon, requires_grad=True)
    # The original gate used Stage 2, which never tuned: compare one launch.
    backend = functools.partial(ci.compile_backend, tuning="baseline")
    compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)
    before = native_launches()
    out = compiled(x, y)
    launched = native_launches() - before
    upstream = torch.randn_like(out)
    actual = torch.autograd.grad(out, (x, y), upstream)
    reference = torch.autograd.grad(fn(x, y), (x, y), upstream)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        compiled(x, y)
        torch.cuda.synchronize()
    names = [e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    expected = fn(x, y) if emulate else fn(x.double(), y.double()).to(dtype)
    rows.append({
        "launched": launched,
        "exact": torch.equal(out, expected)
        and int(torch.count_nonzero(out)) == (0 if emulate else out.numel()),
        "backward": all(torch.equal(a, b) for a, b in zip(actual, reference)),
        "kernels": names,
    })
config.emulate_precision_casts = False
rejected = {}
negative_x = torch.randn(257, device="cuda")
for name, op in [("fmod", torch.fmod), ("remainder", torch.remainder)]:
    before = kernels.COMPILER_INVOCATIONS
    with trace.scoped():
        start = len(trace.EVENTS)
        try:
            torch.compile(op, backend=ci.compile_backend, fullgraph=True, dynamic=True)(negative_x, torch.ones_like(negative_x))
            rejected[name] = "accepted"
        except Exception as error:
            pre_fusion = any(e.get("boundary") == "pre_fusion" and e.get("decision") == "reject" for e in list(trace.EVENTS)[start:])
            rejected[name] = "unsupported" in str(error) and pre_fusion and kernels.COMPILER_INVOCATIONS == before
print(json.dumps({"rows": rows, "rejected": rejected}))
"""
    result = run_python(code, tmp_path)
    for row in result["rows"]:
        assert row["launched"] == 1 and row["exact"] and row["backward"]
        assert len(row["kernels"]) == 1 and row["kernels"][0].startswith("cubecl5_")
    assert result["rejected"] == {"fmod": True, "remainder": True}


@pytest.mark.skipif(
    os.environ.get("CUBECL_TORCH_SLOW") != "1",
    reason="builds cubecl-llvm; set CUBECL_TORCH_SLOW=1",
)
def test_fp_rewrites_rust_suite():
    proc = subprocess.run(
        [
            "cargo", "test", "--manifest-path", str(native.NATIVE / "Cargo.toml"), "--locked",
            "--features", "llvm,hip-probe", "--test", "fp_rewrites",
        ],
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@CUDA
def test_low_precision_rounding_policy(tmp_path):
    """Fused FP16/BF16 math rounds once at storage; graph casts still round."""
    code = """
import functools, itertools
from torch._inductor import config
backend = functools.partial(ci.compile_backend, tuning="baseline")
# No multiply feeds an add, so FP32 eager evaluates each op exactly as fused.
chain = lambda x, y: torch.exp(torch.sin(x) * y) / (y + 2)
casts = {
    dtype: lambda x, dtype=dtype: (x.float() * 3).to(dtype).float().sin() + x.float().exp()
    for dtype in (torch.float16, torch.bfloat16)
}
torch.manual_seed(31)
result = {}
for dtype in (torch.float16, torch.bfloat16):
    x = torch.randn(4099, device="cuda", dtype=dtype)
    y = torch.randn(4099, device="cuda", dtype=dtype)
    for emulate in (False, True):
        config.emulate_precision_casts = emulate
        torch._dynamo.reset()
        out = torch.compile(chain, backend=backend, fullgraph=True)(x, y)
        key = f"{dtype}-{'emulate' if emulate else 'default'}"
        result[key + "-single_rounding"] = torch.equal(out, chain(x.float(), y.float()).to(dtype))
        result[key + "-eager"] = torch.equal(out, chain(x, y))
for (dtype, cast), emulate in itertools.product(casts.items(), (False, True)):
    xc = torch.randn(4099, device="cuda", dtype=dtype)
    config.emulate_precision_casts = emulate
    torch._dynamo.reset()
    out = torch.compile(cast, backend=backend, fullgraph=True)(xc)
    result[f"cast-{dtype}-{'emulate' if emulate else 'default'}"] = torch.equal(out, cast(xc))
print(json.dumps(result))
"""
    result = run_python(code, tmp_path)
    for dtype in ("torch.float16", "torch.bfloat16"):
        assert result[f"{dtype}-default-single_rounding"]
        assert not result[f"{dtype}-default-eager"]
        assert result[f"{dtype}-emulate-eager"]
    assert all(v for k, v in result.items() if k.startswith("cast-")), result


@CUDA
def test_division_rounding_follows_inductor_config(tmp_path):
    """div.full (Triton's x / y) by default; div.rn under division_rounding."""
    code = """
import functools
from torch._inductor import config
from cubecl_torch.inductor import tuning
backend = functools.partial(ci.compile_backend, tuning="baseline")
torch.manual_seed(37)
a, b = torch.randn(1 << 16, device="cuda"), torch.randn(1 << 16, device="cuda")
result = {}
for exact in (False, True):
    config.eager_numerics.division_rounding = exact
    torch._dynamo.reset()
    tuning.CACHE.clear()
    out = torch.compile(lambda a, b: a / b, backend=backend, fullgraph=True)(a, b)
    (kernel,) = [k for k in tuning.CACHE.values()]
    ptx = kernel.last.kernel.artifact.ptx.decode()
    result[str(exact)] = {
        "div": sorted({w for w in ptx.split() if w.startswith("div.") and w.endswith("f32")}),
        "differs_from_eager": int((out != a / b).sum()),
        "max_rel": float(((out - a / b).abs() / (a / b).abs()).max()),
    }
print(json.dumps(result))
"""
    result = run_python(code, tmp_path)
    assert result["False"]["div"] == ["div.full.f32"]
    assert result["False"]["differs_from_eager"] > 0
    assert result["False"]["max_rel"] <= 2 * 2**-23  # div.full: at most 2 ulp
    assert result["True"]["div"] == ["div.rn.f32"]
    assert result["True"]["differs_from_eager"] == 0
