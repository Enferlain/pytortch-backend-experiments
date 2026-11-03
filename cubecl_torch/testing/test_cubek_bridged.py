"""Bridged ``cubek_reduce`` kernels behind ``AutotunedKernel.compile``
(torch-runtime-bridge task 4.1): the call interface of a compiled plan, the
frozen plan's launch contract, and the benchmarker's CUPTI measurement."""

import pytest
import torch

from cubecl_torch.compiler.plan import bridge_catalog
from cubecl_torch.inductor import tuning
from cubecl_torch.inductor.timing import KernelBenchmarker
from cubecl_torch.runtime.launch import LaunchContractError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def row_sum(rows, cols, dtype, out):
    """``x.sum(-1)`` as Inductor describes it for static ``[rows, cols]``."""
    constant = lambda value: {"op": "constant", "value": value}  # noqa: E731
    ops = [
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
                    {"op": "reduction"},
                    {"op": "mul", "args": [constant(cols), {"op": "linear"}]},
                ],
            },
        }
    ]
    value = "x"
    if dtype != "float32":
        ops.append(
            {"id": "f", "op": "cast", "args": ["x"], "dtype": "float32"}
            | {"semantic_dtype": dtype}
        )
        value = "f"
    ops.append(
        {
            "id": "s",
            "op": "reduction",
            "args": [value],
            "dtype": "float32",
            "semantic_dtype": dtype,
            "reduction_type": "sum",
            "source_dtype": dtype,
            "accumulator_dtype": "float32",
        }
    )
    value = "s"
    if out != "float32":
        ops.append(
            {
                "id": "o",
                "op": "cast",
                "args": ["s"],
                "dtype": out,
                "semantic_dtype": out,
            }
        )
        value = "o"
    ops.append(
        {
            "op": "store",
            "args": [value],
            "dtype": out,
            "semantic_dtype": out,
            "index": {"op": "linear"},
            "iteration_scope": "outer",
            "buffer": 1,
        }
    )
    return {
        "schema": 3,
        "ops": ops,
        "buffers": [
            {
                "access": "read",
                "dtype": dtype,
                "size": [constant(rows), constant(cols)],
                "stride": [constant(cols), constant(1)],
                "offset": constant(0),
            },
            {
                "access": "write",
                "dtype": out,
                "size": [constant(rows)],
                "stride": [constant(1)],
                "offset": constant(0),
            },
        ],
        "outer": constant(rows),
        "reduction_extent": constant(cols),
        "numel": constant(rows),
        "index_dtype": "int64",
        "scalars": [],
    }


def assert_row_sums(out, x):
    """Within a summation-order-independent bound of the exact row sums (the
    numerical band, task 4.2, judges accuracy against Triton), plus the
    output's own rounding."""
    exact = x.double().sum(1)
    bound = 64 * torch.finfo(torch.float32).eps * x.double().abs().sum(1)
    bound += exact.abs() * torch.finfo(out.dtype).eps
    assert ((out.double() - exact).abs() <= bound).all()


SHAPES = [(2048, 8192), (300, 4096), (7, 2049), (64, 31)]
DTYPES = [
    ("float32", "float32"),
    ("bfloat16", "bfloat16"),
    ("float16", "float16"),
    ("bfloat16", "float32"),
]


@pytest.mark.parametrize(("rows", "cols"), SHAPES, ids=[f"{r}x{c}" for r, c in SHAPES])
@pytest.mark.parametrize(("dtype", "out"), DTYPES, ids=["-".join(d) for d in DTYPES])
def test_bridged_plans_compute_row_sums(rows, cols, dtype, out):
    kernel = tuning.compile_kernel(
        row_sum(rows, cols, dtype, out), torch.device("cuda")
    )
    x = torch.randn(rows, cols, device="cuda", dtype=getattr(torch, dtype))
    for plan in bridge_catalog():
        result = torch.full(
            (rows,), float("nan"), device="cuda", dtype=getattr(torch, out)
        )
        compiled = kernel.compile(plan)
        assert compiled.plan == plan and compiled.directory is None
        assert compiled((), x, result) is result
        assert_row_sums(result, x)


def test_launch_contract_is_the_frozen_plans():
    """A tensor that does not match the descriptor's layout is refused as it
    would be by the frozen plan, before anything is launched."""
    kernel = tuning.compile_kernel(
        row_sum(64, 1024, "float32", "float32"), torch.device("cuda")
    )
    compiled = kernel.compile(bridge_catalog()[0])
    x = torch.randn(64, 1024, device="cuda")
    with pytest.raises(LaunchContractError, match="layout"):
        compiled((), x.t().contiguous().t(), torch.empty(64, device="cuda"))
    with pytest.raises(LaunchContractError, match="dtype"):
        compiled((), x.half(), torch.empty(64, device="cuda"))
    with pytest.raises(LaunchContractError, match="alias"):
        compiled((), x, x.view(-1)[:64])


def test_benchmarker_times_bridged_kernels():
    """The tuner's measurement (exact CUPTI durations of the named kernel)
    works on a bridged kernel: one CUDA kernel per launch, named for the
    shape's blueprint."""
    kernel = tuning.compile_kernel(
        row_sum(2048, 8192, "bfloat16", "bfloat16"), torch.device("cuda")
    )
    x = torch.randn(2048, 8192, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(2048, device="cuda", dtype=torch.bfloat16)
    compiled = kernel.compile(bridge_catalog()[0])
    with compiled.kernel.prepared((), x, out):
        assert compiled.kernel.artifact.entry.startswith("reduce_kernel_")
        measurement = KernelBenchmarker().benchmark_kernel_gpu(compiled.kernel)
    assert measurement["kernel_count"] == 384
    assert all(v > 0 for v in measurement["kernel_ms"])
    assert_row_sums(out, x)


def test_direct_calls_check_the_runtime_facts():
    """Outside the tuner, a launch is refused before the device when the
    addresses are off CubeK's vector alignment."""
    kernel = tuning.compile_kernel(
        row_sum(64, 1024, "float32", "float32"), torch.device("cuda")
    )
    compiled = kernel.compile(bridge_catalog()[0])
    x = torch.randn(64 * 1024 + 1, device="cuda")[1:].view(64, 1024)
    with pytest.raises(LaunchContractError, match="aligned"):
        compiled((), x, torch.empty(64, device="cuda"))


REFUSAL = """
import glob, os
from cubecl_torch.inductor import legality
from cubecl_torch.runtime import bridge
numerical = legality.numerical_reason
legality.numerical_reason = lambda plan, *a: None if plan.strategy == "cubek_reduce" else numerical(plan, *a)
unchecked = bridge.sum_rows_unchecked
def refusing(*args, **kw):
    if args[4] == "unit":
        raise bridge.BridgeError("injected refusal")
    return unchecked(*args, **kw)
bridge.sum_rows_unchecked = refusing
tuning.INJECTED_TIMINGS = lambda identity, r: [10.0] * 5
c = tuning.compile_kernel(captured("row_sum-float32"), 0)
x = torch.randn(64, 4096, device="cuda")
y = torch.empty(64, device="cuda")
c((4096, 64), x, y)
root = os.environ["CUBECL_TORCH_CACHE_DIR"]
rows = [r for p in glob.glob(root + "/tuning/*/selection.json") for r in json.load(open(p))["candidates"]]
print(json.dumps({
    "close": bool(torch.allclose(y, x.sum(-1), rtol=1e-5, atol=1e-4)),
    "refused": sorted({r["plan"]["routine"] for r in rows if r.get("excluded")}),
    "reasons": sorted({r["reason"] for r in rows if r.get("excluded")}),
    "timed": sorted({r["plan"]["routine"] for r in rows if r["plan"]["strategy"] == "cubek_reduce" and not r.get("excluded")}),
}))
"""


def test_cubek_refusal_excludes_the_candidate(tmp_path):
    """A candidate CubeK refuses at launch is excluded with CubeK's reason;
    the selection completes with the others."""
    from cubecl_torch.testing.fixtures import run_python
    from cubecl_torch.testing.test_contracts_tuning import TUNING_PRELUDE

    result = run_python(
        REFUSAL,
        tmp_path,
        prelude=TUNING_PRELUDE,
        env={"CUBECL_TORCH_STRICT_CANDIDATES": "0"},
    )
    assert result["close"]
    assert result["refused"] == ["unit"]
    assert result["reasons"] == ["CubeK refused: injected refusal"]
    assert result["timed"] == ["cube", "plane"]


def test_inductor_blueprint_is_inductors_config():
    """The copied rule gives what Inductor chose for each recorded row sum on
    this device's capability (``testing.inductor_configs``): its warps as
    planes, ``R0_BLOCK / threads`` elements per unit."""
    import json
    from pathlib import Path

    from cubecl_torch.inductor.bridged import inductor_blueprint

    data = json.loads(
        (Path(__file__).parent / "data" / "inductor_row_configs.json").read_text()
    )
    capability = list(torch.cuda.get_device_capability())
    recorded = [
        c for c in data["configs"] if c["capability"] == capability and "split" not in c
    ]
    assert recorded
    for config in recorded:
        rows, cols = config["shape"]
        warps = config["num_warps"]
        assert inductor_blueprint(rows, cols, torch.device("cuda")) == (
            warps,
            config["R0_BLOCK"] // (32 * warps),
        )
