"""Kernels with several buffers and outputs (OpenSpec multi-output-kernels)."""

import copy
import re

import pytest
import torch

from cubecl_torch.testing.fixtures import captured

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def chained_sum(inputs):
    """A pointwise descriptor loading ``inputs`` float32 buffers and storing
    their left-to-right sum, built from the captured ``add-float32``."""
    base = captured("add-float32")
    read, write = base["buffers"]
    load = next(op for op in base["ops"] if op["op"] == "load")
    store = next(op for op in base["ops"] if op["op"] == "store")
    ops = [{**copy.deepcopy(load), "id": f"x{i}", "buffer": i} for i in range(inputs)]
    total = "x0"
    for i in range(1, inputs):
        ops.append(
            {
                "args": [total, f"x{i}"],
                "dtype": "float32",
                "id": f"s{i}",
                "op": "add",
                "semantic_dtype": "float32",
            }
        )
        total = f"s{i}"
    ops.append({**copy.deepcopy(store), "args": [total], "buffer": inputs})
    return {
        **base,
        "buffers": [copy.deepcopy(read) for _ in range(inputs)] + [write],
        "ops": ops,
    }


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("multi-output-cache")


def test_norm_training_kernels_fit_the_buffer_limit(cache_dir):
    """LayerNorm and GroupNorm (affine, then SiLU) backward kernels read 6 to 9
    buffers (Triton's, measured 2026-10-27); none is rejected for its count."""
    from cubecl_torch.testing.fixtures import run_python

    result = run_python(
        """
import torch.nn.functional as F
import torch._functorch.config
torch._functorch.config.force_non_lazy_backward_lowering = True
reasons = []
def backend(gm, inputs):
    try:
        return ci.compile_backend(gm, inputs, rejections="all")
    except Exception as e:
        reasons.extend(getattr(e, "reasons", None) or [str(e)])
        return gm.forward
def layer_norm(x, w, b):
    return F.layer_norm(x, x.shape[-1:], w, b, 1e-5).square().sum()
def group_norm(x, w, b):
    return F.silu(F.group_norm(x, 8, w, b, 1e-5)).square().sum()
for fn, shape, channels in ((layer_norm, (64, 96), 96), (group_norm, (2, 32, 8, 8), 32)):
    torch._dynamo.reset()
    args = [torch.randn(shape, device="cuda", requires_grad=True)]
    args += [torch.randn(channels, device="cuda", requires_grad=True) for _ in range(2)]
    torch.compile(fn, backend=backend)(*args).backward()
print(json.dumps(sorted(set(reasons))))
""",
        cache_dir,
    )
    assert not [r for r in result if "inputs" in r or "buffers per kernel" in r], result


FUSION_LIMIT = """
from cubecl_torch.runtime import trace
def fn(*xs):
    a, b = xs[:{a}], xs[{a}:]
    row = sum(a[1:], a[0]).sum(-1, keepdim=True)
    return row + sum(b[1:], b[0])
xs = [torch.randn(33, 70, device="cuda") for _ in range({a} + {b})]
torch._dynamo.reset()
with trace.scoped():
    out = torch.compile(fn, backend=ci.compile_backend)(*xs)
    reasons = [e["reason"] for e in trace.EVENTS
               if e.get("boundary") == "fusion" and e.get("reason")]
    kernels = sum(e.get("event") == "kernel" for e in trace.EVENTS)
print(json.dumps({{"close": torch.allclose(out, fn(*xs), rtol=1e-5, atol=1e-4),
                  "reasons": sorted(set(reasons)), "kernels": kernels}}))
"""


def test_fusion_over_the_buffer_limit_is_rejected_and_still_compiles(cache_dir):
    """A row reduction over ``a`` inputs fused with a pointwise epilogue over
    ``b`` more: 9 + 7 inputs and one output need 17 buffers; 8 + 7 fit in 16,
    a reduction kernel with 15 inputs checked against eager."""
    from cubecl_torch.testing.fixtures import run_python

    over = run_python(FUSION_LIMIT.format(a=9, b=7), cache_dir)
    assert over["close"] and over["kernels"] == 2, over
    assert "more than 16 buffers per kernel unsupported" in over["reasons"], over
    fits = run_python(FUSION_LIMIT.format(a=8, b=7), cache_dir)
    assert fits["close"] and fits["kernels"] == 1, fits
    assert not [r for r in fits["reasons"] if "buffers" in r], fits


def test_sixteen_buffer_pointwise_kernel_equals_eager():
    from cubecl_torch.inductor import tuning

    kernel = tuning.compile_kernel(chained_sum(15), 0)
    n = 4099
    xs = [torch.randn(n, device="cuda") for _ in range(15)]
    out = torch.empty_like(xs[0])
    kernel((n,), *xs, out)
    expected = xs[0]
    for x in xs[1:]:
        expected = expected + x
    assert torch.equal(out, expected)


def mean_and_centered():
    """Welford over x[outer, n]: the mean per row and x - mean per element."""
    from cubecl_torch.testing.test_reduction_welford import (
        ROW,
        buffer,
        components,
        descriptor,
        reduction,
        store,
        value,
    )

    return descriptor(
        [
            value("load", "x", [], buffer=0, index=ROW),
            reduction("welford_reduce", ["x"]),
            *components(),
            store("c0", 1),
            value("sub", "d", ["x", "c0"]),
            store("d", 2, scope="reduction"),
        ],
        [buffer("read"), buffer("write", rows=False), buffer("write")],
    )


def test_per_row_and_per_element_outputs_under_every_plan(monkeypatch):
    """Each row's centered output uses exactly the row's stored mean."""
    from cubecl_torch.inductor import legality
    from cubecl_torch.testing.test_reduction_welford import (
        eligible,
        kernel,
        reduction_plans,
        run,
    )

    # Every Welford code path, as test_reduction_welford's every_plan fixture.
    monkeypatch.setattr(legality, "welford_plan_reason", lambda *args: None)
    semantic = mean_and_centered()
    generator = torch.Generator().manual_seed(4211)
    launched = set()
    for plan in filter(eligible, reduction_plans()):
        k = kernel(semantic, plan)
        # Ragged extents, and aligned ones for the vector plans.
        for n in (1, 33, 256, 257, 4096, 4099):
            x = torch.randn(3, n, generator=generator) * 2 + 3
            mean = torch.empty(3, device="cuda")
            centered = torch.empty(3, n, device="cuda")
            if run(k, plan, semantic, (3, n), x.cuda(), mean, centered):
                continue
            launched.add(plan.identity)
            mean, centered = mean.cpu(), centered.cpu()
            torch.testing.assert_close(
                mean.double(), x.double().mean(-1), rtol=1e-5, atol=1e-6
            )
            assert torch.equal(centered, x - mean[:, None]), (plan.strategy, n)
    assert launched == {p.identity for p in filter(eligible, reduction_plans())}


def test_extent_of_a_non_final_output_is_guarded():
    from cubecl_torch.runtime.launch import LaunchContractError
    from cubecl_torch.testing.fixtures import physical

    k = physical(mean_and_centered(), "shared_tree_reduction_baseline")
    x = torch.randn(3, 70, device="cuda")
    centered = torch.empty(3, 70, device="cuda")
    with pytest.raises(LaunchContractError, match="reduction output logical domain"):
        k((3, 70), x, torch.empty(4, device="cuda"), centered)
    with pytest.raises(LaunchContractError, match="disjoint"):
        k((3, 70), x, centered.view(-1)[:3], centered)
    mean = torch.empty(3, device="cuda")
    k((3, 70), x, mean, centered)
    assert torch.equal(centered, x - mean[:, None])


def three_outputs():
    """Five inputs p, g, m, v, d and three outputs, Adam-shaped. No product
    feeds an addition, so no contraction can separate it from eager."""
    base = chained_sum(5)
    load = [op for op in base["ops"] if op["op"] == "load"]
    store = next(op for op in base["ops"] if op["op"] == "store")
    read, write = base["buffers"][0], base["buffers"][-1]

    def op(name, id, *args, **fields):
        return {
            "args": list(args),
            "dtype": "float32",
            "id": id,
            "op": name,
            "semantic_dtype": "float32",
            **fields,
        }

    return {
        **base,
        "buffers": [copy.deepcopy(read) for _ in range(5)]
        + [copy.deepcopy(write) for _ in range(3)],
        "ops": [
            *load,
            op("constant", "k", value=0.25),
            op("sub", "gm", "x1", "x2"),
            op("mul", "o1", "gm", "k"),
            op("add", "pv", "x0", "x3"),
            op("sub", "o2", "pv", "x4"),
            op("mul", "o3", "x1", "x1"),
            *(
                {**copy.deepcopy(store), "args": [o], "buffer": 5 + i}
                for i, o in enumerate(("o1", "o2", "o3"))
            ),
        ],
    }


@pytest.mark.parametrize(
    "plan",
    [
        ("scalar_pointwise_baseline",),
        ("scalar_pointwise", (128, 1, 1), 1, 4),
        ("vector_pointwise", (128, 1, 1), 2),
        ("vector_pointwise", (128, 1, 1), 4),
    ],
)
@pytest.mark.parametrize("n", [4096, 4099])
def test_three_output_pointwise_kernel_equals_eager(plan, n):
    """Full vectors, and a partial tail (per-lane stores) at 4099."""
    from cubecl_torch.testing.fixtures import physical

    k = physical(three_outputs(), *plan)
    p, g, m, v, d = (torch.randn(n, device="cuda") for _ in range(5))
    outs = [torch.empty(n, device="cuda") for _ in range(3)]
    k((n,), p, g, m, v, d, *outs)
    assert torch.equal(outs[0], (g - m) * 0.25)
    assert torch.equal(outs[1], (p + v) - d)
    assert torch.equal(outs[2], g * g)


def test_pointwise_outputs_are_guarded_and_disjoint():
    from cubecl_torch.runtime.launch import LaunchContractError
    from cubecl_torch.testing.fixtures import physical

    k = physical(three_outputs(), "scalar_pointwise_baseline")
    n = 300
    ins = [torch.randn(n, device="cuda") for _ in range(5)]
    a, b, c = (torch.empty(n, device="cuda") for _ in range(3))
    with pytest.raises(LaunchContractError, match="pointwise output extent"):
        k((n,), *ins, torch.empty(n + 1, device="cuda"), b, c)
    with pytest.raises(LaunchContractError, match="disjoint"):
        k((n,), *ins, a, a, c)


def test_output_guard_buffer_round_trips_through_the_contract():
    from cubecl_torch.runtime.artifact import KernelArtifact
    from cubecl_torch.testing.fixtures import physical

    for semantic, buffers in (
        (three_outputs(), [5, 6, None]),
        (chained_sum(2), [None]),
    ):
        artifact = physical(semantic, "scalar_pointwise_baseline").kernel.artifact
        guards = [
            g for g in artifact.contract()["guards"] if g["kind"] == "output_numel"
        ]
        assert [g.get("buffer") for g in guards] == buffers
        # The last output keeps the single-output form: no buffer field.
        assert "buffer" not in guards[-1]
        loaded = KernelArtifact.from_contract(artifact.contract(), artifact.ptx)
        assert loaded.guards == artifact.guards


def masked_region(names, mask="m", other=0.0, index=None):
    """A masked load as CubeCLKernel.masked records it, body ids ``names``."""
    load, cast = names
    return {
        "op": "masked",
        "args": [mask],
        "dtype": "float32",
        "semantic_dtype": "float32",
        "body": [
            {
                "id": load,
                "op": "load",
                "args": [],
                "buffer_name": "x",
                "index": index or {"op": "linear"},
            },
            {"id": cast, "op": "cast", "args": [load]},
        ],
        "result": cast,
        "other": other,
    }


def test_identical_masked_regions_share_a_key():
    import json

    from cubecl_torch.inductor.codegen import canonical_region

    def key(record):
        return json.dumps(canonical_region(record), sort_keys=True)

    first = key(masked_region(("tmp3", "tmp4")))
    assert first == key(masked_region(("tmp13", "tmp14")))
    assert first != key(masked_region(("tmp13", "tmp14"), other=1.0))
    assert first != key(masked_region(("tmp13", "tmp14"), mask="m2"))
    assert first != key(
        masked_region(("tmp13", "tmp14"), index={"op": "constant", "value": 0})
    )

    # An inner op reading a value from outside the body keeps that name, so
    # regions reading different outer values stay apart.
    def reading(outer, names):
        record = masked_region(names)
        record["body"][1]["args"] = [names[0], outer]
        return key(record)

    assert reading("tmp1", ("tmp3", "tmp4")) == reading("tmp1", ("tmp13", "tmp14"))
    assert reading("tmp1", ("tmp3", "tmp4")) != reading("tmp2", ("tmp13", "tmp14"))

    # A nested region is renamed with its parent; a result naming an outer
    # value is kept.
    def nested(names, result=None):
        inner = masked_region(names[1:])
        inner["id"] = names[0]
        outer = masked_region(("tmp90", "tmp91"))
        outer["body"].append(inner)
        outer["result"] = result or names[0]
        return outer

    assert key(nested(("tmp5", "tmp6", "tmp7"))) == key(
        nested(("tmp15", "tmp16", "tmp17"))
    )
    assert key(nested(("tmp5", "tmp6", "tmp7"), "tmp1")) != key(
        nested(("tmp5", "tmp6", "tmp7"), "tmp2")
    )


def statistics_and_two_elementwise(elementwise=2):
    """Mean and m2 per row; x - mean and (x - mean)^2 per element."""
    from cubecl_torch.testing.test_reduction_welford import (
        ROW,
        buffer,
        components,
        descriptor,
        reduction,
        store,
        value,
    )

    ops = [
        value("load", "x", [], buffer=0, index=ROW),
        reduction("welford_reduce", ["x"]),
        *components(),
        store("c0", 1),
        store("c1", 2),
        value("sub", "d", ["x", "c0"]),
        store("d", 3, scope="reduction"),
    ]
    buffers = [buffer("read"), buffer("write", rows=False)]
    buffers += [buffer("write", rows=False), buffer("write")]
    if elementwise == 2:
        ops += [value("mul", "s", ["d", "d"]), store("s", 4, scope="reduction")]
        buffers.append(buffer("write"))
    return descriptor(ops, buffers)


def test_grouped_stores_under_every_plan_and_one_element_pass(monkeypatch):
    from cubecl_torch.inductor import legality
    from cubecl_torch.testing.fixtures import physical
    from cubecl_torch.testing.test_reduction_welford import (
        eligible,
        kernel,
        reduction_plans,
        run,
    )

    monkeypatch.setattr(legality, "welford_plan_reason", lambda *args: None)
    semantic = statistics_and_two_elementwise()
    generator = torch.Generator().manual_seed(4213)
    launched = set()
    for plan in filter(eligible, reduction_plans()):
        k = kernel(semantic, plan)
        for n in (1, 33, 256, 257, 4096, 4099):
            x = torch.randn(3, n, generator=generator) * 2 + 3
            outs = [torch.empty(3, device="cuda") for _ in range(2)]
            outs += [torch.empty(3, n, device="cuda") for _ in range(2)]
            if run(k, plan, semantic, (3, n), x.cuda(), *outs):
                continue
            launched.add(plan.identity)
            mean, m2, d, s = (t.cpu() for t in outs)
            x64 = x.double()
            torch.testing.assert_close(
                mean.double(), x64.mean(-1), rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                m2.double(),
                ((x64 - x64.mean(-1, keepdim=True)) ** 2).sum(-1),
                rtol=1e-4,
                atol=1e-4,
            )
            assert torch.equal(d, x - mean[:, None]), (plan.strategy, n)
            assert torch.equal(s, d * d), (plan.strategy, n)
    assert launched == {p.identity for p in filter(eligible, reduction_plans())}

    # The second per-element store shares the first one's loop.
    def loops(semantic):
        k = physical(semantic, "shared_tree_reduction_baseline")
        source = (k.directory / "kernel.cu").read_text()
        return len(re.findall(r"\bfor\s*\(", source))

    # The reduction loop and one per-element pass.
    assert loops(semantic) == loops(statistics_and_two_elementwise(elementwise=1)) == 2
