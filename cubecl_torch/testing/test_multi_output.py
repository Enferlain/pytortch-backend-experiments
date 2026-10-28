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


def test_per_element_pass_streams_only_the_reduced_rows():
    """The per-element pass re-reads x, which the reduction pass already read,
    as a streaming load (Triton's evict_first second loop, bead 4z8.33); the
    reduction pass loads it normally."""
    from cubecl_torch.testing.fixtures import physical

    semantic = statistics_and_two_elementwise()
    for plan, width in (("shared_tree_reduction_baseline", 1), ("plane_shared", 4)):
        k = (
            physical(semantic, plan)
            if width == 1
            else physical(semantic, plan, (256, 1, 1), width, plane_width=32)
        )
        source = (k.directory / "kernel.cu").read_text()
        ptx = (k.directory / "kernel.ptx").read_text()
        assert source.count("L1::evict_first") == 1, plan
        # After the reduction loop: inside the per-element pass.
        last_loop = [m.start() for m in re.finditer(r"\bfor\s*\(", source)][-1]
        assert source.index("L1::evict_first") > last_loop, plan
        assert "ld.global.nc.L1::evict_first" in ptx, plan


def welford_statistics():
    """Welford over x[outer, n] storing mean, m2 and weight per row."""
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
            *(store(f"c{k}", 1 + k) for k in range(3)),
        ],
        [buffer("read")] + [buffer("write", rows=False) for _ in range(3)],
    )


def test_each_output_is_compared_by_its_own_kind():
    from cubecl_torch.inductor import legality, tuning

    semantic = welford_statistics()
    mean, m2, weight = (tuning.output_semantic(semantic, b) for b in (1, 2, 3))
    # The weight is an element count: compared bit for bit.
    assert legality.welford_weight_output(weight)
    # Mean and m2 are compared under the Welford guard.
    for projected in (mean, m2):
        assert not legality.welford_weight_output(projected)
        assert legality.welford_reduction(projected)
    # A one-output kernel is its own projection.
    single = mean_and_centered()
    single["buffers"].pop(1)
    single["ops"] = [op for op in single["ops"] if op.get("buffer") != 1]
    for op in single["ops"]:
        if op.get("buffer") == 2:
            op["buffer"] = 1
    assert tuning.output_semantic(single, 1) is single


def test_candidate_with_a_corrupted_second_output_is_excluded(monkeypatch, tmp_path):
    from cubecl_torch.inductor import tuning
    from cubecl_torch.runtime import cache

    monkeypatch.setenv(tuning.STRICT_ENV, "0")
    # A private cache, so no earlier selection for this kernel is reused.
    monkeypatch.setattr(cache, "_configured", tmp_path)
    from cubecl_torch.compiler.plan import PhysicalCandidate

    semantic = welford_statistics()
    # A 1024-element float32 Welford row keeps several candidates; its
    # fallback is the widest vector plane plan (wide-reduction-workgroups
    # leaves one plan on longer rows, which selection does not time).
    fallback = PhysicalCandidate("plane_shared", (256, 1, 1), 4, plane_width=32)
    monkeypatch.setattr(tuning, "welford_fallback", lambda semantic, plans: fallback)
    compile = tuning.AutotunedKernel.compile

    def corrupting(self, plan):
        compiled = compile(self, plan)
        # The baseline (frozen) plan is the comparison reference: never corrupted.
        reference = (fallback, tuning.frozen(semantic))
        if plan not in reference and not getattr(compiled, "corrupting", False):
            launch = compiled.kernel

            class Corrupt:
                def __getattr__(self, name):
                    return getattr(launch, name)

                def __call__(self, scalars, *tensors):
                    out = launch(scalars, *tensors)
                    tensors[2].fill_(float("nan"))  # m2
                    return out

            compiled.kernel, compiled.corrupting = Corrupt(), True
        return compiled

    monkeypatch.setattr(tuning.AutotunedKernel, "compile", corrupting)
    tuning.CACHE.clear()
    k = tuning.compile_kernel(semantic, 0)
    x = torch.randn(3, 1024, device="cuda")
    outs = [torch.empty(3, device="cuda") for _ in range(3)]
    from cubecl_torch.runtime import trace

    with trace.scoped():
        start = len(trace.EVENTS)
        k((3, 1024), x, *outs)
        excluded = [
            e
            for e in list(trace.EVENTS)[start:]
            if e.get("event") == "candidate_excluded"
        ]
    assert excluded and all(e["output"] == 2 for e in excluded), excluded
    assert list(k.selections.values()) == [fallback]
    torch.testing.assert_close(outs[0].double(), x.double().mean(-1))
    assert torch.equal(outs[2], torch.full_like(outs[2], 1024))
    tuning.CACHE.clear()


STRUCTURE = """
import torch.nn.functional as F
from cubecl_torch.runtime import trace
g = torch.Generator(device="cuda").manual_seed(7)
def r(*shape, dtype=torch.float32):
    return torch.randn(*shape, device="cuda", generator=g, dtype=dtype)
def adam(param, grad, exp_avg, exp_avg_sq, denom):
    exp_avg = exp_avg.lerp(grad, 0.1)
    exp_avg_sq = exp_avg_sq.mul(0.999).addcmul(grad, grad, value=0.001)
    return param.addcdiv(exp_avg, denom, value=-1e-3), exp_avg, exp_avg_sq
def frexp(x):
    m, e = torch.frexp(x)
    return m, e.to(torch.int64)
cases = {
    "adam": (adam, [r(1 << 16) for _ in range(4)] + [r(1 << 16).abs() + 1e-3], "exact"),
    "frexp": (frexp, [r(1 << 16) * 1e3], "exact"),
    "sum_and_max": (lambda x: (x.sum(-1), x.amax(-1)), [r(300, 4096)], "close"),
    "var_mean": (lambda x: torch.var_mean(x, -1, correction=0), [r(300, 4096)], "close"),
    "var_mean_split": (lambda x: torch.var_mean(x, -1, correction=0), [r(8, 300000)], "close"),
    "layer_norm": (
        lambda x: torch.ops.aten.native_layer_norm(x, x.shape[-1:], None, None, 1e-5),
        [r(512, 1280, dtype=torch.bfloat16)],
        "close",
    ),
    "group_norm": (
        lambda x: torch.ops.aten.native_group_norm(x, None, None, 1, 320, 4096, 32, 1e-5),
        [r(1, 320, 64, 64)],
        "close",
    ),
    "dropout": (lambda x: (F.dropout(x, 0.5, True), x * 2), [r(1 << 12)], "shape"),
}
report = {}
for name, (fn, args, check) in cases.items():
    torch._dynamo.reset()
    with trace.scoped():
        start = len(trace.EVENTS)
        out = torch.compile(fn, backend=ci.compile_backend)(*args)
        events = [e for e in list(trace.EVENTS)[start:] if e.get("event") == "kernel"]
    ref = fn(*args)
    if check == "exact":
        ok = all(torch.equal(a, b) for a, b in zip(out, ref))
    elif check == "close":
        ok = all(
            torch.allclose(a.float(), b.float(), rtol=2e-2, atol=2e-2) for a, b in zip(out, ref)
        )
    else:
        ok = all(a.shape == b.shape for a, b in zip(out, ref))
    report[name] = {
        "ok": ok,
        "outputs": [
            sum(b["access"] == "write" for b in e["descriptor"]["buffers"]) for e in events
        ],
        "reductions": [
            [o["reduction_type"] for o in e["descriptor"]["ops"] if o["op"] == "reduction"]
            for e in events
        ],
        "rand": [any(o["op"] == "rand" for o in e["descriptor"]["ops"]) for e in events],
    }
print(json.dumps(report))
"""


def test_fused_regions_match_tritons_kernel_structure(cache_dir):
    """Kernel count and outputs per kernel as Inductor+Triton (research probe,
    bead 4z8.21, 2026-10-27), one reduction per kernel, results as eager."""
    from cubecl_torch.testing.fixtures import run_python

    result = run_python(STRUCTURE, cache_dir)
    assert all(case["ok"] for case in result.values()), result
    outputs = {name: case["outputs"] for name, case in result.items()}
    assert outputs["adam"] == [3]
    assert outputs["frexp"] == [2]
    assert outputs["sum_and_max"] == [2]
    assert outputs["var_mean"] == [2]
    assert outputs["var_mean_split"] == [3, 2]
    assert outputs["layer_norm"] == [3]
    assert outputs["group_norm"] == [3, 3, 1]
    # The random-number node keeps its own kernel, with one output.
    dropout = result["dropout"]
    assert any(dropout["rand"]), dropout
    assert all(
        n == 1
        for n, rand in zip(dropout["outputs"], dropout["rand"], strict=True)
        if rand
    )
    assert result["var_mean_split"]["reductions"] == [["welford_combine"]] * 2
    assert result["layer_norm"]["reductions"] == [["welford_reduce"]]


PRECISION = """
from cubecl_torch.runtime import trace
def rounded(fn, x):
    torch._dynamo.reset()
    with trace.scoped():
        start = len(trace.EVENTS)
        torch.compile(fn, backend=ci.compile_backend)(x)
        ops = [
            op
            for e in list(trace.EVENTS)[start:]
            if e.get("event") == "kernel"
            for op in e["descriptor"]["ops"]
        ]
    # Values rounded to bfloat16 in the kernel: real casts, and the stores.
    return {
        "casts": sum(op["op"] == "cast" and op["dtype"] == "bfloat16" for op in ops),
        "stores": sum(op["op"] == "store" and op["dtype"] == "bfloat16" for op in ops),
    }
def emulated(fn, x):
    from torch._inductor import config
    with config.patch(emulate_precision_casts=True):
        return rounded(fn, x)
x = torch.randn(37, 129, device="cuda").bfloat16()
print(json.dumps({
    "written": rounded(lambda x: (x.float() * 2).to(torch.bfloat16) + 1, x),
    "normalization": rounded(
        lambda x: (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True), x
    ),
    "emulated": emulated(
        lambda x: (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True), x
    ),
}))
"""


def test_only_written_casts_round_inside_fused_kernels(cache_dir):
    """A cast the user wrote rounds before its consumer; std's decomposition
    cast and the row mean stay float32 until the store, as in Triton."""
    from cubecl_torch.testing.fixtures import run_python

    result = run_python(PRECISION, cache_dir)
    # The written cast, then the store's own rounding.
    assert result["written"] == {"casts": 2, "stores": 1}, result
    # Only the store's rounding.
    assert result["normalization"] == {"casts": 1, "stores": 1}, result
    # Triton's switch brings eager's roundings back.
    assert result["emulated"]["casts"] > 1, result
