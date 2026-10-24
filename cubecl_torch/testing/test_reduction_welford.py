"""Welford reductions (OpenSpec welford-reductions).

Hand-authored descriptors: welford_reduce lifts each element to (x, 0, 1);
welford_combine combines (mean, m2, weight) partials. Both run under every
eligible physical plan and are compared with fp64 statistics. Front end:
var/std/var_mean and normalizations compile through Inductor, unsplit and
split, static and dynamic.
"""

import math

import pytest
import torch

from cubecl_torch.compiler.build import compile_descriptor
from cubecl_torch.compiler.plan import catalog, welford_catalog
from cubecl_torch.inductor import legality
from cubecl_torch.runtime.cache import ArtifactCache
from cubecl_torch.testing.fixtures import physical, run_python

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

OUTER = {"op": "scalar", "slot": 0}
EXTENT = {"op": "scalar", "slot": 1}
ROW = {
    "op": "add",
    "args": [{"op": "reduction"}, {"op": "mul", "args": [{"op": "linear"}, EXTENT]}],
}
# Ragged and short extents, and aligned ones for the vectorized plans.
SHAPES = (1, 5, 31, 33, 256, 257, 4096, 4099)


def buffer(access, rows=True, dtype="float32"):
    size, stride = (
        ([OUTER, EXTENT], [EXTENT, {"op": "constant", "value": 1}])
        if rows
        else ([OUTER], [{"op": "constant", "value": 1}])
    )
    return {
        "access": access,
        "dtype": dtype,
        "offset": {"op": "constant", "value": 0},
        "size": size,
        "stride": stride,
    }


def value(op, id, args, **fields):
    return {
        "op": op,
        "id": id,
        "args": args,
        "dtype": "float32",
        "semantic_dtype": "float32",
        **fields,
    }


def reduction(kind, args):
    return value(
        "reduction",
        "r",
        args,
        reduction_type=kind,
        source_dtype="float32",
        accumulator_dtype="float32",
    )


def components():
    return [value("reduction_component", f"c{k}", ["r"], index=k) for k in range(3)]


def store(arg, buffer_index, scope="outer"):
    return {
        "op": "store",
        "args": [arg],
        "buffer": buffer_index,
        "dtype": "float32",
        "semantic_dtype": "float32",
        "index": {"op": "linear"} if scope == "outer" else ROW,
        "iteration_scope": scope,
    }


def descriptor(ops, buffers):
    return {
        "schema": 3,
        "index_dtype": "int64",
        "buffers": buffers,
        "numel": OUTER,
        "outer": OUTER,
        "reduction_extent": EXTENT,
        "scalars": [
            {"dtype": "int64", "id": 0, "positive": True},
            {"dtype": "int64", "id": 1, "positive": True},
        ],
        "ops": ops,
    }


def reduce_descriptor(component, storage="float32"):
    """welford_reduce over x[outer, n], storing one component per row."""
    load = value("load", "x", [], buffer=0, index=ROW)
    upcast = []
    if storage != "float32":
        load.update(dtype=storage, semantic_dtype=storage, id="s")
        upcast = [{**value("cast", "x", ["s"]), "semantic_dtype": storage}]
    return descriptor(
        [
            load,
            *upcast,
            reduction("welford_reduce", ["x"]),
            *components(),
            store(f"c{component}", 1),
        ],
        [buffer("read", dtype=storage), buffer("write", rows=False)],
    )


def combine_descriptor(component):
    """welford_combine over (mean, m2, weight) partials, one component per row."""
    return descriptor(
        [
            value("load", "m", [], buffer=0, index=ROW),
            value("load", "q", [], buffer=1, index=ROW),
            value("load", "w", [], buffer=2, index=ROW),
            reduction("welford_combine", ["m", "q", "w"]),
            *components(),
            store(f"c{component}", 3),
        ],
        [buffer("read"), buffer("read"), buffer("read"), buffer("write", rows=False)],
    )


def row_mean_descriptor():
    """mean + (x - x) stored for every element: exactly the mean each unit read."""
    return descriptor(
        [
            value("load", "x", [], buffer=0, index=ROW),
            reduction("welford_reduce", ["x"]),
            *components(),
            value("sub", "z", ["x", "x"]),
            value("add", "e", ["c0", "z"]),
            store("e", 1, scope="reduction"),
        ],
        [buffer("read"), buffer("write")],
    )


def reduction_plans():
    return [
        p
        for p in (*catalog(), *welford_catalog())
        if "pointwise" not in p.strategy and p.strategy != legality.FINAL9
    ]


def kernel(semantic, plan):
    return physical(
        semantic,
        plan.strategy,
        plan.workgroup,
        plan.vector_width,
        plan.elements_per_unit,
        plan.outputs_per_workgroup,
        plan.plane_width,
        private_accumulators=plan.private_accumulators,
        private_merge=plan.private_merge,
    )


def eligible(plan, itemsize=4):
    """Welford-eligible and within the 16-byte vector load of the storage."""
    return (
        plan.strategy not in legality.SUM_ONLY_STRATEGIES
        and plan.vector_width * itemsize <= 16
    )


@pytest.fixture
def every_plan(monkeypatch):
    """Lift the Welford plan rule (a selection policy, tested by the band and
    first-use tests) so every Welford code path is executed and checked here."""
    monkeypatch.setattr(legality, "welford_plan_reason", lambda *args: None)


def run(k, plan, semantic, scalars, *tensors):
    """Launch, or return the runtime proof that excludes this plan here."""
    why = legality.reason(plan, semantic, scalars, tensors, k.kernel.resources)
    try:
        k(scalars, *tensors)
    except ValueError as error:
        assert why and why in str(error), (plan.strategy, why, error)
        return why
    return None


def statistics(mean, m2, weight):
    """fp64 Welford combine of partials along the last axis."""
    mean, m2, weight = (t.double() for t in (mean, m2, weight))
    total = weight.sum(-1)
    safe = torch.where(total == 0, 1, total)
    combined = (weight * mean).sum(-1) / safe
    spread = m2.sum(-1) + (weight * (mean - combined[:, None]) ** 2).sum(-1)
    return (
        torch.where(total == 0, 0, combined),
        torch.where(total == 0, 0, spread),
        total,
    )


def close(actual, expected, atol):
    torch.testing.assert_close(actual.double().cpu(), expected, rtol=1e-5, atol=atol)


def check(y, expected, component, n, plan):
    """Weight exact; mean to float32 rounding; m2 relative to its size."""
    y = y.cpu().double()
    if component == 2:
        assert torch.equal(y, expected[2]), (plan.strategy, n)
    elif component == 0:
        close(y, expected[0], 1e-6)
    else:
        close(y, expected[1], 1e-5 * max(1.0, float(expected[1].abs().max())))


@pytest.mark.parametrize("component", [0, 1, 2])
@pytest.mark.parametrize("storage", ["float32", "bfloat16"])
def test_welford_reduce_under_every_eligible_plan(every_plan, component, storage):
    """bfloat16 storage also runs the eight-wide vector plans (pair-tree lanes)."""
    semantic = reduce_descriptor(component, storage)
    dtype = getattr(torch, storage)
    plans = [p for p in reduction_plans() if eligible(p, dtype.itemsize)]
    generator = torch.Generator().manual_seed(1009 + component)
    launched = set()
    for plan in plans:
        k = kernel(semantic, plan)
        for n in SHAPES:
            x = (torch.randn(3, n, generator=generator) * 2 + 3).to(dtype)
            expected = statistics(x.float(), torch.zeros(3, n), torch.ones(3, n))
            y = torch.full((3,), float("nan"), device="cuda")
            if run(k, plan, semantic, (3, n), x.cuda(), y):
                continue
            launched.add(plan.identity)
            assert not y.isnan().any(), (plan.strategy, n)
            check(y, expected, component, n, plan)
    assert launched == {p.identity for p in plans}


# Special rows: NaN first/middle/last, constant, +inf, -inf, and a constant
# so large that delta * delta overflows against an empty partial.
SPECIAL_ROWS = ("finite", "nan0", "nan_mid", "nan_last", "constant", "inf", "-inf", "huge")


def special_rows(n, dtype, generator):
    x = torch.randn(len(SPECIAL_ROWS), n, generator=generator) * 2 + 3
    for r, pattern in enumerate(SPECIAL_ROWS):
        if pattern == "nan0":
            x[r, 0] = float("nan")
        elif pattern == "nan_mid":
            x[r, n // 2] = float("nan")
        elif pattern == "nan_last":
            x[r, -1] = float("nan")
        elif pattern == "constant":
            x[r] = 0.375
        elif pattern in ("inf", "-inf"):
            x[r, n // 2] = float(pattern)
        elif pattern == "huge":
            x[r] = 3e19
    return x.to(dtype)


@pytest.mark.parametrize("storage", ["float32", "bfloat16"])
def test_special_rows_under_every_eligible_plan(every_plan, storage):
    """mean-variance-reductions special values on every plan, vector chains
    and rows shorter than the lanes (empty partials) included: NaN propagates,
    constant rows (also huge ones) give m2 exactly 0, infinities are never
    finite."""
    dtype = getattr(torch, storage)
    plans = [p for p in reduction_plans() if eligible(p, dtype.itemsize)]
    kernels = [reduce_descriptor(c, storage) for c in (0, 1)]
    generator = torch.Generator().manual_seed(4001)
    launched = set()
    for plan in plans:
        mean_kernel, m2_kernel = (kernel(k, plan) for k in kernels)
        for n in (5, 256, 257, 4096, 4099):
            x = special_rows(n, dtype, generator)
            rows = len(SPECIAL_ROWS)
            mean = torch.empty(rows, device="cuda")
            m2 = torch.empty(rows, device="cuda")
            if run(mean_kernel, plan, kernels[0], (rows, n), x.cuda(), mean):
                continue
            run(m2_kernel, plan, kernels[1], (rows, n), x.cuda(), m2)
            launched.add(plan.identity)
            for r, pattern in enumerate(SPECIAL_ROWS):
                m, q = float(mean[r]), float(m2[r])
                where = (plan.identity, n, pattern, m, q)
                if pattern.startswith("nan"):
                    assert math.isnan(m) and math.isnan(q), where
                elif pattern in ("constant", "huge"):
                    assert m == float(x[r, 0]) and q == 0.0, where
                elif pattern in ("inf", "-inf"):
                    assert not (math.isfinite(m) and math.isfinite(q)), where
                else:
                    assert math.isfinite(m) and math.isfinite(q) and q > 0, where
    assert launched == {p.identity for p in plans}


@pytest.mark.parametrize("component", [0, 1, 2])
def test_welford_combine_with_empty_partials(every_plan, component):
    """Some partials have weight 0, as masked split tails do; none give NaN."""
    semantic = combine_descriptor(component)
    generator = torch.Generator().manual_seed(2003 + component)
    launched = set()
    for plan in filter(eligible, reduction_plans()):
        k = kernel(semantic, plan)
        for n in SHAPES:
            weight = torch.randint(0, 4, (3, n), generator=generator).float()
            mean = torch.where(weight > 0, torch.randn(3, n, generator=generator), 0)
            m2 = torch.where(weight > 1, torch.rand(3, n, generator=generator), 0)
            expected = statistics(mean, m2, weight)
            y = torch.full((3,), float("nan"), device="cuda")
            tensors = (mean.cuda(), m2.cuda(), weight.cuda(), y)
            if run(k, plan, semantic, (3, n), *tensors):
                continue
            launched.add(plan.identity)
            assert not y.isnan().any(), (plan.strategy, n)
            check(y, expected, component, n, plan)
    assert launched == {p.identity for p in filter(eligible, reduction_plans())}


def test_every_unit_reads_one_mean_per_row(every_plan):
    """A reduction-scope epilogue sees one mean bit pattern per row."""
    semantic = row_mean_descriptor()
    generator = torch.Generator().manual_seed(3001)
    launched = set()
    for plan in filter(eligible, reduction_plans()):
        k = kernel(semantic, plan)
        for n in (1, 33, 256, 257, 4099):
            x = torch.randn(3, n, generator=generator) * 2 + 3
            y = torch.empty(3, n, device="cuda")
            if run(k, plan, semantic, (3, n), x.cuda(), y):
                continue
            launched.add(plan.identity)
            y = y.cpu()
            assert torch.equal(y, y[:, :1].expand(3, n)), (plan.strategy, n)
            close(y[:, 0], x.double().mean(-1), 1e-5)
    assert launched == {p.identity for p in filter(eligible, reduction_plans())}


@pytest.mark.parametrize("kind", ["welford_reduce", "welford_combine"])
def test_sum_only_plans_reject_welford(tmp_path, kind):
    semantic = (
        reduce_descriptor(0) if kind == "welford_reduce" else combine_descriptor(0)
    )
    for i, plan in enumerate(reduction_plans()):
        if plan.strategy not in legality.SUM_ONLY_STRATEGIES:
            continue
        cache = ArtifactCache(tmp_path / str(i))
        with pytest.raises((ValueError, RuntimeError)) as error:
            compile_descriptor(semantic, plan.to_dict(), 0, cache=cache)
        assert (
            f"strategy {plan.strategy} does not support reduction type {kind}"
            in str(error.value)
        )
        assert not list((tmp_path / str(i)).rglob("kernel.ptx"))


@pytest.mark.parametrize(
    "edit, message",
    [
        (
            lambda d: d["ops"][3].update(args=["m"]),
            "reduction type welford_combine takes 3 operand(s), got 1",
        ),
        (
            lambda d: d["ops"][4].update(index=3),
            "reduction type welford_combine has no component 3",
        ),
    ],
)
def test_malformed_welford_descriptors_rejected(tmp_path, edit, message):
    semantic = combine_descriptor(0)
    edit(semantic)
    plan = next(
        p for p in reduction_plans() if p.strategy == "shared_tree_reduction_baseline"
    )
    with pytest.raises((ValueError, RuntimeError)) as error:
        compile_descriptor(semantic, plan.to_dict(), 0, cache=ArtifactCache(tmp_path))
    assert message in str(error.value)
    assert not list(tmp_path.rglob("kernel.ptx"))


def test_welford_reduce_takes_one_operand(tmp_path):
    semantic = reduce_descriptor(0)
    semantic["ops"][1]["args"] = ["x", "x", "x"]
    plan = next(
        p for p in reduction_plans() if p.strategy == "shared_tree_reduction_baseline"
    )
    with pytest.raises((ValueError, RuntimeError)) as error:
        compile_descriptor(semantic, plan.to_dict(), 0, cache=ArtifactCache(tmp_path))
    assert "reduction type welford_reduce takes 1 operand(s), got 3" in str(error.value)


FRONT_END = """
from functools import partial
import torch.nn.functional as F
from cubecl_torch.inductor import tuning
def kernels():
    return [k for k in tuning.CACHE.values() if k.semantic["schema"] == 3]
def types(k):
    return sorted({op["reduction_type"] for op in k.semantic["ops"] if op["op"] == "reduction"})
def compile_run(f, *args, dynamic=False, **backend):
    tuning.CACHE.clear(); torch._dynamo.reset()
    return torch.compile(f, backend=partial(ci.compile_backend, **backend), dynamic=dynamic)(*args)
def close(y, ref, dtype):
    tol = {torch.float32: (1e-5, 1e-5), torch.float16: (2e-3, 2e-3), torch.bfloat16: (1.6e-2, 1e-2)}[dtype]
    ys, refs = (y, ref) if isinstance(y, tuple) else ((y,), (ref,))
    for a, b in zip(ys, refs):
        torch.testing.assert_close(a, b, rtol=tol[0], atol=tol[1])
"""


def test_welford_operations_compile_and_match_eager(tmp_path):
    code = (
        FRONT_END
        + """
out = {}
g = torch.Generator(device="cuda").manual_seed(4001)
for dt in (torch.float32, torch.float16, torch.bfloat16):
    x = torch.randn(300, 4096, device="cuda", generator=g).to(dt) + 2
    image = torch.randn(1, 320, 32, 32, device="cuda", generator=g).to(dt)
    tokens = torch.randn(512, 1280, device="cuda", generator=g).to(dt)
    w, b = torch.randn(320, device="cuda", dtype=dt), torch.randn(320, device="cuda", dtype=dt)
    cases = {
        "var_mean": (lambda x: torch.var_mean(x, -1), x),
        "var": (lambda x: x.var(-1), x),
        "std": (lambda x: x.std(-1, correction=0), x),
        "group_norm": (lambda x: F.group_norm(x, 32, w, b), image),
        "layer_norm": (lambda x: F.layer_norm(x, (1280,)), tokens),
        "fused_epilogue": (lambda x: (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True), x),
    }
    for name, (f, arg) in cases.items():
        y = compile_run(f, arg)
        close(y, f(arg), dt)
        out[f"{name}-{dt}"] = sorted({t for k in kernels() for t in types(k)})
print(json.dumps(out))
"""
    )
    result = run_python(code, tmp_path)
    assert len(result) == 18
    for name, kinds in result.items():
        assert "welford_reduce" in kinds or "welford_combine" in kinds, (name, kinds)


@pytest.mark.parametrize("dynamic", [False, True])
def test_split_welford_plans(tmp_path, dynamic):
    """Masked split (first stage welford_combine over (x, 0, 1)) and unmasked
    split (first stage welford_reduce); the second stage is welford_combine."""
    code = (
        FRONT_END
        + f"""
out = {{}}
x = torch.randn(8, 300000, device="cuda") * 2 + 3
y = compile_run(lambda x: torch.var_mean(x, -1), x, dynamic={dynamic})
close(y, torch.var_mean(x, -1), torch.float32)
out["masked"] = [types(k) for k in kernels()]
image = torch.randn(1, 320, 64, 64, device="cuda")
f = lambda x: torch.ops.aten.native_group_norm(x, None, None, 1, 320, 4096, 32, 1e-5)
y = compile_run(f, image, dynamic={dynamic})
close(y, f(image), torch.float32)
out["unmasked"] = [types(k) for k in kernels()]
print(json.dumps(out))
"""
    )
    result = run_python(code, tmp_path)
    for case, first in (("masked", "welford_combine"), ("unmasked", "welford_reduce")):
        stages = {tuple(t) for t in result[case]}
        if dynamic:
            # Dynamic shapes may keep one stage (as Triton does).
            assert stages & {("welford_reduce",), ("welford_combine",)}, (case, stages)
        else:
            assert {(first,), ("welford_combine",)} <= stages, (case, stages)


def test_first_use_candidates_exclude_sum_only_plans(tmp_path):
    code = (
        FRONT_END
        + """
x = torch.randn(300, 4096, device="cuda")
compile_run(lambda x: torch.var_mean(x, -1), x)
accepted, rejected = set(), {}
for k in kernels():
    for plans, refused in k._legality.values():
        accepted |= {p.strategy for p in plans}
        rejected.update({r["plan"]["strategy"]: r["reason"] for r in refused})
print(json.dumps({"accepted": sorted(accepted), "rejected": rejected,
    "selected": sorted({p.strategy for k in kernels() for p in k.selections.values()})}))
"""
    )
    result = run_python(code, tmp_path)
    assert result["accepted"]
    assert not set(result["accepted"]) & legality.SUM_ONLY_STRATEGIES
    assert set(result["selected"]) <= set(result["accepted"])
    assert result["rejected"]["shared_tree_compensated"].startswith(
        "strategy shared_tree_compensated does not support reduction type welford"
    )


def test_welford_admission_is_float32_only():
    """Inductor upcasts Welford inputs; any other dtype is named and rejected."""
    import operator

    import torch.fx

    from cubecl_torch.inductor import semantic_checks

    graph = torch.fx.Graph()
    value = graph.placeholder("x")
    reduction = graph.call_method(
        "reduction", (None, torch.float32, torch.float32, "welford_reduce", value)
    )
    component = graph.call_function(operator.getitem, (reduction, 1))
    assert semantic_checks.welford_component(component)
    total = graph.call_method(
        "reduction", (None, torch.float32, torch.float32, "sum", value)
    )
    assert not semantic_checks.welford_component(
        graph.call_function(operator.getitem, (total, 0))
    )
    assert not semantic_checks.welford_component(
        graph.call_function(operator.getitem, (reduction, 3))
    )

    from cubecl_torch.inductor.capabilities import UnsupportedCubeCL
    from cubecl_torch.inductor.reductions import CubeCLKernel

    kernel = object.__new__(CubeCLKernel)
    kernel.features = None  # never consulted: admission fails first
    for dtype in (torch.float16, torch.bfloat16):
        for kind in ("welford_reduce", "welford_combine"):
            with pytest.raises(UnsupportedCubeCL) as error:
                kernel.reduction(dtype, dtype, kind, None)
            assert kind in str(error.value) and str(dtype) in str(error.value)


SPECIAL = """
import math
from cubecl_torch.testing import welford_reference as ref
data = torch.load(ref.DATA, weights_only=False)["cases"]
bad = {}
for case in ref.cases():
    if case["kind"] != "special":
        continue
    torch._dynamo.reset()
    inputs = ref.case_inputs(case)
    assert [[ref.digest(a) for a in args] for args in inputs] == data[case["name"]]["input_sha256"]
    f = torch.compile(ref.var_mean, backend=ci.compile_backend)
    for i, (x,) in enumerate(inputs):
        var, mean = (v.float().cpu() for v in f(x.cuda()))
        for r in range(x.shape[0]):
            pattern = ref.PATTERNS[(r + i) % len(ref.PATTERNS)]
            v, m = float(var[r]), float(mean[r])
            if pattern.startswith("nan"):
                ok = math.isnan(v) and math.isnan(m)
            elif pattern == "constant":
                ok = v == 0.0 and m == float(x[r, 0])
            elif "infinity" in pattern:
                ok = not (math.isfinite(v) and math.isfinite(m))
            else:
                ok = math.isfinite(v) and math.isfinite(m) and v > 0
            if not ok:
                bad.setdefault(case["name"], []).append([i, r, pattern, v, m])
print(json.dumps({k: v[:5] for k, v in bad.items()}))
"""


def test_special_values(tmp_path):
    """NaN rows give NaN; constant rows variance exactly 0, split ones too;
    rows holding an infinity are never finite."""
    assert run_python(SPECIAL, tmp_path) == {}


def test_partial_weights_combine_exactly(every_plan):
    """A ragged split's partial weights (37 parts of 1048579) sum exactly."""
    semantic = combine_descriptor(2)
    block = -(-1048579 // 37)
    weight = torch.tensor(
        [[float(min(block, 1048579 - i * block)) for i in range(37)]] * 3
    )
    mean = torch.randn(3, 37)
    m2 = torch.rand(3, 37) * weight
    for plan in filter(eligible, reduction_plans()):
        k = kernel(semantic, plan)
        y = torch.full((3,), float("nan"), device="cuda")
        if run(k, plan, semantic, (3, 37), mean.cuda(), m2.cuda(), weight.cuda(), y):
            continue
        assert torch.equal(y.cpu(), torch.full((3,), 1048579.0)), plan.strategy


def test_rows_shorter_than_a_workgroup(tmp_path):
    """Most units hold only the identity; statistics match eager, no NaN.

    Rows of at most 8 elements take Inductor's two-step sum lowering, which
    needs the ``square`` op (basic float arithmetic, bead 4x9.2), so the
    shortest rows here have 9 elements.
    """
    code = (
        FRONT_END
        + """
out = {}
for n in (9, 17, 33):
    x = torch.randn(64, n, device="cuda") + 1
    var, mean = compile_run(lambda x: torch.var_mean(x, -1, correction=0), x)
    ref = torch.var_mean(x.double(), -1, correction=0)
    out[n] = [bool(var.isnan().any() or mean.isnan().any()),
              float((var - ref[0]).abs().max()), float((mean - ref[1]).abs().max())]
print(json.dumps(out))
"""
    )
    for n, (nan, var_error, mean_error) in run_python(code, tmp_path).items():
        assert not nan, n
        assert var_error < 1e-5 and mean_error < 1e-6, (n, var_error, mean_error)


BAND = (
    "var_mean-float32-37x129-offset",  # two-pass short rows
    "var_mean-float32-300x4096-offset",  # wide vector lanes
    "var_mean-float32-8x300000-offset",  # split, accumulator plans
    "group_norm-float32-1x320x64x64",
    "layer_norm-bfloat16-4096x1280",  # eight-wide pair-tree lanes
    "layer_norm-float16-4096x1280-dynamic",
)


def test_accuracy_within_tritons_band(tmp_path):
    """mean-variance-reductions: no less accurate than Triton (CubeCL's error
    against fp64 larger on significantly more rows, or larger on average, fails;
    per-row sign and paired mean-difference tests); the full sweeps are recorded
    in results/welford-band-3090.json and results/welford-band-3080.json."""
    code = f"""
from cubecl_torch.testing import welford_band
rows = welford_band.measure({set(BAND)!r})
print(json.dumps({{k: r.get("outside") or r.get("error") for k, r in rows.items()}}))
"""
    result = run_python(code, tmp_path)
    assert set(result) == set(BAND)
    assert all(not v for v in result.values()), result


def test_vector_chain_plan_reproduces_triton(tmp_path):
    """One warp per row with eight-wide vector chains (2048 lanes, each chain
    reduced across the plane before the chains merge) gives bfloat16
    LayerNorm's mean and rstd exactly Triton's error against fp64 on every row
    (no row won or lost)."""
    code = """
from cubecl_torch.compiler.plan import PhysicalCandidate
from cubecl_torch.inductor import tuning
from cubecl_torch.testing import welford_band
plan = PhysicalCandidate("independent_planes", (32, 4, 1), 8,
    outputs_per_workgroup=4, plane_width=32, private_accumulators=8)
select = tuning.AutotunedKernel.select
def pinned(self, key, key_data, plans, rejected, *args):
    # The statistics kernels; the normalized output keeps its own selection.
    return plan if plan in plans else select(self, key, key_data, plans, rejected, *args)
tuning.AutotunedKernel.select = pinned
row = welford_band.measure({"layer_norm-bfloat16-4096x1280"})["layer_norm-bfloat16-4096x1280"]
print(json.dumps({"selected": row["cubecl_selected"], "outside": row["outside"],
    "tests": {k: [t["wins"], t["losses"]] for k, t in row["sign_test"].items()}}))
"""
    result = run_python(code, tmp_path)
    assert result["selected"] == ["independent_planes/32x8v8/halving"], result
    assert not result["outside"], result
    assert result["tests"]["mean"] == [0, 0] and result["tests"]["rstd"] == [0, 0], result


def policy_semantic(extent, kinds=("welford_reduce",)):
    ops = [{"op": "reduction", "reduction_type": k} for k in kinds]
    return {"schema": 3, "ops": ops, "reduction_extent": {"op": "constant", "value": extent}}


def admitted_by_policy(semantic, admitted):
    return [p for p in admitted if not legality.welford_plan_reason(p, semantic, (), admitted)]


def test_policy_keeps_scalar_plans_for_unaligned_16_bit_rows():
    """Without vector plans (an unaligned 16-bit row) the policy still admits
    the four-accumulator plane plan, the tuning baseline, at every extent."""
    from cubecl_torch.inductor.tuning import frozen

    scalar = [p for p in reduction_plans() if p.vector_width == 1]
    for extent in (300, 1500, 2048, 8109):
        semantic = policy_semantic(extent)
        kept = admitted_by_policy(semantic, scalar)
        assert frozen(semantic) in kept, extent


def test_policy_admits_vector_chains_only_on_one_pass_rows():
    """Chains were measured on rows Triton reduces in one pass; past 2048
    elements they are not Triton-shaped."""
    plans = reduction_plans()
    chains = {p for p in plans if p.strategy == "independent_planes" and p.vector_width > 1}
    for extent, expected in ((1280, True), (4096, False)):
        kept = set(admitted_by_policy(policy_semantic(extent), plans))
        assert bool(kept & chains) is expected, extent


def test_fused_sum_keeps_its_qualified_plans():
    """A kernel reducing a sum next to a Welford statistic is not steered by
    the Welford policy and gets neither Welford-catalog plans nor the sum's
    diagnostic-only plans."""
    from cubecl_torch.inductor.tuning import frozen

    semantic = policy_semantic(4096, ("sum", "welford_reduce"))
    assert not legality.welford_only(semantic)
    assert frozen(semantic).strategy == "shared_tree_reduction_baseline"
    for plan in reduction_plans():
        assert legality.welford_plan_reason(plan, semantic, (), reduction_plans()) is None
        why = legality.numerical_reason(plan, semantic, ())
        if plan in welford_catalog() or (
            plan.strategy == "plane_shared" and plan.private_accumulators == 4
        ):
            assert why, plan.identity


def test_eight_wide_plans_are_welford_only():
    """Sum, max and min keep their qualified plan sets (no width-8 or
    Welford-catalog candidate): the shared catalog has none, and legality
    refuses them anyway."""
    assert not [
        p for p in catalog() if "pointwise" not in p.strategy and p.vector_width == 8
    ]
    wide = list(welford_catalog())
    assert wide
    for kind in ("sum", "max", "min"):
        semantic = {"schema": 3, "ops": [{"op": "reduction", "reduction_type": kind}]}
        for plan in wide:
            assert legality.numerical_reason(plan, semantic, ()) == (
                "Welford-catalog plans are Welford-only"
            ), kind
