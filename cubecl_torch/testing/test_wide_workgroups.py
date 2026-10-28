"""512-unit reduction workgroups (OpenSpec wide-reduction-workgroups)."""

import ctypes as C

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def geometry_error(block, shared):
    """The launcher's answer for a geometry, without launching: a null kernel
    handle fails right after the geometry check."""
    from cubecl_torch.runtime import native

    pointers = (C.c_uint64 * 2)(0, 0)
    dims = (C.c_uint * 3)(*block)
    error = C.create_string_buffer(256)
    rc = native.launcher().stage5_launch(
        None, pointers, 2, 1, dims, shared, 1, None, error, len(error)
    )
    assert rc != 0
    return error.value.decode()


@pytest.mark.parametrize(
    ("block", "shared"),
    [
        ((512, 1, 1), 64),  # reference512
        ((512, 1, 1), 192),  # Welford: 16 planes x 12 bytes
        ((512, 1, 1), 4096),  # a compensated sum: 8 bytes per unit
        ((512, 1, 1), 6144),  # 12 bytes per unit
        ((256, 1, 1), 3072),
    ],
)
def test_launcher_accepts_512_unit_blocks(block, shared):
    assert geometry_error(block, shared) == "null kernel handle"


@pytest.mark.parametrize(
    ("block", "shared"),
    [
        ((1024, 1, 1), 0),
        ((512, 2, 1), 0),
        ((256, 2, 1), 0),
        ((512, 1, 1), 6145),
        ((256, 1, 1), 3073),
    ],
)
def test_launcher_refuses_other_geometries(block, shared):
    assert geometry_error(block, shared) == "unsupported launch geometry"


def wide_plans():
    from cubecl_torch.compiler.plan import catalog
    from cubecl_torch.inductor import legality

    return [p for p in catalog() if legality.wide(p)]


def test_wide_plans_are_catalogued_and_offered_from_2048():
    from cubecl_torch.inductor import legality

    plans = wide_plans()
    assert {p.vector_width for p in plans} == {1, 2, 4}
    semantic = {"schema": 3, "reduction_extent": {"op": "constant", "value": 0}}
    for extent, offered in ((1024, False), (2047, False), (2048, True), (8192, True)):
        semantic["reduction_extent"]["value"] = extent
        for plan in plans:
            assert (legality.wide_plan_reason(plan, semantic, ()) is None) is offered
    from cubecl_torch.compiler.plan import catalog

    reference = next(p for p in catalog() if p.strategy == "plane_shared_reference512")
    assert not legality.wide(reference)


CANDIDATES = """
from cubecl_torch.inductor import legality, tuning
functions = {"softmax": lambda x: torch.softmax(x, -1), "amax": lambda x: x.amax(-1)}
out = {}
for name, fn in functions.items():
    for extent in (1024, 2047, 2048, 8192):
        torch._dynamo.reset()
        tuning.CACHE.clear()
        torch.compile(fn, backend=ci.compile_backend)(torch.randn(64, extent, device="cuda"))
        ((facts, (plans, rejected)),) = [
            item for k in tuning.CACHE.values() if k.semantic["schema"] == 3
            for item in k._legality.items()
        ]
        out[f"{name}/{extent}"] = {
            "facts": repr(facts),
            "wide": sorted(p.identity for p in plans if legality.wide(p)),
            "reasons": sorted({r["reason"] for r in rejected if r["plan"]["workgroup"][0] == 512}),
        }
print(json.dumps(out))
"""


def test_wide_candidates_follow_the_extent(tmp_path):
    """Spec 'Long-row candidates': offered from 2048 elements, rejected with the
    rule's reason below; the signature separates 2047 from 2048."""
    from cubecl_torch.testing.fixtures import run_python

    result = run_python(CANDIDATES, tmp_path)
    for name in ("softmax", "amax"):
        for extent in (1024, 2047, 2048, 8192):
            row = result[f"{name}/{extent}"]
            if extent >= 2048:
                assert row["wide"], (name, extent)
            else:
                assert not row["wide"], (name, extent)
                assert any("Triton's 16 warps" in r for r in row["reasons"]), row
        assert result[f"{name}/2047"]["facts"] != result[f"{name}/2048"]["facts"]


PINNED = """
from cubecl_torch.inductor import legality, tuning
legality.numerical_reason = lambda *args: None
select = tuning.AutotunedKernel.select
pinned = {}
def pin(self, key, key_data, plans, rejected, *args):
    if self.semantic["schema"] != 3:
        return select(self, key, key_data, plans, rejected, *args)
    assert pinned["plan"] in plans, pinned["plan"].identity
    return pinned["plan"]
tuning.AutotunedKernel.select = pin
from cubecl_torch.compiler.plan import catalog
functions = {
    "sum": lambda x: x.sum(-1),
    "amax": lambda x: x.amax(-1),
    "amin": lambda x: x.amin(-1),
    "softmax": lambda x: torch.softmax(x, -1),
}
generator = torch.Generator().manual_seed(4351)
failures, runs = [], 0
for plan in [p for p in catalog() if legality.wide(p)]:
    pinned["plan"] = plan
    for dtype in (torch.float32, torch.bfloat16):
        if plan.vector_width * dtype.itemsize > 16:
            continue
        for name, fn in functions.items():
            # Several aligned rows; one ragged row (vector tails need an aligned base).
            for rows, extent in ((5, 2048), (5, 8192), (5, 2056), (1, 2049), (1, 4099)):
                x = torch.randn(rows, extent, generator=generator) * 3 + 1
                # A large row tail: a lost vector tail or chunk moves every
                # result far beyond rounding (and holds the maximum). For
                # softmax, e^8 keeps the rest of the row's mass comparable.
                x[:, -3:] = 8.0 if name == "softmax" else 64.0
                x = x.to(dtype).cuda()
                torch._dynamo.reset()
                tuning.CACHE.clear()
                got = torch.compile(fn, backend=ci.compile_backend)(x).double()
                expected = fn(x.double()).to(dtype).double()
                if name in ("amax", "amin"):
                    ok = torch.equal(got, expected)
                else:
                    # Two float32 ulps of chain error, or two bfloat16 ulps of the
                    # rounded result; no absolute slack beyond the smallest values.
                    rtol = 2.4e-7 * extent if dtype == torch.float32 else 1.6e-2
                    atol = 1e-30 if name == "sum" else 1e-12
                    ok = bool(((got - expected).abs() <= atol + rtol * expected.abs()).all())
                runs += 1
                if not ok:
                    failures.append([plan.identity, str(dtype), name, rows, extent])
print(json.dumps({"failures": failures, "runs": runs}))
"""


def test_every_wide_plan_matches_eager(tmp_path):
    """Spec 'Every reduction type at 512 units': sum, max, min and softmax under
    each 512-unit plan pinned, at aligned and ragged extents with vector tails,
    float32 and bfloat16 (sums pinned past the long-row rule)."""
    from cubecl_torch.testing.fixtures import run_python

    result = run_python(PINNED, tmp_path)
    assert result["runs"] == 3 * 2 * 4 * 5, result
    assert result["failures"] == [], result


WELFORD = """
from cubecl_torch.compiler.plan import PhysicalCandidate
from cubecl_torch.inductor import legality, tuning
wide = PhysicalCandidate(
    "plane_shared", (512, 1, 1), 4, plane_width=32, private_merge=MERGE
)
candidates = tuning.candidates
used = []
def forced(semantic, scalars, tensors, target, **kw):
    if semantic["schema"] != 3:
        return candidates(semantic, scalars, tensors, target, **kw)
    # Past the selection policy (one merge order per row length), not the
    # launch contract.
    assert not legality.reason(wide, semantic, scalars, tensors, target)
    used.append(wide.identity)
    return [wide], []
tuning.candidates = forced
fn = lambda x, w, b: torch.ops.aten.native_layer_norm(x, x.shape[-1:], w, b, 1e-5)
g = torch.Generator().manual_seed(4357)
x = (torch.randn(2048, 8192, generator=g) * 2 + 3).cuda()
w, b = torch.randn(8192, generator=g).cuda(), torch.randn(8192, generator=g).cuda()
out, mean, rstd = torch.compile(fn, backend=ci.compile_backend)(x, w, b)
ref = fn(x.double(), w.double(), b.double())
err = lambda a, r: float(((a.double() - r).abs() / r.abs().clamp_min(1e-3)).max())
print(json.dumps({"used": len(used), "out": float((out.double() - ref[0]).abs().max()),
                  "mean": err(mean, ref[1]), "rstd": err(rstd, ref[2])}))
"""


@pytest.mark.parametrize("merge", ["halving", "lanes"])
def test_float32_layer_norm_at_512_units(tmp_path, merge):
    """Spec 'Welford at 512 units': LayerNorm statistics and the normalized
    output launch under the pinned 512 x v4 plan, folding vector lanes or with
    one accumulator per lane (design decision 7), and match fp64 to float32
    rounding (the band test in welford_band.py is the accuracy contract)."""
    from cubecl_torch.testing.fixtures import run_python

    result = run_python(WELFORD.replace("MERGE", repr(merge)), tmp_path)
    assert result["used"] >= 1, result
    assert result["mean"] < 1e-5 and result["rstd"] < 1e-5, result
    assert result["out"] < 1e-4, result


@pytest.mark.parametrize("component", [0, 1, 2])
@pytest.mark.parametrize(("storage", "width"), [("float32", 4), ("bfloat16", 8)])
def test_lane_accumulators_route_ragged_tails(component, storage, width):
    """Design decision 7: under the lanes variant, a ragged row's tail elements
    go to their own lanes' accumulators. One row (the only layout that admits a
    ragged vector tail), with large tail values so a lost or repeated element
    moves every component, against the fp64 combine."""
    from cubecl_torch.compiler.plan import PhysicalCandidate
    from cubecl_torch.testing.test_reduction_welford import (
        check,
        kernel,
        reduce_descriptor,
        run,
        statistics,
    )

    plan = PhysicalCandidate(
        "plane_shared", (512, 1, 1), width, plane_width=32, private_merge="lanes"
    )
    semantic = reduce_descriptor(component, storage)
    dtype = getattr(torch, storage)
    k = kernel(semantic, plan)
    generator = torch.Generator().manual_seed(4363 + component)
    for n in (2053, 4099, 8197):
        x = torch.randn(1, n, generator=generator) * 2 + 3
        x[0, -3:] = 1000.0
        x = x.to(dtype)
        expected = statistics(x.float(), torch.zeros(1, n), torch.ones(1, n))
        y = torch.full((1,), float("nan"), device="cuda")
        assert run(k, plan, semantic, (1, n), x.cuda(), y) is None, n
        assert not y.isnan().any(), n
        check(y, expected, component, n, plan)
