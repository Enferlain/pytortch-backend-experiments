"""Ported live selection/cache contracts from the frozen gates (contract_inventory.md).

Sources: stage5/proof_controls.py, stage5/cache_probe.py, stage5/controls.py,
stage5/launch_controls.py, stage5/components.py.
"""

import pytest
import torch

from cubecl_torch.testing.fixtures import run_python

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TUNING_PRELUDE = """
import json, sys
from cubecl_torch.runtime import guard, trace
guard.install()
import torch
from cubecl_torch.inductor import kernels, tuning
from cubecl_torch.testing.fixtures import captured
"""


def test_deterministic_cold_cache_and_failed_selection_retry(tmp_path):
    code = """
cold = captured("add-float32")
next(o for o in cold["ops"] if o["op"] == "constant")["value"] = 1.125
x = torch.randn(1001, device="cuda")
y = torch.empty_like(x)
c = tuning.compile_kernel(cold, 0)
state = torch.cuda.get_rng_state().clone()
torch.use_deterministic_algorithms(True)
with trace.scoped():
    start = len(trace.EVENTS)
    try:
        c((1001,), x, y)
    finally:
        torch.use_deterministic_algorithms(False)
    events = [e["event"] for e in list(trace.EVENTS)[start:]]
exact = torch.equal(y, x + 1.125)
rng = torch.equal(state, torch.cuda.get_rng_state())

retry_semantic = captured("add-float32")
next(o for o in retry_semantic["ops"] if o["op"] == "constant")["value"] = 1.0625
retry = tuning.compile_kernel(retry_semantic, 0)
original = retry.select
attempts = []
def fail_once(key, key_data, *args):
    assert isinstance(key_data, dict)
    attempts.append(key)
    if len(attempts) == 1:
        raise RuntimeError("injected profiler failure")
    return original(key, key_data, *args)
retry.select = fail_once
torch.use_deterministic_algorithms(True)
try:
    try:
        retry((1001,), x, y)
        observed = None
    except RuntimeError as error:
        observed = str(error)
    retry((1001,), x, y)
finally:
    torch.use_deterministic_algorithms(False)
print(json.dumps({
    "deterministic_baseline": "deterministic_baseline" in events,
    "tuned": "tune" in events,
    "exact": exact, "rng": rng,
    "observed": observed,
    "same_key": len(attempts) == 2 and attempts[0] == attempts[1],
    "retry_exact": torch.equal(y, x + 1.0625),
}))
"""
    result = run_python(code, tmp_path, prelude=TUNING_PRELUDE)
    assert result == {
        "deterministic_baseline": True,
        "tuned": False,
        "exact": True,
        "rng": True,
        "observed": "injected profiler failure",
        "same_key": True,
        "retry_exact": True,
    }


CACHE_PROBE = """
c = tuning.compile_kernel(captured("add-float32"), 0)
x = torch.randn(1001, device="cuda")
y = torch.empty_like(x)
with trace.scoped():
    start = len(trace.EVENTS)
    c((1001,), x, y)
    events = list(trace.EVENTS)[start:]
guard.state("after_cache_probe")
print(json.dumps({
    "exact": torch.allclose(y, x + 1),
    "compiles": kernels.COMPILER_INVOCATIONS,
    "tuning_cache_hit": any(e["event"] == "tuning_cache_hit" for e in events),
    "artifact_hits": [e["cache_hit"] for e in events if e["event"] == "artifact"],
}))
"""


def test_fresh_process_reload_reuses_artifacts_and_selection(tmp_path):
    first = run_python(CACHE_PROBE, tmp_path, prelude=TUNING_PRELUDE)
    assert first["exact"] and first["compiles"] > 0
    second = run_python(CACHE_PROBE, tmp_path, prelude=TUNING_PRELUDE)
    assert second["exact"] and second["compiles"] == 0
    assert second["tuning_cache_hit"] and all(second["artifact_hits"])


def test_backend_dynamic_buckets_streams_dropout_and_rank0(tmp_path):
    code = """
from functools import partial
from cubecl_torch.inductor import kernels, tuning
compiled = torch.compile(lambda x: x + 1, backend=ci.compile_backend, fullgraph=True, dynamic=True)
counts = []
for n in (257, 259, 1001, 4099, 4101):
    before = kernels.COMPILER_INVOCATIONS
    x = torch.randn(n, device="cuda")
    assert torch.allclose(compiled(x), x + 1)
    counts.append(kernels.COMPILER_INVOCATIONS - before)

stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    torch.cuda._sleep(3000000)
    x = torch.randn(1048583, device="cuda")
    expected = x + 1
    out = compiled(x)
    del x
    trash = [torch.empty(1048583, device="cuda") for _ in range(8)]
    done = torch.cuda.Event()
    done.record()
torch.cuda.current_stream().wait_event(done)
stream_ok = torch.allclose(out, expected)

f = lambda x: torch.nn.functional.dropout(torch.sin(x) + 1, 0.25, True)
a = torch.compile(f, backend=partial(ci.compile_backend, tuning="baseline"), fullgraph=True, dynamic=True)
b = torch.compile(f, backend=ci.compile_backend, fullgraph=True, dynamic=True)
x = torch.randn(257, device="cuda")
state = torch.cuda.get_rng_state().clone()
ya = a(x)
after_a = torch.cuda.get_rng_state().clone()
torch.cuda.set_rng_state(state)
yb = b(x)
after_b = torch.cuda.get_rng_state().clone()

rank0 = torch.compile(lambda z: z + 1, backend=ci.compile_backend, fullgraph=True, dynamic=True)
z = torch.randn((), device="cuda")
rank0_ok = torch.allclose(rank0(z), z + 1)
hits = sum(c.selection_hits for c in tuning.CACHE.values())
rank0(z)
print(json.dumps({
    "counts": counts,
    "stream": stream_ok,
    "dropout_exact": torch.equal(ya, yb),
    "rng_advance_equal": torch.equal(after_a, after_b),
    "rank0": rank0_ok,
    "selection_hit": sum(c.selection_hits for c in tuning.CACHE.values()) > hits,
}))
"""
    result = run_python(code, tmp_path)
    counts = result.pop("counts")
    assert counts[1] == counts[2] == counts[4] == 0, counts
    assert all(result.values()), result


def test_every_catalog_plan_tails_and_rejections():
    """stage5/components.py: each enabled plan, ten tails, three dtypes."""
    from cubecl_torch.compiler.column_plan import STRATEGY as COLUMN
    from cubecl_torch.compiler.column_plan import semantic_reason as column_reason
    from cubecl_torch.compiler.final_plan import STRATEGY as FINAL9
    from cubecl_torch.compiler.final_plan import semantic_reason
    from cubecl_torch.compiler.plan import catalog
    from cubecl_torch.inductor.legality import reason
    from cubecl_torch.testing.fixtures import captured, physical

    torch.manual_seed(6142)
    checked = 0
    for dtype in ("float32", "float16", "bfloat16"):
        td = getattr(torch, dtype)
        tolerance = {
            "float32": (2e-5, 2e-5),
            "float16": (0.002, 0.001),
            "bfloat16": (0.016, 0.008),
        }[dtype]
        for family in ("add", "row_sum"):
            semantic = captured(f"{family}-{dtype}")
            for plan in catalog():
                if (family == "add") != ("pointwise" in plan.strategy):
                    continue
                if dtype == "float32" and plan.vector_width > 4:
                    continue
                if plan.strategy in (FINAL9, COLUMN):
                    assert (
                        semantic_reason if plan.strategy == FINAL9 else column_reason
                    )(semantic)
                    continue
                k = physical(
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
                if plan.strategy == "independent_planes":
                    assert k.kernel.attributes["static_shared_bytes"] == 0
                for n in (0, 1, 31, 32, 33, 255, 256, 257, 4099, 65539):
                    outer = (
                        (1 if plan.vector_width > 1 and n % plan.vector_width else 3)
                        if family == "row_sum"
                        else 1
                    )
                    x = (
                        (torch.arange(outer * n, device="cuda") % 17 - 8).to(td) / 8
                    ).reshape((outer, n) if family == "row_sum" else (n,))
                    if family == "row_sum":
                        x = x.as_strided((outer, n), (n, 1))
                    y = torch.empty(
                        outer if family == "row_sum" else n, device="cuda", dtype=td
                    )
                    scalars, expected = (
                        ((n, outer), x.view(outer, n).sum(-1))
                        if family == "row_sum"
                        else ((n,), x + 1)
                    )
                    why = reason(plan, semantic, scalars, (x, y), k.kernel.resources)
                    if plan.vector_width > 1 and why:
                        with pytest.raises(ValueError) as error:
                            k(scalars, x, y)
                        assert why in str(error.value)
                        continue
                    k(scalars, x, y)
                    torch.testing.assert_close(
                        y,
                        expected,
                        rtol=tolerance[0],
                        atol=tolerance[1],
                        equal_nan=True,
                    )
                    checked += 1
    assert checked > 300


def test_injected_timings_drive_selection(tmp_path):
    code = """
from cubecl_torch.compiler.plan import PhysicalCandidate
target = PhysicalCandidate("vector_pointwise", (128, 1, 1), 4).identity
tuning.INJECTED_TIMINGS = lambda identity, round_id: [1.0 if identity == target else 10.0] * 5
c = tuning.compile_kernel(captured("add-float32"), 0)
x = torch.randn(4096, device="cuda")
y = torch.empty_like(x)
c((4096,), x, y)
print(json.dumps({"winner": [p.identity for p in c.selections.values()], "exact": torch.equal(y, x + 1)}))
"""
    result = run_python(code, tmp_path, prelude=TUNING_PRELUDE)
    from cubecl_torch.compiler.plan import PhysicalCandidate

    assert result["exact"]
    assert result["winner"] == [
        PhysicalCandidate("vector_pointwise", (128, 1, 1), 4).identity
    ]
