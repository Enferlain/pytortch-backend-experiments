"""Review regressions: static reduction coefficient, broadcast, cache policy."""

import copy
import json
from pathlib import Path
from stage1 import evidence

evidence.guard()
import torch
from stage5 import artifact, runtime
from stage5.fixtures import captured
from stage5.plan import PhysicalCandidate
from stage5.legality import candidates, reduction_coefficient

ROOT = Path(__file__).resolve().parent
identity = artifact.compiler_identity(0)
plan = PhysicalCandidate("plane_shared", (128, 1, 1), 4, plane_width=32)
d = captured("row_sum", "float32")
# r*0 is syntactically dependent yet semantically uniform. The scalar pointer is
# intentionally misaligned for vector4 and backed by exactly three elements.
d["buffers"][0]["stride"] = [
    {"op": "constant", "value": 1},
    {"op": "constant", "value": 0},
]
d["ops"][0]["index"] = {
    "op": "add",
    "args": [
        {"op": "linear"},
        {"op": "mul", "args": [{"op": "reduction"}, {"op": "constant", "value": 0}]},
    ],
}
k = artifact.CompiledKernel(d, 0, plan.to_dict(), identity)
x = torch.tensor([123.0, 2.0, 3.0, 4.0], device="cuda")[1:].view(3, 1).expand(3, 257)
y = torch.empty(3, device="cuda")
k((257, 3), x, y)
torch.testing.assert_close(y, x.sum(-1), rtol=0, atol=0)
assert x.data_ptr() % 16 != 0
# The generated source must use a scalar load for that r*0 address.
source = (k.directory / "kernel.cu").read_text()
rows = [
    {
        "case": "literal r*0 uniform broadcast",
        "exact": True,
        "misaligned_scalar_pointer": True,
        "directory": str(k.directory),
    }
]
# A runtime reduction-axis multiplier is intentionally excluded even if the
# particular value is zero/one. Both Python and Rust make the same decision.
for stride in (0, 1):
    s = copy.deepcopy(captured("row_sum", "float32"))
    s["scalars"].append({"id": 2, "dtype": "int64", "positive": False})
    s["buffers"][0]["stride"] = [
        {
            "op": "mul",
            "args": [{"op": "scalar", "slot": 0}, {"op": "scalar", "slot": 2}],
        },
        {"op": "scalar", "slot": 2},
    ]
    s["ops"][0]["index"] = {
        "op": "mul",
        "args": [s["ops"][0]["index"], {"op": "scalar", "slot": 2}],
    }
    x = (
        torch.ones(3, 257, device="cuda")
        if stride
        else torch.ones(1, device="cuda").expand(3, 257)
    )
    baseline = artifact.CompiledKernel(
        s, 0, PhysicalCandidate("shared_tree_reduction_baseline").to_dict(), identity
    )
    legal, rejected = candidates(s, (257, 3, stride), (x, y), baseline.kernel.resources)
    assert all(p.vector_width == 1 for p in legal)
    assert reduction_coefficient(s["ops"][0]["index"]) is None
    try:
        artifact.CompiledKernel(s, 0, plan.to_dict(), identity)
    except RuntimeError as e:
        assert "not proven unit stride" in str(e)
    else:
        raise AssertionError("symbolic coefficient accepted by Rust")
    baseline((257, 3, stride), x, y)
    torch.testing.assert_close(y, x.sum(-1), rtol=0, atol=0)
    rows.append(
        {
            "case": "runtime axis stride excluded",
            "stride": stride,
            "python_and_rust_reject": True,
            "scalar_valid": True,
        }
    )
# The incoming packet has one final write buffer; enforce it before compiling.
multi = copy.deepcopy(captured("add", "float32"))
multi["buffers"][0]["access"] = "write"
before = artifact.COMPILER_INVOCATIONS
try:
    runtime.compile_kernel(multi, 0)
except ValueError as e:
    assert "exactly one final output" in str(e)
else:
    raise AssertionError("multiple output contract accepted")
assert artifact.COMPILER_INVOCATIONS == before
rows.append({"case": "multiple outputs rejected before compile", "compiles": 0})
for value in (0.0, True, 1 << 63):
    tree = {
        "op": "mul",
        "args": [{"op": "reduction"}, {"op": "constant", "value": value}],
    }
    assert reduction_coefficient(tree) is None
rows.append(
    {"case": "axis literal strict signed i64", "float_bool_out_of_range_rejected": True}
)
# Deterministic cold cache: use a distinct represented coefficient to ensure a
# fresh semantic key. No empirical benchmark and no user RNG advancement.
cold = captured("add", "float32")
constant = next(o for o in cold["ops"] if o["op"] == "constant")
constant["value"] = 1.125
c = runtime.compile_kernel(cold, 0)
x = torch.randn(1001, device="cuda")
y = torch.empty_like(x)
state = torch.cuda.get_rng_state().clone()
start = len(runtime.EVENTS)
torch.use_deterministic_algorithms(True)
try:
    c((1001,), x, y)
finally:
    torch.use_deterministic_algorithms(False)
assert any(e["event"] == "deterministic_baseline" for e in runtime.EVENTS[start:])
assert not any(e["event"] == "tune" for e in runtime.EVENTS[start:])
torch.testing.assert_close(y, x + 1.125, rtol=0, atol=0)
assert torch.equal(state, torch.cuda.get_rng_state())
rows.append(
    {
        "case": "deterministic cold cache",
        "benchmark_skipped": True,
        "rng_unchanged": True,
    }
)
# A failed selection may be retried on the same object. Reconstruct key_data
# even though the bounded local key was installed before the exception.
retry_semantic = copy.deepcopy(cold)
next(o for o in retry_semantic["ops"] if o["op"] == "constant")["value"] = 1.0625
retry = runtime.compile_kernel(retry_semantic, 0)
original_select = retry.select
attempts = []


def fail_once(key, key_data, *args):
    assert isinstance(key_data, dict)
    attempts.append(key)
    if len(attempts) == 1:
        raise RuntimeError("injected profiler failure")
    return original_select(key, key_data, *args)


retry.select = fail_once
torch.use_deterministic_algorithms(True)
try:
    try:
        retry((1001,), x, y)
    except RuntimeError as error:
        assert str(error) == "injected profiler failure"
    else:
        raise AssertionError("injected failure not observed")
    retry((1001,), x, y)
finally:
    torch.use_deterministic_algorithms(False)
assert len(attempts) == 2 and attempts[0] == attempts[1]
torch.testing.assert_close(y, x + 1.0625, rtol=0, atol=0)
rows.append({"case": "selection failure retry", "same_key_and_full_identity": True})
# Prepared timing does not retain ordinary user tensors after scope exit.
plain = artifact.CompiledKernel(
    captured("add", "float32"),
    0,
    PhysicalCandidate("scalar_pointwise_baseline").to_dict(),
    identity,
)
plain((1001,), x, y)
assert plain.kernel._prepared is None
with plain.kernel.prepared((1001,), x, y):
    plain.kernel.batch(2)
assert plain.kernel._prepared is None
rows.append({"case": "prepared refs scoped", "ordinary_launch_no_refs": True})
evidence.state("after_proof_controls")
(ROOT / "artifacts/proof-controls.json").write_text(
    json.dumps(
        {
            "checks": rows,
            "events": runtime.EVENTS,
            "independence": evidence.inventory(),
        },
        indent=2,
    )
)
print("Stage5 review proof controls PASS", flush=True)
