"""Cache, effect-reset, alignment and current-stream regression controls."""

import copy
import ctypes as C
from functools import partial
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from stage1 import evidence

evidence.guard()
import torch
from stage1.overlay import install

install()
from stage5 import artifact, runtime
from stage5.plan import PhysicalCandidate
from stage5.fixtures import captured
from stage5.backend import compile_backend, TRACE

native = C.CDLL(str(ROOT.parent / "stage0/libstage0_launcher.so"))
native.stage0_kernel_launches.restype = C.c_uint64
identity = artifact.compiler_identity(0)
rows = []
semantic = captured("add", "float32")
vector = PhysicalCandidate("vector_pointwise", (128, 1, 1), 4)
k = artifact.CompiledKernel(semantic, 0, vector.to_dict(), identity)
x = torch.randn(258, device="cuda")[1:]
y = torch.empty_like(x)
before = native.stage0_kernel_launches()
try:
    k((257,), x, y)
except ValueError as e:
    assert "misaligned" in str(e)
else:
    raise AssertionError("misaligned vector accepted")
assert native.stage0_kernel_launches() == before
rows.append({"case": "vector misalignment", "rejected_before_launch": True})
# Component tested 760 exact tail outcomes; special values check both vector
# memory and conservative arithmetic. Output canaries use aligned subviews.
for dtype in ("float32", "float16", "bfloat16"):
    for width in (2, 4, 8):
        if dtype == "float32" and width == 8:
            continue
        p = PhysicalCandidate("vector_pointwise", (128, 1, 1), width)
        v = artifact.CompiledKernel(captured("add", dtype), 0, p.to_dict(), identity)
        x = torch.tensor(
            [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1e-30, -1e-30] * 37,
            device="cuda",
            dtype=getattr(torch, dtype),
        )
        backing = torch.full(
            (x.numel() + 2 * width,), 123, device="cuda", dtype=x.dtype
        )
        y = backing[width:-width]
        v((x.numel(),), x, y)
        torch.testing.assert_close(y, x + 1, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(
            backing[:width], torch.full((width,), 123, device="cuda", dtype=x.dtype)
        )
        assert torch.equal(backing[-width:], backing[:width])
rows.append({"case": "special values and aligned output canaries", "exact": True})

# Dynamic same-bucket extents/artifacts, fresh allocations, no pointer keys.
compiled = torch.compile(
    lambda x: x + 1, backend=compile_backend, fullgraph=True, dynamic=True
)
counts = []
for n in (257, 259, 1001, 4099, 4101):
    before = artifact.COMPILER_INVOCATIONS
    x = torch.randn(n, device="cuda")
    torch.testing.assert_close(compiled(x), x + 1)
    counts.append({"n": n, "new_compiles": artifact.COMPILER_INVOCATIONS - before})
assert (
    counts[1]["new_compiles"] == 0
    and counts[2]["new_compiles"] == 0
    and counts[4]["new_compiles"] == 0
)
rows.append({"case": "dynamic same-bucket no candidate recompilation", "calls": counts})

# Non-default stream with explicit producer readiness; allocator churn while
# earlier launches remain asynchronous; retains normal record_stream contract.
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
torch.testing.assert_close(out, expected)
rows.append(
    {"case": "current stream and allocator pressure", "stream": stream.cuda_stream}
)

# Alternate scalar effect plans are checked with equivalent explicit state.
# Automatic tuning remains frozen; batch repeat is rejected for effects.
artifacts = json.loads((ROOT / "artifacts/baseline-cubecl_baseline.json").read_text())[
    "artifacts"
]
rng = next(
    copy.deepcopy(a["semantic"])
    for a in artifacts
    if a["semantic"].get("effects", {}).get("rng")
    and len(a["semantic"]["buffers"]) == 2
)
from stage2.indexing import evaluate

for p in (
    PhysicalCandidate("scalar_pointwise_baseline"),
    PhysicalCandidate("scalar_pointwise", (64, 1, 1)),
    PhysicalCandidate("scalar_pointwise", (128, 1, 1)),
):
    r = artifact.CompiledKernel(rng, 0, p.to_dict(), identity)
    for seed in (0, -1, 123456):
        scalars = (257,)
        buffers = [
            torch.tensor([seed], device="cuda", dtype=torch.int64),
            torch.empty(257, device="cuda"),
        ]
        state = torch.cuda.get_rng_state().clone()
        r(scalars, *buffers)
        from stage4.reference import uniform

        expected = torch.tensor([uniform(seed, i) for i in range(257)], device="cuda")
        torch.testing.assert_close(buffers[-1], expected, rtol=0, atol=0)
        assert torch.equal(state, torch.cuda.get_rng_state())
        try:
            r.kernel.batch(2)
        except ValueError as e:
            assert "pure" in str(e)
        else:
            raise AssertionError("RNG repeated batch accepted")
rows.append(
    {
        "case": "controlled RNG candidates",
        "equivalent_seed_counter": True,
        "generator_unchanged": True,
        "batch_veto": True,
    }
)

# Compare generated stochastic graph user-visible RNG advancement with frozen
# backend after first-use tuning of its pure consumer. No artificial RNG retry.
f = lambda x: torch.nn.functional.dropout(torch.sin(x) + 1, 0.25, True)
a = torch.compile(
    f, backend=partial(compile_backend, tuning="baseline"), fullgraph=True, dynamic=True
)
b = torch.compile(f, backend=compile_backend, fullgraph=True, dynamic=True)
x = torch.randn(257, device="cuda")
state = torch.cuda.get_rng_state().clone()
torch.cuda.set_rng_state(state)
ya = a(x)
after_a = torch.cuda.get_rng_state().clone()
torch.cuda.set_rng_state(state)
yb = b(x)
after_b = torch.cuda.get_rng_state().clone()
torch.testing.assert_close(ya, yb, rtol=0, atol=0)
assert torch.equal(after_a, after_b)
rows.append(
    {
        "case": "first-use dropout reset",
        "output_exact": True,
        "user_rng_advance_equal": True,
    }
)

# Functional atomic destination gets zeros for each trial; different plan/order
# sees identical dyadic updates. Never benchmark on a previous candidate's output.
atomic = next(
    copy.deepcopy(a["semantic"])
    for a in artifacts
    if a["semantic"].get("effects", {}).get("mutation")
    and len(a["semantic"]["buffers"]) == 3
)
print("atomic descriptor", atomic, flush=True)
# Actual fixture contract is discovered from captured index/symbol declarations.
for p in (
    PhysicalCandidate("scalar_pointwise_baseline"),
    PhysicalCandidate("scalar_pointwise", (64, 1, 1)),
    PhysicalCandidate("scalar_pointwise", (128, 1, 1)),
):
    c = artifact.CompiledKernel(atomic, 0, p.to_dict(), identity)
    # The captured scatter dynamic scalar is the source numel; output size may
    # be symbolically separate. Read buffer shapes to build exactly that ABI.
    scalars = tuple(257 for _ in atomic["scalars"])
    tensors = []
    for buf in atomic["buffers"]:
        size = tuple(evaluate(e, scalars) for e in buf["size"])
        td = c.kernel.storage_dtypes[buf["dtype"]]
        if buf["access"] == "write":
            t = torch.zeros(size, device="cuda", dtype=td)
        elif td == torch.int64:
            t = (
                torch.arange(__import__("math").prod(size), device="cuda").reshape(size)
                % 17
            )
        else:
            t = (
                (
                    torch.arange(__import__("math").prod(size), device="cuda") % 17 - 8
                ).float()
                / 8
            ).reshape(size)
        tensors.append(t)
    expected = torch.zeros_like(tensors[-1]).scatter_add_(
        0,
        tensors[0] if tensors[0].dtype == torch.int64 else tensors[1],
        tensors[1] if tensors[0].dtype == torch.int64 else tensors[0],
    )
    for _ in range(3):
        tensors[-1].zero_()
        c(scalars, *tensors)
        torch.testing.assert_close(tensors[-1], expected, rtol=0, atol=0)
    try:
        c.kernel.batch(2)
    except ValueError as e:
        assert "pure" in str(e)
    else:
        raise AssertionError("atomic repeated batch accepted")
rows.append(
    {
        "case": "controlled atomic candidates",
        "destination_reset_each_trial": True,
        "exact_dyadic_updates": True,
        "batch_veto": True,
    }
)

evidence.state("after_stage5_controls")
(ROOT / "artifacts/controls.json").write_text(
    json.dumps(
        {
            "checks": rows,
            "events": runtime.EVENTS,
            "trace": TRACE,
            "independence": evidence.inventory(),
        },
        indent=2,
    )
)
print("Stage5 controls PASS", flush=True)
