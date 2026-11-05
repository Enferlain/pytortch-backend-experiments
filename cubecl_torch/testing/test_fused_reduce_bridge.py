"""The fused ecosystem reduce as a replay and index check (fused-reduction-bridge
task 3.2, design 9): each covered kernel's recorded body run inside CubeK's
reduce, with each CubeK routine (inferred blueprint), on small-integer inputs,
so every sum is exact in any order. Every output must equal a float64
evaluation of the recorded ops exactly, and the guard region around each
output must be untouched. Numerical acceptance is the band (group 5), not
this test."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import json, sys
import cubecl_torch.inductor as ci
ci.install()
import torch
from cubecl_torch.inductor import tuning
from cubecl_torch.inductor.legality import fused_reduce_reason
from cubecl_torch.inductor.operand_layout import operand_layout
from cubecl_torch.runtime import bridge

GUARD = 64
SENTINEL = 12345.0
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

captured = []
init = tuning.AutotunedKernel.__init__
def record(self, semantic, *args, **kwargs):
    init(self, semantic, *args, **kwargs)
    if semantic.get("schema") == 3 and fused_reduce_reason(semantic, self.layouts) is None:
        captured.append((semantic, self.layouts))
tuning.AutotunedKernel.__init__ = record


def index(v, outer, r):
    op = v["op"]
    if op == "linear":
        return outer
    if op == "reduction":
        return r
    if op == "constant":
        return torch.tensor(v["value"], dtype=torch.int64, device=outer.device)
    args = [index(a, outer, r) for a in v["args"]]
    out = args[0]
    for a in args[1:]:
        out = {"add": torch.add, "mul": torch.mul,
               "floor_div": torch.floor_divide, "mod": torch.remainder}[op](out, a)
    return out


def evaluate(semantic, tensors):
    # float64 values of the recorded ops over the (output, reduction) grid.
    n, extent = semantic["outer"]["value"], semantic["reduction_extent"]["value"]
    dev = tensors[0].device
    outer = torch.arange(n, device=dev)[:, None]
    r = torch.arange(extent, device=dev)[None, :]
    values, out = {}, {}
    for op in semantic["ops"]:
        name = op["op"]
        if name == "load":
            t = tensors[op["buffer"]]
            base = torch.as_strided(t, (t.numel() and 1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride())),), (1,))
            v = base[index(op["index"], outer, r)].double()
        elif name == "store":
            out[op["buffer"]] = (index(op["index"], outer, r), values[op["args"][0]])
            continue
        elif name == "reduction":
            v = values[op["args"][0]].sum(-1, keepdim=True)
        elif name == "constant":
            v = torch.tensor(float(op["value"]), dtype=torch.float64, device=dev)
        elif name == "cast":
            v = values[op["args"][0]].to(DTYPES[op["dtype"]]).double()
        else:
            a = [values[x] for x in op["args"]]
            v = {"mul": torch.mul, "add": torch.add, "sub": torch.sub,
                 "truediv": torch.div}[name](*a)
        values[op["id"]] = v.expand(n, -1) if v.dim() == 2 else v
    return out


def guarded(t):
    # ``t`` inside a buffer with SENTINEL before and after it.
    size, stride = list(t.shape), list(t.stride())
    span = 1 + sum((s - 1) * st for s, st in zip(size, stride)) if t.numel() else 0
    whole = torch.full((span + 2 * GUARD,), SENTINEL, dtype=t.dtype, device=t.device)
    return whole, torch.as_strided(whole, size, stride, GUARD)


case, rows, cols, dtype = json.loads(sys.argv[1])
dtype = DTYPES[dtype]
device = next(i for i in range(torch.cuda.device_count()))
compiled = lambda fn: torch.compile(fn, backend=ci.compile_backend, dynamic=False)
if case == "rms_backward":
    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    x = torch.randn(rows, cols, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(cols, device="cuda", dtype=dtype, requires_grad=True)
    compiled(rms)(x, w).sum().backward()
elif case == "shared":
    compiled(lambda x: (x * 2).sum(-1) * 2)(torch.randn(rows, cols, device="cuda", dtype=dtype))
elif case == "two_outputs":
    def f(x):
        s = x.sum(-1)
        return s, s * 2
    compiled(f)(torch.randn(rows, cols, device="cuda", dtype=dtype))
torch.cuda.synchronize()

results = []
g = torch.Generator(device="cuda").manual_seed(0)
for semantic, layouts in captured:
    loads = [op for op in semantic["ops"] if op["op"] == "load"]
    agreed, reason = operand_layout([layouts.get(op["id"]) for op in loads])
    inputs = []
    for b in semantic["buffers"]:
        size = [s["value"] for s in b["size"]]
        stride = [s["value"] for s in b["stride"]]
        t = torch.empty_strided(size, stride, device="cuda", dtype=DTYPES[b["dtype"]])
        if b["access"] == "read":
            t.copy_(torch.randint(-3, 4, size, device="cuda", generator=g).to(t.dtype))
        inputs.append(t)
    expected = evaluate(semantic, inputs)
    stores = [op for op in semantic["ops"] if op["op"] == "store"]
    for routine in ("unit", "plane", "cube_unit", "cube"):
        tensors, wholes = [], {}
        for i, (b, t) in enumerate(zip(semantic["buffers"], inputs)):
            if b["access"] == "write":
                wholes[i], t = guarded(t)
            tensors.append(t)
        bridge.fused_reduce(semantic, agreed, tensors, routine)
        torch.cuda.synchronize()
        exact, guard = True, True
        for buffer, (where, value) in expected.items():
            t = tensors[buffer]
            base = torch.as_strided(t, (1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride())),), (1,))
            got = base[where.reshape(-1)].double()
            want = value.reshape(-1).to(t.dtype).double()
            exact &= torch.equal(got, want)
            whole = wholes[buffer]
            guard &= bool((whole[:GUARD] == SENTINEL).all() and (whole[-GUARD:] == SENTINEL).all())
        results.append({"outer": semantic["outer"]["value"],
                        "extent": semantic["reduction_extent"]["value"],
                        "stores": len(stores), "routine": routine,
                        "exact": exact, "guard": guard})
print(json.dumps(results))
"""


def run(case, rows, cols, dtype, tmp_path):
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT, json.dumps([case, rows, cols, dtype])],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO),
            "CUBECL_TORCH_CACHE_DIR": str(tmp_path / "cache"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("rows, cols", [(64, 1024), (8192, 1280)])
def test_weight_gradient_replays_exactly(tmp_path, rows, cols, dtype):
    results = run("rms_backward", rows, cols, dtype, tmp_path)
    # Every covered kernel (one, or both split stages) with every routine.
    assert {r["routine"] for r in results} == {"unit", "plane", "cube_unit", "cube"}
    assert len(results) in (4, 8), results
    assert all(r["exact"] and r["guard"] for r in results), results


def test_shared_ops_are_replayed_after_the_reduction(tmp_path):
    """An op after the reduction may read the operand's row-invariant ops (a
    constant Inductor emits once): covered, and exact."""
    results = run("shared", 64, 1024, "float32", tmp_path)
    assert results, "the kernel was not covered"
    assert all(r["exact"] and r["guard"] for r in results), results


def test_two_outputs_of_one_reduction(tmp_path):
    """Both stores of one reduction are written (a row sum: CubeK's parallel
    vectorization, the reduction axis contiguous)."""
    results = run("two_outputs", 64, 1024, "float32", tmp_path)
    assert results and all(r["stores"] == 2 for r in results), results
    assert all(r["exact"] and r["guard"] for r in results), results


DIVISION = r"""
import json, sys
import cubecl_torch.inductor as ci
ci.install()
import torch
from cubecl_torch.inductor import tuning
from cubecl_torch.inductor.legality import fused_reduce_reason
from cubecl_torch.inductor.operand_layout import operand_layout
from cubecl_torch.runtime import bridge

captured = []
init = tuning.AutotunedKernel.__init__
def record(self, semantic, *args, **kwargs):
    init(self, semantic, *args, **kwargs)
    if semantic.get("schema") == 3 and fused_reduce_reason(semantic, self.layouts) is None:
        captured.append((semantic, self.layouts))
tuning.AutotunedKernel.__init__ = record
x = torch.randn(64, 1000, device="cuda")
y = torch.rand(64, device="cuda") + 1
torch.compile(lambda x, y: x.sum(-1) / y, backend=ci.compile_backend, dynamic=False)(x, y)
torch.cuda.synchronize()
(semantic, layouts), = captured
assert any(op["op"] == "truediv" for op in semantic["ops"]), semantic["ops"]
loads = [op for op in semantic["ops"] if op["op"] == "load"]
agreed, _ = operand_layout([layouts.get(op["id"]) for op in loads])
out = {}
for exact in (True, False):
    before = set(bridge._KERNELS)
    tensors = [torch.empty_strided([s["value"] for s in b["size"]], [s["value"] for s in b["stride"]],
               device="cuda") for b in semantic["buffers"]]
    tensors[0].copy_(x); tensors[1].copy_(y)
    bridge.fused_reduce(semantic, agreed, tensors, "cube", exact_division=exact)
    torch.cuda.synchronize()
    (device, entry), = set(bridge._KERNELS) - before
    out[str(exact)] = bridge.compiled_kernel(device, entry)[1].decode(errors="replace")
print(json.dumps(out))
"""


def test_division_follows_the_numerics_policy(tmp_path):
    """An epilogue division compiles with the policy's precision: ``div.rn``
    under exact division, the approximate division otherwise (task 3.3)."""
    out = subprocess.run(
        [sys.executable, "-c", DIVISION],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO),
            "CUBECL_TORCH_CACHE_DIR": str(tmp_path / "cache"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    ptx = json.loads(out.stdout.strip().splitlines()[-1])
    assert "div.rn.f32" in ptx["True"]
    assert "div.rn.f32" not in ptx["False"]
    assert "div.full.f32" in ptx["False"] or "div.approx" in ptx["False"]
