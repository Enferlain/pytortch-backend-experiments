"""State/effect acceptance with real capture, exact RNG state and execution census."""

import copy
import ctypes as C
import functools
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache")
from stage1 import evidence

evidence.guard()
import torch
from stage1.overlay import install

install()
from stage4.backend import compile_backend, TRACE
from stage4 import runtime
from stage3 import runtime as bridge
from stage4.reference import uniform
from fp_hardening.patches import SOURCE, ROOT as PROJECT_ROOT
import subprocess

assert (
    subprocess.check_output(["git", "-C", str(SOURCE), "diff", "HEAD"])
    == (PROJECT_ROOT / "fp_hardening/cubecl-fp.patch").read_bytes()
)
assert not torch._dynamo.config.suppress_errors
torch._dynamo.config.recompile_limit = 128
native = C.CDLL(str(PROJECT_ROOT / "stage0/libstage0_launcher.so"))
for name in (
    "stage0_kernel_launches",
    "stage3_kernel_launches",
    "stage3_total_shared_bytes",
    "stage4_kernel_launches",
    "stage4_five_input_launches",
):
    getattr(native, name).restype = C.c_uint64
native_begin = native.stage0_kernel_launches()
reduction_begin = native.stage3_kernel_launches()
shared_begin = native.stage3_total_shared_bytes()
stage4_begin = native.stage4_kernel_launches()
five_begin = native.stage4_five_input_launches()
evidence.state("after_torch_inductor")
evidence.SELECTED |= {
    "replace_random",
    "replace_random_passes",
    "fuse_seed_creation_pass",
    "inductor_random",
    "inductor_seeds",
    "inductor_lookup_seed",
    "RandomSeeds.__init__",
    "gather",
    "embedding",
    "scatter_reduce_",
    "scatter_add",
    "index_put_impl_",
    "CSEProxy.indirect_indexing",
    "CubeCLKernel.indirect_indexing",
    "Kernel._enqueue",
    "FallbackKernel.create",
    "ExternKernel.codegen",
    "ExternKernelOut.codegen",
}
sys.setprofile(evidence.profile)
RESULT = {"checks": [], "failures": []}
CENSUS = []


def compiled(fn, policy="generated"):
    return torch.compile(
        fn,
        backend=functools.partial(compile_backend, rng_policy=policy),
        fullgraph=True,
        dynamic=True,
    )


def run(name, fn):
    start = len(TRACE)
    events = len(runtime.EVENTS)
    try:
        detail = fn() or {}
        RESULT["checks"].append(
            {
                "name": name,
                "trace_range": [start, len(TRACE)],
                "event_range": [events, len(runtime.EVENTS)],
                **detail,
            }
        )
        print(name, "PASS", flush=True)
    except Exception as e:
        RESULT["failures"].append(
            {"name": name, "type": type(e).__name__, "error": str(e)}
        )
        print(name, "FAIL", str(e), flush=True)


def seeds_from_state(state, count):
    torch.cuda.set_rng_state(state)
    bounds = torch.iinfo(torch.int64)
    result = torch.randint(
        bounds.min, bounds.max, (count,), device="cuda", dtype=torch.int64
    )
    after = torch.cuda.get_rng_state().clone()
    return list(map(int, result.cpu())), after


def reference_random(seed, n):
    return torch.tensor(
        [uniform(seed, i) for i in range(n)], device="cuda", dtype=torch.float32
    )


def profile(label, fn):
    start = len(runtime.EVENTS)
    native_count = native.stage0_kernel_launches()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        fn()
    path = ROOT / "artifacts" / f"{label}-profiler.json"
    p.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())["traceEvents"]
    kernels = [e for e in trace if e.get("cat") == "kernel"]
    launches = [
        e for e in runtime.EVENTS[start:] if e["event"] == "launch" and e["grid"][0] > 0
    ]
    expected = [e["entrypoint"] for e in launches]
    actual = [e["name"] for e in kernels if e["name"] in expected]
    assert (
        len(actual) == len(expected) == native.stage0_kernel_launches() - native_count
    )
    foreign = [e["name"] for e in kernels if e["name"] not in expected]
    assert not any("triton" in x.lower() for x in foreign)
    cpu = [e["name"] for e in trace if e.get("cat") == "cpu_op"]
    value = {
        "label": label,
        "generated_kernel_count": len(expected),
        "generated_entrypoints": actual,
        "non_cubecl_kernel_count": len(foreign),
        "non_cubecl_kernel_names": foreign,
        "cpu_operator_names": cpu,
        "graph_breaks": 0,
        "triton_kernels": 0,
        "launch_range": [start, len(runtime.EVENTS)],
    }
    CENSUS.append(value)
    return value


for dtype in (torch.float32, torch.float16, torch.bfloat16):

    def rng(dtype=dtype):
        f = compiled(lambda x: torch.rand_like(x))
        rows = []
        base = None
        for n in (0, 1, 255, 256, 257, 65539):
            x = torch.ones(n, device="cuda", dtype=dtype)
            torch.manual_seed(321)
            state = torch.cuda.get_rng_state().clone()
            cold = f(x)
            after = torch.cuda.get_rng_state().clone()
            before = bridge.COMPILER_INVOCATIONS
            torch.cuda.set_rng_state(state)
            warm = f(x)
            assert torch.equal(cold, warm) and torch.equal(
                after, torch.cuda.get_rng_state()
            )
            assert bridge.COMPILER_INVOCATIONS == before
            if n:
                seeds, expected_state = seeds_from_state(state, 1)
                assert torch.equal(after, expected_state)
                torch.testing.assert_close(
                    cold[: min(n, 1024)],
                    reference_random(seeds[0], min(n, 1024)).to(dtype),
                    rtol=0,
                    atol=0,
                )
                assert cold.dtype == dtype and bool(((cold >= 0) & (cold <= 1)).all())
                if n == 65539:
                    assert abs(cold.float().mean().item() - 0.5) < 0.012
                torch.cuda.set_rng_state(after)
                advanced = f(x)
                assert not torch.equal(cold, advanced) if n > 1 else True
            else:
                assert torch.equal(state, after)
            if n == 255:
                base = before
            if n in (256, 257, 65539):
                assert before == base, "compatible dynamic artifact recompiled"
            rows.append(
                {
                    "n": n,
                    "cold_warm_exact": True,
                    "state_exact": True,
                    "compiles": before,
                }
            )
        profile_x = torch.ones(257, device="cuda", dtype=dtype)
        return {
            "rows": rows,
            "profile": profile(
                "rng-" + str(dtype).removeprefix("torch."),
                lambda: f(profile_x),
            ),
        }

    run("uniform_rng_" + str(dtype), rng)


def multi_rng():
    f = compiled(lambda x: torch.rand_like(x) + torch.rand_like(x))
    x = torch.ones(257, device="cuda")
    torch.manual_seed(22)
    state = torch.cuda.get_rng_state()
    y = f(x)
    after = torch.cuda.get_rng_state()
    seeds, expected = seeds_from_state(state, 2)
    assert seeds[0] != seeds[1] and torch.equal(after, expected)
    torch.testing.assert_close(
        y,
        reference_random(seeds[0], 257) + reference_random(seeds[1], 257),
        rtol=0,
        atol=0,
    )
    return {"distinct_seeds": seeds, "profile": profile("multi-rng", lambda: f(x))}


run("multiple_random_operations", multi_rng)

for dtype in (torch.float32, torch.float16, torch.bfloat16):

    def dropout(dtype=dtype):
        f = compiled(lambda x: torch.nn.functional.dropout(x, 0.25, True))
        rows = []
        for n in (1, 255, 256, 257, 65539):
            x = torch.ones(n, device="cuda", dtype=dtype, requires_grad=True)
            torch.manual_seed(45)
            state = torch.cuda.get_rng_state().clone()
            y = f(x)
            after = torch.cuda.get_rng_state()
            seeds, expected = seeds_from_state(state, 1)
            assert torch.equal(after, expected)
            keep = reference_random(seeds[0], n).to(dtype) > 0.25
            xr = x.detach().clone().requires_grad_()
            ref = (keep.to(dtype) * xr) / 0.75
            torch.testing.assert_close(
                y,
                ref,
                rtol=0.016
                if dtype == torch.bfloat16
                else 0.002
                if dtype == torch.float16
                else 2e-6,
                atol=0.008
                if dtype == torch.bfloat16
                else 0.001
                if dtype == torch.float16
                else 2e-6,
            )
            upstream = torch.linspace(-1, 1, n, device="cuda", dtype=dtype)
            y.backward(upstream)
            ref.backward(upstream)
            # Separate reference leaf checks the compiled gradient directly.
            torch.testing.assert_close(
                x.grad,
                xr.grad,
                rtol=0.016
                if dtype == torch.bfloat16
                else 0.002
                if dtype == torch.float16
                else 2e-6,
                atol=0.008
                if dtype == torch.bfloat16
                else 0.001
                if dtype == torch.float16
                else 2e-6,
            )
            if n == 65539:
                assert abs(keep.float().mean().item() - 0.75) < 0.02
            rows.append({"n": n, "kept": int(keep.sum()), "arbitrary_upstream": True})
        for p, train in ((0.0, True), (1.0, True), (0.25, False)):
            x = torch.ones(257, device="cuda", dtype=dtype, requires_grad=True)
            state = torch.cuda.get_rng_state().clone()
            g = compiled(
                (lambda x: torch.nn.functional.dropout(x, 0.0, True))
                if p == 0
                else (lambda x: torch.nn.functional.dropout(x, 1.0, True))
                if p == 1
                else (lambda x: torch.nn.functional.dropout(x, 0.25, False))
            )
            y = g(x)
            torch.testing.assert_close(y, torch.nn.functional.dropout(x, p, train))
            assert torch.equal(state, torch.cuda.get_rng_state())
            y.sum().backward()
            torch.testing.assert_close(
                x.grad, torch.full_like(x, 0 if p == 1 and train else 1)
            )
        profile_x = torch.ones(257, device="cuda", dtype=dtype, requires_grad=True)
        profile_upstream = torch.linspace(-1, 1, 257, device="cuda", dtype=dtype)

        def profiled_backward():
            profile_x.grad = None
            f(profile_x).backward(profile_upstream)

        return {
            "rows": rows,
            "profile": profile(
                "dropout-" + str(dtype).removeprefix("torch."), profiled_backward
            ),
        }

    run("dropout_forward_backward_" + str(dtype), dropout)


def indirect():
    gather = compiled(lambda x, i: torch.gather(x, 0, i))
    index = compiled(lambda x, i: x[i])
    rows = []
    for n, m in ((7, 1), (255, 255), (256, 256), (257, 257), (513, 65539)):
        base = torch.arange(n * 2 + 3, device="cuda", dtype=torch.float32)
        x = base[3::2]
        i = torch.arange(m, device="cuda") % n
        if m > 1:
            i[-1] = n - 1
        y = gather(x, i)
        torch.testing.assert_close(y, torch.gather(x, 0, i), rtol=0, atol=0)
        neg = i.clone()
        neg[::2] -= n
        torch.testing.assert_close(index(x, neg), x[neg], rtol=0, atol=0)
        rows.append(
            {
                "n": n,
                "m": m,
                "source_stride": list(x.stride()),
                "offset": x.storage_offset(),
            }
        )
    for n in (0, 7):
        x = torch.empty(n, device="cuda")
        i = torch.empty(0, device="cuda", dtype=torch.int64)
        assert gather(x, i).numel() == index(x, i).numel() == 0
    profile_x = torch.arange(257, device="cuda", dtype=torch.float32)
    profile_i = torch.arange(257, device="cuda") % 17
    return {
        "rows": rows,
        "profile": profile("indirect-gather", lambda: gather(profile_x, profile_i)),
    }


run("direct_storage_indirect_addresses", indirect)

for dtype in (torch.float32, torch.float16, torch.bfloat16):

    def atomic(dtype=dtype):
        f = compiled(
            lambda x, i, d: torch.zeros(
                d, device=x.device, dtype=torch.float32
            ).scatter_add(0, i, x.float())
        )
        rows = []
        for n in (0, 1, 255, 256, 257, 4099):
            for form in ("distinct", "two_way", "many_to_one", "sparse"):
                d = n + 7 if form == "distinct" else 17
                i = (
                    torch.arange(n, device="cuda")
                    if form == "distinct"
                    else torch.arange(n, device="cuda") // 2 % 17
                    if form == "two_way"
                    else torch.zeros(n, device="cuda", dtype=torch.int64)
                    if form == "many_to_one"
                    else (torch.arange(n, device="cuda") % 4) * 4
                )
                x = ((torch.arange(n, device="cuda") % 9) - 4).to(dtype) / 8
                y = f(x, i, d)
                ref = torch.zeros(d, device="cuda").scatter_add(0, i, x.float())
                # Dyadic values have exact accumulated results at these bounded counts.
                torch.testing.assert_close(y, ref, rtol=0, atol=0)
                rows.append({"n": n, "d": d, "collisions": form})
        x = torch.linspace(-1, 1, 257, device="cuda", dtype=dtype)
        i = torch.arange(257, device="cuda") % 17
        return {
            "rows": rows,
            "profile": profile(
                "atomic-" + str(dtype).removeprefix("torch."), lambda: f(x, i, 17)
            ),
        }

    run("atomic_fp32_destination_" + str(dtype), atomic)


def embedding():
    f = compiled(lambda w, i: torch.nn.functional.embedding(i, w))
    rows = []
    for n, m in ((7, 1), (17, 255), (19, 256), (23, 257)):
        w = torch.linspace(-1, 1, n * 7, device="cuda").reshape(n, 7).requires_grad_()
        i = torch.arange(m, device="cuda") % n
        y = f(w, i)
        up = torch.linspace(-2, 2, y.numel(), device="cuda").reshape(y.shape)
        y.backward(up)
        wr = w.detach().clone().requires_grad_()
        yr = torch.nn.functional.embedding(i, wr)
        yr.backward(up)
        torch.testing.assert_close(y, yr, rtol=0, atol=0)
        torch.testing.assert_close(w.grad, wr.grad, rtol=2e-5, atol=2e-5)
        rows.append({"n": n, "m": m, "arbitrary_upstream": True})

    def profiled_backward():
        w.grad = None
        f(w, i).backward(up)

    return {"rows": rows, "profile": profile("embedding-backward", profiled_backward)}


run("embedding_generated_atomic_backward", embedding)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(17, 7))
        self.linear = torch.nn.Linear(7, 9, bias=False)

    def forward(self, idx, mask=None):
        x = self.linear(torch.nn.functional.embedding(idx, self.weight))
        x = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5)
        return (
            torch.nn.functional.dropout(x, 0.25, True)
            if mask is None
            else x * mask.to(x.dtype) * (1 / 0.75)
        )


def training():
    torch.manual_seed(313)
    model = Model().cuda()
    ref = copy.deepcopy(model)
    opt = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, foreach=False, fused=False
    )
    refopt = torch.optim.SGD(
        ref.parameters(), lr=0.01, momentum=0.9, foreach=False, fused=False
    )
    f = compiled(lambda i: (model(i) ** 2).flatten().mean())
    i = torch.tensor([0, 2, 2, 16, 1], device="cuda")
    rows = []
    for step in range(4):
        opt.zero_grad(set_to_none=True)
        refopt.zero_grad(set_to_none=True)
        state = torch.cuda.get_rng_state().clone()
        seeds, expected = seeds_from_state(state, 1)
        torch.cuda.set_rng_state(state)
        loss = f(i)
        state_after = torch.cuda.get_rng_state().clone()
        assert torch.equal(state_after, expected)
        mask = reference_random(seeds[0], i.numel() * 9).reshape(i.numel(), 9) > 0.25
        loss.backward()
        reference = (ref(i, mask) ** 2).flatten().mean()
        reference.backward()
        torch.testing.assert_close(loss, reference, rtol=2e-5, atol=2e-5)
        for a, b in zip(model.parameters(), ref.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-5)
        opt.step()
        refopt.step()
        for a, b in zip(model.parameters(), ref.parameters()):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(
                opt.state[a]["momentum_buffer"],
                refopt.state[b]["momentum_buffer"],
                rtol=2e-5,
                atol=2e-5,
            )
        rows.append(
            {
                "step": step,
                "loss": loss.item(),
                "reference_loss": reference.item(),
                "gradients_parameters_momentum_match": True,
            }
        )

    def train_step():
        opt.zero_grad(set_to_none=True)
        loss = f(i)
        loss.backward()
        opt.step()

    return {
        "rows": rows,
        "profile": profile("training-step", train_step),
        "optimizer_boundary": "eager ATen outside fullgraph forward/backward; non-fused/non-foreach SGD momentum",
    }


run("representative_training_optimizer_state", training)


def external_policy():
    f = compiled(lambda x: torch.randn_like(x) + 1, policy="aten_extern")
    x = torch.ones(257, device="cuda")
    torch.manual_seed(777)
    state = torch.cuda.get_rng_state().clone()
    y = f(x)
    after = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    ref = torch.randn_like(x) + 1
    torch.testing.assert_close(y, ref, rtol=0, atol=0)
    assert torch.equal(after, torch.cuda.get_rng_state())
    return {
        "profile": profile("explicit-aten-rng", lambda: f(x)),
        "policy": "existing Inductor fallback_random selected before random replacement/lowering",
    }


run("explicit_existing_rng_fallback", external_policy)


def negative_capabilities():
    rows = []
    for name, fn, args in (
        (
            "randn_generated",
            lambda x: torch.randn_like(x),
            (torch.ones(257, device="cuda"),),
        ),
        (
            "index_int32",
            lambda x, i: x[i],
            (
                torch.ones(257, device="cuda"),
                torch.arange(257, device="cuda", dtype=torch.int32),
            ),
        ),
        (
            "atomic_half",
            lambda x, i: torch.zeros_like(x).scatter_add(0, i, x),
            (
                torch.ones(257, device="cuda", dtype=torch.float16),
                torch.arange(257, device="cuda"),
            ),
        ),
        ("user_mutation", lambda x: x.add_(1), (torch.ones(257, device="cuda"),)),
        (
            "out_mutation",
            lambda x: torch.add(x, 1, out=x),
            (torch.ones(257, device="cuda"),),
        ),
        (
            "inplace_keyword",
            lambda x: torch.nn.functional.dropout(x, 0.25, True, inplace=True),
            (torch.ones(257, device="cuda"),),
        ),
    ):
        before = bridge.COMPILER_INVOCATIONS
        state = torch.cuda.get_rng_state().clone()
        start = len(TRACE)
        try:
            compiled(fn)(*args)
        except Exception as e:
            assert "unsupported" in str(e).lower(), str(e)
            assert bridge.COMPILER_INVOCATIONS == before and torch.equal(
                state, torch.cuda.get_rng_state()
            )
            rows.append(
                {
                    "case": name,
                    "error": str(e),
                    "trace_range": [start, len(TRACE)],
                    "rng_state_unchanged": True,
                }
            )
        else:
            raise AssertionError(name + " accepted")
    return {"cases": rows}


run("explicit_unsupported_capabilities", negative_capabilities)

sys.setprofile(None)
(ROOT / "artifacts/compiler-path.json").write_text(json.dumps(evidence.CALLS, indent=2))
evidence.state("after_stage4")
RESULT.update(
    compiles=bridge.COMPILER_INVOCATIONS,
    launches=bridge.KERNEL_LAUNCHES,
    native_launches=native.stage0_kernel_launches() - native_begin,
    native_reductions=native.stage3_kernel_launches() - reduction_begin,
    native_shared_bytes=native.stage3_total_shared_bytes() - shared_begin,
    native_stage4_launches=native.stage4_kernel_launches() - stage4_begin,
    native_five_input_launches=native.stage4_five_input_launches() - five_begin,
    independence=evidence.inventory(),
    fp_patch_sha256=hashlib.sha256(
        (PROJECT_ROOT / "fp_hardening/cubecl-fp.patch").read_bytes()
    ).hexdigest(),
)
assert RESULT["native_launches"] == RESULT["launches"]
assert RESULT["native_reductions"] * 1024 == RESULT["native_shared_bytes"]
assert (
    RESULT["native_stage4_launches"] + RESULT["native_reductions"]
    == RESULT["native_launches"]
)
assert RESULT["native_five_input_launches"] > 0
(ROOT / "artifacts/results.json").write_text(json.dumps(RESULT, indent=2))
(ROOT / "artifacts/fallback-census.json").write_text(
    json.dumps(
        {
            "execution_regions": CENSUS,
            "scheduler": TRACE,
            "notes": "Cross-check pre-fusion owners, wrapper runtime/native counters and per-region CPU/CUDA profiler events; raw profiler files remain local.",
        },
        indent=2,
    )
)
(ROOT / "artifacts/schedule-replay.json").write_text(
    json.dumps(TRACE, indent=2, default=str)
)
(ROOT / "artifacts/runtime-events.json").write_text(
    json.dumps(runtime.EVENTS, indent=2)
)
if RESULT["failures"]:
    raise SystemExit(1)
print("Stage4 acceptance PASS", len(RESULT["checks"]), flush=True)
