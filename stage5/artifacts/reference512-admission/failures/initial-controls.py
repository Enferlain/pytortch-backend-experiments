"""Qualification and normal-selector controls for frozen split9 admission."""

import argparse
import copy
import ctypes as C
from itertools import chain
import json
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/reference512-admission"


def setup():
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    from stage1.overlay import install

    install()
    import torch

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache/reference512-admission")
    OUT.mkdir(parents=True, exist_ok=True)
    return torch


def save(name, data):
    path = OUT / (name + ".json")
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, indent=2) + "\n")


def cache(mode):
    torch = setup()
    from stage5.backend import compile_backend
    from stage5 import runtime, artifact
    from stage5.reference512_plan import STRATEGY
    from stage1 import evidence
    from stage5.reference_reduction_probe import tensor_sha

    f = torch.compile(
        lambda x: x.sum(-1), backend=compile_backend, fullgraph=True, dynamic=True
    )
    torch.manual_seed(8801)
    x = torch.randn((8, 65539), device="cuda")
    y = f(x)
    objects = list(runtime.CACHE.values())
    assert {o.reduction_context["role"] for o in objects} == {"producer", "final"}
    p = next(o for o in objects if o.reduction_context["role"] == "producer")
    assert any(
        plan.strategy == STRATEGY for plans, _ in p._legality.values() for plan in plans
    )
    count = sum(e["event"] == "reference512_model_check" for e in runtime.EVENTS)
    for _ in range(3):
        assert tensor_sha(f(x)) == tensor_sha(y)
    assert count == sum(
        e["event"] == "reference512_model_check" for e in runtime.EVENTS
    )
    if mode == "reload":
        assert count == 0 and artifact.COMPILER_INVOCATIONS == 0
        assert any(e["event"] == "tuning_cache_hit" for e in runtime.EVENTS)
    else:
        assert count == 1, "init must exercise a fresh validation/tuning key"
    details = [
        json.loads((ROOT / "cache/tuning" / k / "selection.json").read_text())
        for o in objects
        for k in o.selections
    ]
    evidence.state("after_reference512_cache")
    save(
        "cache-" + mode,
        {
            "complete": True,
            "model_checks": count,
            "compiler_invocations": artifact.COMPILER_INVOCATIONS,
            "output_sha256": tensor_sha(y),
            "selections": details,
            "environment": evidence.inventory(),
        },
    )
    print("normal pair cache", mode, "PASS", flush=True)


def controls():
    torch = setup()
    from stage5 import runtime
    from stage5.reference512_plan import templates, context, STRATEGY, admits
    from stage5.reference512_oracle import partials, check, environment, fenv
    from stage5.plan import catalog
    from stage5.legality import candidates, signature
    from stage5.reduction_codegen_mechanisms import Component
    from stage5.reference_reduction_probe import tensor_sha
    from stage1 import evidence

    t = templates()
    ctx = context(t["producer"], t["final"], "producer")
    obj = runtime.compile_kernel(t["producer"], 0, reduction_context=ctx)
    t["producer"]["schema"] = 99
    ctx["producer"]["schema"] = 99
    assert obj.semantic["schema"] == obj.reduction_context["producer"]["schema"] == 3
    assert templates()["producer"]["schema"] == 3
    frozen = Component(ROOT / "artifacts/reduction-structure/codegen/uniform/float32")
    x = torch.zeros((5, 65539), device="cuda")
    x[0, [0, 256, 512]] = torch.tensor([1e8, 1, -1e8], device="cuda")
    x[1] = -x[0]
    x[2, [0, 256, 512]] = torch.tensor([2**24, 0.25, -(2**24)], device="cuda")
    x[3, :1024] = 2**-149
    x[4, ::2], x[4, 1::2] = 1024, -1024
    p = torch.empty((5, 9), device="cuda")
    frozen((65539, 5), x, p)
    target = frozen.kernel.resources
    plan = next(p for p in catalog() if p.strategy == STRATEGY)
    for b, n, allowed in (
        (5, 65539, True),
        (8, 65539, True),
        (64, 65539, True),
        (8, 131071, True),
        (8, 1048576, True),
        (8, 65540, False),
        (7, 65539, False),
    ):
        assert admits(obj.reduction_context, (n, b)) == allowed
        plans, _ = candidates(
            obj.semantic, (n, b), (), target, reduction_context=obj.reduction_context
        )
        assert (plan in plans) == allowed
    plain, _ = candidates(obj.semantic, (65539, 5), (), target)
    assert plan not in plain
    for mutation in (
        lambda v: v.update(version="wrong"),
        lambda v: v["producer"].update(schema=99),
        lambda v: v["final"]["reduction_extent"].update(value=8),
    ):
        value = copy.deepcopy(obj.reduction_context)
        mutation(value)
        try:
            runtime.compile_kernel(obj.semantic, 0, reduction_context=value)
        except ValueError:
            pass
        else:
            raise AssertionError("unqualified context")
    # Caller mode/flags are restored on success and on an exception, with
    # multiple PyTorch thread counts. No model-side thread-setting operation.
    lib = fenv()
    lib.fegetenv.argtypes, lib.fegetenv.restype = [C.c_void_p], C.c_int

    def state():
        buf = (C.c_longdouble * 32)()
        assert lib.fegetenv(C.byref(buf)) == 0
        return bytes(buf)[:32]

    original = (C.c_longdouble * 32)()
    assert lib.fegetenv(C.byref(original)) == 0
    threads = torch.get_num_threads()
    env_rows = []
    try:
        for workers in (1, 4):
            torch.set_num_threads(workers)
            for rounding in (0, 0x400, 0x800, 0xC00):
                for flush in (False, True):
                    assert lib.fesetround(rounding) == 0
                    assert torch.set_flush_denormal(flush)
                    before = state()
                    expected = partials(x)
                    assert state() == before, (workers, rounding, flush, "model")
                    check(p, expected)
                    assert state() == before, (workers, rounding, flush, "check")
                    try:
                        with environment():
                            raise RuntimeError("intentional")
                    except RuntimeError:
                        pass
                    assert state() == before, (workers, rounding, flush, "exception")
                    assert torch.get_num_threads() == workers
                    env_rows.append(
                        {
                            "threads": workers,
                            "rounding": rounding,
                            "flush": flush,
                            "restored": True,
                        }
                    )
    finally:
        torch.set_num_threads(threads)
        assert lib.fesetenv(C.byref(original)) == 0
    # An actual first-use selection must reject corrupted model expectations
    # before timing or touching the caller's destination.
    scratch = torch.full((5, 9), 12345.0, device="cuda")
    saved = tensor_sha(scratch)
    plans, rejected = candidates(
        obj.semantic,
        (65539, 5),
        (x, scratch),
        target,
        reduction_context=obj.reduction_context,
    )
    corrupt = partials(x)
    corrupt[0, 0] += 1e6
    with patch("stage5.reference512_oracle.partials", return_value=corrupt):
        try:
            obj.select(
                "reference512-negative-control",
                {},
                plans,
                rejected,
                (65539, 5),
                (x, scratch),
                obj.compile(obj.baseline_plan),
            )
        except AssertionError as error:
            assert "frozen arithmetic model" in str(error)
        else:
            raise AssertionError("corrupted oracle was accepted")
    assert tensor_sha(scratch) == saved
    evidence.state("after_reference512_controls")
    save(
        "controls",
        {
            "complete": True,
            "fp_environment": env_rows,
            "caller_isolation": True,
            "corrupted_oracle_rejected": True,
            "environment": evidence.inventory(),
        },
    )
    print(
        "FP state32 combinations, admission/context and corruption controls PASS",
        flush=True,
    )


def corpus():
    torch = setup()
    from stage5.artifact import CompiledKernel
    from stage5.reference512_oracle import partials, check
    from stage5.reduction_codegen_mechanisms import Component
    from stage5.reduction_merge_qualification import core_inputs, extra_inputs, row
    from stage5.reduction_policy import qualify_case
    from stage5.long_row_analyze import reference_for_extent
    from stage5.reference_reduction_probe import tensor_sha, sha
    from stage5.plan import catalog
    from stage1 import evidence

    source = ROOT / "artifacts/reduction-structure"
    originals = {
        r["case"]: r
        for r in json.loads(
            (ROOT / "artifacts/admission-boundary/results.json").read_text()
        )["rows"]
    }
    references = {
        f"triton-{h}": json.loads(
            (ROOT / f"artifacts/long-row/triton-{h}.json").read_text()
        )
        for h in ("short", "mid", "medium", "million")
    }
    references["triton-large"] = json.loads((source / "triton-r0.json").read_text())
    extra = json.loads((source / "rounding/new-triton-reference.json").read_text())
    plan = next(p for p in catalog() if p.strategy == "plane_shared_reference512")
    groups = {}
    for dtype in ("float32", "float16", "bfloat16"):
        frozen = Component(source / f"codegen/uniform/{dtype}")
        groups[dtype] = (
            CompiledKernel(frozen.semantic, 0, plan.to_dict()),
            Component(source / f"rounding/topology/kernels/{dtype}/fixed_final"),
        )
    checks = []
    torch.set_num_threads(4)
    for label, x in chain(core_inputs(torch), extra_inputs(torch)):
        b, n = x.shape
        previous = originals[label]
        assert tensor_sha(x) == previous["input_sha256"], label
        producer, final = groups[str(x.dtype).removeprefix("torch.")]
        p = torch.empty((b, 9), device="cuda")
        y = torch.empty(b, dtype=x.dtype, device="cuda")
        producer((n, b), x, p)
        final((b,), p, y)
        assert tensor_sha(p) == previous["hashes"]["reference512"]["partial"], label
        assert tensor_sha(y) == previous["hashes"]["reference512"]["output"], label
        check(p, partials(x))
        reference = next(
            r
            for r in (
                extra
                if label.startswith("new_")
                else references[reference_for_extent(n)]
            )["rows"]
            if r["case"] == label
        )
        outcome = qualify_case(row(torch, label, x, y), reference)
        assert outcome["passes"], label
        checks.append(
            {
                "case": label,
                "input_sha256": tensor_sha(x),
                "partial_sha256": tensor_sha(p),
                "output_sha256": tensor_sha(y),
                "qualification": outcome,
            }
        )
    assert len(checks) == 148
    evidence.state("after_reference512_corpus")
    save(
        "corpus",
        {
            "complete": True,
            "checks": checks,
            "harness_sha256": sha(__file__),
            "policy_changed": False,
            "environment": evidence.inventory(),
        },
    )
    print(
        "148 exact inputs, GPU partial/final hashes, row-wise CPU model and full policy PASS",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("init", "reload", "controls", "corpus"))
    args = parser.parse_args()
    {
        "init": lambda: cache("init"),
        "reload": lambda: cache("reload"),
        "controls": controls,
        "corpus": corpus,
    }[args.mode]()
