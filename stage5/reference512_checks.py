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
    from stage5.legality import candidates
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
    try:
        obj.select(
            "reference512-missing-fallback-control",
            {},
            [plan],
            [],
            (65539, 5),
            (x, scratch),
            obj.compile(obj.baseline_plan),
        )
    except ValueError as error:
        assert "qualified fallback" in str(error)
    else:
        raise AssertionError("missing fallback was not rejected")
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        chosen = obj.select(
            "reference512-deterministic-control",
            {},
            plans,
            rejected,
            (65539, 5),
            (x, scratch),
            obj.compile(obj.baseline_plan),
        )
        assert chosen.strategy == "shared_tree_compensated"
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    corrupt = partials(x).clone()
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
            "missing_fallback_rejected": True,
            "deterministic_fallback": "shared_tree_compensated",
            "environment": evidence.inventory(),
        },
    )
    print(
        "FP state16 combinations, admission/context and corruption controls PASS",
        flush=True,
    )


def graphs():
    torch = setup()
    import sys
    from stage5 import runtime
    from stage5.backend import compile_backend
    from stage5.reference_reduction_probe import sha
    from stage1 import evidence

    observed, rows = [], []
    original = runtime.AutotunedKernel.__call__

    def observe(obj, scalars, *args):
        result = original(obj, scalars, *args)
        wrapper = Path(sys._getframe(1).f_code.co_filename)
        assert wrapper.is_relative_to(ROOT / "cache/reference512-admission"), wrapper
        saved = OUT / "wrappers" / wrapper.name
        saved.parent.mkdir(exist_ok=True)
        saved.write_bytes(wrapper.read_bytes())
        observed.append(
            {
                "role": obj.reduction_context and obj.reduction_context["role"],
                "plan": obj.last.kernel.plan.to_dict(),
                "semantic": obj.semantic,
                "wrapper": str(saved),
                "wrapper_sha256": sha(saved),
                "directory": str(obj.last.directory),
                "ptx_sha256": sha(obj.last.directory / "kernel.ptx"),
            }
        )
        return result

    f = torch.compile(
        lambda x: x.sum(-1), backend=compile_backend, fullgraph=True, dynamic=True
    )
    with patch.object(runtime.AutotunedKernel, "__call__", observe):
        for b, n, paired in (
            (8, 65539, True),
            (8, 65540, False),
            (8, 65539, True),
            (7, 65539, False),
            (8, 1048576, False),
            (64, 65539, False),
        ):
            observed.clear()
            y = f(torch.ones((b, n), device="cuda"))
            assert y.tolist() == [float(n)] * b
            assert len(observed) == 2
            assert (observed[0]["role"] == "producer") == paired
            assert (observed[1]["role"] == "final") == paired
            if paired:
                assert observed[1]["plan"]["strategy"] in (
                    "shared_tree_reduction_baseline",
                    "independent_planes_final9",
                )
            else:
                assert observed[0]["plan"]["strategy"] != "plane_shared_reference512"
            rows.append({"shape": [b, n], "paired": paired, "kernels": list(observed)})
        observed.clear()
        x = torch.zeros((5, 65539), device="cuda")
        x[0, [0, 256, 512]] = torch.tensor([1e8, 1, -1e8], device="cuda")
        x[1] = -x[0]
        x[2, [0, 256, 512]] = torch.tensor([2**24, 0.25, -(2**24)], device="cuda")
        values = f(x).tolist()
        assert values[:3] == [1.0, -1.0, 0.25], values
        assert [o["role"] for o in observed] == ["producer", "final"]
        rows.append(
            {
                "shape": [5, 65539],
                "adversary": True,
                "values": values,
                "kernels": list(observed),
            }
        )
    from stage5.reduction_codegen_mechanisms import Component
    from stage5.reference_reduction_probe import tensor_sha

    producer = next(
        o
        for o in runtime.CACHE.values()
        if o.reduction_context and o.reduction_context["role"] == "producer"
    )
    final = next(
        o
        for o in runtime.CACHE.values()
        if o.reduction_context and o.reduction_context["role"] == "final"
    )
    frozen_final = Component(
        ROOT
        / "artifacts/reduction-structure/rounding/topology/kernels/float32/fixed_final"
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = torch.randn(8 * 65539 + 1, device="cuda")[1:].view(8, 65539)
        ps = torch.full((8 * 9 + 2,), 12345.0, device="cuda")
        ys = torch.full((10,), 12345.0, device="cuda")
        p, y = ps[1:-1].view(8, 9), ys[1:-1]
        producer((65539, 8), x, p)
        final((8,), p, y)
        expected = torch.empty_like(y)
        frozen_final((8,), p, expected)
        assert tensor_sha(y) == tensor_sha(expected)
    stream.synchronize()
    assert ps[[0, -1]].tolist() == ys[[0, -1]].tolist() == [12345.0] * 2
    evidence.state("after_reference512_graphs")
    save(
        "graphs",
        {
            "complete": True,
            "rows": rows,
            "offset_stream_canaries": True,
            "environment": evidence.inventory(),
        },
    )
    print(
        "6 actual graph/neighbor transitions, invoked wrappers and split32/4 fallbacks PASS",
        flush=True,
    )


def specials():
    torch = setup()
    from stage5.reference512_plan import templates, context, STRATEGY
    from stage5.reference512_oracle import partials, check
    from stage5.runtime import compile_kernel
    from stage5.plan import catalog
    from stage5.reduction_codegen_mechanisms import Component
    from stage5.reference_reduction_probe import tensor_sha

    t = templates()
    obj = compile_kernel(
        t["producer"],
        0,
        reduction_context=context(t["producer"], t["final"], "producer"),
    )
    candidate = obj.compile(next(p for p in catalog() if p.strategy == STRATEGY))
    frozen = Component(ROOT / "artifacts/reduction-structure/codegen/uniform/float32")
    x = torch.zeros((5, 65539), device="cuda", dtype=torch.float32)
    x[0].fill_(-0.0)
    x[1, 0], x[2, 0], x[3, 0] = float("nan"), float("inf"), -float("inf")
    x[4, 0], x[4, 256] = float("inf"), -float("inf")
    a, b = (torch.empty((5, 9), device="cuda", dtype=torch.float32) for _ in range(2))
    candidate((65539, 5), x, a)
    frozen((65539, 5), x, b)
    assert tensor_sha(a) == tensor_sha(b)
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("cuda"):
            expected = partials(x)
            assert expected.device.type == "cpu" and expected.dtype == torch.float32
            check(a, expected)
            assert torch.empty(0).device.type == "cuda"
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(dtype)
    save(
        "specials",
        {
            "complete": True,
            "gpu_special_bits_unchanged": True,
            "oracle_classes_and_finite_bits": True,
            "default_dtype_device_preserved": True,
        },
    )
    print("special values, default dtype/device and frozen GPU bytes PASS", flush=True)


def identity():
    torch = setup()
    from stage0.bridge import nvrtc_compile
    from stage5.reduction_codegen_checks import compile_packet
    from stage5.reference_reduction_probe import sha

    source = ROOT / "artifacts/reduction-structure"
    includes = [str(Path(torch.__file__).parent.parent / "nvidia/cu13/include")]
    rows = []
    for dtype in ("float32", "float16", "bfloat16"):
        for family, original in (
            ("partial", source / f"codegen/uniform/{dtype}"),
            ("final", source / f"rounding/topology/kernels/{dtype}/fixed_final"),
        ):
            packet = json.loads((original / "descriptor.json").read_text())
            dest = OUT / "identity" / dtype / family
            result = compile_packet(packet, "cpp", dest)
            assert result.returncode == 0, result.stderr
            nvrtc_compile((dest / "kernel.cu").read_text(), dest, 0, includes)
            same = {
                name: sha(dest / name) == sha(original / name)
                for name in ("cubecl.ir", "kernel.cu", "kernel.ptx")
            }
            assert all(same.values()), (dtype, family, same)
            rows.append({"dtype": dtype, "family": family, "byte_identical": same})
    save("identity", {"complete": True, "rows": rows})
    print("6 frozen partial/final IR/CUDA/PTX identities PASS", flush=True)


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
    import faulthandler

    faulthandler.dump_traceback_later(15, repeat=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "init",
            "reload",
            "controls",
            "corpus",
            "graphs",
            "identity",
            "specials",
        ),
    )
    args = parser.parse_args()
    {
        "init": lambda: cache("init"),
        "reload": lambda: cache("reload"),
        "controls": controls,
        "corpus": corpus,
        "graphs": graphs,
        "identity": identity,
        "specials": specials,
    }[args.mode]()
