"""Bounded production admission, cache, generated-graph and timing controls.

No compiler/selector monkeypatch or artifact substitution. The frozen-producer
graph control uses a test scheduling subclass solely to emit existing per-kernel
policies: reference512_diagnostic for the producer, first_use for its final.
Ordinary backend graphs are checked separately with their normal producer.
"""

import argparse
import copy
import ctypes as C
from itertools import chain
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts/reduction-structure"
OUT = SOURCE / "final/promotion"


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2))


def semantic(dtype="float32"):
    return json.loads(
        (
            SOURCE / f"final/validated-plane/candidate/{dtype}/descriptor.json"
        ).read_text()
    )["semantic"]


def setup():
    from stage1 import evidence

    evidence.guard()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache/final-promotion")
    from stage1.overlay import install

    install()
    import torch

    torch.set_num_threads(1)
    return torch


def controls():
    torch = setup()
    from stage0.bridge import nvrtc_compile
    from stage5 import artifact, runtime
    from stage5.final_plan import STRATEGY, semantic_reason
    from stage5.legality import candidates, structural_reason
    from stage5.plan import catalog, PhysicalCandidate
    from stage5.reduction_codegen_checks import compile_packet
    from stage5.reduction_codegen_mechanisms import Component
    from stage5.reduction_final_probe import final_model
    from stage5.reduction_merge_qualification import core_inputs, extra_inputs
    from stage5.reference_reduction_probe import sha, tensor_sha
    from stage1 import evidence

    assert json.loads((OUT / "cpu.json").read_text())["complete"]
    plan = next(p for p in catalog() if p.strategy == STRATEGY)
    assert (
        plan.identity
        != PhysicalCandidate(
            "independent_planes", (32, 4, 1), outputs_per_workgroup=4, plane_width=32
        ).identity
    )
    identity, compiled = [], {}
    includes = [str(Path(torch.__file__).parent.parent / "nvidia/cu13/include")]
    for dtype in ("float32", "float16", "bfloat16"):
        old = SOURCE / f"final/validated-plane/candidate/{dtype}"
        packet = json.loads((old / "descriptor.json").read_text())
        packet["physical_plan"] = plan.to_dict()
        # Same symbol allows exact bytes to prove reuse through ordinary cpp.
        dest = OUT / "identity" / dtype
        run = compile_packet(packet, "cpp", dest)
        assert run.returncode == 0, run.stderr
        nvrtc_compile((dest / "kernel.cu").read_text(), dest, 0, includes)
        matches = {
            n: sha(dest / n) == sha(old / n)
            for n in ("cubecl.ir", "kernel.cu", "kernel.ptx")
        }
        assert all(matches.values()), (dtype, matches)
        identity.append({"dtype": dtype, "matches_qualified_probe": matches})
        # Exercise normal artifact/cache construction and its actual new symbol.
        compiled[dtype] = artifact.CompiledKernel(semantic(dtype), 0, plan.to_dict())

    native = C.CDLL(str(ROOT.parent / "stage0/libstage0_launcher.so"))
    native.stage0_kernel_launches.restype = C.c_uint64
    direct, rejected, corpus = [], [], []
    target = compiled["float32"].kernel.resources
    for dtype, kernel in compiled.items():
        td = getattr(torch, dtype)
        baseline = Component(SOURCE / f"rounding/topology/kernels/{dtype}/fixed_final")
        for b in (1, 3, 4, 5, 8, 31, 32, 33, 64, 257):
            torch.manual_seed(9201 + b)
            storage = torch.randn(b * 9 + 2, device="cuda")
            p = storage[1:-1].view(b, 9)
            if b >= 31:
                p[:9].zero_()
                for lane in range(9):
                    p[lane, lane] = float("nan")
                p[9].fill_(-0.0)
                p[10].fill_(2**-149)
                p[11, 0], p[12, 0] = float("inf"), -float("inf")
                p[13, 0], p[13, 8] = float("inf"), -float("inf")
                p[14, :3] = torch.tensor([1e8, 1, -1e8], device="cuda")
            stores = [
                torch.full((b + 2,), 12345.0, device="cuda", dtype=td) for _ in range(2)
            ]
            for k, y in zip((baseline, kernel), stores):
                k((b,), p, y[1:-1])
            assert tensor_sha(stores[0]) == tensor_sha(stores[1]), (dtype, b)
            model = (
                final_model(p.cpu()).to(td).clone(memory_format=torch.contiguous_format)
            )
            gpu = stores[1][1:-1].cpu()
            # CPU and CUDA can encode NaNs differently; GPU-to-frozen parity
            # above remains byte-exact, including those encodings.
            torch.testing.assert_close(model, gpu, rtol=0, atol=0, equal_nan=True)
            finite = torch.isfinite(model)
            assert tensor_sha(model[finite]) == tensor_sha(gpu[finite])
            assert torch.equal(
                torch.signbit(model[~torch.isnan(model)]),
                torch.signbit(gpu[~torch.isnan(gpu)]),
            )
            assert (
                stores[1][[0, -1]].tolist()
                == torch.tensor([12345.0] * 2, dtype=td).tolist()
            )
            direct.append(
                {
                    "dtype": dtype,
                    "outer": b,
                    "gpu_frozen_bitwise": True,
                    "model_finite_bitwise_and_classes": True,
                }
            )
        # Warm descriptor/layout caches must not suppress any live ABI checks.
        p, y = (
            torch.randn((5, 9), device="cuda"),
            torch.empty(5, device="cuda", dtype=td),
        )
        kernel((5,), p, y)
        before = native.stage0_kernel_launches()
        try:
            kernel((0,), p[:0], y[:0])
        except ValueError as error:
            rejected.append(
                {
                    "dtype": dtype,
                    "runtime": "zero violates positive scalar domain",
                    "reason": str(error),
                }
            )
        else:
            raise AssertionError("positive scalar domain was bypassed")
        assert native.stage0_kernel_launches() == before
        for label, args in (
            ("shape", (p[:4], y)),
            ("stride", (torch.empty_strided((5, 9), (10, 1), device="cuda"), y)),
            ("dtype", (p.to(torch.float16), y)),
            ("device", (p.cpu(), y)),
            ("alias", (p, p.flatten()[:5] if td == torch.float32 else y.double())),
        ):
            before = native.stage0_kernel_launches()
            try:
                kernel((5,), *args)
            except ValueError as error:
                rejected.append(
                    {"dtype": dtype, "runtime": label, "reason": str(error)}
                )
            else:
                raise AssertionError(label)
            assert before == native.stage0_kernel_launches()
        # Views with offsets and a nondefault current stream remain supported.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            p = torch.randn(5 * 9 + 2, device="cuda")[1:-1].view(5, 9)
            y = torch.empty(7, device="cuda", dtype=td)[1:-1]
            kernel((5,), p, y)
            expected = final_model(p.cpu()).to(td)
        stream.synchronize()
        assert tensor_sha(y) == tensor_sha(expected)
        assert kernel.kernel._call_pointers is None

    for label, x in chain(core_inputs(torch), extra_inputs(torch)):
        dtype = str(x.dtype).removeprefix("torch.")
        b, n = x.shape
        producer = Component(SOURCE / f"codegen/uniform/{dtype}")
        baseline = Component(SOURCE / f"rounding/topology/kernels/{dtype}/fixed_final")
        p = torch.empty((b, 9), device="cuda")
        a, z = (torch.empty(b, device="cuda", dtype=x.dtype) for _ in range(2))
        producer((n, b), x, p)
        baseline((b,), p, a)
        compiled[dtype]((b,), p, z)
        assert tensor_sha(a) == tensor_sha(z), label
        corpus.append(
            {
                "case": label,
                "input_sha256": tensor_sha(x),
                "output_sha256": tensor_sha(z),
                "bitwise": True,
            }
        )

    def mutate(label, fn):
        d = semantic()
        fn(d)
        assert semantic_reason(d), label
        # Structural rejection happens before choosing/compiling any candidate.
        assert structural_reason(plan, d, (5,), (), target)
        dest = OUT / "rejected" / label
        packet = {
            "schema": 5,
            "semantic": d,
            "physical_plan": plan.to_dict(),
            "entrypoint": "cubecl5_rejected",
        }
        run = compile_packet(packet, "cpp", dest)
        assert run.returncode, label
        rejected.append(
            {"descriptor": label, "python": semantic_reason(d), "native": run.stderr}
        )

    for n in (0, 1, 8, 10, 16, 32, 33):
        mutate(
            f"extent{n}",
            lambda d, n=n: d.update(reduction_extent={"op": "constant", "value": n}),
        )
    mutate("dynamic", lambda d: d.update(reduction_extent={"op": "scalar", "slot": 0}))
    mutate("owner", lambda d: d["ops"][-1].update(iteration_scope="reduction"))
    mutate("epilogue", lambda d: d["ops"][-1].update(args=["tmp0"]))
    mutate("input-half", lambda d: d["buffers"][0].update(dtype="float16"))
    mutate("output-double", lambda d: d["buffers"][1].update(dtype="float64"))
    mutate("accumulator", lambda d: d["ops"][1].update(accumulator_dtype="float64"))
    mutate("address", lambda d: d["ops"][0].update(index={"op": "reduction"}))
    mutate(
        "stride",
        lambda d: d["buffers"][0].update(
            stride=[{"op": "constant", "value": 10}, {"op": "constant", "value": 1}]
        ),
    )
    mutate("effect", lambda d: d.update(effects={"mutation": True}))
    assert structural_reason(plan, semantic(), (5,), (), {**target, "plane_width": 64})
    assert structural_reason(
        plan, semantic(), (5,), (), {**target, "device_max_threads": 64}
    )
    # A dynamic generic reduction must still have ordinary legal fallbacks.
    from stage5.fixtures import captured

    d = captured("row_sum", "float32")
    admitted, _ = candidates(
        d,
        (9, 5),
        (torch.randn((5, 9), device="cuda"), torch.empty(5, device="cuda")),
        target,
    )
    assert plan not in admitted and any(
        p.strategy == "shared_tree_reduction_baseline" for p in admitted
    )
    from stage5.reduction_policy import qualified_for_runtime

    assert not qualified_for_runtime(semantic(), plan)
    evidence.state("after_final_plan_controls")
    write(
        "controls.json",
        {
            "complete": True,
            "identity": identity,
            "direct": direct,
            "corpus": corpus,
            "rejected": rejected,
            "target": target,
            "coarse_guard_excluded": True,
            "fallback_preserved": True,
        },
    )
    print(
        "production final controls PASS",
        len(corpus),
        len(direct),
        len(rejected),
        flush=True,
    )


def cache(mode):
    torch = setup()
    from stage5 import runtime, artifact
    from stage5.final_plan import STRATEGY
    from stage5.reference_reduction_probe import tensor_sha
    from stage1 import evidence

    result = []
    torch.manual_seed(108)
    for dtype in ("float32", "float16", "bfloat16"):
        d = semantic(dtype)
        obj = runtime.compile_kernel(d, 0, policy="first_use")
        p = torch.randn((64, 9), device="cuda")
        y = torch.empty(64, device="cuda", dtype=getattr(torch, dtype))
        obj((64,), p, y)
        first = tensor_sha(y)
        before = obj.selection_hits
        obj((64,), p, y)
        assert obj.selection_hits > before and tensor_sha(y) == first
        # Production candidate was actually compiled/qualified by cold tuning,
        # or exists in the validated persistent catalog on a fresh reload.
        catalogs = [
            json.loads((ROOT / "cache/tuning" / key / "selection.json").read_text())
            for key in obj.selections
        ]
        assert all(
            any(c["plan"]["strategy"] == STRATEGY for c in s["candidates"])
            for s in catalogs
        )
        result.append(
            {
                "dtype": dtype,
                "selected": obj.last.kernel.plan.to_dict(),
                "compiled": [p.to_dict() for p in obj.compiled],
                "selection_keys": list(obj.selections),
                "warm_hits": obj.selection_hits,
                "output_sha256": first,
            }
        )
    if mode == "reload":
        assert artifact.COMPILER_INVOCATIONS == 0
        assert any(e["event"] == "tuning_cache_hit" for e in runtime.EVENTS)
    evidence.state("after_final_plan_cache")
    write(
        f"cache-{mode}.json",
        {
            "complete": True,
            "compiler_invocations": artifact.COMPILER_INVOCATIONS,
            "rows": result,
            "events": runtime.EVENTS,
        },
    )
    print("production final cache", mode, "PASS", flush=True)


def transitions():
    torch = setup()
    from stage5 import runtime
    from stage5.final_plan import STRATEGY
    from stage5.reference_reduction_probe import tensor_sha

    rows, seen = [], {}
    for n in (9, 8, 9, 10, 9):

        def replace(v):
            if v == {"op": "constant", "value": 9}:
                return {"op": "constant", "value": n}
            if isinstance(v, dict):
                return {k: replace(x) for k, x in v.items()}
            if isinstance(v, list):
                return [replace(x) for x in v]
            return v

        d = replace(semantic())
        obj = runtime.compile_kernel(d, 0, policy="first_use")
        if n in seen:
            assert seen[n] is obj
        seen[n] = obj
        p, y = torch.randn((64, n), device="cuda"), torch.empty(64, device="cuda")
        obj((64,), p, y)
        torch.testing.assert_close(y, p.sum(-1), rtol=2e-5, atol=2e-5)
        admitted = any(
            p.strategy == STRATEGY for plans, _ in obj._legality.values() for p in plans
        )
        assert admitted == (n == 9)
        assert n == 9 or obj.last.kernel.plan.strategy != STRATEGY
        rows.append(
            {
                "extent": n,
                "eligible": admitted,
                "winner": obj.last.kernel.plan.to_dict(),
                "selection_keys": list(obj.selections),
                "output_sha256": tensor_sha(y),
            }
        )
    assert len({key for obj in seen.values() for key in obj.selections}) == 3
    write("transitions.json", {"complete": True, "rows": rows})
    print("production final guard/cache transitions PASS", flush=True)


def counter():
    torch = setup()
    from stage5.artifact import CompiledKernel
    from stage5.final_plan import STRATEGY
    from stage5.plan import catalog, PhysicalCandidate

    plan = next(p for p in catalog() if p.strategy == STRATEGY)
    kernels = [
        CompiledKernel(semantic(), 0, p.to_dict())
        for p in (PhysicalCandidate("shared_tree_reduction_baseline"), plan)
    ]
    p, y = torch.randn((64, 9), device="cuda"), torch.empty(64, device="cuda")
    for k in kernels:
        k((64,), p, y)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for k in kernels:
        k((64,), p, y)
    torch.cuda.cudart().cudaProfilerStop()
    write(
        "counter-kernels.json",
        [
            {
                "entrypoint": k.entrypoint,
                "directory": str(k.directory),
                "attributes": k.kernel.attributes,
                "resources": k.kernel.resources,
            }
            for k in kernels
        ],
    )


def graphs():
    torch = setup()
    from functools import partial
    from torch._inductor import config
    from stage4 import backend as semantic_backend
    from stage5 import runtime
    from stage5.backend import compile_backend, CubeCLScheduling
    from stage5.final_plan import semantic_reason, STRATEGY
    from stage5.index_masks import widen_mask_indices
    from stage5.attribution import inventory
    from stage5.reference_reduction_probe import tensor_sha, sha
    from stage1 import evidence

    assert json.loads((OUT / "controls.json").read_text())["complete"]
    torch._dynamo.config.recompile_limit = 256
    result = {
        "complete": False,
        "environment": inventory(torch, "cubecl"),
        "cases": [],
        "scope": "ordinary first_use generated graphs; test scheduling only selects existing policies for frozen-producer controls; no artifact substitution; CUPTI kernel durations, not host latency",
    }
    # Exercise ordinary backend emission of the final itself with a static
    # reduction axis and a dynamic outer dimension, independent of split choice.
    runtime.CACHE.clear()
    direct_input = torch.randn((64, 9), device="cuda")
    torch._dynamo.mark_static(direct_input, 1)
    direct_graph = torch.compile(
        lambda z: z.sum(-1), backend=compile_backend, fullgraph=True, dynamic=True
    )
    direct_output = direct_graph(direct_input)
    direct_objects = list(runtime.CACHE.values())
    assert len(direct_objects) == 1
    direct_obj = direct_objects[0]
    assert semantic_reason(direct_obj.semantic) is None
    assert any(
        p.strategy == STRATEGY
        for plans, _ in direct_obj._legality.values()
        for p in plans
    )
    torch.testing.assert_close(
        direct_output, direct_input.sum(-1), rtol=2e-5, atol=2e-5
    )
    result["ordinary_static9_graph"] = {
        "directory": str(direct_obj.last.directory),
        "selected": direct_obj.last.kernel.plan.to_dict(),
        "eligible": True,
    }

    def frozen_backend(gm, inputs, *, final_policy):
        class Scheduling(CubeCLScheduling):
            def define_kernel(self, src_code, node_schedule, kernel):
                self.tuning = (
                    final_policy
                    if semantic_reason(kernel.descriptor) is None
                    else "reference512_diagnostic"
                )
                return super().define_kernel(src_code, node_schedule, kernel)

        previous = config._pre_fusion_custom_pass

        def prepare(nodes):
            return widen_mask_indices(previous(nodes) if previous else nodes)

        with config.patch({"_pre_fusion_custom_pass": prepare}):
            return semantic_backend.compile_backend(
                gm,
                inputs,
                scheduling=partial(Scheduling, tuning=final_policy),
                backend_name="cwk9_frozen_producer_" + final_policy,
                choices_class=None,
            )

    def capture(tag, fn, names, round_id, shape, count=16):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(count):
                fn()
        path = OUT / f"trace-{shape[0]}-{shape[1]}-r{round_id}-{tag}.json"
        prof.export_chrome_trace(str(path))
        events = sorted(
            (
                e
                for e in json.loads(path.read_text())["traceEvents"]
                if e.get("cat") == "kernel"
            ),
            key=lambda e: e["ts"],
        )
        assert [e["name"] for e in events] == names * count, (
            tag,
            names,
            [e["name"] for e in events],
        )
        stride = len(names)
        return {
            "variant": tag,
            "round": round_id,
            "durations_us": [
                [e["dur"] for e in events[i : i + stride]]
                for i in range(0, len(events), stride)
            ],
            "trace": str(path),
            "sha256": sha(path),
        }

    for b, n in ((8, 1048576), (64, 65539)):
        torch.manual_seed(7301)
        x = torch.randn((b, n), device="cuda")
        functions, records, objects = {}, {}, {}
        # Normal production selection with its normal compensated producer.
        for dtype in ("float32", "float16", "bfloat16"):
            runtime.CACHE.clear()
            f = torch.compile(
                lambda z: z.sum(-1),
                backend=compile_backend,
                fullgraph=True,
                dynamic=True,
            )
            data = x.to(getattr(torch, dtype))
            z = f(data)
            torch.testing.assert_close(
                z,
                data.sum(-1),
                rtol=0.016
                if dtype == "bfloat16"
                else 0.002
                if dtype == "float16"
                else 2e-5,
                atol=0.008
                if dtype == "bfloat16"
                else 0.001
                if dtype == "float16"
                else 2e-5,
            )
            final_objects = [
                o for o in runtime.CACHE.values() if semantic_reason(o.semantic) is None
            ]
            # The existing guarded planner may choose another split count.
            # Only a matching descriptor is newly eligible; no planner change.
            for obj in runtime.CACHE.values():
                admitted = any(
                    p.strategy == STRATEGY
                    for plans, _ in obj._legality.values()
                    for p in plans
                )
                assert admitted == (semantic_reason(obj.semantic) is None)
            # A different split count is an ordinary fallback control, not a
            # reason to alter the planner to manufacture final9 eligibility.
            # Record actual emitted policies, choices and candidate evidence.
            records["normal-" + dtype] = [
                {
                    "directory": str(o.last.directory),
                    "plan": o.last.kernel.plan.to_dict(),
                    "policy": o.policy,
                    "entrypoint": o.last.entrypoint,
                    "selection_keys": list(o.selections),
                }
                for o in runtime.CACHE.values()
            ]
            if dtype == "float32":
                functions["normal"] = lambda f=f, data=data: f(data)
                objects["normal"] = list(runtime.CACHE.values())
        # Two real generated wrappers share the same frozen partial bytes.
        for tag, policy in (("frozen", "baseline"), ("promoted", "first_use")):
            runtime.CACHE.clear()
            f = torch.compile(
                lambda z: z.sum(-1),
                backend=partial(frozen_backend, final_policy=policy),
                fullgraph=True,
                dynamic=True,
            )
            f(torch.randn((8, 65539), device="cuda"))
            z = f(x)
            functions[tag] = lambda f=f: f(x)
            objects[tag] = list(runtime.CACHE.values())
            records[tag] = [
                {
                    "directory": str(o.last.directory),
                    "plan": o.last.kernel.plan.to_dict(),
                    "policy": o.policy,
                    "entrypoint": o.last.entrypoint,
                    "selection_keys": list(o.selections),
                }
                for o in objects[tag]
            ]
            assert len(records[tag]) == 2
        assert tensor_sha(functions["frozen"]()) == tensor_sha(functions["promoted"]())
        for name in ("cubecl.ir", "kernel.cu", "kernel.ptx"):
            a = next(
                o.last.directory
                for o in objects["frozen"]
                if o.last.kernel.plan.strategy == "plane_shared_reference512"
            )
            z = next(
                o.last.directory
                for o in objects["promoted"]
                if o.last.kernel.plan.strategy == "plane_shared_reference512"
            )
            assert (a / name).read_bytes() == (z / name).read_bytes()
            retained = SOURCE / "codegen/uniform/float32" / name
            # Retained component has its own symbol; exact graph partial bytes
            # are independently checked against the prior timed graph below.
            assert retained.is_file()
        old = json.loads((SOURCE / "final/graphs/timing.json").read_text())
        retained_partial = Path(
            next(c for c in old["cases"] if c["shape"] == [b, n])["artifacts"][
                "frozen"
            ][0]["directory"]
        )
        assert all(
            (a / name).read_bytes() == (retained_partial / name).read_bytes()
            for name in ("cubecl.ir", "kernel.cu", "kernel.ptx")
        )
        # Isolated final timings use the candidate itself even if the ordinary
        # empirical selector selects a different eligible plan in this capture.
        from stage5.artifact import CompiledKernel
        from stage5.plan import catalog, PhysicalCandidate

        plan = next(p for p in catalog() if p.strategy == STRATEGY)
        direct = CompiledKernel(semantic(), 0, plan.to_dict())
        tree = CompiledKernel(
            semantic(), 0, PhysicalCandidate("shared_tree_reduction_baseline").to_dict()
        )
        p, y = torch.randn((b, 9), device="cuda"), torch.empty(b, device="cuda")
        functions["isolated-tree"] = lambda: tree((b,), p, y)
        functions["isolated-plane"] = lambda: direct((b,), p, y)
        names = {
            tag: [
                r["entrypoint"]
                for r in records[tag if tag != "normal" else "normal-float32"]
            ]
            for tag in ("normal", "frozen", "promoted")
        }
        names.update(
            {"isolated-tree": [tree.entrypoint], "isolated-plane": [direct.entrypoint]}
        )
        # Actual paths execute only for compilation/module readiness before
        # capture; no synthetic GPU activity or allocator/memory primers.
        for _ in range(10):
            for f in functions.values():
                f()
        torch.cuda.synchronize()
        captures = []
        labels = list(functions)
        for r in range(6):
            order = labels[r % len(labels) :] + labels[: r % len(labels)]
            if r >= 3:
                order.reverse()
            for tag in order:
                capture_row = capture(tag, functions[tag], names[tag], r, (b, n))
                capture_row["order"] = order
                captures.append(capture_row)
        summary = {
            tag: {
                "median_gpu_sum_us": statistics.median(
                    sum(cell)
                    for c in captures
                    if c["variant"] == tag
                    for cell in c["durations_us"]
                ),
                "round_medians_us": [
                    statistics.median(
                        sum(cell)
                        for c in captures
                        if c["variant"] == tag and c["round"] == r
                        for cell in c["durations_us"]
                    )
                    for r in range(6)
                ],
            }
            for tag in labels
        }
        result["cases"].append(
            {
                "shape": [b, n],
                "input_sha256": tensor_sha(x),
                "records": records,
                "captures": captures,
                "summary": summary,
                "partial_byte_identity": True,
                "frozen_pair_outputs_bitwise": True,
            }
        )
        write("graphs.json", result)
        print("production generated graph", b, n, summary, flush=True)
    evidence.state("after_final_plan_graphs")
    result["complete"] = True
    write("graphs.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("controls", "init", "reload", "graphs", "transitions", "counter"),
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    (OUT / f"harness-{args.mode}.py").write_bytes(Path(__file__).read_bytes())
    if args.mode == "controls":
        controls()
    elif args.mode == "graphs":
        graphs()
    elif args.mode == "transitions":
        transitions()
    elif args.mode == "counter":
        counter()
    else:
        cache(args.mode)


if __name__ == "__main__":
    main()
