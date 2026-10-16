"""cwk.7: explicit plane-final substitution in real generated graphs.

The adapter lives only in this executable harness. No catalog, normal compiler
route, runtime policy, partial topology, or production warmup changes.
"""

import argparse
import copy
import ctypes as C
from contextlib import contextmanager
from itertools import chain
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts/reduction-structure"
PACKET = SOURCE / "final/graphs"
SIZES = (1, 4, 8, 9, 10, 16, 17, 31, 32)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def cpu_gate():
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.reduction_final_probe import final_model
    from stage5.reduction_topology_model import model
    from stage5.reduction_merge_qualification import core_inputs, extra_inputs, row
    from stage5.reduction_rounding_probe import finish_model
    from stage5.reduction_policy import qualify_case, VERSION
    from stage5.reference_reduction_probe import tensor_sha
    from stage5.long_row_analyze import reference_for_extent

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    frozen = json.loads((SOURCE / "rounding/topology/qualification.json").read_text())
    references = {
        "triton-" + h: json.loads((ROOT / f"artifacts/long-row/triton-{h}.json").read_text())
        for h in ("short", "mid", "medium", "million")
    }
    references["triton-large"] = json.loads((SOURCE / "triton-r0.json").read_text())
    extra = json.loads((SOURCE / "rounding/new-triton-reference.json").read_text())
    checks, neighbors = [], []
    for label, x in chain(core_inputs(torch), extra_inputs(torch)):
        m = model(x.cpu())
        out = final_model(m["partials"]).to(x.dtype)
        actual = row(torch, label, x, out)
        old = next(r for r in frozen["rows"] if r["case"] == label)
        assert actual == old, label
        assert tensor_sha(out) == tensor_sha(m["output"]), label
        ref = extra if label.startswith("new_") else references[reference_for_extent(x.shape[1])]
        check = qualify_case(actual, next(r for r in ref["rows"] if r["case"] == label))
        assert check["passes"], check
        checks.append(check)
        # Exercise the same tree over nearby sizes with the corpus partials.
        for n in SIZES:
            p = m["partials"].repeat(1, 4)[:, :n].contiguous()
            initialized = torch.zeros_like(p) + p
            candidate, baseline = final_model(p), finish_model(initialized)
            assert tensor_sha(candidate) == tensor_sha(baseline), (label, n)
            neighbors.append({"case": label, "extent": n, "input_sha256": tensor_sha(p),
                              "output_sha256": tensor_sha(candidate), "bitwise": True})
    # Lane-position and FP32 exceptional/rounding controls before GPU extension.
    direct = []
    for n in SIZES:
        p = torch.zeros((n + 8, n), dtype=torch.float32)
        for lane in range(n):
            p[lane, lane] = float("nan")
        p[n].fill_(-0.0)
        p[n + 1].fill_(2**-149)
        p[n + 2, 0] = float("inf")
        p[n + 3, 0] = -float("inf")
        if n > 1:
            p[n + 4, 0], p[n + 4, -1] = float("inf"), -float("inf")
        if n >= 3:
            p[n + 5, :3] = torch.tensor([1e8, 1, -1e8])
            p[n + 6, :3] = torch.tensor([2**24, 0.25, -(2**24)])
        out, old = final_model(p), finish_model(torch.zeros_like(p) + p)
        assert tensor_sha(out) == tensor_sha(old), n
        direct.append({"extent": n, "input": p.tolist(), "input_sha256": tensor_sha(p),
                       "output_sha256": tensor_sha(out), "bitwise": True})
    evidence.state("after_cwk7_cpu")
    write(PACKET / "cpu.json", {"complete": True, "policy": VERSION, "checks": checks,
                               "neighbors": neighbors, "direct": direct})
    print(len(checks), "corpus and", len(neighbors), "neighbor CPU checks pass", flush=True)


def compile_final(semantic, device, directory, *, plane=True):
    import torch
    from stage0.bridge import native_path
    from stage5.artifact import Kernel, digest
    from stage5.plan import PhysicalCandidate

    plan = PhysicalCandidate("independent_planes", (32, 4, 1),
                             outputs_per_workgroup=4, plane_width=32) if plane else PhysicalCandidate("shared_tree_reduction_baseline")
    packet = {"schema": 5, "semantic": semantic, "physical_plan": plan.to_dict()}
    packet["entrypoint"] = "cubecl5_" + digest(json.dumps(packet, sort_keys=True).encode())[:16]
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "descriptor.json", packet)
    run = subprocess.run([str(native_path("cubecl-stage0")), str(directory),
                          "cpp-probe-final-plane" if plane else "cpp",
                          str(directory / "descriptor.json")], capture_output=True, text=True)
    (directory / "compiler.log").write_text(run.stdout + run.stderr)
    run.check_returncode()
    return Kernel(directory, semantic, device, plan.to_dict())


def controls():
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage0.bridge import native_path, nvrtc_compile
    from stage5.reduction_codegen_checks import compile_packet
    from stage5.reference_reduction_probe import sha, tensor_sha

    cpu = json.loads((PACKET / "cpu.json").read_text())
    assert cpu["complete"] and len(cpu["checks"]) == 148 and len(cpu["neighbors"]) == 1332
    torch.set_num_threads(1)
    includes = [str(Path(torch.__file__).parent.parent / "nvidia/cu13/include")]
    identities, checks, rejected = [], [], []
    for dtype in ("float32", "float16", "bfloat16"):
        for tag, original, route in (
            ("partial", SOURCE / "codegen/uniform" / dtype, "cpp"),
            ("ordinary_final", SOURCE / "rounding/topology/kernels" / dtype / "fixed_final", "cpp"),
            ("plane", SOURCE / "final/validated-plane/candidate" / dtype, "cpp-probe-final-plane"),
        ):
            packet = json.loads((original / "descriptor.json").read_text())
            destination = PACKET / "identity" / dtype / tag
            run = compile_packet(packet, route, destination)
            assert run.returncode == 0, run.stderr
            nvrtc_compile((destination / "kernel.cu").read_text(), destination, 0, includes)
            matches = {n: sha(destination / n) == sha(original / n)
                       for n in ("cubecl.ir", "kernel.cu", "kernel.ptx")}
            assert all(matches.values()), (dtype, tag, matches)
            identities.append({"dtype": dtype, "tag": tag, "matches": matches})
        source = json.loads((SOURCE / "rounding/topology/kernels" / dtype / "fixed_final/descriptor.json").read_text())
        for n in SIZES:
            def replace(value):
                if value == {"op": "constant", "value": 9}:
                    return {"op": "constant", "value": n}
                if isinstance(value, dict):
                    return {k: replace(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [replace(v) for v in value]
                return value
            semantic = replace(source["semantic"])
            destination = PACKET / "neighbors" / dtype / str(n)
            candidate = compile_final(semantic, 0, destination / "plane")
            baseline = compile_final(semantic, 0, destination / "tree", plane=False)
            direct = torch.tensor(next(d["input"] for d in cpu["direct"] if d["extent"] == n), device="cuda")
            for b in (1, 3, 4, 5, 8, 64):
                torch.manual_seed(7301)
                data = torch.randn((b, n), device="cuda")
                for label, p in (("random", data), ("direct", direct)):
                    count = p.shape[0]
                    backing = torch.full((p.numel() + 2,), 12345.0, device="cuda")
                    view = backing[1:-1].view_as(p)
                    view.copy_(p)
                    outputs = [torch.full((count + 2,), 12345.0, device="cuda", dtype=getattr(torch, dtype)) for _ in range(2)]
                    for k, out in zip((baseline, candidate), outputs, strict=True):
                        k((count,), view, out[1:-1])
                    assert tensor_sha(outputs[0]) == tensor_sha(outputs[1]), (dtype, n, b, label)
                    assert tensor_sha(view) == tensor_sha(p)
                    expected = torch.tensor([12345.0] * 2, dtype=getattr(torch, dtype)).tolist()
                    assert outputs[1][[0, -1]].tolist() == expected
                    assert backing[[0, -1]].tolist() == [12345.0] * 2
                    checks.append({"dtype": dtype, "extent": n, "outputs": count, "kind": label,
                                   "input_sha256": tensor_sha(view), "output_sha256": tensor_sha(outputs[1][1:-1]),
                                   "bitwise": True, "canaries": True})
    base = json.loads((SOURCE / "final/validated-plane/candidate/float32/descriptor.json").read_text())
    bad = []
    for n in (0, 33, 64, -1):
        packet = copy.deepcopy(base)
        packet["semantic"]["reduction_extent"] = {"op": "constant", "value": n}
        bad.append((f"extent-{n}", packet))
    packet = copy.deepcopy(base)
    packet["semantic"]["reduction_extent"] = {"op": "scalar", "slot": 0}
    bad.append(("dynamic", packet))
    packet = copy.deepcopy(base)
    packet["physical_plan"]["workgroup"] = [32, 8, 1]
    packet["physical_plan"]["outputs_per_workgroup"] = 8
    bad.append(("eight-planes", packet))
    packet = copy.deepcopy(base)
    packet["semantic"]["ops"][-1]["iteration_scope"] = "reduction"
    bad.append(("non-owner", packet))
    packet = copy.deepcopy(base)
    packet["physical_plan"] = source["physical_plan"]
    bad.append(("wrong-plan", packet))
    for label, packet in bad:
        directory = PACKET / "rejections" / label
        run = compile_packet(packet, "cpp-probe-final-plane", directory)
        assert run.returncode and not (directory / "kernel.cu").exists(), (label, run.stderr)
        rejected.append({"case": label, "exit_code": run.returncode, "stderr": run.stderr, "no_target_code": True})
    evidence.state("after_cwk7_controls")
    write(PACKET / "controls.json", {"complete": True, "identities": identities,
                                    "gpu_neighbors": checks, "rejections": rejected,
                                    "compiler_sha256": sha(native_path("cubecl-stage0"))})
    print(len(checks), "GPU neighbor controls, 27 unchanged IR/CUDA/PTX files,", len(rejected), "rejections pass", flush=True)


@contextmanager
def graph_adapter(plane, destination):
    """Tag each runtime object once; baseline and plane graphs coexist safely."""
    from stage5 import runtime, artifact

    original_init = runtime.AutotunedKernel.__init__
    original_compile = runtime.AutotunedKernel.compile
    records = []

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._cwk7_plane = plane

    def compile(self, plan):
        is_final = (getattr(self, "_cwk7_plane", False)
                    and self.policy == "reference512_diagnostic"
                    and self.semantic.get("reduction_extent") == {"op": "constant", "value": 9}
                    and plan.strategy == "shared_tree_reduction_baseline")
        if not is_final:
            result = original_compile(self, plan)
        else:
            if plan not in self.compiled:
                # Keep the original semantic packet and validated Kernel ABI.
                result = artifact.CompiledKernel.__new__(artifact.CompiledKernel)
                result.semantic = copy.deepcopy(self.semantic)
                key = artifact.digest(json.dumps(self.semantic, sort_keys=True).encode())
                result.directory = destination / key
                result.kernel = compile_final(self.semantic, self.device, result.directory)
                result.entrypoint = result.kernel.manifest["entrypoint"]
                result.cache_hit = False
                result.load_compile_ms = 0
                self.compiled[plan] = result
            result = self.compiled[plan]
        if str(result.directory) not in {r["directory"] for r in records}:
            records.append({"directory": str(result.directory), "semantic": result.semantic,
                            "plan": result.kernel.plan.to_dict(), "substituted": is_final})
        return result

    runtime.AutotunedKernel.__init__ = init
    runtime.AutotunedKernel.compile = compile
    try:
        yield records
    finally:
        runtime.AutotunedKernel.__init__ = original_init
        runtime.AutotunedKernel.compile = original_compile


def qualification():
    """The existing 178-row whole-graph corpus, with timing explicitly disabled."""
    from stage5 import reference_reduction_probe as harness

    assert json.loads((PACKET / "cpu.json").read_text())["complete"]
    args = argparse.Namespace(impl="cubecl", history="short_first",
                              output=PACKET / "qualification.json", qualification_only=True,
                              cubecl_policy="reference512_diagnostic", cubecl_split_policy="guarded",
                              long_row=True, rounds=1, ncu_case=None)
    result = {"qualification_complete": False}
    try:
        # Import Torch only after the existing fresh-process Triton/GIL guard.
        from stage1 import evidence
        evidence.guard()
        with graph_adapter(True, PACKET / "qualification-kernels") as records:
            result = harness.run(args, result)
        result["substitutions"] = records
        old = json.loads((SOURCE / "codegen/graph-cubecl.json").read_text())
        assert len(result["rows"]) == len(old["rows"]) == 178
        for a, b in zip(result["rows"], old["rows"], strict=True):
            for key in ("case", "shape", "dtype", "input_sha256", "values", "aten", "fp64"):
                assert json.dumps(a[key]) == json.dumps(b[key]), (a["case"], key)
        assert result["numerical_failures"] == old["numerical_failures"]
        assert sum(r["substituted"] for r in records) > 0
        result["frozen_graph_rows_identical"] = True
        assert result["timings"] == []
    finally:
        write(args.output, result)
    print("178 generated-graph rows preserve frozen outputs/failures", flush=True)


def timing():
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage1.overlay import install
    from stage5 import runtime
    from stage5.backend import compile_backend
    from stage5.reduction_codegen_compare import DriverKernel
    from stage5.reference_reduction_probe import tensor_sha, sha
    from stage5.attribution import inventory

    assert json.loads((PACKET / "qualification.json").read_text())["frozen_graph_rows_identical"]
    torch.set_num_threads(1)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache/cwk7-graphs")
    install()
    torch._dynamo.config.recompile_limit = 256
    old = json.loads((SOURCE / "codegen/fixed-graph-triton-r0.json").read_text())
    result = {"complete": False, "cases": [], "environment": inventory(torch, "cubecl"),
              "scope": "real CubeCL generated wrappers; retained Triton generated two-kernel device sequence via DriverKernel; GPU durations only; no synthetic activity/warmup/primer"}
    write(PACKET / "timing.json", result)
    for b, n in ((8, 1048576), (64, 65539)):
        torch.manual_seed(7301)
        x = torch.randn((b, n), device="cuda")
        closures, objects, records = {}, {}, {}
        # Keep the adapter installed while both variants execute. Each object
        # retains the tag from its own creation, even after the cache is cleared.
        with graph_adapter(False, PACKET / f"timing-{b}-{n}") as all_records:
            for label, plane in (("frozen", False), ("plane", True)):
                runtime.CACHE.clear()
                def backend(gm, inputs):
                    return compile_backend(gm, inputs, tuning="reference512_diagnostic", split_policy="guarded")
                f = torch.compile(lambda z: z.sum(-1), backend=backend, fullgraph=True, dynamic=True)
                # Creation tag is changed explicitly only for this build.
                # Objects are created lazily during the first execution.
                original_init = runtime.AutotunedKernel.__init__
                def tagged(self, *a, **kw):
                    original_init(self, *a, **kw)
                    self._cwk7_plane = plane
                runtime.AutotunedKernel.__init__ = tagged
                try:
                    f(x)
                finally:
                    runtime.AutotunedKernel.__init__ = original_init
                closures[label] = f
                objects[label] = list(runtime.CACHE.values())
                records[label] = [{"directory": str(o.last.directory), "plan": o.last.kernel.plan.to_dict(),
                                   "entrypoint": o.last.kernel.manifest["entrypoint"]} for o in objects[label]]
            assert tensor_sha(closures["frozen"](x)) == tensor_sha(closures["plane"](x))
            output_sha = tensor_sha(closures["plane"](x))
            partials = [[r for r in records[tag] if r["plan"]["strategy"] == "plane_shared_reference512"] for tag in records]
            assert len(partials[0]) == len(partials[1]) == 1
            for name in ("cubecl.ir", "kernel.cu", "kernel.ptx"):
                assert (Path(partials[0][0]["directory"]) / name).read_bytes() == (Path(partials[1][0]["directory"]) / name).read_bytes()
            tref = next(t for t in old["timings"] if t["case"] == f"float32-{b}-{n}")
            assert tensor_sha(x) == tref["input_sha256"]
            tmp, y = torch.empty((b, 9), device="cuda"), torch.empty(b, device="cuda")
            drivers = []
            for i, k in enumerate(tref["selected_kernels"]):
                path = ROOT / "cache/reference-reduction/fixed-graph-triton-r0/triton" / k["cache_hash"] / (k["entrypoint"] + ".cubin")
                d = DriverKernel(path, k["entrypoint"], (512 if i == 0 else 64, 1, 1), 64 if i == 0 else 0)
                d.prepare(b * 9 if i == 0 else b, (x, tmp) if i == 0 else (tmp, y),
                          (C.c_int64(n), C.c_int32(b * 9), C.c_int32((n + 8) // 9), C.c_uint64(0), C.c_uint64(0)) if i == 0 else
                          (C.c_int32(b), C.c_int32(9), C.c_uint64(0), C.c_uint64(0)))
                drivers.append(d)
            def reference():
                for d in drivers:
                    d.batch(1)
                return y
            closures["triton_device_sequence"] = reference
            reference()
            assert tensor_sha(y) == output_sha
            captures = []
            labels = list(closures)
            # Normal execution of the actual evaluated paths for compilation and
            # module readiness. No GPU sleeps, clock anchors or memory primers.
            for _ in range(20):
                for f in closures.values():
                    f(x) if f is not reference else f()
            torch.cuda.synchronize()
            for round_id in range(6):
                order = labels[round_id % 3:] + labels[:round_id % 3]
                if round_id >= 3:
                    order.reverse()
                for tag in order:
                    f = closures[tag]
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(16):
                            f(x) if f is not reference else f()
                    path = PACKET / f"trace-{b}-{n}-r{round_id}-{tag}.json"
                    prof.export_chrome_trace(str(path))
                    events = sorted((e for e in json.loads(path.read_text())["traceEvents"] if e.get("cat") == "kernel"), key=lambda e: e["ts"])
                    names = [d.manifest["entrypoint"] for d in drivers] if f is reference else [r["entrypoint"] for r in records[tag]]
                    assert [e["name"] for e in events] == names * 16, (tag, [e["name"] for e in events])
                    assert tensor_sha(f() if f is reference else f(x)) == output_sha
                    captures.append({"round": round_id, "order": order, "variant": tag,
                                     "partial_us": [e["dur"] for e in events[::2]],
                                     "final_us": [e["dur"] for e in events[1::2]],
                                     "gap_us": [z["ts"] - a["ts"] - a["dur"] for a, z in zip(events[::2], events[1::2], strict=True)],
                                     "trace": str(path), "sha256": sha(path)})
            result["cases"].append({"shape": [b, n], "input_sha256": tensor_sha(x), "output_sha256": output_sha,
                                    "artifacts": records, "compiled": all_records, "captures": captures})
            write(PACKET / "timing.json", result)
        print("timed generated graphs", b, n, flush=True)
    evidence.state("after_cwk7_graph_timing")
    result["complete"] = True
    write(PACKET / "timing.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("cpu", "controls", "qualification", "timing"))
    args = parser.parse_args()
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    PACKET.mkdir(parents=True, exist_ok=True)
    archive = PACKET / ("harness-" + args.mode + ".py")
    if archive.exists():
        raise FileExistsError(archive)
    archive.write_bytes(Path(__file__).read_bytes())
    {"cpu": cpu_gate, "controls": controls, "qualification": qualification, "timing": timing}[args.mode]()


if __name__ == "__main__":
    main()
