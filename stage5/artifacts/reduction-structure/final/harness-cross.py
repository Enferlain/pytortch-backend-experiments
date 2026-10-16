"""One explicit final-plane candidate, with a CPU gate before any GPU candidate."""

import argparse
from itertools import chain
import json
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/reduction-structure/final"
SOURCE = ROOT / "artifacts/reduction-structure"


def final_model(partials):
    import torch
    from stage5.reduction_rounding_probe import butterfly

    assert partials.device.type == "cpu" and partials.shape[-1] == 9
    lanes = torch.zeros((*partials.shape[:-1], 32), dtype=torch.float32)
    # Retain the original +0 private initialization, including signed-zero rules.
    lanes[..., :9] = torch.zeros_like(partials) + partials
    return butterfly(lanes, (16, 8, 4, 2, 1))[..., 0]


def cpu_gate():
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.long_row_analyze import reference_for_extent
    from stage5.reduction_merge_qualification import core_inputs, extra_inputs, row
    from stage5.reduction_policy import qualify_case, VERSION
    from stage5.reduction_topology_model import model
    from stage5.reference_reduction_probe import tensor_sha

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    references = {f"triton-{h}": json.loads((ROOT / f"artifacts/long-row/triton-{h}.json").read_text())
                  for h in ("short", "mid", "medium", "million")}
    references["triton-large"] = json.loads((SOURCE / "triton-r0.json").read_text())
    extra = json.loads((SOURCE / "rounding/new-triton-reference.json").read_text())
    frozen = json.loads((SOURCE / "rounding/topology/qualification.json").read_text())
    checks = []
    for label, x in chain(core_inputs(torch), extra_inputs(torch)):
        m = model(x.cpu())
        candidate = final_model(m["partials"]).to(x.dtype)
        reference = next(r for r in (extra if label.startswith("new_") else references[reference_for_extent(x.shape[1])])["rows"] if r["case"] == label)
        actual = row(torch, label, x, candidate)
        previous = next(r for r in frozen["rows"] if r["case"] == label)
        assert actual["input_sha256"] == previous["input_sha256"]
        assert tensor_sha(candidate) == tensor_sha(m["output"]), label
        assert actual["values"] == previous["values"], label
        check = qualify_case(actual, reference)
        check["frozen_final_bitwise"] = True
        checks.append(check)
    result = {"complete": True, "policy": VERSION, "checks": checks,
              "failures": [c for c in checks if not c["passes"]],
              "candidate": "one 32-lane plane per output; same positive-zero initialization and descending offsets16,8,4,2,1",
              "gpu_candidate_implemented": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cpu-qualification.json").write_text(json.dumps(result, indent=2))
    assert not result["failures"]
    print(len(checks), "CPU-model cases qualify; all frozen final bytes identical", flush=True)


def gpu_probe(mechanism="plane"):
    from stage1 import evidence

    evidence.guard()
    import torch
    import ctypes as C
    from stage0.bridge import nvrtc_compile
    from stage5.plan import PhysicalCandidate
    from stage5.reduction_codegen_mechanisms import Component
    from stage5.reduction_codegen_compare import DriverKernel
    from stage5.reduction_merge_qualification import core_inputs, extra_inputs
    from stage5.reference_reduction_probe import sha, tensor_sha
    from stage5.attribution import inventory

    gate = json.loads((SOURCE / "final/cpu-qualification.json").read_text())
    assert gate["complete"] and not gate["failures"] and len(gate["checks"]) == 148
    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    original = SOURCE / "rounding/topology/kernels"
    uniform = SOURCE / "codegen/uniform"
    frozen_hashes = {str(p): sha(p) for root in (uniform, original) for p in root.rglob("*") if p.is_file()}
    plan = PhysicalCandidate("independent_planes", (32, 4, 1), outputs_per_workgroup=4, plane_width=32)
    kernels = {}
    for dtype in ("float32", "float16", "bfloat16"):
        directory = OUT / "candidate" / dtype
        directory.mkdir(parents=True, exist_ok=False)
        packet = json.loads((original / dtype / "fixed_final/descriptor.json").read_text())
        semantic = packet["semantic"]
        packet["physical_plan"] = plan.to_dict()
        packet["entrypoint"] = "cubecl5_final_plane_" + dtype
        (directory / "descriptor.json").write_text(json.dumps(packet, indent=2))
        route = "cpp-probe-final-plane" + ("-uniform" if mechanism == "uniform" else "")
        subprocess.run([str(ROOT.parent / "stage0/target/release/cubecl-stage0"), str(directory), route, str(directory / "descriptor.json")], check=True, stdout=subprocess.DEVNULL)
        includes = [str(Path(torch.__file__).parent.parent / "nvidia/cu13/include")]
        nvrtc_compile((directory / "kernel.cu").read_text(), directory, 0, includes)
        assert semantic == json.loads((original / dtype / "fixed_final/descriptor.json").read_text())["semantic"]
        kernels[dtype] = (Component(uniform / dtype), Component(original / dtype / "fixed_final"), Component(directory))

    checks = []

    def check(label, x):
        b, n = x.shape
        partial, final, candidate = kernels[str(x.dtype).removeprefix("torch.")]
        tmp_store = torch.full((b * 9 + 2,), 12345.0, device="cuda")
        tmp = tmp_store[1:-1].view(b, 9)
        outs = [torch.full((b + 2,), 12345.0, dtype=x.dtype, device="cuda") for _ in range(2)]
        partial((n, b), x, tmp)
        partial_hash = tensor_sha(tmp)
        for k, out in zip((final, candidate), outs, strict=True):
            k((b,), tmp, out[1:-1])
        assert torch.equal(outs[0].view(torch.uint8), outs[1].view(torch.uint8)), label
        assert tensor_sha(tmp) == partial_hash
        assert tmp_store[[0, -1]].tolist() == [12345.0] * 2
        assert outs[1][[0, -1]].tolist() == torch.tensor([12345.0] * 2, dtype=x.dtype).tolist()
        checks.append({"case": label, "input_sha256": tensor_sha(x), "partial_sha256": partial_hash,
                       "output_sha256": tensor_sha(outs[1][1:-1]), "bitwise_final": True, "canaries": True})

    for label, x in chain(core_inputs(torch), extra_inputs(torch)):
        check(label, x)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.zeros((5, 65539), dtype=dtype, device="cuda")
        x[0].fill_(-0.0)
        x[1, 0], x[2, 0], x[3, 0] = float("nan"), float("inf"), -float("inf")
        x[4, 0], x[4, 256] = float("inf"), -float("inf")
        check("specials_" + str(dtype), x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.manual_seed(7301)
        check("current_stream", torch.randn((8, 65539), device="cuda"))
    torch.cuda.current_stream().wait_stream(stream)
    # Direct partial buffers stress both signs of zero and final-only boundaries.
    direct = torch.zeros((16, 9), device="cuda")
    direct[0].fill_(-0.0)
    for i in range(1, 10):
        direct[i, i - 1] = -0.0
    direct[10, :3] = torch.tensor([1e8, 1, -1e8], device="cuda")
    direct[11, :3] = torch.tensor([2**24, 0.25, -(2**24)], device="cuda")
    direct[12].fill_(2**-149)
    direct[13, 0], direct[14, 0] = float("inf"), -float("inf")
    direct[15, 0], direct[15, 8] = float("inf"), -float("inf")
    direct_checks = []
    for dtype, (_, final, candidate) in kernels.items():
        outs = [torch.empty(16, dtype=getattr(torch, dtype), device="cuda") for _ in range(2)]
        for k, out in zip((final, candidate), outs, strict=True):
            k((16,), direct, out)
        assert torch.equal(outs[0].view(torch.uint8), outs[1].view(torch.uint8))
        direct_checks.append({"dtype": dtype, "input_sha256": tensor_sha(direct), "output_sha256": tensor_sha(outs[1]), "bitwise": True})
    qualification = {"complete": True, "checks": checks, "direct_checks": direct_checks,
                     "frozen_files": frozen_hashes, "partial_byte_identity": True,
                     "resources": {dtype: [k.kernel.attributes for k in group] for dtype, group in kernels.items()}}
    assert all(sha(p) == digest for p, digest in frozen_hashes.items())
    (OUT / "gpu-qualification.json").write_text(json.dumps(qualification, indent=2))
    print(len(checks), "bitwise GPU corpus checks and", len(direct_checks), "final-only checks pass", flush=True)

    # Retained efficient Triton artifacts, launched through the existing driver
    # binding on PyTorch storage/current stream. No Triton frontend or retuning.
    old = json.loads((SOURCE / "codegen/fixed-graph-triton-r0.json").read_text())
    cases = []
    for b, n in ((8, 1048576), (64, 65539)):
        torch.manual_seed(7301)
        x = torch.randn((b, n), device="cuda")
        row = next(t for t in old["timings"] if t["case"] == f"float32-{b}-{n}")
        assert tensor_sha(x) == row["input_sha256"]
        tmp = torch.empty((b, 9), device="cuda")
        outputs = {tag: torch.empty(b, device="cuda") for tag in ("frozen", "candidate", "triton")}
        partial, final, candidate = kernels["float32"]
        tmeta = row["selected_kernels"][1]
        tpath = ROOT / "cache/reference-reduction/fixed-graph-triton-r0/triton" / tmeta["cache_hash"] / (tmeta["entrypoint"] + ".cubin")
        tref = DriverKernel(tpath, tmeta["entrypoint"], (64, 1, 1))
        tref.prepare(b, (tmp, outputs["triton"]), (C.c_int32(b), C.c_int32(9), C.c_uint64(0), C.c_uint64(0)))
        partial((n, b), x, tmp)
        final((b,), tmp, outputs["frozen"])
        candidate((b,), tmp, outputs["candidate"])
        tref.batch(1)
        assert all(torch.equal(outputs["frozen"].view(torch.int32), v.view(torch.int32)) for v in outputs.values())
        captures = []
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(partial.kernel.prepared((n, b), x, tmp))
            stack.enter_context(final.kernel.prepared((b,), tmp, outputs["frozen"]))
            stack.enter_context(candidate.kernel.prepared((b,), tmp, outputs["candidate"]))
            bound = {"frozen": final.kernel, "candidate": candidate.kernel, "triton": tref}
            for round_id in range(6):
                tags = list(bound)
                rotate = round_id % 3
                tags = tags[rotate:] + tags[:rotate]
                for scope in ("final_only", "fixed_partial_pair"):
                    for tag in tags:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                            torch.cuda._sleep(50_000_000)
                            for _ in range(16):
                                if scope == "fixed_partial_pair":
                                    partial.kernel.batch(1)
                                bound[tag].batch(1)
                        trace = OUT / f"timing-{b}-{n}-r{round_id}-{scope}-{tag}.json"
                        prof.export_chrome_trace(str(trace))
                        events = sorted((e for e in json.loads(trace.read_text())["traceEvents"] if e.get("cat") == "kernel"), key=lambda e: e["ts"])
                        assert "spin_kernel" in events[0]["name"]
                        names = ([partial.kernel.manifest["entrypoint"]] if scope == "fixed_partial_pair" else []) + [bound[tag].manifest["entrypoint"]]
                        assert [e["name"] for e in events[1:]] == names * 16
                        durations = [e["dur"] for e in events[1:]]
                        captures.append({"round": round_id, "scope": scope, "impl": tag,
                                         "final_us": durations if scope == "final_only" else durations[1::2],
                                         "partial_us": [] if scope == "final_only" else durations[::2],
                                         "trace": str(trace), "trace_sha256": sha(trace)})
                        assert all(torch.equal(outputs["frozen"].view(torch.int32), v.view(torch.int32)) for v in outputs.values())
        case = {"shape": [b, n], "input_sha256": tensor_sha(x), "partial_sha256": tensor_sha(tmp),
                "triton_final_sha256": sha(tpath), "captures": captures,
                "summary": {f"{scope}/{tag}": statistics.median(v for c in captures if c["scope"] == scope and c["impl"] == tag for v in c["final_us"])
                            for scope in ("final_only", "fixed_partial_pair") for tag in bound}}
        cases.append(case)
        print([b, n], case["summary"], flush=True)
    assert all(sha(p) == digest for p, digest in frozen_hashes.items())
    (OUT / "timing.json").write_text(json.dumps({"complete": True, "cases": cases, "environment": inventory(torch, "cubecl"), "harness_sha256": sha(__file__)}, indent=2))


def counter_final():
    from stage1 import evidence

    evidence.guard()
    import torch
    import ctypes as C
    from contextlib import ExitStack
    from stage5.reduction_codegen_mechanisms import Component
    from stage5.reduction_codegen_compare import DriverKernel

    assert json.loads((OUT / "gpu-qualification.json").read_text())["complete"]
    tmp = torch.ones((64, 9), device="cuda")
    outputs = [torch.empty(64, device="cuda") for _ in range(3)]
    baseline = Component(SOURCE / "rounding/topology/kernels/float32/fixed_final")
    candidate = Component(OUT / "candidate/float32")
    tpath = ROOT / "cache/reference-reduction/fixed-graph-triton-r0/triton/ZWLRMTU6YDG5HLDOHWP7PKIOB7HJ253YERMOM7RVLITAN37PPPPA/triton_per_fused_sum_1.cubin"
    tref = DriverKernel(tpath, "triton_per_fused_sum_1", (64, 1, 1))
    tref.prepare(64, (tmp, outputs[2]), (C.c_int32(64), C.c_int32(9), C.c_uint64(0), C.c_uint64(0)))
    with ExitStack() as stack:
        for k, y in zip((baseline, candidate), outputs[:2], strict=True):
            stack.enter_context(k.kernel.prepared((64,), tmp, y))
        for _ in range(20):
            baseline.kernel.batch(1)
            candidate.kernel.batch(1)
            tref.batch(1)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        baseline.kernel.batch(1)
        candidate.kernel.batch(1)
        tref.batch(1)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    assert all(torch.equal(y, torch.full_like(y, 9)) for y in outputs)


def cross_producers():
    from stage1 import evidence

    evidence.guard()
    import torch
    import ctypes as C
    from contextlib import ExitStack
    from stage5.reduction_codegen_mechanisms import Component
    from stage5.reduction_codegen_compare import DriverKernel
    from stage5.reference_reduction_probe import sha, tensor_sha

    assert json.loads((OUT / "gpu-qualification.json").read_text())["complete"]
    old = json.loads((SOURCE / "codegen/fixed-graph-triton-r0.json").read_text())
    cases = []
    for b, n in ((8, 1048576), (64, 65539)):
        torch.manual_seed(7301)
        x = torch.randn((b, n), device="cuda")
        oldcase = next(t for t in old["timings"] if t["case"] == f"float32-{b}-{n}")
        assert tensor_sha(x) == oldcase["input_sha256"]
        tmp = torch.empty((b, 9), device="cuda")
        y = torch.empty(b, device="cuda")
        cpartial = Component(SOURCE / "codegen/uniform/float32")
        finals = {"frozen": Component(SOURCE / "rounding/topology/kernels/float32/fixed_final"),
                  "candidate": Component(OUT / "candidate/float32")}
        tpaths = [ROOT / "cache/reference-reduction/fixed-graph-triton-r0/triton" / k["cache_hash"] / (k["entrypoint"] + ".cubin") for k in oldcase["selected_kernels"]]
        tpartial = DriverKernel(tpaths[0], oldcase["selected_kernels"][0]["entrypoint"], (512, 1, 1), 64)
        tpartial.prepare(b * 9, (x, tmp), (C.c_int64(n), C.c_int32(b * 9), C.c_int32((n + 8) // 9), C.c_uint64(0), C.c_uint64(0)))
        tfinal = DriverKernel(tpaths[1], oldcase["selected_kernels"][1]["entrypoint"], (64, 1, 1))
        tfinal.prepare(b, (tmp, y), (C.c_int32(b), C.c_int32(9), C.c_uint64(0), C.c_uint64(0)))
        artifacts = {str(k.directory / "kernel.ptx"): sha(k.directory / "kernel.ptx") for k in (cpartial, *finals.values())}
        artifacts.update({str(p): sha(p) for p in tpaths})
        captures = []
        with ExitStack() as stack:
            stack.enter_context(cpartial.kernel.prepared((n, b), x, tmp))
            for k in finals.values():
                stack.enter_context(k.kernel.prepared((b,), tmp, y))
            producers = {"cubecl": cpartial.kernel, "triton": tpartial}
            consumers = {**{tag: k.kernel for tag, k in finals.items()}, "triton": tfinal}
            cpartial.kernel.batch(1)
            expected_tmp = tensor_sha(tmp)
            tpartial.batch(1)
            assert tensor_sha(tmp) == expected_tmp
            tfinal.batch(1)
            expected_y = tensor_sha(y)
            for round_id in range(8):
                combinations = [(p, f) for p in producers for f in consumers]
                import random

                random.Random(62000 + round_id).shuffle(combinations)
                for producer, consumer in combinations:
                    p, f = producers[producer], consumers[consumer]
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                        torch.cuda._sleep(50_000_000)
                        for _ in range(16):
                            p.batch(1)
                            f.batch(1)
                    trace = OUT / f"cross-{b}-{n}-r{round_id}-{producer}-{consumer}.json"
                    prof.export_chrome_trace(str(trace))
                    events = json.loads(trace.read_text())["traceEvents"]
                    kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
                    assert [e["name"] for e in kernels[1:]] == [p.manifest["entrypoint"], f.manifest["entrypoint"]] * 16
                    apis = [e for e in events if "LaunchKernel" in e.get("name", "") and e.get("cat") in ("cuda_driver", "cuda_runtime")]
                    assert len(apis) == 33
                    lead = kernels[0]["ts"] + kernels[0]["dur"] - max(e["ts"] + e["dur"] for e in apis)
                    assert lead > 0
                    assert tensor_sha(tmp) == expected_tmp and tensor_sha(y) == expected_y
                    captures.append({"round": round_id, "producer": producer, "consumer": consumer,
                                     "partial_us": [e["dur"] for e in kernels[1::2]],
                                     "final_us": [e["dur"] for e in kernels[2::2]],
                                     "prequeue_lead_us": lead, "bitwise": True,
                                     "trace": str(trace), "trace_sha256": sha(trace)})
        assert all(sha(p) == digest for p, digest in artifacts.items())
        cases.append({"shape": [b, n], "input_sha256": tensor_sha(x), "partial_sha256": expected_tmp,
                      "output_sha256": expected_y, "artifacts": artifacts, "captures": captures})
    (OUT / "cross-producer.json").write_text(json.dumps({"complete": True, "cases": cases}, indent=2))


def main():
    global OUT
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cpu", action="store_true")
    group.add_argument("--gpu", action="store_true")
    group.add_argument("--counter", action="store_true")
    group.add_argument("--cross", action="store_true")
    parser.add_argument("--mechanism", choices=["plane", "uniform"], default="plane")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        OUT = args.output
        if args.gpu or args.cpu:
            OUT.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    if args.cpu:
        cpu_gate()
    elif args.gpu:
        gpu_probe(args.mechanism)
    elif args.counter:
        counter_final()
    else:
        cross_producers()


if __name__ == "__main__":
    main()
