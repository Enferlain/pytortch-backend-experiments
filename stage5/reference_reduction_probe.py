"""Matched Triton/CubeCL reduction qualification, not a relaxed acceptance gate.

Run each backend/history in a fresh process. Input hashes must match across
processes before comparing outputs. No eager-relative failure is hidden. Timing
uses actual CUPTI kernel events, not queue-inclusive host/event intervals.
"""

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import traceback

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(x):
    import torch

    return hashlib.sha256(
        x.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def run(args, result):
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    cache = ROOT / "cache/reference-reduction" / args.output.stem
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    os.environ["CUBECL_STAGE5_TRACE"] = "0"
    cube = args.impl == "cubecl"
    if cube:
        from stage1 import evidence

        evidence.guard()
    import torch
    from stage1.overlay import install

    install()
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor.choices import InductorChoices
    from stage5.attribution import inventory

    patches = {
        "triton.cudagraphs": False,
        "compile_threads": 1,
        "fx_graph_cache": False,
        "emulate_precision_casts": True,
        "joint_graph_constant_folding": False,
        "eager_numerics.division_rounding": True,
        "eager_numerics.disable_ftz": True,
        "eager_numerics.use_pytorch_libdevice": True,
    }
    config.compile_threads = 1
    config.fx_graph_cache = False
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.suppress_errors = False
    graphs, splits = [], []
    native = C.CDLL(str(PROJECT / "stage0/libstage0_launcher.so")) if cube else None
    if native:
        native.stage0_kernel_launches.restype = C.c_uint64

    def backend(gm, inputs):
        graphs.append(
            {
                "graph": str(gm.graph),
                "inputs": [str(getattr(x, "shape", x)) for x in inputs],
            }
        )
        if cube:
            from stage5.backend import compile_backend

            return compile_backend(
                gm,
                inputs,
                tuning=args.cubecl_policy,
                split_policy=getattr(args, "cubecl_split_policy", "inherited"),
            )
        with config.patch(patches):
            return compile_fx(gm, inputs, config_patches=patches)

    previous = InductorChoices.reduction_split_factor

    def traced(self, device, reduction_numel_hint, numel_hint, inner_reduction):
        value = previous(
            self, device, reduction_numel_hint, numel_hint, inner_reduction
        )
        splits.append(
            {
                "reduction_hint": reduction_numel_hint,
                "outer_hint": numel_hint,
                "inner": inner_reduction,
                "split": value,
            }
        )
        return value

    InductorChoices.reduction_split_factor = traced
    compiled = torch.compile(
        lambda x: x.sum(-1), backend=backend, fullgraph=True, dynamic=True
    )
    rows, timings = [], []
    result.update(
        {
            "qualification_complete": False,
            "implementation": args.impl,
            "history": args.history,
            "cubecl_policy": args.cubecl_policy if cube else None,
            "rows": rows,
            "timings": timings,
            "graphs": graphs,
            "splits": splits,
            "triton_config": patches,
            "cubecl_split_policy": getattr(args, "cubecl_split_policy", "inherited")
            if cube
            else None,
        }
    )
    representatives = args.output.parent / "representatives" / args.output.stem
    representatives.mkdir(parents=True, exist_ok=True)

    def selected(x, names=None):
        if cube:
            from stage5 import runtime

            return [
                {
                    "entrypoint": k.last.kernel.manifest["entrypoint"],
                    "plan": k.last.kernel.plan.to_dict(),
                    "directory": str(k.last.directory),
                    "resources": k.last.kernel.resources,
                }
                for k in runtime.CACHE.values()
                if k.last is not None
                and (
                    names is None
                    or any(k.last.kernel.manifest["entrypoint"] in n for n in names)
                )
            ]
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        invoked = []
        previous_run = CachingAutotuner.run

        def observe(obj, *args, **kwargs):
            answer = previous_run(obj, *args, **kwargs)
            invoked.append(obj)
            return answer

        CachingAutotuner.run = observe
        try:
            compiled(x)
        finally:
            CachingAutotuner.run = previous_run
        kernels = []
        for obj in invoked:
            assert len(obj.launchers) == 1, "executed Triton configuration is ambiguous"
            for launcher in obj.launchers:
                kernels.append(
                    {
                        "entrypoint": obj.inductor_meta["kernel_name"],
                        "config": str(launcher.config),
                        "registers": launcher.n_regs,
                        "spills": launcher.n_spills,
                        "shared": launcher.shared,
                        "cache_hash": getattr(launcher, "cache_hash", None),
                        "wrapper": obj.filename,
                    }
                )
        assert kernels, "Triton resource observer did not see the executed launch"
        return kernels

    def check(label, x):
        eager, truth = x.sum(-1), x.double().sum(-1)
        out = compiled(x)
        assert out.dtype == eager.dtype and out.shape == eager.shape
        rtol, atol = {
            torch.float32: (2e-5, 2e-5),
            torch.float16: (0.002, 0.001),
            torch.bfloat16: (0.016, 0.008),
        }[x.dtype]
        row = {
            "case": label,
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "input_sha256": tensor_sha(x),
            "values": out.tolist(),
            "aten": eager.tolist(),
            "fp64": truth.tolist(),
            "gate_rtol": rtol,
            "gate_atol": atol,
            "eager_gate_passes": bool(
                torch.all((out - eager).abs() <= atol + rtol * eager.abs())
            ),
            "max_abs_error_fp64": (out.double() - truth).abs().max().item(),
            "max_abs_error_eager": (out - eager).abs().max().item(),
        }
        rows.append(row)
        (cache / "progress.json").write_text(json.dumps(result, indent=2))

    def measure(label, x):
        if getattr(args, "qualification_only", False):
            return
        for _ in range(20):
            compiled(x)
        torch.cuda.synchronize()
        profiles = []
        for repeat in range(7 * getattr(args, "rounds", 1)):
            # Reuse the executed Stage5 bench activity warmup. This consumes no
            # RNG/model state and is outside every measured region. Sparse host
            # enqueue alone can leave this WSL device in its idle power regime.
            torch.cuda._sleep(50_000_000)
            torch.cuda.synchronize()
            before = native.stage0_kernel_launches() if native else None
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                compiled(x)
            raw = cache / (label + "-" + str(repeat) + ".json")
            prof.export_chrome_trace(str(raw))
            events = [
                e
                for e in prof.profiler.kineto_results.events()
                if e.device_type() == torch.autograd.DeviceType.CUDA
                and e.activity_type() == "kernel"
            ]
            assert events, "incomplete GPU capture"
            names = [e.name() for e in events]
            expected = "cubecl5_" if cube else "triton_"
            assert all(n.startswith(expected) for n in names), names
            per_call = len(events)
            if profiles:
                assert names == profiles[0]["names"], "kernel census changed"
            if native:
                assert native.stage0_kernel_launches() - before == len(events)
            samples = [
                sum(
                    (e.end_ns() - e.start_ns()) / 1000 for e in events[i : i + per_call]
                )
                for i in range(0, len(events), per_call)
            ]
            profiles.append(
                {
                    "kernels_per_call": per_call,
                    "round": repeat // 7,
                    "names": names[:per_call],
                    "median_us": statistics.median(samples),
                    "samples_us": samples,
                    "raw_trace": str(raw),
                    "sha256": sha(raw),
                }
            )
        timings.append(
            {
                "case": label,
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "input_sha256": tensor_sha(x),
                "profiles": profiles,
                "median_us": statistics.median(p["median_us"] for p in profiles),
                "timing_method": "seven independent CUPTI kernel sums; same 50M-cycle activity warmup outside each capture, as Stage5 bench; graph capture remains unsupported",
                "selected_kernels": selected(x, profiles[0]["names"]),
            }
        )
        print(args.impl, args.history, label, timings[-1]["median_us"], flush=True)

    try:
        old = json.loads(
            (
                ROOT / "artifacts/attribution/cancellation-counterexample.json"
            ).read_text()
        )
        warm_shapes = {
            "short_first": old["rng_history"],
            "large_first": [old["rng_history"][-1]],
            "mid_first": [(8, 8192)],
            "medium_first": [(8, 131071)],
            "million_first": [(8, 1048576)],
        }[args.history]
        torch.manual_seed(old["seed"])
        for shape in warm_shapes:
            compiled(torch.randn(shape, device="cuda"))
        if args.ncu_case:
            dtype_name, count = args.ncu_case.split("-")
            torch.manual_seed(7301)
            x = torch.randn((int(count), 65539), device="cuda").to(
                getattr(torch, dtype_name)
            )
            for _ in range(20):
                out = compiled(x)
            torch.cuda._sleep(50_000_000)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            compiled(x)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            result.update(
                qualification_complete=True,
                resources_only=True,
                shape=list(x.shape),
                dtype=str(x.dtype),
                input_sha256=tensor_sha(x),
                values=out.tolist(),
                selected_kernels=selected(x),
                environment=inventory(torch, args.impl),
                harness_sha256=sha(__file__),
            )
            return result
        torch.manual_seed(old["seed"])
        for shape in old["rng_history"]:
            original = torch.randn(shape, device="cuda")
        check("original_rng_history", original)
        for n in (31, 255, 256, 257, 1023, 1024, 1025, 4099, 65539, 131071, 1048576):
            for seed in (9241, 11, 101, 7301):
                torch.manual_seed(seed)
                check(f"random_{seed}_{n}", torch.randn((8, n), device="cuda"))
        for seed in (*range(10000, 10032), *range(30000, 30032)):
            torch.manual_seed(seed)
            check(f"heldout_{seed}", torch.randn((8, 65539), device="cuda"))
        # Different storage types exercise conversion and the same reduction op;
        # ordinary samples above preserve the exact historical FP32 input bytes.
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            special = torch.zeros((5, 65539), device="cuda", dtype=dtype)
            special[0].fill_(-0.0)
            special[1, 0] = float("nan")
            special[2, 0] = float("inf")
            special[3, 0] = -float("inf")
            special[4, 0], special[4, 256] = float("inf"), -float("inf")
            a, b = compiled(special), special.sum(-1)
            torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
            assert torch.equal(torch.signbit(a[a == 0]), torch.signbit(b[b == 0]))
            result.setdefault("special_checks", []).append(
                {
                    "dtype": str(dtype),
                    "values": a.tolist(),
                    "classification_and_zero_sign": True,
                }
            )
            for shape in ((8, 65539), (64, 65539)):
                torch.manual_seed(7301)
                x = torch.randn(shape, device="cuda").to(dtype)
                check(f"storage_{dtype}_{shape[0]}", x)
                measure(
                    str(dtype).replace("torch.", "") + "-" + str(shape[0]) + "-65539", x
                )
        for shape in ((64, 31), (64, 257), (8, 131071), (64, 131071), (8, 1048576)):
            torch.manual_seed(7301)
            measure(
                "float32-" + str(shape[0]) + "-" + str(shape[1]),
                torch.randn(shape, device="cuda"),
            )
        if getattr(args, "long_row", False):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for n in (
                    0,
                    1,
                    255,
                    256,
                    257,
                    8191,
                    8192,
                    8193,
                    65536,
                    65537,
                    73727,
                    73728,
                    73729,
                    131071,
                    131072,
                    131073,
                    1048575,
                    1048576,
                    1048577,
                ):
                    torch.manual_seed(7301)
                    check(
                        f"boundary_{dtype}_{n}",
                        torch.randn((8, n), device="cuda").to(dtype),
                    )
                for shape in ((8, 131071), (8, 1048576)):
                    torch.manual_seed(7301)
                    x = torch.randn(shape, device="cuda").to(dtype)
                    check(f"neighbor_{dtype}_{shape[0]}_{shape[1]}", x)
                    if dtype != torch.float32:
                        measure(
                            f"{str(dtype).replace('torch.', '')}-{shape[0]}-{shape[1]}",
                            x,
                        )
        # Retain actual wrappers and target IR. Binary caches stay local/ignored.
        retained = []
        for root, suffixes in (
            (cache / "inductor", (".py",)),
            (cache / "triton", (".ttir", ".ttgir", ".llir", ".ptx", ".json")),
        ):
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.suffix in suffixes:
                    dst = representatives / root.name / p.relative_to(root)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, dst)
                    retained.append({"path": str(dst), "sha256": sha(dst)})
        if cube:
            from stage5.attribution import artifacts
            from stage5 import artifact
            from stage5.choices import EVENTS as regime_events
            from stage5.index_masks import EVENTS as index_events

            result.update(
                artifacts=artifacts(),
                compiler_invocations=artifact.COMPILER_INVOCATIONS,
                native_launches=native.stage0_kernel_launches(),
                split_regimes=list(regime_events),
                index_mask_widening=list(index_events),
            )
            evidence.state("after_reference_reduction")
        result.update(
            qualification_complete=True,
            environment=inventory(torch, args.impl),
            retained=retained,
            harness_sha256=sha(__file__),
            numerical_failures=[
                {"case": r["case"], "shape": r["shape"]}
                for r in rows
                if r["eager_gate_passes"] is False
            ],
            interpretation="comparison only; does not waive an eager gate or accept a production policy",
        )
    finally:
        InductorChoices.reduction_split_factor = previous
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("cubecl", "triton"), required=True)
    parser.add_argument(
        "--history",
        choices=(
            "short_first",
            "large_first",
            "mid_first",
            "medium_first",
            "million_first",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cubecl-policy",
        choices=(
            "first_use",
            "baseline",
            "combined_diagnostic",
            "compensated_diagnostic",
            "load_batch_diagnostic",
            "reference512_diagnostic",
            "adjacent_merge_diagnostic",
        ),
        default="first_use",
    )
    parser.add_argument(
        "--cubecl-split-policy", choices=("inherited", "guarded"), default="inherited"
    )
    parser.add_argument("--long-row", action="store_true")
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--rounds", type=int, choices=(1, 3), default=1)
    parser.add_argument(
        "--ncu-case",
        choices=(
            "float32-8",
            "float32-64",
            "float16-8",
            "float16-64",
            "bfloat16-8",
            "bfloat16-64",
        ),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"qualification_complete": False}
    try:
        result = run(args, result)
    except Exception as error:
        result.update(failure=repr(error), traceback=traceback.format_exc())
        args.output = args.output.with_suffix(".failed.json")
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
