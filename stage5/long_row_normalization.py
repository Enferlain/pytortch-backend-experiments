"""Matched RMS forward/arbitrary-gradient capture, with raw arrays in cache."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/long-row"
CACHE = ROOT / "cache/long-row"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("cubecl", "triton"), required=True)
    parser.add_argument(
        "--cubecl-policy",
        choices=(
            "first_use",
            "load_batch_diagnostic",
            "adjacent_merge_diagnostic",
            "reference512_diagnostic",
        ),
        default="first_use",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    if args.cubecl_policy != "first_use":
        if args.impl != "cubecl":
            parser.error("--cubecl-policy applies only to --impl cubecl")
        if args.output_dir is None or args.cache_dir is None:
            parser.error(
                "diagnostic policy requires explicit --output-dir and --cache-dir"
            )
        if (
            args.output_dir.resolve() == OUTPUT.resolve()
            or args.cache_dir.resolve() == CACHE.resolve()
        ):
            parser.error("diagnostic directories must differ from baseline directories")
    output, cache = args.output_dir or OUTPUT, args.cache_dir or CACHE
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / ("norm-" + args.impl))
    if args.impl == "cubecl":
        from stage1 import evidence

        evidence.guard()
    import numpy as np
    import torch
    from stage1.overlay import install

    install()
    from torch._inductor.compile_fx import compile_fx
    from stage5.corpus import rms
    from stage5.attribution import inventory
    from stage5.reference_reduction_probe import tensor_sha, sha

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

    def backend(gm, inputs):
        if args.impl == "cubecl":
            from stage5.backend import compile_backend

            return compile_backend(gm, inputs, tuning=args.cubecl_policy)
        return compile_fx(gm, inputs, config_patches=patches)

    cache.mkdir(parents=True, exist_ok=True)
    cases, arrays = [], {}
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        torch._dynamo.reset()
        f = torch.compile(rms, backend=backend, dynamic=True, fullgraph=True)
        # Efficient long capture for the reference; the CubeCL guard then gets
        # exercised crossing back to short and returning to the long regime.
        order = (65539, 31, 65539)
        for position, n in enumerate(order):
            torch.manual_seed(24001 + n)
            x = torch.randn((3, n), device="cuda").to(dtype).requires_grad_()
            upstream = torch.randn((3, n), device="cuda").to(dtype)
            out = f(x)
            gradient = torch.autograd.grad(out, x, upstream)[0]
            refx = x.detach().clone().requires_grad_()
            reference = rms(refx)
            refgrad = torch.autograd.grad(reference, refx, upstream)[0]
            label = f"{dtype}-{n}-{position}"
            for name, value in (
                ("out", out),
                ("gradient", gradient),
                ("eager", reference),
                ("eager_gradient", refgrad),
            ):
                arrays[label + ":" + name] = value.detach().float().cpu().numpy()
            for phase in ("forward", "backward"):
                if phase == "backward":
                    saved = f(x)
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    if phase == "forward":
                        f(x)
                    else:
                        torch.autograd.grad(saved, x, upstream)
                kernels = [
                    e.name()
                    for e in prof.profiler.kineto_results.events()
                    if e.device_type() == torch.autograd.DeviceType.CUDA
                    and e.activity_type() == "kernel"
                ]
                prefix = "cubecl5_" if args.impl == "cubecl" else "triton_"
                assert kernels and all(k.startswith(prefix) for k in kernels), kernels
                raw = cache / (f"norm-{args.impl}-{label}-{phase}.json")
                prof.export_chrome_trace(str(raw))
                cases.append(
                    {
                        "case": label,
                        "phase": phase,
                        "dtype": str(dtype),
                        "shape": list(x.shape),
                        "input_sha256": tensor_sha(x),
                        "upstream_sha256": tensor_sha(upstream),
                        "kernels": kernels,
                        "raw_trace": str(raw),
                        "trace_sha256": sha(raw),
                    }
                )
    np.savez_compressed(cache / ("norm-" + args.impl + ".npz"), **arrays)
    result = {
        "complete": True,
        "cubecl_policy": args.cubecl_policy if args.impl == "cubecl" else None,
        "output_directory": str(output.resolve()),
        "cache_directory": str(cache.resolve()),
        "cases": cases,
        "environment": inventory(torch, args.impl),
        "raw_arrays_sha256": sha(cache / ("norm-" + args.impl + ".npz")),
        "reference_config": patches,
        "harness_sha256": sha(__file__),
    }
    (output / ("normalization-" + args.impl + ".json")).write_text(
        json.dumps(result, indent=2)
    )
    print("Matched normalization", args.impl, "PASS", flush=True)


if __name__ == "__main__":
    main()
