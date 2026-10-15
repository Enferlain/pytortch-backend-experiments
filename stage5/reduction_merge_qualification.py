"""Fixed-split qualification of the one adjacent-private-merge candidate.

Run --impl triton first in the separate reference environment for additional
held-out inputs. Historical references remain unchanged and hash matched.
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts/reduction-structure"
OUTPUT = SOURCE / "rounding"


def core_inputs(torch):
    old = json.loads(
        (ROOT / "artifacts/attribution/cancellation-counterexample.json").read_text()
    )
    torch.manual_seed(old["seed"])
    for shape in old["rng_history"]:
        x = torch.randn(shape, device="cuda")
    yield "original_rng_history", x
    for n in (31, 255, 256, 257, 1023, 1024, 1025, 4099, 65539, 131071, 1048576):
        for seed in (9241, 11, 101, 7301):
            torch.manual_seed(seed)
            yield f"random_{seed}_{n}", torch.randn((8, n), device="cuda")
    for seed in (*range(10000, 10032), *range(30000, 30032)):
        torch.manual_seed(seed)
        yield f"heldout_{seed}", torch.randn((8, 65539), device="cuda")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for b in (8, 64):
            torch.manual_seed(7301)
            yield (
                f"storage_{dtype}_{b}",
                torch.randn((b, 65539), device="cuda").to(dtype),
            )


def extra_inputs(torch):
    # Not inspected when selecting the single candidate from the original trace.
    for seed in range(40000, 40032):
        torch.manual_seed(seed)
        yield f"new_heldout_{seed}", torch.randn((8, 65539), device="cuda")
    # Existing accumulation-suite finite adversaries, now evaluated under the
    # corresponding efficient-reference policy rather than exact-FP64 demands.
    x = torch.zeros((5, 65539), device="cuda")
    x[0, [0, 256, 512]] = torch.tensor([1e8, 1, -1e8], device="cuda")
    x[1] = -x[0]
    x[2, [0, 256, 512]] = torch.tensor([2**24, 0.25, -(2**24)], device="cuda")
    x[3, :1024] = 2**-149
    x[4, ::2], x[4, 1::2] = 1024, -1024
    yield "new_existing_finite_adversaries", x


def row(torch, label, x, out):
    from stage5.reference_reduction_probe import tensor_sha

    return {
        "case": label,
        "dtype": str(x.dtype),
        "shape": list(x.shape),
        "input_sha256": tensor_sha(x),
        "aten": x.sum(-1).tolist(),
        "fp64": x.double().sum(-1).tolist(),
        "values": out.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("cubecl", "triton"), required=True)
    args = parser.parse_args()
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(
        ROOT / "cache/reduction-rounding" / args.impl
    )
    if args.impl == "cubecl":
        from stage1 import evidence

        evidence.guard()
    import torch

    from stage5.attribution import inventory
    from stage5.reference_reduction_probe import sha, tensor_sha

    if args.impl == "triton":
        from stage1.overlay import install

        install()
        from torch._inductor import config
        from torch._inductor.compile_fx import compile_fx
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        patches = json.loads((SOURCE / "triton-r0.json").read_text())["triton_config"]

        def backend(gm, inputs):
            return compile_fx(gm, inputs, config_patches=patches)

        f = torch.compile(
            lambda x: x.sum(-1), backend=backend, fullgraph=True, dynamic=True
        )
        observed = []
        adversarial_partial = {}
        previous = CachingAutotuner.run

        def observe(obj, *values, **kwargs):
            result = previous(obj, *values, **kwargs)
            if obj.inductor_meta["kernel_name"] == "triton_red_fused_sum_0" and tuple(
                values[0].shape
            ) == (5, 65539):
                adversarial_partial.update(
                    values=values[1].tolist(), sha256=tensor_sha(values[1])
                )
            assert len(obj.launchers) == 1
            observed.append(
                {
                    "name": obj.inductor_meta["kernel_name"],
                    "config": str(obj.launchers[0].config),
                    "wrapper": obj.filename,
                }
            )
            return result

        with config.patch(patches):
            torch.manual_seed(7301)
            f(torch.randn((8, 65539), device="cuda"))
            CachingAutotuner.run = observe
            try:
                rows = [row(torch, label, x, f(x)) for label, x in extra_inputs(torch)]
            finally:
                CachingAutotuner.run = previous
        configs = [
            json.loads(v)
            for v in sorted({json.dumps(v, sort_keys=True) for v in observed})
        ]
        assert len(configs) == 2 and any(
            "R0_BLOCK: 2048" in v["config"] and "num_warps: 16" in v["config"]
            for v in configs
        )
        (OUTPUT / "new-triton-reference.json").write_text(
            json.dumps(
                {
                    "rows": rows,
                    "config": patches,
                    "kernels": configs,
                    "adversarial_partial": adversarial_partial,
                    "environment": inventory(torch, args.impl),
                    "harness_sha256": sha(__file__),
                },
                indent=2,
            )
        )
        print("Captured", len(rows), "new held-out reference inputs")
        return

    from stage5.artifact import CompiledKernel
    from stage5.long_row_analyze import reference_for_extent
    from stage5.plan import PhysicalCandidate
    from stage5.reduction_policy import qualify_case
    from stage5.reduction_rounding_probe import model

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    current = json.loads((SOURCE / "cubecl-r0.json").read_text())
    references = {
        f"triton-{h}": json.loads(
            (ROOT / f"artifacts/long-row/triton-{h}.json").read_text()
        )
        for h in ("short", "mid", "large", "medium", "million")
    }
    references["triton-large"] = json.loads((SOURCE / "triton-r0.json").read_text())
    extra = json.loads((OUTPUT / "new-triton-reference.json").read_text())
    plans = {
        "baseline": PhysicalCandidate(
            "plane_shared_load_batch", plane_width=32, private_accumulators=4
        ),
        "adjacent": PhysicalCandidate(
            "plane_shared_load_batch",
            plane_width=32,
            private_accumulators=4,
            private_merge="adjacent",
        ),
    }
    kernels = {}
    for dtype in ("float32", "float16", "bfloat16"):
        timing = next(t for t in current["timings"] if t["case"] == f"{dtype}-8-65539")
        p, f = [
            json.loads((Path(k["directory"]) / "descriptor.json").read_text())
            for k in timing["selected_kernels"]
        ]
        kernels[dtype] = (
            {
                name: CompiledKernel(p["semantic"], 0, plan.to_dict())
                for name, plan in plans.items()
            },
            CompiledKernel(f["semantic"], 0, f["physical_plan"]),
        )
    rows, checks, adversarial_partials = [], [], {}

    def check(label, x):
        dtype = str(x.dtype).removeprefix("torch.")
        parts, finish = kernels[dtype]
        b, n = x.shape
        tmp = torch.empty((b, 9), device=x.device, dtype=torch.float32)
        values, partial_hashes = {}, {}
        for name, k in parts.items():
            out = torch.empty((b,), device=x.device, dtype=x.dtype)
            k((n, b), x, tmp)
            partial_hashes[name] = tensor_sha(tmp)
            if label == "new_existing_finite_adversaries":
                adversarial_partials[name] = tmp.tolist()
            finish((b,), tmp, out)
            values[name] = out
            m = model(x.cpu(), adjacent=name == "adjacent")
            assert tensor_sha(m["partials"]) == partial_hashes[name], (
                label,
                name,
                "partials",
            )
            assert tensor_sha(m["output"].to(x.dtype)) == tensor_sha(out), (
                label,
                name,
                "output",
            )
        actual = row(torch, label, x, values["adjacent"])
        reference = next(
            r
            for r in (
                extra
                if label.startswith("new_")
                else references[reference_for_extent(n)]
            )["rows"]
            if r["case"] == label
        )
        outcome = qualify_case(actual, reference)
        baseline = qualify_case(
            {**actual, "values": values["baseline"].tolist()}, reference
        )
        checks.append(
            {
                **outcome,
                "baseline_passes": baseline["passes"],
                "cpu_model_bitwise": True,
                "partial_hashes": partial_hashes,
                "baseline_values": values["baseline"].tolist(),
            }
        )
        rows.append(actual)

    for label, x in core_inputs(torch):
        check(label, x)
    assert len(rows) == 115
    for a, b in zip(rows[:109], current["rows"][:109], strict=True):
        assert a["case"] == b["case"] and a["input_sha256"] == b["input_sha256"]
    for label, x in extra_inputs(torch):
        check(label, x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.manual_seed(7301)
        check("storage_torch.float32_8", torch.randn((8, 65539), device="cuda"))
        churn = [torch.empty((100_000,), device="cuda") for _ in range(12)]
        del churn
    torch.cuda.current_stream().wait_stream(stream)
    specials = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.zeros((5, 65539), device="cuda", dtype=dtype)
        x[0].fill_(-0.0)
        x[1, 0], x[2, 0], x[3, 0] = float("nan"), float("inf"), -float("inf")
        x[4, 0], x[4, 256] = float("inf"), -float("inf")
        parts, finish = kernels[str(dtype).removeprefix("torch.")]
        tmp, out = (
            torch.empty((5, 9), device="cuda"),
            torch.empty(5, device="cuda", dtype=dtype),
        )
        parts["adjacent"]((65539, 5), x, tmp)
        finish((5,), tmp, out)
        ref = x.sum(-1)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(torch.signbit(out[out == 0]), torch.signbit(ref[ref == 0]))
        specials.append({"dtype": str(dtype), "classifications_and_zero_sign": True})
    evidence.state("after_adjacent_merge_qualification")
    result = {
        "complete": True,
        "diagnostic_only": True,
        "rows": rows,
        "checks": checks,
        "failures": [c for c in checks if not c["passes"]],
        "specials": specials,
        "adversarial_partials": adversarial_partials,
        "environment": inventory(torch, "cubecl"),
        "harness_sha256": sha(__file__),
        "kernels": {
            dtype: {name: str(k.directory) for name, k in parts.items()}
            | {"fixed_final": str(f.directory)}
            for dtype, (parts, f) in kernels.items()
        },
    }
    (OUTPUT / "merge-qualification.json").write_text(json.dumps(result, indent=2))
    print(
        len(checks),
        "cases;",
        len(result["failures"]),
        "qualification failures; CPU model matches all partial/final bytes",
    )


if __name__ == "__main__":
    main()
