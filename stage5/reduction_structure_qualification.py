"""Qualify the sole load-batch candidate using fixed real semantic packets."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/reduction-structure"


def main():
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch

    from stage5.artifact import CompiledKernel
    from stage5.long_row_analyze import reference_for_extent
    from stage5.plan import PhysicalCandidate
    from stage5.reduction_policy import qualify_case
    from stage5.reference_reduction_probe import sha, tensor_sha

    current = json.loads((OUTPUT / "cubecl-r0.json").read_text())
    references = {
        f"triton-{h}": json.loads(
            (ROOT / f"artifacts/long-row/triton-{h}.json").read_text()
        )
        for h in ("short", "mid", "large", "medium", "million")
    }
    references["triton-large"] = json.loads((OUTPUT / "triton-r0.json").read_text())
    plans = {
        name: PhysicalCandidate(strategy, plane_width=32, private_accumulators=4)
        for name, strategy in (
            ("four", "plane_shared"),
            ("batch", "plane_shared_load_batch"),
        )
    }
    kernels = {}
    rows, checks = [], []
    for dtype in ("float32", "float16", "bfloat16"):
        t = next(t for t in current["timings"] if t["case"] == f"{dtype}-8-65539")
        p, f = [
            json.loads((Path(k["directory"]) / "descriptor.json").read_text())
            for k in t["selected_kernels"]
        ]
        kernels[dtype] = (
            {
                name: CompiledKernel(p["semantic"], 0, plan.to_dict())
                for name, plan in plans.items()
            },
            CompiledKernel(f["semantic"], 0, f["physical_plan"]),
        )

    def check(label, x):
        dtype = str(x.dtype).removeprefix("torch.")
        parts, finish = kernels[dtype]
        n, b = x.shape[1], x.shape[0]
        tmp = torch.empty(
            (b, 9),
            device=x.device,
            dtype=getattr(torch, parts["batch"].semantic["buffers"][-1]["dtype"]),
        )
        values, partial_hashes = {}, {}
        for name, k in parts.items():
            out = torch.empty((b,), device=x.device, dtype=x.dtype)
            k((n, b), x, tmp)
            partial_hashes[name] = tensor_sha(tmp)
            finish((b,), tmp, out)
            values[name] = out
        row = {
            "case": label,
            "dtype": str(x.dtype),
            "shape": list(x.shape),
            "input_sha256": tensor_sha(x),
            "aten": x.sum(-1).tolist(),
            "fp64": x.double().sum(-1).tolist(),
            "values": values["batch"].tolist(),
        }
        reference = next(
            r for r in references[reference_for_extent(n)]["rows"] if r["case"] == label
        )
        outcome = qualify_case(row, reference)
        # This mechanism changes load organization only, preserving every
        # active private chain and its merge. Compare identical packets' bytes.
        assert torch.equal(
            values["four"].view(torch.uint8), values["batch"].view(torch.uint8)
        ), label
        assert partial_hashes["four"] == partial_hashes["batch"], label
        rows.append(row)
        checks.append(
            {
                **outcome,
                "old_four_bitwise_identical": True,
                "partials_bitwise_identical": True,
                "partial_sha256": partial_hashes["batch"],
            }
        )

    old = json.loads(
        (ROOT / "artifacts/attribution/cancellation-counterexample.json").read_text()
    )
    torch.manual_seed(old["seed"])
    for shape in old["rng_history"]:
        x = torch.randn(shape, device="cuda")
    check("original_rng_history", x)
    for n in (31, 255, 256, 257, 1023, 1024, 1025, 4099, 65539, 131071, 1048576):
        for seed in (9241, 11, 101, 7301):
            torch.manual_seed(seed)
            check(f"random_{seed}_{n}", torch.randn((8, n), device="cuda"))
    for seed in (*range(10000, 10032), *range(30000, 30032)):
        torch.manual_seed(seed)
        check(f"heldout_{seed}", torch.randn((8, 65539), device="cuda"))
    assert len(rows) == 109
    for a, b in zip(rows, current["rows"][:109], strict=True):
        assert a["case"] == b["case"] and a["input_sha256"] == b["input_sha256"]
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for b in (8, 64):
            torch.manual_seed(7301)
            check(
                f"storage_{dtype}_{b}", torch.randn((b, 65539), device="cuda").to(dtype)
            )
    # Explicit stream and lifetime exercise, including concurrent scratch users.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.manual_seed(7301)
        check("storage_torch.float32_8", torch.randn((8, 65539), device="cuda"))
        churn = [torch.empty((100_000,), device="cuda") for _ in range(12)]
        del churn
    torch.cuda.current_stream().wait_stream(stream)
    evidence.state("after_batch_qualification")
    result = {
        "complete": True,
        "diagnostic_only": True,
        "rows": rows,
        "checks": checks,
        "failures": [c for c in checks if not c["passes"]],
        "independence": evidence.inventory(),
        "harness_sha256": sha(__file__),
        "kernels": {
            dtype: {name: str(k.directory) for name, k in v[0].items()}
            for dtype, v in kernels.items()
        },
    }
    (OUTPUT / "candidate-qualification.json").write_text(json.dumps(result, indent=2))
    print(
        len(checks),
        "cases;",
        len(result["failures"]),
        "qualification failures; all old-four packets bitwise identical",
    )


if __name__ == "__main__":
    main()
