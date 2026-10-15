"""Execute only the CPU-qualified reference512 candidate, fixed split/final."""

import json
import os
import shutil
from itertools import chain

from stage5.reduction_topology_model import ROOT, SOURCE, OUTPUT, model


def candidate():
    from stage5.plan import PhysicalCandidate

    return PhysicalCandidate(
        "plane_shared_reference512",
        (512, 1, 1),
        plane_width=32,
        private_accumulators=4,
        private_merge="adjacent",
    )


def kernels():
    from stage5.artifact import CompiledKernel
    from stage5.reference_reduction_probe import sha

    result = {}
    for dtype in ("float32", "float16", "bfloat16"):
        base = SOURCE / "rounding/kernels" / dtype
        partial_packet = json.loads((base / "baseline/descriptor.json").read_text())
        final_packet = json.loads((base / "fixed_final/descriptor.json").read_text())
        partial = CompiledKernel(partial_packet["semantic"], 0, candidate().to_dict())
        final = CompiledKernel(
            final_packet["semantic"], 0, final_packet["physical_plan"]
        )
        for name in ("kernel.cu", "kernel.ptx"):
            assert sha(final.directory / name) == sha(base / "fixed_final" / name)
        result[dtype] = partial, final
    return result


def main():
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch

    from stage5.reduction_merge_qualification import core_inputs, extra_inputs, row
    from stage5.reference_reduction_probe import tensor_sha, sha
    from stage5.attribution import inventory

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    qualified = json.loads((OUTPUT / "qualification.json").read_text())
    assert qualified["complete"] and not qualified["failures"]
    assert qualified["harness_sha256"] == sha(ROOT / "reduction_topology_model.py")
    parts = kernels()
    checks = []

    def check(label, x):
        b, n = x.shape
        partial, final = parts[str(x.dtype).removeprefix("torch.")]
        tmp_backing = torch.full((b * 9 + 2,), 12345.0, device="cuda")
        out_backing = torch.full((b + 2,), 12345.0, device="cuda", dtype=x.dtype)
        tmp, out = tmp_backing[1:-1].view(b, 9), out_backing[1:-1]
        original_pointer = x.data_ptr()
        before = partial.kernel.launch_count
        partial((n, b), x, tmp)
        final((b,), tmp, out)
        assert x.data_ptr() == original_pointer
        assert partial.kernel.launch_count == before + 1
        m = model(x.cpu())
        assert tensor_sha(tmp) == tensor_sha(m["partials"]), (label, "partials")
        assert tensor_sha(out) == tensor_sha(m["output"]), (label, "output")
        assert tmp_backing[[0, -1]].tolist() == [12345.0] * 2
        assert (
            out_backing[[0, -1]].float().tolist()
            == torch.tensor([12345.0] * 2, dtype=x.dtype).float().tolist()
        )
        expected = next(r for r in qualified["rows"] if r["case"] == label)
        actual = row(torch, label, x, out)
        # Recheck input bytes, eager and FP64 diagnostics and stored model values.
        assert actual == expected
        checks.append(
            {
                "case": label,
                "partials_bitwise": True,
                "output_bitwise": True,
                "partial_sha256": tensor_sha(tmp),
                "output_sha256": tensor_sha(out),
                "canaries": True,
                "original_pointer": True,
                "qualification": next(
                    c for c in qualified["checks"] if c["case"] == label
                ),
            }
        )

    for label, x in chain(core_inputs(torch), extra_inputs(torch)):
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
        p, f = parts[str(dtype).removeprefix("torch.")]
        tmp, out = (
            torch.empty((5, 9), device="cuda"),
            torch.empty(5, device="cuda", dtype=dtype),
        )
        p((65539, 5), x, tmp)
        f((5,), tmp, out)
        ref = x.sum(-1)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(torch.signbit(out[out == 0]), torch.signbit(ref[ref == 0]))
        specials.append({"dtype": str(dtype), "classes_and_zero_sign": True})
    for dtype, (p, f) in parts.items():
        for name, k in (("candidate", p), ("fixed_final", f)):
            dest = OUTPUT / "kernels" / dtype / name
            dest.mkdir(parents=True, exist_ok=True)
            for path in k.directory.iterdir():
                if path.suffix in (".json", ".cu", ".ptx", ".ir"):
                    shutil.copyfile(path, dest / path.name)
    evidence.state("after_reference512_gpu_qualification")
    result = {
        "complete": True,
        "checks": checks,
        "specials": specials,
        "model_qualification_sha256": sha(OUTPUT / "qualification.json"),
        "environment": inventory(torch, "cubecl"),
        "harness_sha256": sha(__file__),
        "kernels": {
            d: {"candidate": str(p.directory), "fixed_final": str(f.directory)}
            for d, (p, f) in parts.items()
        },
    }
    (OUTPUT / "gpu-qualification.json").write_text(json.dumps(result, indent=2))
    print(
        len(checks),
        "partial/final bitwise matches; all canaries and special classifications pass",
        flush=True,
    )


if __name__ == "__main__":
    main()
