"""Isolated expression changes behind the frozen semantic/physical boundary.

No new topology or planner eligibility. Every change is checked against the
retained frozen GPU binary before it is timed; original final remains fixed.
"""

import argparse
from itertools import chain
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/reduction-structure/codegen"
FROZEN = ROOT / "artifacts/reduction-structure/rounding/topology"


class Component:
    def __init__(self, directory, *, compile_ptx=False):
        from stage5.artifact import Kernel

        self.directory = directory
        packet = json.loads((directory / "descriptor.json").read_text())
        self.semantic = packet["semantic"]
        self.kernel = Kernel(
            directory,
            self.semantic,
            0,
            packet["physical_plan"],
            compile_ptx=compile_ptx,
        )

    def __call__(self, scalars, *tensors):
        self.kernel(scalars, *tensors)


def compile_variant(mechanism, dtype):
    from stage0.bridge import nvrtc_compile

    directory = OUT / mechanism / dtype
    directory.mkdir(parents=True, exist_ok=True)
    source = FROZEN / "kernels" / dtype / "candidate"
    packet = json.loads((source / "descriptor.json").read_text())
    (directory / "descriptor.json").write_text(json.dumps(packet, indent=2))
    subprocess.run(
        [
            str(ROOT.parent / "stage0/target/release/cubecl-stage0"),
            str(directory),
            "cpp-probe-" + mechanism,
            str(directory / "descriptor.json"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    includes = [
        str(Path(__import__("torch").__file__).parent.parent / "nvidia/cu13/include")
    ]
    nvrtc_compile((directory / "kernel.cu").read_text(), directory, 0, includes)
    return Component(directory)


def parts(mechanism, build=False):
    return {
        d: (
            Component(FROZEN / "kernels" / d / "candidate"),
            compile_variant(mechanism, d) if build else Component(OUT / mechanism / d),
            Component(FROZEN / "kernels" / d / "fixed_final"),
        )
        for d in ("float32", "float16", "bfloat16")
    }


def qualification(mechanism, kernels):
    import torch
    from stage5.reduction_merge_qualification import core_inputs, extra_inputs
    from stage5.reference_reduction_probe import tensor_sha

    checks = []

    def check(label, x):
        b, n = x.shape
        frozen, changed, final = kernels[str(x.dtype).removeprefix("torch.")]
        tmp_store = [torch.full((b * 9 + 2,), 12345.0, device="cuda") for _ in range(2)]
        outputs = [
            torch.full((b + 2,), 12345.0, dtype=x.dtype, device="cuda")
            for _ in range(2)
        ]
        for k, tmp, out in zip((frozen, changed), tmp_store, outputs, strict=True):
            k((n, b), x, tmp[1:-1].view(b, 9))
            final((b,), tmp[1:-1].view(b, 9), out[1:-1])
        for a, bvalue in zip(
            (tmp_store[0], outputs[0]), (tmp_store[1], outputs[1]), strict=True
        ):
            assert torch.equal(a.view(torch.uint8), bvalue.view(torch.uint8)), (
                mechanism,
                label,
                "bitwise",
            )
        assert tmp_store[1][[0, -1]].tolist() == [12345.0] * 2
        expected = torch.tensor([12345.0] * 2, dtype=x.dtype).tolist()
        assert outputs[1][[0, -1]].tolist() == expected
        checks.append(
            {
                "case": label,
                "input_sha256": tensor_sha(x),
                "partial_sha256": tensor_sha(tmp_store[1][1:-1]),
                "output_sha256": tensor_sha(outputs[1][1:-1]),
                "partials_bitwise": True,
                "final_bitwise": True,
                "canaries": True,
            }
        )

    for label, x in chain(core_inputs(torch), extra_inputs(torch)):
        check(label, x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.manual_seed(7301)
        check("storage_torch.float32_8", torch.randn((8, 65539), device="cuda"))
    torch.cuda.current_stream().wait_stream(stream)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.zeros((5, 65539), device="cuda", dtype=dtype)
        x[0].fill_(-0.0)
        x[1, 0] = float("nan")
        x[2, 0] = float("inf")
        x[3, 0] = -float("inf")
        x[4, 0] = float("inf")
        x[4, 256] = -float("inf")
        check("specials_" + str(dtype), x)
    result = {
        "complete": True,
        "mechanism": mechanism,
        "checks": checks,
        "frozen_reference": str(FROZEN / "gpu-qualification.json"),
        "policy_changed": False,
    }
    (OUT / f"{mechanism}-qualification.json").write_text(json.dumps(result, indent=2))
    print(
        mechanism, len(checks), "bitwise partial/final/canary checks pass", flush=True
    )
    return result


def timing(mechanism, kernels, ncu_case=None):
    import torch
    from stage5.attribution import inventory, summary
    from stage5.reduction_codegen_compare import reference
    from stage5.timing import KernelBenchmarker
    from stage5.reference_reduction_probe import tensor_sha

    assert json.loads((OUT / f"{mechanism}-qualification.json").read_text())["complete"]
    cases = []
    for b, n in (
        ([(8, 1048576)] if ncu_case == "million" else [(64, 65539)])
        if ncu_case
        else [(8, 1048576), (64, 65539)]
    ):
        torch.manual_seed(7301)
        x = torch.randn((b, n), device="cuda")
        tmp = [torch.empty((b, 9), device="cuda") for _ in range(3)]
        frozen, changed, final = kernels["float32"]
        frozen((n, b), x, tmp[0])
        changed((n, b), x, tmp[1])
        ref = reference(b, n, x, tmp[2])
        ref.batch(1)
        assert all(
            torch.equal(tmp[0].view(torch.int32), v.view(torch.int32)) for v in tmp[1:]
        )
        row = {
            "shape": [b, n],
            "input_sha256": tensor_sha(x),
            "rounds": [],
            "attributes": {
                name: k.kernel.attributes
                for name, k in [("frozen", frozen), (mechanism, changed)]
            },
            "resources": {
                name: k.kernel.resources
                for name, k in [("frozen", frozen), (mechanism, changed)]
            },
            "reference_attributes": ref.attributes,
        }
        with (
            frozen.kernel.prepared((n, b), x, tmp[0]),
            changed.kernel.prepared((n, b), x, tmp[1]),
        ):
            paths = {"frozen": frozen.kernel, mechanism: changed.kernel, "triton": ref}
            if ncu_case:
                for k in paths.values():
                    for _ in range(20):
                        k.batch(1)
                torch.cuda._sleep(50_000_000)
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                for k in paths.values():
                    k.batch(1)
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            else:
                names = list(paths)
                for r in range(3):
                    for name in names[r:] + names[:r]:
                        measured = KernelBenchmarker().benchmark_kernel_gpu(paths[name])
                        row["rounds"].append(
                            {
                                "round": r,
                                "name": name,
                                **summary([v * 1000 for v in measured["kernel_ms"]]),
                            }
                        )
        cases.append(row)
        print(
            mechanism,
            [b, n],
            {
                name: [v["median_us"] for v in row["rounds"] if v["name"] == name]
                for name in paths
            },
            flush=True,
        )
    result = {
        "complete": True,
        "mechanism": mechanism,
        "scope": "prepared partial only; fixed9 and original final, same input and ownership",
        "cases": cases,
        "environment": inventory(torch, "cubecl"),
    }
    (
        OUT
        / (
            f"{mechanism}-ncu-{ncu_case}.json"
            if ncu_case
            else f"{mechanism}-timing.json"
        )
    ).write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mechanism", choices=("uniform", "owner", "narrow", "uniform_narrow")
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--ncu", choices=("million", "64"))
    parser.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    kernels = parts(args.mechanism, args.build)
    if not args.ncu:
        qualification(args.mechanism, kernels)
    if not args.qualify_only:
        timing(args.mechanism, kernels, args.ncu)


if __name__ == "__main__":
    main()
