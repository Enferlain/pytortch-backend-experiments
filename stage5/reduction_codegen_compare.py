"""Executed CubeK codegen comparisons; no CubeK client, runtime or allocator.

The CubeK stock sums are codegen controls, not frozen-topology candidates.
The reader controls retain four private chains and expose their per-thread sum.
"""

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/reduction-structure/codegen"


class DriverKernel:
    """Diagnostic-only PTX/cubin execution on caller-retained PyTorch storage."""

    def __init__(self, path, name, block, shared=0):
        import torch

        self.device = torch.device("cuda", 0)
        torch.cuda.init()
        self.library = C.CDLL("libcuda.so.1")
        self.module, self.function = C.c_void_p(), C.c_void_p()
        self.manifest = {"entrypoint": name}
        self.block, self.shared = block, shared
        self.library.cuModuleLoadData.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
        self.library.cuModuleGetFunction.argtypes = [
            C.POINTER(C.c_void_p),
            C.c_void_p,
            C.c_char_p,
        ]
        self.library.cuLaunchKernel.argtypes = [
            C.c_void_p,
            *([C.c_uint] * 7),
            C.c_void_p,
            C.POINTER(C.c_void_p),
            C.c_void_p,
        ]
        self.check(
            self.library.cuModuleLoadData(C.byref(self.module), Path(path).read_bytes())
        )
        self.check(
            self.library.cuModuleGetFunction(
                C.byref(self.function), self.module, name.encode()
            )
        )
        self.attributes = {}
        for label, attribute in [
            ("registers", 4),
            ("local_bytes", 3),
            ("static_shared_bytes", 1),
        ]:
            value = C.c_int()
            self.check(
                self.library.cuFuncGetAttribute(
                    C.byref(value), attribute, self.function
                )
            )
            self.attributes[label] = value.value
        self.arguments = None

    def check(self, result):
        if result:
            detail = C.c_char_p()
            self.library.cuGetErrorString(result, C.byref(detail))
            raise RuntimeError((result, detail.value))

    def prepare(self, grid, tensors, scalars=()):
        import torch

        self.grid, self.tensors = grid, tensors
        self.values = [C.c_uint64(t.data_ptr()) for t in tensors] + [
            C.c_int64(s) if isinstance(s, int) else s for s in scalars
        ]
        self.arguments = (C.c_void_p * len(self.values))(
            *[C.cast(C.pointer(v), C.c_void_p) for v in self.values]
        )
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream())

    def batch(self, repeat):
        import torch

        stream = torch.cuda.current_stream()
        stream.wait_event(self.ready)
        for t in self.tensors:
            t.record_stream(stream)
        for _ in range(repeat):
            self.check(
                self.library.cuLaunchKernel(
                    self.function,
                    self.grid,
                    1,
                    1,
                    *self.block,
                    self.shared,
                    stream.cuda_stream,
                    self.arguments,
                    None,
                )
            )


def compile_controls():
    from stage0.bridge import nvrtc_compile

    for mode in ("branch", "mask", "cube-branch", "cube-mask", "plane-branch"):
        directory = OUT / f"cubek-{mode}"
        subprocess.run(
            [
                str(ROOT.parent / "stage0/target/release/stage5-codegen-probe"),
                str(directory),
                mode,
            ],
            check=True,
        )
        nvrtc_compile((directory / "kernel.cu").read_text(), directory, 0)


def controls(b, n):
    import torch

    e, rows = (n + 8) // 9, b * 9
    torch.manual_seed(7301)
    original = torch.randn((b, n), device="cuda")
    x = torch.zeros((b, 9 * e), device="cuda")
    x[:, :n] = original
    x = x.reshape(rows, e)
    kernels = {}
    for mode in ("branch", "mask", "cube-branch", "cube-mask", "plane-branch"):
        directory = OUT / f"cubek-{mode}"
        cube, plane = mode.startswith("cube"), mode.startswith("plane")
        component = cube or plane
        y = torch.empty(rows if component else (rows, 512), device="cuda")
        # Reuse CubeCL InfoBuilder; no diagnostic metadata convention is invented.
        words = json.loads(
            subprocess.check_output(
                [
                    str(ROOT.parent / "stage0/target/release/stage5-codegen-probe"),
                    "--metadata",
                    mode,
                    str(e),
                    str(rows),
                ],
                text=True,
            )
        )
        info = torch.tensor(words, device="cuda", dtype=torch.uint64)
        block = (32, 8, 1) if plane else (32, 16, 1) if cube else (512, 1, 1)
        name = "cubek_reader_" + mode.replace("-", "_")
        k = DriverKernel(directory / "kernel.ptx", name, block, 64 if cube else 0)
        k.prepare((rows + 7) // 8 if plane else rows, (x, y, info))
        k.batch(1)
        if component:
            torch.testing.assert_close(y, x.sum(-1), rtol=2e-4, atol=2e-3)
        kernels[mode] = k
    assert torch.equal(
        kernels["branch"].tensors[1].view(torch.int32),
        kernels["mask"].tensors[1].view(torch.int32),
    )
    # Independently verify the exact four-chain reader outputs.
    import numpy as np

    data = x.cpu().numpy()
    a = np.zeros((4, rows, 512), dtype=np.float32)
    for i in range((e + 2047) // 2048):
        for j in range(4):
            start = i * 2048 + j * 512
            count = min(512, max(0, e - start))
            if count:
                a[j, :, :count] = np.add(
                    a[j, :, :count], data[:, start : start + count], dtype=np.float32
                )
    expected = np.add(np.add(a[0], a[1]), np.add(a[2], a[3]))
    assert np.array_equal(
        expected.view(np.int32),
        kernels["branch"].tensors[1].cpu().numpy().view(np.int32),
    )
    return kernels


def reference(b, n, x, tmp):
    """Load retained Triton cubin, preserving its actual assembler output."""
    directory = (
        ROOT
        / "cache/attribution/history-triton-large/triton/0/22FRMEHQY7GOBHBJC6OITADZEDJIOXOLC3DDHYLIPDIIOU5CHWAA"
    )
    retained = (
        ROOT
        / "artifacts/reduction-structure/selected/22FRMEHQY7GOBHBJC6OITADZEDJIOXOLC3DDHYLIPDIIOU5CHWAA"
    )
    # Retained signature: input/output pointers, i64 row length, i32 output
    # count and split extent, then two unused auxiliary pointers.
    signature = re.search(
        r"\.entry\s+\w+\s*\((.*?)\)",
        (retained / "triton_red_fused_sum_0.ptx").read_text(),
        re.S,
    )[1]
    assert signature.count(".param") == 7, signature
    k = DriverKernel(
        directory / "triton_red_fused_sum_0.cubin",
        "triton_red_fused_sum_0",
        (512, 1, 1),
        64,
    )
    k.prepare(
        b * 9,
        (x, tmp),
        (
            C.c_int64(n),
            C.c_int32(b * 9),
            C.c_int32((n + 8) // 9),
            C.c_uint64(0),
            C.c_uint64(0),
        ),
    )
    return k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--ncu",
        choices=(
            "branch",
            "mask",
            "cube-branch",
            "cube-mask",
            "plane-branch",
            "frozen",
            "triton",
            "all",
        ),
    )
    parser.add_argument("--case", choices=("million", "64"), default="million")
    args = parser.parse_args()
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.attribution import inventory, summary
    from stage5.reduction_topology_probe import kernels as frozen_kernels
    from stage5.timing import KernelBenchmarker
    from stage5.reference_reduction_probe import tensor_sha

    torch.set_num_threads(1)
    if args.compile:
        compile_controls()
    results = []
    for b, n in (
        ([(8, 1048576)] if args.case == "million" else [(64, 65539)])
        if args.ncu
        else [(8, 1048576), (64, 65539)]
    ):
        control = controls(b, n)
        torch.manual_seed(7301)
        x = torch.randn((b, n), device="cuda")
        p, f = frozen_kernels()["float32"]
        tmp, out = torch.empty((b, 9), device="cuda"), torch.empty(b, device="cuda")
        p((n, b), x, tmp)
        f((b,), tmp, out)
        ref_tmp = torch.empty_like(tmp)
        triton = reference(b, n, x, ref_tmp)
        triton.batch(1)
        torch.cuda.synchronize()
        assert torch.equal(tmp.view(torch.int32), ref_tmp.view(torch.int32)), (
            "retained reference signature/output"
        )
        row = {
            "shape": [b, n],
            "input_sha256": tensor_sha(x),
            "controls": {},
            "timings": [],
        }
        for name, k in control.items():
            row["controls"][name] = {
                "attributes": k.attributes,
                "block": k.block,
                "shared": k.shared,
            }
        row["controls"]["frozen"] = {
            "attributes": p.kernel.attributes,
            "resources": p.kernel.resources,
        }
        row["controls"]["triton"] = {"attributes": triton.attributes}
        with p.kernel.prepared((n, b), x, tmp):
            paths = {**control, "frozen": p.kernel, "triton": triton}
            if args.ncu:
                selected = (
                    list(paths.values()) if args.ncu == "all" else [paths[args.ncu]]
                )
                for k in selected:
                    for _ in range(20):
                        k.batch(1)
                torch.cuda._sleep(50_000_000)
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                for k in selected:
                    k.batch(1)
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            else:
                names = list(paths)
                for r in range(3):
                    for name in names[r:] + names[:r]:
                        measured = KernelBenchmarker().benchmark_kernel_gpu(paths[name])
                        row["timings"].append(
                            {
                                "round": r,
                                "name": name,
                                **summary([v * 1000 for v in measured["kernel_ms"]]),
                            }
                        )
            results.append(row)
        print([b, n], row["controls"], flush=True)
    result = {
        "complete": True,
        "scope": "CubeK reader and stock component codegen controls use padded split rows; only frozen/reference use original unpadded input; no stock topology qualification or adoption",
        "environment": inventory(torch, "cubecl"),
        "cases": results,
    }
    (
        OUT / (f"ncu-{args.ncu}-{args.case}.json" if args.ncu else "comparisons.json")
    ).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
