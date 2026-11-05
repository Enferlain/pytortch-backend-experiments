"""Inductor+Triton reference data for RMSNorm's weight gradient
(fused-reduction-bridge task 5.1), for the band (``welford_band.measure``).

The graph is the timing harness's ``weight_gradient``: a column sum, which
Inductor splits at 8192x1280. The split count depends on the GPU, so the
references are recorded per GPU (``DATA[gpu]``) and the band compares on the
same GPU, at matched settings. Inputs are regenerated from seeds with the CPU
generator; only their SHA-256, fp64 sums and Triton's results are stored.
Regenerate with the Triton environment from the repository root::

    /home/imi/Projects/sd-scripts/.venv-wsl/bin/python \\
        -m cubecl_torch.testing.fused_reduction_reference --gpu 3080

Imports only torch: the CubeCL backend never loads here.
"""

import argparse
import io
import sys
from pathlib import Path

import torch

from cubecl_torch.testing.extrema_timing import DEVICES
from cubecl_torch.testing.fused_reduction_timing import SHAPES, weight_gradient
from cubecl_torch.testing.welford_reference import digest, kernels, row_errors, split

__all__ = ["digest", "row_errors"]

DIRECTORY = Path(__file__).resolve().parent / "data"
DATA = {gpu: DIRECTORY / f"fused_reduction_triton-{gpu}.pt" for gpu in DEVICES}
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
SEED = 83


def gradient(g, x, rstd):
    return (weight_gradient(g, x, rstd),)


FUNCTIONS = {"weight_gradient": gradient}


def cases():
    for dtype in DTYPES:
        name = str(dtype).removeprefix("torch.")
        for shape in SHAPES:
            yield {
                "name": f"weight_gradient-{name}-{shape[0]}x{shape[1]}",
                "kind": "weight_gradient",
                "dtype": dtype,
                "shape": shape,
                "dynamic": False,
                "seed": SEED,
                # Each output column is one pair; the large shape has 1280 per input.
                "count": 2 if shape[0] >= 1024 else 8,
            }


def case_inputs(case):
    """A list of ``(g, x, rstd)`` tuples, one per input."""
    generator = torch.Generator().manual_seed(case["seed"])
    rows, _ = case["shape"]
    dtype = case["dtype"]
    return [
        (
            torch.randn(case["shape"], generator=generator).to(dtype),
            torch.randn(case["shape"], generator=generator).to(dtype),
            (torch.rand(rows, generator=generator) + 0.5).to(dtype),
        )
        for _ in range(case["count"])
    ]


def fp64_reference(case, args):
    """The fp64 sum of the stored (already rounded) inputs."""
    g, x, rstd = (a.double() for a in args)
    return {"sum": (g * x * rstd[:, None]).sum(0)}


def generate(gpu):
    import triton  # noqa: F401  the reference must come from Inductor+Triton
    from torch._inductor.utils import run_and_get_code

    torch.cuda.set_device(
        next(
            i
            for i in range(torch.cuda.device_count())
            if torch.cuda.get_device_name(i) == DEVICES[gpu]
        )
    )
    records = {}
    for case in cases():
        torch._dynamo.reset()
        inputs = case_inputs(case)
        compiled = torch.compile(FUNCTIONS[case["kind"]], dynamic=case["dynamic"])
        results, sources = [], []
        for i, args in enumerate(inputs):
            device_args = [a.cuda() for a in args]
            if i == 0:
                value, sources = run_and_get_code(compiled, *device_args)
            else:
                value = compiled(*device_args)
            results.append({"sum": value[0].cpu(), "fp64": fp64_reference(case, args)})
        found = kernels(sources)
        records[case["name"]] = {
            "torch": str(torch.__version__),
            "device": torch.cuda.get_device_name(),
            "input_sha256": [[digest(a) for a in args] for args in inputs],
            "results": results,
            "kernels": found,
            "split": split(found),
        }
    buffer = io.BytesIO()
    torch.save({"torch": str(torch.__version__), "cases": records}, buffer)
    DATA[gpu].write_bytes(buffer.getvalue())
    return DATA[gpu]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=sorted(DEVICES), required=True)
    options = parser.parse_args(argv)
    sys.stdout.write(f"wrote {generate(options.gpu)}\n")


if __name__ == "__main__":
    main()
