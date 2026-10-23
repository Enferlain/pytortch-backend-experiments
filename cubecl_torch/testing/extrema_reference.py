"""Inductor+Triton reference outputs for float extrema (generic-reduction-combiners).

Inputs are regenerated from seeds with the CPU generator, so only their
SHA-256 and the reference outputs are stored. Regenerate the data file with
the Triton environment from the repository root::

    /home/imi/Projects/sd-scripts/.venv-wsl/bin/python -m cubecl_torch.testing.extrema_reference

Imports only torch: the CubeCL backend never loads here.
"""

import hashlib
import io
import sys
from pathlib import Path

import torch

DATA = Path(__file__).resolve().parent / "data" / "extrema_triton.pt"
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
SPECIALS = (float("nan"), float("inf"), -float("inf"), 0.0, -0.0)

POINTWISE = {
    "maximum": lambda a, b: torch.maximum(a, b),
    "minimum": lambda a, b: torch.minimum(a, b),
    "fmax": lambda a, b: torch.fmax(a, b),
    "fmin": lambda a, b: torch.fmin(a, b),
    "clamp": lambda a, b: torch.clamp(a, -0.5, 0.5),
    "clamp_min": lambda a, b: torch.clamp_min(a, 0.0),
    "isnan": lambda a, b: torch.isnan(a),
    "fused_maximum": lambda a, b: torch.maximum(a * 2, b) + 1,
}


def inputs(seed, shape, dtype, count):
    """Seeded normals with specials, every (+-0, +-0) order and NaN position."""
    g = torch.Generator().manual_seed(seed)
    tensors = []
    for i in range(count):
        x = torch.randn(shape, generator=g)
        flat = x.view(-1)
        n = flat.numel()
        pick = torch.randint(0, len(SPECIALS), (n,), generator=g)
        where = torch.rand(n, generator=g) < 0.125
        flat[where] = torch.tensor(SPECIALS)[pick[where]]
        # Fixed head: both zero orders against each operand, and NaN pairs.
        head = (
            [-0.0, 0.0, 0.0, -0.0, float("nan"), 1.0, float("nan")]
            if i == 0
            else [0.0, -0.0, -0.0, 0.0, 1.0, float("nan"), float("nan")]
        )
        flat[: len(head)] = torch.tensor(head[: min(len(head), n)])
        tensors.append(x.to(dtype))
    return tensors


REDUCTIONS = {
    "amax_amin": lambda x: torch.stack([x.amax(-1), x.amin(-1)]),
}
REDUCTION_SHAPES = ((37, 129), (300, 4096), (8, 65539), (4, 262147), (2, 1048579))
# Row patterns, cycled over rows and shifted per input so that every shape,
# split ones included, sees each of them.
PATTERNS = (
    "finite",
    "nan_middle",
    "nan_first",
    "nan_last",
    "zeros_mixed",
    "negative_zeros_one_positive",
    "negative_infinity",
    "infinities",
)


def reduction_inputs(seed, shape, dtype, count):
    g = torch.Generator().manual_seed(seed)
    rows, n = shape
    tensors = []
    for i in range(count):
        x = torch.randn(shape, generator=g)
        for r in range(rows):
            pattern = PATTERNS[(r + i) % len(PATTERNS)]
            row = x[r]
            spot = int(torch.randint(1, n - 1, (1,), generator=g))
            if pattern == "nan_middle":
                row[spot] = float("nan")
            elif pattern == "nan_first":
                row[0] = float("nan")
            elif pattern == "nan_last":
                row[-1] = float("nan")
            elif pattern == "zeros_mixed":
                negative = torch.rand(n, generator=g) < 0.5
                row.copy_(torch.where(negative, -0.0, 0.0))
                row[0], row[spot] = -0.0, 0.0
            elif pattern == "negative_zeros_one_positive":
                row.fill_(-0.0)
                row[spot] = 0.0
            elif pattern == "negative_infinity":
                row.fill_(-float("inf"))
            elif pattern == "infinities":
                row[spot] = float("inf")
                row[(spot + n // 2) % n] = -float("inf")
        tensors.append(x.to(dtype))
    return tensors


def digest(t):
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def cases():
    for dtype in DTYPES:
        for name in POINTWISE:
            for dynamic in (
                (False, True) if name in ("maximum", "minimum") else (False,)
            ):
                yield {
                    "name": f"{name}-{str(dtype).removeprefix('torch.')}"
                    + ("-dynamic" if dynamic else ""),
                    "kind": "pointwise",
                    "fn": name,
                    "dtype": dtype,
                    "shape": (4099,),
                    "dynamic": dynamic,
                    "seed": 31,
                    "count": 2,
                }

    for dtype in DTYPES:
        for shape in REDUCTION_SHAPES:
            for dynamic in (False, True):
                yield {
                    "name": f"amax_amin-{str(dtype).removeprefix('torch.')}"
                    + f"-{shape[0]}x{shape[1]}"
                    + ("-dynamic" if dynamic else ""),
                    "kind": "reduction",
                    "fn": "amax_amin",
                    "dtype": dtype,
                    "shape": shape,
                    "dynamic": dynamic,
                    "seed": 37,
                    "count": len(PATTERNS),
                }


def function(case):
    return (POINTWISE if case["kind"] == "pointwise" else REDUCTIONS)[case["fn"]]


def case_inputs(case):
    make = inputs if case["kind"] == "pointwise" else reduction_inputs
    return make(case["seed"], case["shape"], case["dtype"], case["count"])


def run(compiled, case, args):
    """Pointwise cases take every input at once; reductions run per input."""
    if case["kind"] == "pointwise":
        return compiled(*(a.cuda() for a in args)).cpu()
    return torch.stack([compiled(a.cuda()).cpu() for a in args])


def generate(path=DATA):
    import triton  # noqa: F401  the reference must come from Inductor+Triton

    records = {}
    for case in cases():
        torch._dynamo.reset()
        args = case_inputs(case)
        compiled = torch.compile(function(case), dynamic=case["dynamic"])
        records[case["name"]] = {
            "input_sha256": [digest(a) for a in args],
            "output": run(compiled, case, args),
        }
    payload = {"torch": str(torch.__version__), "cases": records}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path.write_bytes(buffer.getvalue())
    return path


if __name__ == "__main__":
    sys.stdout.write(f"wrote {generate()}\n")
