"""Inductor+Triton reference data for Welford reductions (welford-reductions).

Inputs are regenerated from seeds with the CPU generator, so only their
SHA-256, fp64 statistics and Triton's results are stored. Normalized norm
outputs are large, so for those only Triton's error against fp64 is kept: in
total, and per normalized row (the band's sign test pairs rows).
Each case also records the Triton kernels Inductor generated: persistent or
looped, the algorithm (Welford, or two-pass: Inductor's sum-based two-step
variance, or a persistent kernel's welford_reduce_fallback), and the split.
Low-precision norm statistics are stored in the dtype Triton returned; compare
them against the fp64 values, not against eager's dtype. Regenerate the data file with
the Triton environment from the repository root (``--new`` adds only the cases
missing from the file and keeps the stored ones)::

    /home/imi/Projects/sd-scripts/.venv-wsl/bin/python -m cubecl_torch.testing.welford_reference [--new]

Imports only torch: the CubeCL backend never loads here.
"""

import hashlib
import io
import re
import sys
from pathlib import Path

import torch

DATA = Path(__file__).resolve().parent / "data" / "welford_triton.pt"
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
ROW_SHAPES = ((37, 129), (300, 4096), (8, 300000), (2, 1048579))
# Zero-mean normals, and an offset that stresses the variance formula.
DATA_KINDS = {"normal": (0.0, 1.0), "offset": (1e3, 1.0)}
# Eight inputs per case; the band test pairs rows across all of them.
COUNT = 8
EPS = 1e-5
GROUP_NORM_SHAPES = ((1, 320, 64, 64), (1, 640, 32, 32), (2, 64, 8, 8))
# A sum fused with a Welford variance over the same rows (multi-output-kernels).
MIXED_SHAPES = ((37, 129), (300, 4096), (8, 300000))
MIXED_DTYPES = (torch.float32, torch.bfloat16)
# A low-precision row sum consumed in its own kernel: Triton keeps it in
# float32 there (multi-output-kernels group 5).
SUM_CONSUMER_DTYPES = (torch.float16, torch.bfloat16)
LAYER_NORM_SHAPES = ((4096, 1280),)
# Rows where Triton runs 16 warps: one 2048-element pass, and a long row
# (wide-reduction-workgroups). Two inputs each: the rows give the pairs.
WIDE_LAYER_NORM_SHAPES = ((4096, 2048), (2048, 8192))
WIDE_COUNT = 2
# Rows of the special-value case, cycled over rows.
PATTERNS = (
    "finite",
    "nan_first",
    "nan_middle",
    "nan_last",
    "constant",
    "positive_infinity",
    "negative_infinity",
    "all_positive_infinity",
)


def digest(t):
    h = hashlib.sha256(f"{t.dtype}{tuple(t.shape)}".encode())
    h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def row_inputs(seed, shape, dtype, kind, count=COUNT):
    g = torch.Generator().manual_seed(seed)
    mean, std = DATA_KINDS[kind]
    return [
        (torch.randn(shape, generator=g) * std + mean).to(dtype) for _ in range(count)
    ]


def special_inputs(seed, shape, dtype, count=COUNT):
    g = torch.Generator().manual_seed(seed)
    rows, n = shape
    tensors = []
    for i in range(count):
        x = torch.randn(shape, generator=g)
        for r in range(rows):
            pattern = PATTERNS[(r + i) % len(PATTERNS)]
            row = x[r]
            spot = int(torch.randint(1, n - 1, (1,), generator=g))
            if pattern == "nan_first":
                row[0] = float("nan")
            elif pattern == "nan_middle":
                row[spot] = float("nan")
            elif pattern == "nan_last":
                row[-1] = float("nan")
            elif pattern == "constant":
                row.fill_(0.375 + r)
            elif pattern == "positive_infinity":
                row[spot] = float("inf")
            elif pattern == "negative_infinity":
                row[spot] = -float("inf")
            elif pattern == "all_positive_infinity":
                row.fill_(float("inf"))
        tensors.append(x.to(dtype))
    return tensors


def norm_inputs(seed, case):
    g = torch.Generator().manual_seed(seed)
    shape, dtype = case["shape"], case["dtype"]
    channels = shape[1] if case["kind"] == "group_norm" else shape[-1]
    result = []
    for _ in range(case.get("count", COUNT)):
        x = torch.randn(shape, generator=g).to(dtype)
        if case["kind"] == "layer_norm_plain":
            result.append((x,))
            continue
        weight = (1 + 0.1 * torch.randn(channels, generator=g)).to(dtype)
        bias = (0.1 * torch.randn(channels, generator=g)).to(dtype)
        result.append((x, weight, bias))
    return result


def var_mean(x):
    return torch.var_mean(x, -1, correction=0)


def group_norm(x, weight, bias):
    n, c = x.shape[:2]
    return torch.ops.aten.native_group_norm(
        x, weight, bias, n, c, x[0, 0].numel(), 32, EPS
    )


def layer_norm(x, weight, bias):
    return torch.ops.aten.native_layer_norm(x, x.shape[-1:], weight, bias, EPS)


def layer_norm_plain(x):
    """LayerNorm without affine parameters."""
    return layer_norm(x, None, None)


def mixed(x):
    """x.mean is a sum, x.std a Welford variance: one kernel reduces both,
    beside a plain row sum."""
    normalized = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
    return normalized, x.sum(-1)


def sum_consumer(x):
    return x * x.sum(-1, keepdim=True), x.sum(-1)


FUNCTIONS = {
    "mixed": mixed,
    "sum_consumer": sum_consumer,
    "var_mean": var_mean,
    "special": var_mean,
    "group_norm": group_norm,
    "layer_norm": layer_norm,
    "layer_norm_plain": layer_norm_plain,
}


def cases():
    yield from base_cases()
    yield from mixed_cases()
    yield from wide_cases()


def wide_cases():
    for dtype in DTYPES:
        name = str(dtype).removeprefix("torch.")
        for shape in WIDE_LAYER_NORM_SHAPES:
            for kind in ("layer_norm", "layer_norm_plain"):
                yield {
                    "name": f"{kind}-{name}-{shape[0]}x{shape[1]}",
                    "kind": kind,
                    "dtype": dtype,
                    "shape": shape,
                    "dynamic": False,
                    "seed": 67,
                    "count": WIDE_COUNT,
                }


def base_cases():
    for dtype in DTYPES:
        name = str(dtype).removeprefix("torch.")
        for shape in ROW_SHAPES:
            for kind in DATA_KINDS:
                for dynamic in (False, True):
                    yield {
                        "name": f"var_mean-{name}-{shape[0]}x{shape[1]}-{kind}"
                        + ("-dynamic" if dynamic else ""),
                        "kind": "var_mean",
                        "data": kind,
                        "dtype": dtype,
                        "shape": shape,
                        "dynamic": dynamic,
                        "seed": 41,
                    }
        for shape in ((37, 129), (8, 300000), (2, 1048579)):
            yield {
                "name": f"special-{name}-{shape[0]}x{shape[1]}",
                "kind": "special",
                "dtype": dtype,
                "shape": shape,
                "dynamic": False,
                "seed": 43,
            }
        for kind, shapes, seed in (
            ("group_norm", GROUP_NORM_SHAPES, 47),
            ("layer_norm", LAYER_NORM_SHAPES, 53),
        ):
            for shape in shapes:
                # The largest shape also runs dynamic: an unsplit Welford there.
                for dynamic in (False, True) if shape == shapes[0] else (False,):
                    yield {
                        "name": f"{kind}-{name}-"
                        + "x".join(map(str, shape))
                        + ("-dynamic" if dynamic else ""),
                        "kind": kind,
                        "dtype": dtype,
                        "shape": shape,
                        "dynamic": dynamic,
                        "seed": seed,
                    }


def mixed_cases():
    for dtype in SUM_CONSUMER_DTYPES:
        name = str(dtype).removeprefix("torch.")
        for shape in MIXED_SHAPES:
            yield {
                "name": f"sum_consumer-{name}-{shape[0]}x{shape[1]}-normal",
                "kind": "sum_consumer",
                "data": "normal",
                "dtype": dtype,
                "shape": shape,
                "dynamic": False,
                "seed": 61,
            }
    for dtype in MIXED_DTYPES:
        name = str(dtype).removeprefix("torch.")
        for shape in MIXED_SHAPES:
            for kind in DATA_KINDS:
                yield {
                    "name": f"mixed-{name}-{shape[0]}x{shape[1]}-{kind}",
                    "kind": "mixed",
                    "data": kind,
                    "dtype": dtype,
                    "shape": shape,
                    "dynamic": False,
                    "seed": 59,
                }


def case_inputs(case):
    """A list of argument tuples, one per input."""
    if case["kind"] in ("var_mean", "mixed", "sum_consumer"):
        tensors = row_inputs(case["seed"], case["shape"], case["dtype"], case["data"])
    elif case["kind"] == "special":
        tensors = special_inputs(case["seed"], case["shape"], case["dtype"])
    else:
        return norm_inputs(case["seed"], case)
    return [(t,) for t in tensors]


def fp64_reference(case, args):
    """fp64 statistics of the stored (already rounded) inputs."""
    x = args[0].double()
    if case["kind"] in ("var_mean", "special"):
        var, mean = torch.var_mean(x, -1, correction=0)
        return {"var": var, "mean": mean}
    if case["kind"] in ("mixed", "sum_consumer"):
        out, total = FUNCTIONS[case["kind"]](x)
        return {"out": out, "sum": total}
    if case["kind"] == "group_norm":
        out, mean, rstd = group_norm(x, args[1].double(), args[2].double())
    elif case["kind"] == "layer_norm_plain":
        out, mean, rstd = layer_norm_plain(x)
    else:
        out, mean, rstd = layer_norm(x, args[1].double(), args[2].double())
    return {"out": out, "mean": mean, "rstd": rstd}


def errors(value, reference):
    """Error statistics against fp64 over finite reference elements."""
    finite = reference.isfinite()
    if not finite.any():
        raise ValueError("no finite reference elements")
    diff = (value.double() - reference).abs()[finite]
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "finite": int(finite.sum()),
        "nonfinite_mismatch": int(
            (value.double().isfinite() != finite).sum()
            + (value.isnan() != reference.isnan()).sum()
        ),
    }


def row_errors(value, reference, rows):
    """Mean absolute error against fp64 of each of ``rows`` contiguous rows,
    over finite reference elements. A NaN in ``value`` there propagates; a row
    with no finite reference element is NaN."""
    diff = (value.double() - reference).abs().reshape(rows, -1)
    finite = reference.reshape(rows, -1).isfinite()
    return diff.where(finite, 0.0).sum(1) / finite.sum(1)


def algorithm(body):
    if "welford_reduce(" in body:
        return "welford"
    if "welford_combine(" in body or "triton_helpers.welford(" in body:
        return "welford_combine"
    return "two_pass" if "tl.sum(" in body else None


def split(found):
    """Partials per row: the second reduction stage's extent, else 1."""
    reductions = [k for k in found if not k["name"].startswith("triton_poi")]
    return reductions[1]["r0_numel"] if len(reductions) > 1 else 1


def kernels(sources):
    """Generated Triton kernels: persistent or looped, and their algorithm."""
    found = []
    for source in sources:
        for match in re.finditer(
            r"def (triton_(per|red|poi)_\w+)\(.*?\n(.*?)(?=\n@triton|\n'''|\Z)",
            source,
            re.DOTALL,
        ):
            body = match.group(3)
            numel = dict(re.findall(r"(xnumel|r0_numel) = (\d+)", body))
            found.append(
                {
                    "name": match.group(1),
                    "persistent": match.group(2) == "per",
                    "algorithm": algorithm(body),
                    "xnumel": int(numel["xnumel"]) if "xnumel" in numel else None,
                    "r0_numel": int(numel["r0_numel"]) if "r0_numel" in numel else None,
                }
            )
    return found


def generate(path=DATA, *, new_only=False):
    """Record every case, or with ``new_only`` add the cases missing from the
    data file and keep the stored ones (Triton's autotuning can pick another
    config on a rerun, which would move unrelated stored results)."""
    import triton  # noqa: F401  the reference must come from Inductor+Triton
    from torch._inductor.utils import run_and_get_code

    records = (
        torch.load(path, weights_only=False)["cases"]
        if new_only and path.exists()
        else {}
    )
    for case in cases():
        if case["name"] in records:
            continue
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
            value = [v.cpu() for v in value]
            reference = fp64_reference(case, args)
            if case["kind"] in ("var_mean", "special"):
                result = {"var": value[0], "mean": value[1]}
            elif case["kind"] in ("mixed", "sum_consumer"):
                result = {
                    "sum": value[1],
                    "out_error": errors(value[0], reference["out"]),
                    "out_row_error": row_errors(
                        value[0], reference["out"], value[1].numel()
                    ),
                }
            else:
                result = {
                    "mean": value[1],
                    "rstd": value[2],
                    "out_error": errors(value[0], reference["out"]),
                    "out_row_error": row_errors(
                        value[0], reference["out"], value[1].numel()
                    ),
                }
            result["fp64"] = {k: v for k, v in reference.items() if k != "out"}
            results.append(result)
        records[case["name"]] = {
            "torch": str(torch.__version__),
            "input_sha256": [[digest(a) for a in args] for args in inputs],
            "results": results,
            "kernels": kernels(sources),
            "split": split(kernels(sources)),
        }
    # Each case records the torch it was generated with; the file's label is
    # the one of its first full generation.
    version = (
        torch.load(path, weights_only=False)["torch"]
        if new_only and path.exists()
        else str(torch.__version__)
    )
    payload = {"torch": version, "cases": records}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path.write_bytes(buffer.getvalue())
    return path


if __name__ == "__main__":
    sys.stdout.write(f"wrote {generate(new_only='--new' in sys.argv)}\n")
