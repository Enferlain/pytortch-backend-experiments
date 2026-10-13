"""Per-operation numerical comparison, independent of performance configuration."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "--impl",
    required=True,
    choices=("cubecl", "triton_matched", "triton_default", "aten"),
)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache/math" / args.impl)
if args.impl == "cubecl":
    from stage1 import evidence

    evidence.guard()
import torch

if args.impl == "cubecl":
    from stage1.overlay import install

    install()
    from stage5.backend import compile_backend
    from functools import partial

    backend = partial(compile_backend, tuning="baseline")
elif args.impl.startswith("triton"):
    from torch._inductor.compile_fx import compile_fx

    patches = {
        "triton.cudagraphs": False,
        "compile_threads": 1,
        "fx_graph_cache": False,
    }
    if args.impl == "triton_matched":
        patches.update(
            {
                "emulate_precision_casts": True,
                "joint_graph_constant_folding": False,
                "eager_numerics.division_rounding": True,
                "eager_numerics.disable_ftz": True,
                "eager_numerics.use_pytorch_libdevice": True,
            }
        )
    backend = lambda gm, inputs: compile_fx(gm, inputs, config_patches=patches)
functions = {
    "add": lambda x, y: x + y,
    "mul": lambda x, y: x * y,
    "contract_sensitive": lambda x, y: x * y + x,
    "division": lambda x, y: x / y,
    "sin": lambda x, y: torch.sin(x),
    "cos": lambda x, y: torch.cos(x),
    "exp": lambda x, y: torch.exp(x),
    "sqrt": lambda x, y: torch.sqrt(x),
    "reciprocal": lambda x, y: torch.reciprocal(x),
    "precision_chain": lambda x, y: (x * y + x) * 0.333333333333,
    "where": lambda x, y: torch.where(x > y, x, y),
    "tensor_constant": lambda x, y: torch.full_like(x, 0.333333333333) * x,
    "constant_output": lambda x, y: torch.full_like(x, 0.333333333333),
    "explicit_cast": lambda x, y: (x.float() * y.float()).to(x.dtype),
}
rows = []
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    torch.manual_seed(6789)
    x = torch.cat(
        [
            torch.tensor(
                [
                    0.0,
                    -0.0,
                    float("nan"),
                    float("inf"),
                    -float("inf"),
                    1e-38,
                    -1e-38,
                    1e-30,
                    -1e-30,
                    1e10,
                    -1e10,
                    1.0,
                    -1.0,
                    3.14159265,
                    65504.0,
                    0.00006103515625,
                ],
                device="cuda",
                dtype=dtype,
            ),
            torch.randn(4099, device="cuda", dtype=dtype) * 3,
        ]
    )
    y = torch.cat(
        [
            torch.tensor(
                [
                    -0.0,
                    0.0,
                    2.0,
                    float("inf"),
                    -float("inf"),
                    2.0,
                    2.0,
                    0.3,
                    0.3,
                    1.000000119,
                    1.000000119,
                    -1.0,
                    1.0,
                    0.33333,
                    2.0,
                    0.0001220703125,
                ],
                device="cuda",
                dtype=dtype,
            ),
            torch.randn(4099, device="cuda", dtype=dtype),
        ]
    )
    for name, fn in functions.items():
        compiled = (
            fn
            if args.impl == "aten"
            else torch.compile(fn, backend=backend, fullgraph=True)
        )
        out = compiled(x, y)
        ref = fn(x, y)
        torch.cuda.synchronize()
        raw = out.view(torch.uint8).cpu().numpy().tobytes()
        raw_ref = ref.view(torch.uint8).cpu().numpy().tobytes()
        equal = (out == ref) | (torch.isnan(out) & torch.isnan(ref))
        finite = torch.isfinite(out) & torch.isfinite(ref)
        zeros = (out == 0) & (ref == 0)
        classification = torch.equal(
            torch.isnan(out), torch.isnan(ref)
        ) and torch.equal(torch.isinf(out), torch.isinf(ref))
        sign_zero = torch.equal(torch.signbit(out[zeros]), torch.signbit(ref[zeros]))
        rows.append(
            {
                "op": name,
                "dtype": str(dtype),
                "classification_equal": classification,
                "zero_sign_equal": sign_zero,
                "numeric_equal_fraction": float(equal.float().mean()),
                "bitwise_equal": raw == raw_ref,
                "max_abs_error_finite": float((out[finite] - ref[finite]).abs().max())
                if finite.any()
                else 0,
                "output_sha256": hashlib.sha256(raw).hexdigest(),
                "reference_sha256": hashlib.sha256(raw_ref).hexdigest(),
                "output_bytes_hex": raw.hex(),
                "reference_bytes_hex": raw_ref.hex(),
            }
        )
        print(
            args.impl,
            name,
            dtype,
            "classification",
            classification,
            "zero sign",
            sign_zero,
            "exact",
            float(equal.float().mean()),
            flush=True,
        )
result = {
    "implementation": args.impl,
    "checks": rows,
    "environment": {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gil": sys._is_gil_enabled(),
        "triton_libdevice": os.environ.get("TRITON_LIBDEVICE_PATH"),
    },
    "independence": evidence.inventory() if args.impl == "cubecl" else None,
}
(ROOT / "artifacts" / ("math-" + args.impl + ".json")).write_text(
    json.dumps(result, indent=2)
)
