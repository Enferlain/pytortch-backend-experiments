"""Fixed inputs and exact CPU order for the single CubeK RMS trial."""

from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/reuse-first-rms"
WIDTHS = (31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 4095, 4096, 4099)


def tensor_sha(t):
    import torch

    return hashlib.sha256(
        t.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def fixtures(torch):
    for dtype in ("float32", "float16", "bfloat16"):
        td = getattr(torch, dtype)
        for b, n in ((64, 1024), (64, 255), (16, 4099)):
            torch.manual_seed(8020 + b + n)
            yield (
                f"corpus_{dtype}_{b}_{n}",
                torch.randn((b, n), device="cuda", dtype=td),
            )
        for n in WIDTHS:
            # Each row isolates one arithmetic or value-class obligation.
            x = torch.zeros((10, n), dtype=torch.float32)
            exponent = torch.arange(n) % 41 - 20
            x[0] = torch.pow(2.0, exponent.float())
            x[1, 0], x[1, -1] = 2**20, 2**-20
            x[2, ::2], x[2, 1::2] = 1024, -1024
            x[4].fill_(-0.0)
            x[5].fill_(torch.finfo(td).tiny / 2)
            x[6, 0] = 2**100
            x[7, n // 2] = float("nan")
            x[8, n // 2], x[9, n // 2] = float("inf"), -float("inf")
            yield f"adversarial_{dtype}_10_{n}", x.to(device="cuda", dtype=td)
        for b in (1, 3, 8, 9, 16, 64, 65):
            torch.manual_seed(9810 + b)
            yield (
                f"row_tail_{dtype}_{b}_33",
                torch.randn((b, 33), device="cuda", dtype=td),
            )


def model(x):
    """Stock Lazy PlaneReader/Sum: one chain/lane, then ascending XOR."""
    import torch

    assert x.device.type == "cpu" and x.ndim == 2
    b, n = x.shape
    operand = x.float() * x.float()
    padded = torch.zeros((b, (n + 31) // 32 * 32), dtype=torch.float32)
    padded[:, :n] = operand
    accumulator = torch.zeros((b, 32), dtype=torch.float32)
    for k in range(padded.shape[1] // 32):
        accumulator = accumulator + padded[:, k * 32 : (k + 1) * 32]
    for offset in (1, 2, 4, 8, 16):
        accumulator = accumulator + accumulator[:, torch.arange(32) ^ offset]
    # Use every lane's own result; do not presume root-only interchangeability.
    factors = torch.rsqrt(accumulator / n + torch.tensor(1e-5, dtype=torch.float32))
    output = (x.float() * factors[:, torch.arange(n) % 32]).to(x.dtype)
    return accumulator, output
