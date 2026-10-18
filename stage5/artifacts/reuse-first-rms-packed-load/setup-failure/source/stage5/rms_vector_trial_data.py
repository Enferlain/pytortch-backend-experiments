"""Exact vector4 PlaneReader/Sum order and bounded trial admission."""

from stage5.rms_trial_data import ROOT, tensor_sha as tensor_sha

OUT = ROOT / "artifacts/reuse-first-rms-vector4"
OLD = ROOT / "artifacts/reuse-first-rms"
WIDTHS = (32, 256, 1024, 4096)
ROWS = (1, 3, 8, 9, 16, 64, 65)


def admitted(shape):
    return len(shape) == 2 and 1 <= shape[0] <= 65 and shape[1] in WIDTHS


def model(x):
    import torch

    assert x.device.type == "cpu" and admitted(x.shape)
    b, n = x.shape
    operand = x.float() * x.float()
    padded = torch.zeros((b, (n + 127) // 128 * 128), dtype=torch.float32)
    padded[:, :n] = operand
    chunks = padded.view(b, -1, 32, 4)
    acc = torch.zeros((b, 32, 4), dtype=torch.float32)
    for k in range(chunks.shape[1]):
        acc = acc + chunks[:, k]
    for offset in (1, 2, 4, 8, 16):
        acc = acc + acc[:, torch.arange(32) ^ offset, :]
    sums = torch.zeros((b, 32), dtype=torch.float32)
    for component in range(4):
        sums = sums + acc[:, :, component]
    factors = torch.rsqrt(sums / n + torch.tensor(1e-5, dtype=torch.float32))
    # Preserve the original scalar output loop lane + 32*k.
    output = (x.float() * factors[:, torch.arange(n) % 32]).to(x.dtype)
    return sums, output
