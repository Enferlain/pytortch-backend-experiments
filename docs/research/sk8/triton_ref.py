"""Inputs, Triton outputs and float64 references for the sum consumer (bead sk8).

Run with the Triton reference venv. Writes <data>/<kind>.x.bin, .triton.bin, .triton.sums.bin,
.f64.bin (float64 out) and .f64sums.bin, plus triton_config.txt and triton_code.py.
"""

import os
import sys
from pathlib import Path

import torch
from torch._inductor.runtime import triton_heuristics
from torch._inductor.utils import run_and_get_code

data = sys.argv[1]
rows, cols = int(sys.argv[2]), int(sys.argv[3])
dev = next(
    f"cuda:{i}"
    for i in range(torch.cuda.device_count())
    if "3080" in torch.cuda.get_device_name(i)
)
KINDS = {"normal": (0.0, 1.0), "offset": (1e3, 1.0)}

configs = []
orig = triton_heuristics.CachingAutotuner.run


def run(self, *a, **k):
    configs.append(
        (self.fn.__name__, [str(launcher.config) for launcher in self.launchers])
    )
    return orig(self, *a, **k)


triton_heuristics.CachingAutotuner.run = run


def consumer(x):
    return x * x.sum(-1, keepdim=True)


def sums(x):
    return x.sum(-1)


fc = torch.compile(consumer, dynamic=False)
fs = torch.compile(sums, dynamic=False)
for i, (kind, (mean, std)) in enumerate(KINDS.items()):
    g = torch.Generator().manual_seed(1234 + i)
    x = torch.randn((rows, cols), generator=g) * std + mean
    x.numpy().tofile(f"{data}/{kind}.x.bin")
    xd = x.to(dev)
    out, code = run_and_get_code(fc, xd)
    out.cpu().numpy().tofile(f"{data}/{kind}.triton.bin")
    fs(xd).cpu().numpy().tofile(f"{data}/{kind}.triton.sums.bin")
    s64 = x.double().sum(-1, keepdim=True)
    (x.double() * s64).numpy().tofile(f"{data}/{kind}.f64.bin")
    s64.numpy().tofile(f"{data}/{kind}.f64sums.bin")
    if i == 0:
        with open(f"{data}/triton_code.py", "w") as f:
            f.write(code[0])
with open(f"{data}/triton_config.txt", "w") as f:
    f.write(
        f"{torch.cuda.get_device_name(dev)} {os.environ.get('TORCHINDUCTOR_CACHE_DIR')}\n"
    )
    f.writelines(f"{c}\n" for c in configs)
print(Path(f"{data}/triton_config.txt").read_text())
