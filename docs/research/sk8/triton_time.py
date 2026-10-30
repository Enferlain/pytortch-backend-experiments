"""Run the compiled sum consumer a few times, for ncu (bead sk8).

triton_time.py <x.bin>; run with the Triton reference venv."""

import sys

import numpy as np
import torch

dev = next(
    f"cuda:{i}"
    for i in range(torch.cuda.device_count())
    if sys.argv[2] in torch.cuda.get_device_name(i)
)
x = torch.from_numpy(np.fromfile(sys.argv[1], np.float32).reshape(2048, 8192)).to(dev)
f = torch.compile(lambda x: x * x.sum(-1, keepdim=True), dynamic=False)
for _ in range(4):
    f(x)
torch.cuda.synchronize()
