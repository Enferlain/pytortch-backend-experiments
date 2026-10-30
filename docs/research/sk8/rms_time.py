"""RMSNorm with a weight, compiled 4 times by one backend (cubecl or triton), for ncu."""

import sys

if sys.argv[1] == "cubecl":
    from cubecl_torch.runtime import guard

    guard.install()
    import cubecl_torch.inductor as ci

    ci.install()
    backend = ci.compile_backend
else:
    backend = "inductor"
import torch

dev = next(
    f"cuda:{i}"
    for i in range(torch.cuda.device_count())
    if "3080" in torch.cuda.get_device_name(i)
)


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w


g = torch.Generator().manual_seed(0)
x = torch.randn(2048, 8192, generator=g).to(dev)
w = torch.randn(8192, generator=g).to(dev)
f = torch.compile(rms, backend=backend, dynamic=False)
for _ in range(6):
    f(x, w)
torch.cuda.synchronize()
