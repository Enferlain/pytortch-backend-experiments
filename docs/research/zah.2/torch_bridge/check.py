"""Bridge prototype checks (bead zah.2): CubeCL/CubeK kernels on PyTorch tensors and streams.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
        stage0/.venv/bin/python docs/research/zah.2/torch_bridge/check.py [graph]
"""

import ctypes
import sys
import time
from pathlib import Path

import torch

LIB = ctypes.CDLL(str(Path(__file__).parent / "target/release/libtorch_bridge.so"))
for name, args in {
    "bridge_init": [],
    "bridge_touch": [ctypes.c_uint64] * 3,
    "bridge_sum_rows": [ctypes.c_uint64] * 5,
}.items():
    getattr(LIB, name).argtypes = args
    getattr(LIB, name).restype = ctypes.c_int


def stream():
    return torch.cuda.current_stream().cuda_stream


def sum_rows(x, out):
    rows, cols = x.shape
    assert LIB.bridge_sum_rows(x.data_ptr(), out.data_ptr(), rows, cols, stream()) == 0


def correctness():
    torch.manual_seed(0)
    for rows, cols in [(2048, 8192), (300, 4096), (7, 2049), (64, 31)]:
        x = torch.randn(rows, cols, device="cuda")
        out = torch.empty(rows, device="cuda")
        sum_rows(x, out)
        ref = x.double().sum(-1)
        # Float32 summation: within 64 units of rounding over the summed magnitudes.
        bound = 64 * torch.finfo(torch.float32).eps * x.double().abs().sum(-1)
        ratio = ((out.double() - ref).abs() / bound).max().item()
        print(f"correctness {rows}x{cols}: max error {ratio:.3f} of the float32 bound")
        assert ratio <= 1


def ordering(side_stream):
    """PyTorch work queued right before and after the kernel, no host sync between."""
    torch.manual_seed(1)
    s = torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    with torch.cuda.stream(s):
        x = torch.randn(4096, 8192, device="cuda")
        out = torch.empty(4096, device="cuda")
        for i in range(20):
            y = x * (i + 1)  # producer: must finish before the kernel reads y
            sum_rows(y, out)
            z = out + 0  # consumer: must see this iteration's sums
            ref = x.sum(-1) * (i + 1)
            torch.testing.assert_close(z, ref, rtol=1e-4, atol=1e-2)
        s.synchronize()
    print(f"ordering on {'a side' if side_stream else 'the default'} stream: ok")


def dispatch():
    x = torch.zeros(32, device="cuda")
    n = 20000
    for _ in range(2000):
        LIB.bridge_touch(x.data_ptr(), 32, stream())
    torch.cuda.synchronize()
    for r in range(3):
        t = time.perf_counter()
        for _ in range(n):
            LIB.bridge_touch(x.data_ptr(), 32, stream())
        sub = time.perf_counter() - t
        torch.cuda.synchronize()
        print(f"dispatch round {r}: {sub / n * 1e6:.2f} us per call from Python")


def graph():
    torch.manual_seed(2)
    x = torch.randn(1024, 4096, device="cuda")
    out = torch.empty(1024, device="cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        sum_rows(x, out)  # warm: compile before capture
    s.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        sum_rows(x, out)
    x.copy_(torch.randn_like(x))
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x.sum(-1), rtol=1e-4, atol=1e-2)
    print("graph capture and replay: ok")


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0))
    LIB.bridge_init()
    correctness()
    ordering(side_stream=False)
    ordering(side_stream=True)
    dispatch()
    if "graph" in sys.argv:
        graph()
