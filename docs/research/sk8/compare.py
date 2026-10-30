"""Compare a candidate's sum-consumer output with Triton's and float64 (bead sk8).

compare.py <data dir> <kind> <candidate out.bin> [label]
"""

import sys

import numpy as np

data, kind, cand = sys.argv[1], sys.argv[2], sys.argv[3]
label = sys.argv[4] if len(sys.argv) > 4 else cand
x = np.fromfile(f"{data}/{kind}.x.bin", np.float32)
rows = np.fromfile(f"{data}/{kind}.f64sums.bin", np.float64).size
x = x.reshape(rows, -1)
ref = np.fromfile(f"{data}/{kind}.f64.bin", np.float64).reshape(x.shape)
s_ref = np.fromfile(f"{data}/{kind}.f64sums.bin", np.float64)
tri = np.fromfile(f"{data}/{kind}.triton.bin", np.float32).reshape(x.shape)
got = np.fromfile(cand, np.float32).reshape(x.shape)

# The row sum each backend used, recovered from out / x where x is not tiny.
col = np.argmax(np.abs(x), axis=1)
r = np.arange(rows)


def implied(out):
    return out[r, col].astype(np.float64) / x[r, col]


s_tri, s_got = implied(tri), implied(got)
e_tri, e_got = np.abs(s_tri - s_ref), np.abs(s_got - s_ref)
same_out = np.mean(got.view(np.uint32) == tri.view(np.uint32))
same_rows = np.mean(np.all(got.view(np.uint32) == tri.view(np.uint32), axis=1))
closer, farther = int(np.sum(e_got < e_tri)), int(np.sum(e_got > e_tri))
print(
    f"{label} [{kind}] out bitwise = Triton: {same_out:.4%} of elements, {same_rows:.2%} of rows | "
    f"sum |err| mean tile {e_got.mean():.3e} triton {e_tri.mean():.3e} | "
    f"rows closer/farther than Triton: {closer}/{farther} | "
    f"max rel out err tile {np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)):.3e} "
    f"triton {np.max(np.abs(tri - ref) / np.maximum(np.abs(ref), 1e-30)):.3e}"
)
