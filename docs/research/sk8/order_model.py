"""Float32 models of the row-sum orders, checked bit for bit against the GPU (bead sk8).

order_model.py <data dir> <kind> [tile lanes out.bin]

A row of 8192 is indexed by 13 bits. Triton (R0_BLOCK 8192, 16 warps, 4 values a load) holds
column c = 4 * (32 * warp + lane) + vec + 2048 * rep and sums as a binary tree: inside the thread
over its 16 registers (vec bits, then rep bits), then across lanes xor 16 down to 1, then across
warps the same way. The tile `lanes` kernel (planes split the row into contiguous slices, walk 4,
32 units, 4 lanes) keeps a sum per lane over its 4 steps in sequence, combines the plane's units
xor 1 up to 16, the 16 planes in sequence through the cyclic rounds (plane 0, then 15 down to 1),
then the 4 lanes in sequence.
"""

import sys

import numpy as np

data, kind = sys.argv[1], sys.argv[2]
x = np.fromfile(f"{data}/{kind}.x.bin", np.float32).reshape(-1, 8192)
rows = x.shape[0]
s_ref = np.fromfile(f"{data}/{kind}.f64sums.bin", np.float64)
tri = np.fromfile(f"{data}/{kind}.triton.bin", np.float32).reshape(x.shape)


def bits(row_major):
    """The row as 13 index bits, most significant first (axis 1 + 12 - b is column bit b)."""
    return row_major.reshape(rows, *([2] * 13))


def axis(bit, ndim):
    """The axis of column bit ``bit`` in an array that still holds ``ndim`` bit axes."""
    return 1 + (ndim - 1) - bit


def tree(v, order):
    """Pairwise sums: halves split on each column bit in ``order`` added in float32."""
    live = list(range(13))
    for b in order:
        a = 1 + (len(live) - 1) - live.index(b)
        lo, hi = np.take(v, 0, axis=a), np.take(v, 1, axis=a)
        v = (lo + hi).astype(np.float32)
        live.remove(b)
    return v.reshape(rows)


def seq(v, a):
    """Sequential float32 sum along axis ``a``, starting from 0."""
    acc = np.zeros(np.delete(np.array(v.shape), a), np.float32)
    for i in range(v.shape[a]):
        acc = (acc + np.take(v, i, axis=a)).astype(np.float32)
    return acc


def xor_tree(v, a, descending):
    """Butterfly combine along axis ``a`` of length 2**n, xor bits in the given direction."""
    n = int(np.log2(v.shape[a]))
    order = range(n - 1, -1, -1) if descending else range(n)
    for b in order:
        idx = np.arange(v.shape[a])
        v = (v + np.take(v, idx ^ (1 << b), axis=a)).astype(np.float32)
    return np.take(v, 0, axis=a)


triton_order = [0, 1, 11, 12, 6, 5, 4, 3, 2, 10, 9, 8, 7]


def tile_lanes(units_desc=False, planes="cyclic", lanes="seq", steps="seq"):
    # column c = 512 * plane + 128 * step + 4 * unit + vec
    v = x.reshape(rows, 16, 4, 32, 4)
    per_step = (
        seq(v, 2) if steps == "seq" else xor_tree(v, 2, True)
    )  # (rows, plane, unit, vec)
    per_plane = xor_tree(per_step, 2, units_desc)  # (rows, plane, vec)
    if planes == "cyclic":
        # The four lanes are one line, one chunk: round r takes plane (16 - r) % 16, so plane 0
        # replaces, then planes 15, 14, ..., 1 add.
        out = per_plane[:, 0].copy()
        for r in range(1, 16):
            out = (out + per_plane[:, 16 - r]).astype(np.float32)
    else:
        out = xor_tree(per_plane, 1, True)
    return (
        seq(out[:, :, None].transpose(0, 2, 1), 2)[:, 0]
        if lanes == "seq"
        else xor_tree(out, 1, True)
    )


s_tri = tree(bits(x), triton_order)
e_tri = np.abs(s_tri.astype(np.float64) - s_ref)


def report(name, s32, check=None):
    out = (x * s32[:, None]).astype(np.float32)
    e = np.abs(s32.astype(np.float64) - s_ref)
    same_tri = np.mean(np.all(out.view(np.uint32) == tri.view(np.uint32), axis=1))
    line = (
        f"{name:34s} rows = Triton {same_tri:7.2%} | mean |err| {e.mean():.3e} | "
        f"closer/farther than Triton {int(np.sum(e < e_tri))}/{int(np.sum(e > e_tri))}"
    )
    if check is not None:
        line += f" | rows = GPU {np.mean(np.all(out.view(np.uint32) == check.view(np.uint32), axis=1)):.2%}"
    print(line)


report("triton model", s_tri, tri)
gpu = None
if len(sys.argv) > 3:
    gpu = np.fromfile(sys.argv[3], np.float32).reshape(x.shape)
report("tile lanes model", tile_lanes(), gpu)
report("  + units xor descending", tile_lanes(units_desc=True))
report("  + planes as a tree", tile_lanes(planes="tree"))
report("  + lanes as a tree", tile_lanes(lanes="tree"))
report("  + all three", tile_lanes(units_desc=True, planes="tree", lanes="tree"))
report("  + steps as a tree too", tile_lanes(True, "tree", "tree", "tree"))
