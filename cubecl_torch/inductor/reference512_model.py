"""CPU FP32 model of the frozen reference512 partial (from Stage 5's
``reduction_topology_model`` and ``reduction_rounding_probe``).

Only the arithmetic the first-use oracle needs: no evidence or file IO.
"""


def butterfly(values, distances):
    import torch

    lanes = torch.arange(values.shape[-1])
    for distance in distances:
        values = values + values[..., lanes ^ distance]
    return values


def finish_model(partials):
    import torch

    values = torch.zeros((*partials.shape[:-1], 256), dtype=torch.float32)
    values[..., : partials.shape[-1]] = partials
    for distance in (128, 64, 32, 16, 8, 4, 2, 1):
        values = values[..., :distance] + values[..., distance : 2 * distance]
    return values[..., 0]


def model(x):
    import torch

    assert x.device.type == "cpu" and x.ndim == 2
    b, n = x.shape
    extent = (n + 8) // 9
    data = torch.zeros((b, 9 * extent), dtype=torch.float32)
    data[:, :n] = x.float()
    data = data.reshape(b, 9, extent)
    steps = (extent + 2047) // 2048
    tiled = torch.zeros((b, 9, steps * 2048), dtype=torch.float32)
    tiled[..., :extent] = data
    chains = torch.zeros((b, 9, 4, 512), dtype=torch.float32)
    # Thread t, chain j, loop k: r=t+512*j+2048*k. LLVM selects the
    # old chain when r>=extent, but adds a protected zero when part*extent+r>=n.
    coords = torch.arange(2048).reshape(4, 512)
    for k in range(steps):
        pending = tiled[..., k * 2048 : (k + 1) * 2048].reshape(b, 9, 4, 512)
        chains = torch.where(coords + k * 2048 < extent, chains + pending, chains)
    a0, a1, a2, a3 = chains.unbind(-2)
    private = (a0 + a1) + (a2 + a3)
    # Every lane has the warp sum. All 32 lanes store that same value to
    # shared[warp] in the retained code (byte address (tid>>3)&60).
    warps = butterfly(private.reshape(b, 9, 16, 32), (16, 8, 4, 2, 1))
    shared = warps[..., 0]
    # Every warp reloads shared[tid&15]; each half-warp gets all 16 values.
    replicated = shared[..., torch.arange(32) & 15]
    reduced = butterfly(replicated, (8, 4, 2, 1))
    partials = reduced[..., 0]  # Only tid==0 owns the global partial store.
    reference_final = torch.zeros((b, 16), dtype=torch.float32)
    reference_final[:, :9] = partials
    return {
        "chains": chains,
        "private": private,
        "warps": shared,
        "partials": partials,
        "output": finish_model(partials).to(x.dtype),
        "reference_output": butterfly(reference_final, (8, 4, 2, 1))[:, 0].to(x.dtype),
        "truth_partials": data.double().sum(-1),
    }
