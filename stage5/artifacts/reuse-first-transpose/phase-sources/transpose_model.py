"""CPU address/ownership gate, before the transpose GPU adapter exists."""

import json
from pathlib import Path

import numpy as np

from stage5.transpose_trial import OUT, ROOT, sha, write

SHAPES = ((33, 257), (256, 513), (256, 512), (31, 32), (32, 33), (32, 32), (256, 256), (257, 256))


def width(m, n, dtype):
    # CubeCL's 128-bit I/O ceiling and perpendicular/parallel/axis divisibility.
    maximum = 4 if dtype == "float32" else 8
    return max(v for v in (1, 2, 4, 8) if v <= maximum and m % v == n % v == 0)


def verify(m, n, v):
    # Literal upstream coordinate/gather/register-transpose/store work, with
    # vector indices distinct from scalar coordinates. Values are scalar input
    # physical addresses; compare independently with a dense array transpose.
    source = np.arange(m * n, dtype=np.int64)
    actual = np.full(m * n, -1, dtype=np.int64)
    owners = np.zeros(m * n, dtype=np.int32)
    num_batch = m // v
    channel_in_stride, channel_out_stride = n // v, m // v
    num_runs = (m * n // v) // (num_batch * v)
    for worker in range(num_runs):
        batch_index = worker * num_batch
        global_index = (batch_index // channel_out_stride) * m + batch_index % channel_out_stride
        for b in range(num_batch):
            offset_out = global_index + b
            scalar_out = offset_out * v
            row, col = (scalar_out // m) % n, scalar_out % m
            batch_offset = (row + col * n) // v
            gathered = np.stack([source[(batch_offset + i * channel_in_stride) * v: (batch_offset + i * channel_in_stride + 1) * v] for i in range(v)])
            registers = gathered.T
            for o in range(v):
                start = (offset_out + o * channel_out_stride) * v
                actual[start:start + v] = registers[o]
                owners[start:start + v] += 1
    assert np.all(owners == 1)
    assert np.array_equal(actual.reshape(n, m), source.reshape(m, n).T)
    return dict(m=m, n=n, width=v, working_units=num_runs, batch_iterations=num_batch, scalar_identities=m*n, exactly_one_owner=True)


def main():
    assert not (OUT / "cpu.json").exists()
    assert not (ROOT / "stage5/transpose_probe/src/main.rs").exists()
    rows = [verify(m, n, width(m, n, d)) | dict(dtype=d) for d in ("float32", "float16", "bfloat16") for m, n in SHAPES]
    write("cpu.json", dict(complete=True, gpu_source_existed=False, rows=rows, model_sha256=sha(__file__), capture_sha256=sha(OUT / "capture.json"), note="Address/ownership proof only. Pointwise arithmetic is unchanged descriptor replay; no CPU sin approximation is a GPU numerical gate."))
    print(json.dumps(dict(cases=len(rows), scalar_identities=sum(r['scalar_identities'] for r in rows))))


if __name__ == "__main__":
    main()
