"""CPU proof of the actual layout maps and bounded CubeCL FastDivmod arithmetic."""

import json

import numpy as np

from stage5.layout_trial import OUT, ROOT, sha, write

LIMIT = 1 << 21
MASK = (1 << 64) - 1


def evaluate(tree, scalars, linear=0):
    op = tree["op"]
    if op == "linear":
        return linear
    if op == "constant":
        return tree["value"]
    if op == "scalar":
        return scalars[tree["slot"]]
    args = [evaluate(t, scalars, linear) for t in tree["args"]]
    if op == "add":
        return sum(args)
    if op == "mul":
        value = 1
        for a in args:
            value = value * a
        return value
    if op == "mod":
        return args[0] % args[1]
    if op == "floor_div":
        return args[0] // args[1]
    raise ValueError(op)


def params(divisor):
    """Literal host translation of CubeCL fast_math::find_params_u64."""
    if not 1 <= divisor <= LIMIT:
        raise ValueError("divisor outside proved domain")
    shift = (divisor - 1).bit_length()
    multiplier = ((1 << 64) * ((1 << shift) - divisor)) // divisor + 1
    assert 0 <= multiplier <= MASK
    return multiplier, shift


def signed_bits(v):
    return v if v < (1 << 63) else v - (1 << 64)


def divmod_fast(x, d):
    multiplier, shift = params(d)
    x = np.asarray(x, dtype=np.uint64)
    assert np.all(x < LIMIT)
    # For x<2^21, these two products and their sum fit u64. This computes
    # the exact high64(x*multiplier), independently of GPU MulHi lowering.
    hi = (
        x * np.uint64(multiplier >> 32)
        + ((x * np.uint64(multiplier & ((1 << 32) - 1))) >> np.uint64(32))
    ) >> np.uint64(32)
    q = (hi + x) >> np.uint64(shift)
    r = x - q * np.uint64(d)
    assert np.array_equal(q, x // np.uint64(d)) and np.array_equal(r, x % np.uint64(d))
    assert (signed_bits(multiplier) & MASK) == multiplier
    return q, r


def maps(semantic, scalars):
    ref = semantic["buffers"][-1]
    shape = [int(evaluate(d, scalars)) for d in ref["size"]]
    if len(shape) not in (1, 2) or min(shape) < 2 or np.prod(shape) > LIMIT:
        raise ValueError("layout extent outside trial")
    contiguous = []
    stride = 1
    for d in reversed(shape):
        contiguous.insert(0, stride)
        stride *= d
    if [evaluate(d, scalars) for d in ref["stride"]] != contiguous or evaluate(
        ref["offset"], scalars
    ) != 0:
        raise ValueError("output must be contiguous with relative offset zero")
    result = []
    loads = [op for op in semantic["ops"] if op["op"] == "load"]
    for load in loads:
        b = semantic["buffers"][load["buffer"]]
        if evaluate(b["offset"], scalars) != 0:
            raise ValueError("relative input offset")
        if load["index"] == {"op": "linear"}:
            result.append(dict(buffer=load["buffer"], kind="plain", stride=[]))
            continue
        size = [{"op": "constant", "value": 1}] * (len(shape) - len(b["size"])) + b[
            "size"
        ]
        strides = [{"op": "constant", "value": 0}] * (len(shape) - len(b["size"])) + b[
            "stride"
        ]
        if len(size) != len(shape):
            raise ValueError("rank mismatch")
        normalized = []
        for a, r, st in zip(size, ref["size"], strides):
            if a == r:
                normalized.append(int(evaluate(st, scalars)))
            elif a == {"op": "constant", "value": 1}:
                normalized.append(0)
            else:
                raise ValueError("broadcast not structurally proved")
        if any(st < 0 or st > LIMIT for st in normalized):
            raise ValueError("stride domain")
        result.append(dict(buffer=load["buffer"], kind="permuted", stride=normalized))
    return shape, result


def verify(sem, scalars):
    shape, layouts = maps(sem, scalars)
    count = int(np.prod(shape))
    linear = np.arange(count, dtype=np.uint64)
    loads = [op for op in sem["ops"] if op["op"] == "load"]
    identities = 0
    for load, layout in zip(loads, layouts):
        if layout["kind"] == "plain":
            actual = linear
        else:
            actual = np.zeros(count, dtype=np.uint64)
            remainder = linear
            for d, st in zip(reversed(shape), reversed(layout["stride"])):
                remainder, coordinate = divmod_fast(remainder, d)
                actual += coordinate * np.uint64(st)
            assert np.all(remainder == 0)
        expected = evaluate(load["index"], scalars, linear.astype(np.int64))
        assert np.array_equal(actual, np.broadcast_to(expected, actual.shape))
        b = sem["buffers"][load["buffer"]]
        span = 1 + sum(
            (evaluate(sz, scalars) - 1) * evaluate(st, scalars)
            for sz, st in zip(b["size"], b["stride"])
        )
        assert int(actual.min()) >= 0 and int(actual.max()) < span
        identities += count
    return dict(
        scalars=list(scalars),
        shape=shape,
        layouts=layouts,
        scalar_identities=identities,
        ownership="unchanged scalar linear output; one active thread per output",
        metadata=[x for d in shape for x in (signed_bits(params(d)[0]), params(d)[1])],
    )


def main():
    assert not (ROOT / "stage5/layout_probe/src/main.rs").exists()
    assert not (OUT / "cpu.json").exists()
    cap = json.loads((OUT / "capture.json").read_text())
    rows = []
    divisors = set()
    for r in cap["rows"]:
        sem = json.loads((OUT / r["baseline"] / "semantic.json").read_text())
        row = verify(sem, r["scalars"])
        row["case"] = r["case"]["name"]
        rows.append(row)
        divisors.update(row["shape"])
    for d in sorted(divisors | {1, 2, 3, 31, 32, 33, LIMIT - 1, LIMIT}):
        divmod_fast(
            np.asarray(
                sorted({0, d - 1, min(d, LIMIT - 1), min(d + 1, LIMIT - 1), LIMIT - 1}),
                dtype=np.uint64,
            ),
            d,
        )
    write(
        "cpu.json",
        dict(
            complete=True,
            gpu_source_existed=False,
            rows=rows,
            divisors=sorted(divisors),
            capture_sha256=sha(OUT / "capture.json"),
            source_sha256=sha(__file__),
            physical_domain=f"0 <= dividend < {LIMIT}; high64 <= dividend, so high64+dividend < 2^22 and no u64 overflow",
        ),
    )
    print(
        "PASS",
        len(rows),
        "cases",
        sum(r["scalar_identities"] for r in rows),
        "scalar address identities",
    )


if __name__ == "__main__":
    main()
