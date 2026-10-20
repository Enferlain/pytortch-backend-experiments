"""Independent unsigned Python implementation of the published Philox4x32-10 round.

Copied from stage4/reference.py; the reference for generated RNG kernels.
"""

MASK = (1 << 32) - 1


def philox_words(seed, counter):
    c = [counter & MASK, (counter >> 32) & MASK, 0, 0]
    k = [seed & MASK, (seed >> 32) & MASK]
    for r in range(10):
        p0 = c[0] * 0xD2511F53
        p1 = c[2] * 0xCD9E8D57
        c = [
            ((p1 >> 32) ^ c[1] ^ k[0]) & MASK,
            p1 & MASK,
            ((p0 >> 32) ^ c[3] ^ k[1]) & MASK,
            p0 & MASK,
        ]
        if r < 9:
            k = [(k[0] + 0x9E3779B9) & MASK, (k[1] + 0xBB67AE85) & MASK]
    return c


def uniform(seed, counter):
    return (philox_words(seed, counter)[0] >> 8) / 16777216


assert philox_words(0, 0) == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
