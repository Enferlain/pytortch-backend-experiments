"""Whether two PTX files are the same program up to register allocation.

NVRTC can number registers differently for CUDA that differs only in code it then
removes (a CubeCL upgrade adding unused locals, for example). Two files are
equivalent here when one consistent renaming of registers, within each register
class, maps every instruction of one onto the other, with one allowance: a run of
float registers set to zero before any of them is used again (directly, or by an
unpredicated copy of a register just zeroed) compares as a set, since the order in
which they become zero, and which copies which, cannot change what is computed.
Comments, `.reg` declarations and kernel names are ignored.

The check is conservative: a register stops counting as zero at any other mention,
and a run ends at a label or at any mention of its registers, so anything it cannot
place reads as a difference.

    python -m cubecl_torch.testing.ptx_equivalence OLD.ptx NEW.ptx
"""

import re
import sys
from pathlib import Path

REGISTER = re.compile(r"%[a-z]+\d+")
ZERO = re.compile(r"\s*mov\.f32\s+(%f\d+), 0f00000000;$")
COPY = re.compile(r"\s*mov\.f32\s+(%f\d+), (%f\d+);$")
LABEL = re.compile(r"^\S+:\s*$")


def _class(register):
    return re.match(r"%[a-z]+", register).group(0)


def _lines(text):
    """Lines to compare, with each copy of a register just zeroed written as a zero."""
    text = re.sub(r"cubecl5_[0-9a-f]+", "K", text)
    out, zeros = [], set()
    for line in text.split("\n"):
        if line.startswith("//") or re.match(r"\s*\.reg ", line):
            continue
        if LABEL.match(line):
            zeros = set()
        copied = COPY.match(line)
        if copied and copied.group(2) in zeros:
            line = f"\tmov.f32 \t{copied.group(1)}, 0f00000000;"
        zeroed = ZERO.match(line)
        if zeroed:
            zeros.add(zeroed.group(1))
        else:
            zeros -= set(REGISTER.findall(line))
        out.append(line)
    return out


def difference(old, new):
    """``None`` when the two PTX texts are equivalent, else the first difference."""
    a, b = _lines(old), _lines(new)
    if len(a) != len(b):
        return f"line count {len(a)} vs {len(b)}"
    forward, backward = {}, {}
    runs, run = [], (set(), set())

    def close():
        nonlocal run
        if run[0] or run[1]:
            runs.append(run)
        run = (set(), set())

    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        zx, zy = ZERO.match(x), ZERO.match(y)
        if zx and zy:
            run[0].add(zx.group(1))
            run[1].add(zy.group(1))
            continue
        if REGISTER.sub("R", x) != REGISTER.sub("R", y):
            return f"instruction {i} differs: {x.strip()!r} / {y.strip()!r}"
        used_x, used_y = REGISTER.findall(x), REGISTER.findall(y)
        if LABEL.match(x) or run[0] & set(used_x) or run[1] & set(used_y):
            close()
        for r, s in zip(used_x, used_y, strict=True):
            if _class(r) != _class(s):
                return f"instruction {i}: {r} and {s} are different register classes"
            if forward.setdefault(r, s) != s or backward.setdefault(s, r) != r:
                return f"instruction {i}: {r} maps to both {forward[r]} and {s}"
    close()
    for old_zeros, new_zeros in runs:
        if len(old_zeros) != len(new_zeros):
            return f"zeroed registers {sorted(old_zeros)} vs {sorted(new_zeros)}"
        for s in new_zeros:
            if s in backward and backward[s] not in old_zeros:
                return f"zeroed {s} stands for {backward[s]}, not zeroed with it"
        for r in old_zeros:
            if r in forward and forward[r] not in new_zeros:
                return f"zeroed {r} stands for {forward[r]}, not zeroed with it"
    return None


def main(argv=None):
    old, new = (Path(p).read_text() for p in (argv or sys.argv[1:]))
    found = difference(old, new)
    print("equivalent" if found is None else found)
    return 0 if found is None else 1


if __name__ == "__main__":
    sys.exit(main())
