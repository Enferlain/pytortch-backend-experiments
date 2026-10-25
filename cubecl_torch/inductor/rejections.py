"""Record mode: collect every reason a region is rejected, not only the first.

Normal compilation rejects at the first unsupported feature. Inside
``collecting()`` the backend-selection checks and the pre-fusion scan run to the
end of every graph Inductor lowers (forward and backward), and the region is
rejected once, by ``rejected(reasons)``, whose ``reasons`` map each distinct
reason to the number of times it was found (nodes, for pre-fusion reasons).

A graph's scan stops at its pre-fusion boundary: rejected nodes are never
scheduled or generated, so later code-generation rejections of that graph stay
hidden until its pre-fusion reasons are fixed.
"""

import contextlib
import contextvars
import itertools

from cubecl_torch.compiler.errors import UnsupportedCubeCL

_REASONS = contextvars.ContextVar("cubecl_torch_rejections", default=None)


class Stop(UnsupportedCubeCL):
    """Record mode: this graph cannot be scanned further; its reasons are recorded."""


@contextlib.contextmanager
def collecting():
    reasons = {}
    token = _REASONS.set(reasons)
    try:
        yield reasons
    finally:
        _REASONS.reset(token)


def recording():
    return _REASONS.get() is not None


def reject(*messages, stop=False):
    """Raise on the first message, or in record mode record all of them.

    ``stop`` ends the current graph's scan after recording (record mode only).
    """
    reasons = _REASONS.get()
    if reasons is None:
        raise UnsupportedCubeCL(messages[0])
    for message in messages:
        reasons[message] = reasons.get(message, 0) + 1
    if stop:
        raise Stop(messages[0])


def rejected(reasons):
    error = UnsupportedCubeCL(
        f"{len(reasons)} unsupported feature(s): " + "; ".join(reasons)
    )
    error.reasons = dict(reasons)
    return error


def scan(reasons):
    """All of a reason generator in record mode, else only its first (fail-fast) one."""
    return list(reasons) if recording() else list(itertools.islice(reasons, 1))
