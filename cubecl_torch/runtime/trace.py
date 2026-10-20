"""Opt-in, bounded evidence recording; never changes compilation or launches.

Off by default. ``CUBECL_TORCH_TRACE=1`` enables it process-wide; ``scoped()``
enables it for the current context only (ContextVar, so independent
free-threaded callers keep separate preferences). Events go to a bounded deque.
"""

import os
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar

LIMIT = 10000
EVENTS = deque(maxlen=LIMIT)
_ENABLED = ContextVar(
    "cubecl_torch_trace", default=os.environ.get("CUBECL_TORCH_TRACE") == "1"
)


def enabled():
    return _ENABLED.get()


def record(event):
    if _ENABLED.get():
        EVENTS.append(event)


@contextmanager
def scoped(on=True):
    token = _ENABLED.set(bool(on))
    try:
        yield
    finally:
        _ENABLED.reset(token)
