"""Experimental per-context evidence switch; counters stay enabled.

Set CUBECL_STAGE5_TRACE=1 for subprocess acceptance suites, or use the scoped
API. ContextVar keeps independent free-threaded callers' evidence preferences
separate. This never changes compilation, numerical policy or stream handling.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import os

_DETAIL = ContextVar(
    "cubecl_stage5_trace", default=os.environ.get("CUBECL_STAGE5_TRACE") == "1"
)


def tracing():
    return _DETAIL.get()


@contextmanager
def detailed_trace(enabled=True):
    token = _DETAIL.set(bool(enabled))
    try:
        yield
    finally:
        _DETAIL.reset(token)
