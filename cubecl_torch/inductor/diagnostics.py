"""Front-end evidence: an append-only view onto the runtime's opt-in trace.

The Stage 1-5 modules appended full graph strings to unbounded module lists on
every compile. Here ``TraceLog.append`` records only when tracing is enabled
(``CUBECL_TORCH_TRACE=1`` or ``runtime.trace.scoped()``), into the bounded
runtime deque. Callers building expensive events check ``tracing()`` first.
"""

from cubecl_torch.runtime import trace


def tracing():
    return trace.enabled()


detailed_trace = trace.scoped


class TraceLog:
    def append(self, event):
        trace.record(event)

    def __iter__(self):
        return iter(list(trace.EVENTS))
