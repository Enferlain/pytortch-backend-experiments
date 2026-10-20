"""Free-threading evidence: GIL state per phase and a Triton import veto.

Call ``install()`` before importing torch. It refuses a forced ``PYTHON_GIL``
or an environment where Triton is importable, makes any later Triton import
fail loudly, and records the GIL state. Imports nothing from torch.
"""

import builtins
import importlib.abc
import importlib.util
import os
import sys


class TritonProbe(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.probes = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            self.probes.append(fullname)


PROBE = TritonProbe()
ATTEMPTS = []
GIL = {}
_original_import = builtins.__import__
_installed = False


def _is_triton(name):
    return name == "triton" or name.startswith("triton.")


def install():
    global _installed
    if _installed:
        return
    if "PYTHON_GIL" in os.environ:
        raise RuntimeError("run without PYTHON_GIL forcing")
    sys.meta_path.insert(0, PROBE)
    if importlib.util.find_spec("triton") is not None:
        raise RuntimeError("run in an environment without Triton (stage0/.venv)")

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and _is_triton(name):
            frame = sys._getframe(1)
            ATTEMPTS.append(
                {
                    "module": name,
                    "caller": frame.f_code.co_filename,
                    "line": frame.f_lineno,
                }
            )
            raise ImportError("Triton deliberately unavailable")
        return _original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded
    _installed = True
    state("before_torch")


def state(phase):
    """Record the GIL state for ``phase``; raise if the GIL is enabled."""
    GIL[phase] = sys._is_gil_enabled()
    if GIL[phase]:
        raise RuntimeError("GIL enabled at " + phase)


def inventory():
    return {
        "availability_probes": list(PROBE.probes),
        "import_attempts": list(ATTEMPTS),
        "loaded_modules": [m for m in sys.modules if _is_triton(m)],
        "gil_states": dict(GIL),
        "python": sys.version,
        "PYTHON_GIL": os.environ.get("PYTHON_GIL"),
    }
