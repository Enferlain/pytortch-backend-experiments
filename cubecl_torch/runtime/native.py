"""ctypes bindings for the native launcher and the CubeCL metadata library.

Both are built by ``cubecl_torch/native/build.sh``. The launcher loads PTX into
the current CUDA context and enqueues on a caller-supplied stream; the metadata
library serialises CubeCL ``InfoBuilder`` words. Neither generates code.
"""

import ctypes as C
import hashlib
import os
import sys
import threading
from functools import cache
from pathlib import Path

NATIVE = Path(__file__).resolve().parents[1] / "native"
# Audited CubeCL source state. Artifacts recording anything else are refused.
CUBECL_PIN = "907676dca8c4cd407237892cc2915ff6aadc4f1e"
FP_PATCH = NATIVE / "cubecl-fp.patch"
_LOCK = threading.Lock()


class NativeUnavailable(RuntimeError):
    """A native library is missing; run cubecl_torch/native/build.sh."""


def _shared(name):
    if sys.platform == "win32":
        return f"{name}.dll"
    return f"lib{name}.dylib" if sys.platform == "darwin" else f"lib{name}.so"


def launcher_path():
    return Path(
        os.environ.get(
            "CUBECL_TORCH_LAUNCHER", NATIVE / _shared("cubecl_torch_launcher")
        )
    )


def metadata_library_path():
    return Path(
        os.environ.get(
            "CUBECL_TORCH_METADATA_LIB",
            NATIVE / "target" / "release" / _shared("cubecl_stage0"),
        )
    )


def _require(path):
    if not path.is_file():
        raise NativeUnavailable(f"missing {path}; run cubecl_torch/native/build.sh")
    return path


@cache
def audited_patch_sha256():
    return hashlib.sha256(FP_PATCH.read_bytes()).hexdigest()


@cache
def launcher():
    with _LOCK:
        lib = C.CDLL(str(_require(launcher_path())))
        lib.stage0_load.argtypes = [
            C.c_char_p,
            C.c_char_p,
            C.POINTER(C.c_void_p),
            C.c_void_p,
            C.c_size_t,
        ]
        lib.stage0_load.restype = C.c_int
        lib.stage5_launch.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_uint64),
            C.c_size_t,
            C.c_uint64,
            C.POINTER(C.c_uint),
            C.c_uint,
            C.c_uint,
            C.c_void_p,
            C.c_void_p,
            C.c_size_t,
        ]
        lib.stage5_launch.restype = C.c_int
        lib.stage5_resources.argtypes = [
            C.c_void_p,
            C.c_uint,
            C.c_uint,
            C.POINTER(C.c_int),
            C.c_void_p,
            C.c_size_t,
        ]
        lib.stage5_resources.restype = C.c_int
        lib.stage0_kernel_attributes.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_int),
            C.c_void_p,
            C.c_size_t,
        ]
        lib.stage0_kernel_attributes.restype = C.c_int
        return lib


@cache
def metadata_library():
    with _LOCK:
        lib = C.CDLL(str(_require(metadata_library_path())))
        lib.cubecl_torch_metadata.argtypes = [
            C.POINTER(C.c_uint64),
            C.c_size_t,
            C.POINTER(C.c_int64),
            C.c_size_t,
            C.POINTER(C.c_uint64),
            C.c_size_t,
        ]
        lib.cubecl_torch_metadata.restype = C.c_size_t
        return lib


def metadata_words(lengths, scalars):
    """CubeCL InfoBuilder words: signed scalars, then per-buffer lengths."""
    lib = metadata_library()
    ns = (C.c_uint64 * len(lengths))(*lengths)
    ss = (C.c_int64 * len(scalars))(*scalars)
    count = lib.cubecl_torch_metadata(ns, len(ns), ss, len(ss), None, 0)
    if not count:
        raise ValueError("CubeCL metadata serialisation rejected its arguments")
    words = (C.c_uint64 * count)()
    if lib.cubecl_torch_metadata(ns, len(ns), ss, len(ss), words, count) != count:
        raise RuntimeError("metadata serialization failed")
    return list(words)
