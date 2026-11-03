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
CUBECL_PIN = "fed6c0c0ca5f06250be8cdb15da0b12458ac083f"
# Audited CubeK source state: this revision with CUBEK_PATCHES applied.
CUBEK_PIN = "50cb94b245f1bc693fbfdf4bd376a4c8de9c890e"
# The audited patch stacks: ``series`` names a stack's patches in application
# order.
PATCHES = NATIVE / "patches"
CUBEK_PATCHES = PATCHES / "cubek"
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


def patch_stack(patches=PATCHES):
    """The stack's patch files, in application order: one name per line of
    ``series``, blank lines ignored."""
    lines = (Path(patches) / "series").read_text().splitlines()
    return [Path(patches) / line.strip() for line in lines if line.strip()]


@cache
def patch_stack_sha256(patches=PATCHES):
    """Digest of the stack: each patch's name and contents, in order."""
    h = hashlib.sha256()
    for patch in patch_stack(patches):
        data = patch.read_bytes()
        h.update(f"{patch.name}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()


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
