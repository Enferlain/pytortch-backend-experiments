"""CubeCL's CUDA runtime and CubeK on PyTorch's memory and streams.

Loads the bridge library (``native/bridge``, built by ``native/build.sh``).
Every launch:

- runs on the current stream of the tensors' device, and is on it when the
  call returns, so PyTorch's work before and after it on that stream is
  ordered around it;
- borrows the tensors' storage for the call only. Like PyTorch's own ops and
  Inductor's Triton kernels, it does not record the tensors on the stream: a
  caller that frees a tensor while another stream still uses it records it
  there itself (``Tensor.record_stream``). Recording by the bridge is opt-in:
  ``record_stream=True`` on a call, or ``CUBECL_TORCH_BRIDGE_RECORD_STREAM=1``
  when the module loads, makes it the default;
- takes CubeCL's own device memory (launch metadata, workspaces) from
  PyTorch's caching allocator on the launch's stream, so allocator statistics
  and graph capture cover it.

Only PyTorch's own streams (the default stream and pooled streams) are
accepted: CubeCL keeps a slot for each stream it is given for the life of the
process, and an external stream can be destroyed under it.

Kernels compile in this process on first launch, through CubeCL's C++ dialect
and the NVRTC library and headers of ``runtime/toolkit.py``.
"""

import ctypes as C
import os
import threading
from pathlib import Path

import torch

from cubecl_torch.runtime import native, toolkit


class BridgeError(RuntimeError):
    """The bridge refused or failed a call."""


# Whether launches record their tensors on the launch stream unless the call
# says otherwise.
RECORD_STREAM = os.environ.get("CUBECL_TORCH_BRIDGE_RECORD_STREAM", "") == "1"


def library_path():
    return Path(
        os.environ.get(
            "CUBECL_TORCH_BRIDGE",
            native.NATIVE
            / "bridge"
            / "target"
            / "release"
            / native._shared("cubecl_torch_bridge"),
        )
    )


_ALLOCATE = C.CFUNCTYPE(C.c_uint64, C.c_uint64, C.c_uint64, C.c_uint32)
_FREE = C.CFUNCTYPE(None, C.c_uint64, C.c_uint64, C.c_uint32)
_COMPILED = C.CFUNCTYPE(
    None, C.c_char_p, C.POINTER(C.c_uint8), C.c_size_t, C.POINTER(C.c_uint8), C.c_size_t
)


def _allocate(size, stream, device):
    try:
        return torch.cuda.caching_allocator_alloc(size, device, stream)
    except (torch.OutOfMemoryError, RuntimeError):  # CubeCL sees no memory.
        return 0


def _free(ptr, stream, device):
    torch.cuda.caching_allocator_delete(ptr)


# Kept for the life of the process: the library holds these pointers.
_CALLBACKS = (_ALLOCATE(_allocate), _FREE(_free))
_OBSERVERS = []
_LOCK = threading.Lock()


_LIBRARY = None


def library():
    """The loaded library, set up once: concurrent first calls wait for it."""
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    with _LOCK:
        if _LIBRARY is None:
            _LIBRARY = _load()
        return _LIBRARY


def _load():
    path = library_path()
    if not path.is_file():
        raise native.NativeUnavailable(
            f"missing {path}; run cubecl_torch/native/build.sh"
        )
    # CubeCL reads the headers from CUDA_PATH, and its NVRTC loader finds the
    # library already loaded under its soname: both are PyTorch's.
    root = str(toolkit.root())
    if os.environ.setdefault("CUDA_PATH", root) != root:
        raise BridgeError(
            f"CUDA_PATH is {os.environ['CUDA_PATH']}; bridged kernels compile with the "
            f"headers of PyTorch's NVRTC ({root}): unset it or set it to that"
        )
    C.CDLL(toolkit.library_path(), mode=C.RTLD_GLOBAL)
    lib = C.CDLL(str(path))
    lib.bridge_last_error.restype = C.c_char_p
    lib.bridge_set_allocator.argtypes = [_ALLOCATE, _FREE]
    lib.bridge_init.argtypes = [C.c_uint32]
    lib.bridge_set_observer.argtypes = [C.c_uint32, _COMPILED]
    lib.bridge_test_add_one.argtypes = [
        C.c_uint32,
        C.c_uint64,
        C.c_uint64,
        C.c_uint64,
        C.c_uint64,
    ]
    lib.bridge_sum_rows.argtypes = [
        C.c_uint32,
        *(C.c_uint64,) * 5,
        *(C.c_uint32,) * 3,
        C.c_int32,
        C.c_int32,
    ]
    _check(lib, lib.bridge_set_allocator(*_CALLBACKS))
    return lib


def _check(lib, code):
    if code:
        raise BridgeError(lib.bridge_last_error().decode(errors="replace"))


# PyTorch's streams, by id and device: they live for the process anyway
# (external streams are refused).
_STREAMS = {}


def _stream(device):
    """The current stream of ``device``: its handle and ``torch.cuda.Stream``."""
    stream_id, index, device_type = torch._C._cuda_getCurrentStream(device)
    stream = _STREAMS.get((stream_id, index))
    if stream is None:
        if stream_id and not stream_id & 1:
            # c10 gives external streams their cudaStream_t as id (low bit clear).
            raise BridgeError(
                "the bridge runs only on PyTorch's own streams, not external ones"
            )
        stream = torch.cuda.Stream(
            stream_id=stream_id, device_index=index, device_type=device_type
        )
        _STREAMS[(stream_id, index)] = stream
    return stream


def _launch(code, lib, stream, tensors, record_stream):
    """Raises the bridge's error for a failed call and, if ``record_stream``
    (``None``: ``RECORD_STREAM``), records the tensors on the stream the call
    ran on."""
    # Recorded whether or not the call failed: a launch that failed its check
    # may still have been enqueued.
    if RECORD_STREAM if record_stream is None else record_stream:
        for tensor in tensors:
            tensor.record_stream(stream)
    if code:
        raise BridgeError(lib.bridge_last_error().decode(errors="replace"))


def _device(*tensors):
    """The CUDA device index all ``tensors`` are on."""
    device = tensors[0].get_device()
    if device < 0 or any(t.get_device() != device for t in tensors[1:]):
        raise BridgeError("bridged launches take CUDA tensors on one device")
    return device


def set_observer(device, observer):
    """Calls ``observer(kernel, source, binary)`` for every kernel compiled for
    ``device`` from now on: the kernel id (text), its CUDA source and its PTX."""

    def compiled(kernel, source, source_len, binary, binary_len):
        observer(
            kernel.decode(errors="replace"),
            C.string_at(source, source_len).decode(errors="replace"),
            C.string_at(binary, binary_len),
        )

    callback = _COMPILED(compiled)
    lib = library()
    # Every callback is kept: a compilation on another thread may still be
    # calling the one this replaces.
    _OBSERVERS.append(callback)
    _check(lib, lib.bridge_set_observer(device, callback))


def _numerics(contraction, exact_division):
    return (1 if contraction else 0, 1 if exact_division else 0)


def test_add_one(input, output, *, record_stream=None):
    """``output = input + 1`` for contiguous float32 tensors: the bridge's own
    test kernel."""
    if not (input.is_contiguous() and output.is_contiguous()) or not (
        input.dtype is output.dtype is torch.float32
    ):
        raise BridgeError("test_add_one takes contiguous float32 tensors")
    n = output.numel()
    if input.numel() != n:
        raise BridgeError("test_add_one takes tensors of one size")
    lib = _LIBRARY or library()
    device = _device(input, output)
    stream = _stream(device)
    code = lib.bridge_test_add_one(
        device, stream.cuda_stream, input.data_ptr(), output.data_ptr(), n
    )
    _launch(code, lib, stream, (input, output), record_stream)


FLOAT_CODES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}

# CubeK's reduce routines, each with its inferred blueprint (see the library).
ROUTINES = {"unit": 0, "plane": 1, "plane_shared": 2, "cube": 3, "cube_unit": 4}


def sum_rows(
    input,
    output,
    routine,
    *,
    contraction=True,
    exact_division=False,
    record_stream=None,
):
    """``output[r] = input[r, :].sum()`` with CubeK's reduce, accumulated in
    float32: ``input`` a contiguous matrix, ``output`` a contiguous vector of
    its rows."""
    if input.dim() != 2 or not input.is_contiguous() or not output.is_contiguous():
        raise BridgeError("sum_rows takes a contiguous matrix and a contiguous vector")
    rows, cols = input.shape
    if output.numel() != rows:
        raise BridgeError("sum_rows writes one value per row")
    if input.dtype not in FLOAT_CODES or output.dtype not in FLOAT_CODES:
        raise BridgeError("sum_rows takes float32, float16 or bfloat16")
    lib = _LIBRARY or library()
    device = _device(input, output)
    stream = _stream(device)
    code = lib.bridge_sum_rows(
        device,
        stream.cuda_stream,
        input.data_ptr(),
        output.data_ptr(),
        rows,
        cols,
        FLOAT_CODES[input.dtype],
        FLOAT_CODES[output.dtype],
        ROUTINES[routine],
        *_numerics(contraction, exact_division),
    )
    _launch(code, lib, stream, (input, output), record_stream)
