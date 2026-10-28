"""Load artifacts and launch them on PyTorch storage and the current stream.

Lifetime: CUDA modules stay loaded for the process (asynchronous launches can
outlive Python wrappers). Every launch records all tensors and the metadata
buffer on the launching stream, so the allocator does not reuse them early.
Cross-stream inputs need caller waits for readiness; ``record_stream`` adds no
execution dependency. CUDA graph capture is refused.
"""

import ctypes as C
import re
import struct
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager

import torch

from cubecl_torch.runtime import native, trace
from cubecl_torch.runtime.artifact import check_provenance
from cubecl_torch.runtime.rules import LIMIT, RuleDomainError, evaluate

STORAGE_DTYPES = {
    str(t).removeprefix("torch."): t
    for t in (torch.float32, torch.float16, torch.bfloat16, torch.bool, torch.int64)
}
RESOURCE_NAMES = (
    "device_max_threads",
    "plane_width",
    "multiprocessors",
    "threads_per_sm",
    "function_max_threads",
    "active_blocks_per_sm",
)
GRID_LIMIT = 2147483647 * 256
# Native-endian like the metadata words: the kernel reinterprets the i64 slot.
_F64 = struct.Struct("=d")
_I64 = struct.Struct("=q")


class LaunchContractError(ValueError):
    """A launch was refused because its arguments violate the artifact contract.

    ``kind`` names the mismatch class: ``arguments``, ``scalar``, ``device``,
    ``dtype``, ``layout`` (shape, stride or offset span), ``extent``, ``alias`` or ``resources``.
    """

    def __init__(self, kind, message):
        super().__init__(f"{message} [{kind} mismatch]")
        self.kind = kind


def _capturing(device):
    """Whether ``device``'s current stream is capturing (not the current device's)."""
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def _check_signature(artifact):
    signature = re.search(
        rb"\.entry\s+" + re.escape(artifact.entry.encode()) + rb"\s*\((.*?)\)",
        artifact.ptx,
        re.DOTALL,
    )
    count = len(artifact.tensors) + 1
    if (
        not signature
        or len(re.findall(rb"\.param\s+\.u64", signature[1])) != count
        or len(re.findall(rb"\.param", signature[1])) != count
    ):
        raise LaunchContractError("arguments", "PTX does not match the pointer ABI")


class LoadedKernel:
    """An artifact loaded on one device.

    ``launch_check(scalars, tensors)`` is an optional front-end admission hook
    run before each enqueue; it raises to refuse a launch. It is not part of
    the cached contract.
    """

    def __init__(self, artifact, device=0, *, launch_check=None):
        check_provenance(artifact.provenance)
        _check_signature(artifact)
        self.artifact = artifact
        self.launch_check = launch_check
        self.device = (
            torch.device("cuda", device) if isinstance(device, int) else device
        )
        self.native = native.launcher()
        native.metadata_library()
        self.handle = C.c_void_p()
        error = C.create_string_buffer(2048)
        with torch.cuda.device(self.device):
            torch.cuda.init()
            # Establish PyTorch's primary context using its native allocation path.
            torch.empty(1, device=self.device)
            started = time.perf_counter()
            if self.native.stage0_load(
                artifact.ptx,
                artifact.entry.encode(),
                C.byref(self.handle),
                error,
                len(error),
            ):
                raise RuntimeError(error.value.decode())
            self.load_ms = (time.perf_counter() - started) * 1000
            values = (C.c_int * 3)()
            if self.native.stage0_kernel_attributes(
                self.handle, values, error, len(error)
            ):
                raise RuntimeError(error.value.decode())
            self.attributes = dict(
                zip(
                    (
                        "registers_per_thread",
                        "static_shared_bytes",
                        "local_bytes_per_thread",
                    ),
                    values,
                    strict=True,
                )
            )
            values = (C.c_int * 6)()
            if self.native.stage5_resources(
                self.handle,
                artifact.threads,
                artifact.launch.shared_bytes,
                values,
                error,
                len(error),
            ):
                raise RuntimeError(error.value.decode())
        self.resources = dict(zip(RESOURCE_NAMES, values, strict=True))
        if artifact.threads > min(values[0], values[4]) or (
            artifact.plane_width and artifact.plane_width != values[1]
        ):
            raise LaunchContractError("resources", "unsupported target physical plan")
        self.resources["theoretical_occupancy"] = (
            values[5] * artifact.threads / values[3]
        )
        self._lock = threading.RLock()
        self._layouts = OrderedDict()
        self._metadata = {}
        self._metadata_complete = set()
        self._tensors = tuple((p, STORAGE_DTYPES[p.dtype]) for p in artifact.tensors)
        self._outputs = tuple(
            i for i, p in enumerate(artifact.tensors) if p.access == "write"
        )
        self._positive = tuple(s.positive for s in artifact.scalars)
        self._float = tuple(s.dtype == "float64" for s in artifact.scalars)
        self._pointer_type = C.c_uint64 * (len(artifact.tensors) + 1)
        self._dimensions = (C.c_uint * 3)(*artifact.launch.block)
        self._shared = artifact.launch.shared_bytes
        self._call_pointers = None
        self._prepared = None
        self._for_batch = False
        self._launch_count = 0

    @property
    def launch_count(self):
        with self._lock:
            return self._launch_count

    def slots(self, scalars):
        """Launch scalars as i64 metadata slot values (float64 values as their bits)."""
        return tuple(
            _I64.unpack(_F64.pack(float(s)))[0] if f else int(s)
            for s, f in zip(scalars, self._float, strict=True)
        )

    def geometry(self, scalars):
        return evaluate(self.artifact.launch.grid, self.slots(scalars))

    def _layout(self, scalars):
        """Scalar-only rule results, bounded LRU; never tensor or storage facts."""
        if scalars in self._layouts:
            self._layouts.move_to_end(scalars)
            return self._layouts[scalars]
        try:
            value = self._evaluate_layout(scalars)
        except RuleDomainError as error:
            # Overflow or zero divisor outside a guard: refused all the same.
            raise LaunchContractError("extent", str(error)) from error
        self._layouts[scalars] = value
        if len(self._layouts) > 64:
            self._layouts.popitem(last=False)
        return value

    def _evaluate_layout(self, scalars):
        artifact = self.artifact
        layouts = []
        for param, dtype in self._tensors:
            size = tuple(evaluate(v, scalars) for v in param.size)
            stride = tuple(evaluate(v, scalars) for v in param.stride)
            offset = evaluate(param.offset, scalars)
            span = (
                offset + 1 + sum((s - 1) * t for s, t in zip(size, stride, strict=True))
                if all(size)
                else 0
            )
            layouts.append((size, stride, offset, span, dtype))
        expected = []
        for guard in artifact.guards:
            try:
                values = tuple(evaluate(r, scalars) for r in guard.rules)
            except RuleDomainError as error:
                raise LaunchContractError("extent", guard.message) from error
            if guard.kind == "output_numel":
                buffer = -1 if guard.buffer is None else guard.buffer
                expected.append((buffer, values[0], guard.message))
            elif guard.kind == "range":
                (value,) = values
                if (guard.minimum is not None and value < guard.minimum) or (
                    guard.maximum is not None and value > guard.maximum
                ):
                    raise LaunchContractError("extent", guard.message)
            elif values not in guard.allowed:
                raise LaunchContractError("extent", guard.message)
        n = evaluate(artifact.extent, scalars)
        if not 0 <= n <= GRID_LIMIT:
            raise LaunchContractError("extent", "runtime numel exceeds physical grid")
        try:
            info = tuple(evaluate(r, scalars) for r in artifact.info_scalars)
        except RuleDomainError as error:
            raise LaunchContractError("extent", str(error)) from error
        return (
            tuple(layouts),
            n,
            tuple(expected),
            info,
            evaluate(artifact.launch.grid, scalars),
        )

    def _validate(self, scalars, tensors):
        if len(tensors) != len(self._tensors) or len(scalars) != len(self._positive):
            raise LaunchContractError("arguments", "runtime argument count")
        try:
            scalars = self.slots(scalars)
        except (TypeError, ValueError, OverflowError) as error:
            raise LaunchContractError("scalar", f"runtime scalar: {error}") from error
        # Every float bit pattern (i64::MIN is -0.0) is a valid slot value.
        if any(
            not f and not -LIMIT <= s <= LIMIT
            for s, f in zip(scalars, self._float, strict=True)
        ):
            raise LaunchContractError("scalar", "runtime scalar outside i64")
        if any(p and s <= 0 for p, s in zip(self._positive, scalars, strict=True)):
            raise LaunchContractError(
                "scalar", "runtime scalar violates positive index domain"
            )
        layouts, n, expected, info, grid = self._layout(scalars)
        for buffer, numel, message in expected:
            if tensors[buffer].numel() != numel:
                raise LaunchContractError("extent", message)
        storage_pointers, pointers, lengths = [], [], []
        for tensor, (size, stride, offset, span, dtype) in zip(
            tensors, layouts, strict=True
        ):
            if tensor.device != self.device:
                raise LaunchContractError(
                    "device",
                    f"runtime tensor on {tensor.device}, kernel loaded on {self.device}",
                )
            if tensor.dtype != dtype:
                raise LaunchContractError(
                    "dtype",
                    f"runtime tensor dtype {tensor.dtype}, contract declares {dtype}",
                )
            if tuple(tensor.shape) != size or tuple(tensor.stride()) != stride:
                raise LaunchContractError(
                    "layout",
                    f"runtime tensor layout {tuple(tensor.shape)}/{tuple(tensor.stride())}"
                    f" differs from the guard {size}/{stride}",
                )
            storage = tensor.untyped_storage()
            available = (
                storage.nbytes() // tensor.element_size() - tensor.storage_offset()
            )
            if offset < 0 or any(s < 0 for s in stride) or not 0 <= span <= available:
                raise LaunchContractError(
                    "layout", "direct layout exceeds original storage"
                )
            storage_pointers.append(storage.data_ptr())
            pointers.append(tensor.data_ptr())
            lengths.append(span)
        # An output never shares storage with an input. Outputs may share one
        # storage as disjoint views (Inductor's cat writes slices of one
        # buffer), never the same view; an empty output writes nothing.
        if n and any(
            storage_pointers[i] == storage_pointers[j]
            and (j not in self._outputs or pointers[i] == pointers[j])
            for i in self._outputs
            if tensors[i].numel()
            for j in range(len(tensors))
            if j != i
        ):
            raise LaunchContractError("alias", "output must be disjoint")
        return (tuple(lengths), info), n, tuple(pointers), grid

    def __call__(self, scalars, *tensors):
        with self._lock:
            if _capturing(self.device):
                raise NotImplementedError("CubeCL CUDA graph capture unsupported")
            key, n, pointers, grid = self._validate(scalars, tensors)
            if self.launch_check is not None:
                self.launch_check(scalars, tensors)
            result = self._enqueue(tensors, key, pointers, grid)
            if n:
                self._launch_count += 1
            if trace.enabled():
                trace.record(
                    {
                        "event": "launch",
                        "entrypoint": self.artifact.entry,
                        "grid": [grid, 1, 1],
                        "block": list(self.artifact.launch.block),
                        "stream": torch.cuda.current_stream(self.device).cuda_stream,
                        "pointers": list(pointers),
                    }
                )
            return result

    def _info(self, key):
        """Immutable metadata per (spans, info scalars); one setup upload each."""
        if key not in self._metadata:
            if _capturing(self.device):
                raise RuntimeError("prepare metadata before capture")
            words = native.metadata_words(*key)
            if len(words) != self.artifact.metadata_words:
                raise RuntimeError("InfoBuilder metadata ABI mismatch")
            tensor = torch.tensor(words, dtype=torch.uint64, device=self.device)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            self._metadata[key] = (tensor, ready)
        return self._metadata[key]

    def _enqueue(self, tensors, key, pointers, grid):
        with torch.cuda.device(self.device):
            info, ready = self._info(key)
            stream = torch.cuda.current_stream(self.device)
            # Completion is a property of immutable metadata, not a stream ID.
            if key not in self._metadata_complete:
                if ready.query():
                    self._metadata_complete.add(key)
                else:
                    stream.wait_event(ready)
            for tensor in tensors:
                tensor.record_stream(stream)
            info.record_stream(stream)
            args = self._pointer_type(*pointers, info.data_ptr())
            if self._for_batch:
                # Kept alive explicitly for prepared timing only.
                self._prepared = (tensors, info, args, grid, stream)
            error = C.create_string_buffer(2048)
            if self.native.stage5_launch(
                self.handle,
                args,
                len(args),
                grid,
                self._dimensions,
                self._shared,
                1,
                stream.cuda_stream,
                error,
                len(error),
            ):
                raise RuntimeError(error.value.decode())
        return tensors[-1]

    @contextmanager
    def prepared(self, scalars, *tensors):
        """Offline timing scope; ordinary launches retain no user tensor refs."""
        if not self.artifact.repeatable:
            raise ValueError("prepared repeated launch only permits pure kernels")
        with self._lock, torch.cuda.device(self.device):
            if self._for_batch:
                raise ValueError("nested prepared timing scope")
            self._for_batch = True
            try:
                self(scalars, *tensors)
                yield self
            finally:
                self._for_batch = False
                self._prepared = None

    def batch(self, repeat=1):
        if not self.artifact.repeatable:
            raise ValueError("prepared repeated launch only permits pure kernels")
        if not isinstance(repeat, int) or not 1 <= repeat <= 10000:
            raise ValueError("invalid prepared batch repeat")
        if self._prepared is None:
            raise ValueError("prepare with a validated launch first")
        _tensors, _info, args, grid, stream = self._prepared
        if torch.cuda.current_stream(self.device).cuda_stream != stream.cuda_stream:
            raise ValueError("prepare again after changing current stream")
        error = C.create_string_buffer(2048)
        if self.native.stage5_launch(
            self.handle,
            args,
            len(args),
            grid,
            self._dimensions,
            self._shared,
            repeat,
            stream.cuda_stream,
            error,
            len(error),
        ):
            raise RuntimeError(error.value.decode())
