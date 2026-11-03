"""Bridged plans: CubeK's reduce through the runtime bridge, behind the call
interface of a compiled plan (torch-runtime-bridge design decision 8).

A bridged kernel compiles nothing of ours: CubeK builds and compiles its own
kernel in this process on first launch for a shape. Its launch contract
(argument count, scalars, dtypes, layouts, storage spans, output aliasing) is
the frozen plan's for the same descriptor, so a launch is refused exactly where
that plan's would be. Legality (``legality.cubek_reason``) proved the body;
each launch checks again the runtime facts it proved for its class (aligned
addresses, contiguous non-empty rows), since a misaligned address would fault
on the device, and the kernel may be called outside the tuner.
"""

import threading
from contextlib import contextmanager
from functools import cache
from types import SimpleNamespace

import torch

from cubecl_torch.compiler.indexing import evaluate
from cubecl_torch.inductor.legality import bridged_rows
from cubecl_torch.runtime import bridge
from cubecl_torch.runtime.launch import LaunchContractError


def inductor_blueprint(cols, device):
    """``(planes, unit_elements)`` for a cube that reduces a row as Triton
    does under Inductor's config for this row sum, a row it loops over (over
    ``legality.INDUCTOR_PERSISTENT_INNER`` elements): its warps are the planes,
    and each unit keeps the elements a Triton thread holds of a row,
    ``R0_BLOCK / threads``.

    The config is the one Inductor gives a looped INNER reduction of one load
    without max-autotune, copied from torch 2.14.1 (``ReductionHeuristic.
    get_configs``, ``contiguous_config`` through ``triton_config_reduction``;
    it needs Triton to run, which this environment lacks): ``R0_BLOCK =
    min(rnumel hint, 2048)``, 1024 from compute capability 10, and ``num_warps =
    next_pow2(clamp(R0_BLOCK // 128, 2, 16))``. The row count changes only
    ``XBLOCK``, which keeps each row's partials; ``add_persistent_rblock``,
    the other config Inductor may add, needs a body that a persistent
    reduction would read less of, never a plain sum.
    ``testing/data/inductor_row_configs.json`` holds what Inductor chose
    (``testing.inductor_configs``), and the tests hold this to it.

    For 16-bit input CubeK loads 8 elements per vector, so a unit keeps 8 where
    a Triton thread keeps 4: twice Triton's partials, as measured in the band."""
    return _inductor_blueprint(cols, torch.cuda.get_device_capability(device)[0])


@cache
def _inductor_blueprint(cols, major):
    hint = 1 << (cols - 1).bit_length()
    r0_block = min(hint, 1024 if major >= 10 else 2048)
    warps = 1 << (min(max(r0_block // 128, 2), 16) - 1).bit_length()
    return warps, max(1, r0_block // (32 * warps))


class BridgedKernel:
    """``LoadedKernel``'s interface over ``bridge.sum_rows_unchecked``.

    ``contract`` is the frozen plan's ``LoadedKernel`` for the same descriptor:
    each launch is validated by it, and its resources describe the device.
    ``artifact.entry`` names the CUDA kernel CubeK launches for the shape last
    prepared, for the benchmarker's CUPTI events.
    """

    def __init__(self, semantic, plan, contract, numerics):
        self.contract = contract
        self.device = contract.device
        self.resources = contract.resources
        self.attributes = {}
        self.launch_check = contract.launch_check
        ops = semantic["ops"]
        self._input = next(op["buffer"] for op in ops if op["op"] == "load")
        self._output = next(op["buffer"] for op in ops if op["op"] == "store")
        self._semantic = semantic
        self._rows = {}
        self._outer = semantic["outer"]
        self._extent = semantic["reduction_extent"]
        self._options = {
            "parallel_output": plan.parallel_output,
            "forced_planes": plan.planes,
            "contraction": numerics.fp_contract,
            "exact_division": numerics.exact_division,
        }
        self._mapping = plan.mapping
        self._major = torch.cuda.get_device_capability(self.device)[0]
        self._routine = plan.routine
        self.artifact = SimpleNamespace(entry=None, repeatable=True)
        self._entries = {}
        self._prepared = None
        self._lock = threading.RLock()
        self._launch_count = 0

    @property
    def launch_count(self):
        with self._lock:
            return self._launch_count

    def geometry(self, scalars):
        return self.contract.geometry(scalars)

    def _launch(self, scalars, tensors):
        rows = evaluate(self._outer, scalars)
        if rows:
            cols = evaluate(self._extent, scalars)
            blueprint = {}
            if self._mapping == "inductor":
                planes, elements = _inductor_blueprint(cols, self._major)
                blueprint = {"forced_planes": planes, "unit_elements": elements}
            bridge.sum_rows_unchecked(
                tensors[self._input],
                tensors[self._output],
                rows,
                cols,
                self._routine,
                **(self._options | blueprint),
            )
        return rows

    def _check(self, scalars, tensors):
        """The runtime facts legality proved for the tuner's class."""
        if any(
            tensors[i].data_ptr() % bridge.ALIGNMENT
            for i in (self._input, self._output)
        ):
            raise LaunchContractError(
                "layout", "bridged rows and output must be 16-byte aligned"
            )
        key = tuple(scalars)
        if key not in self._rows:
            if len(self._rows) >= 64:
                self._rows.clear()
            self._rows[key] = bridged_rows(self._semantic, scalars)
        if not self._rows[key]:
            raise LaunchContractError(
                "layout", "bridged rows must be contiguous and non-empty"
            )

    def __call__(self, scalars, *tensors):
        with self._lock:
            self.contract._validate(scalars, tensors)
            self._check(scalars, tensors)
            if self.launch_check is not None:
                self.launch_check(scalars, tensors)
            if self._launch(scalars, tensors):
                self._launch_count += 1
            return tensors[-1]

    def _entry(self, scalars, tensors):
        """The CUDA kernel CubeK launches for this shape, from one profiled
        launch (CubeK names it by the blueprint it infers)."""
        key = tuple(scalars)
        if key not in self._entries:
            torch.cuda.synchronize(self.device)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as profiler:
                self(scalars, *tensors)
                torch.cuda.synchronize(self.device)
            names = {
                e.name()
                for e in profiler.profiler.kineto_results.events()
                if e.device_type() == torch.autograd.DeviceType.CUDA
                and e.activity_type() != "gpu_user_annotation"
            }
            if len(names) != 1:
                raise RuntimeError(
                    f"bridged launch expected one CUDA kernel, observed {sorted(names)}"
                )
            self._entries[key] = names.pop()
        return self._entries[key]

    @contextmanager
    def prepared(self, scalars, *tensors):
        """Offline timing scope; ordinary launches retain no user tensor refs."""
        with self._lock, torch.cuda.device(self.device):
            if self._prepared is not None:
                raise ValueError("nested prepared timing scope")
            self(scalars, *tensors)
            self.artifact.entry = self._entry(scalars, tensors)
            self._prepared = (scalars, tensors)
            try:
                yield self
            finally:
                self._prepared = None

    def observed(self):
        """The CUDA source and PTX of each kernel this kernel's prepared shapes
        launched, as NVRTC took and returned them (``bridge.compiled_kernel``),
        in entry order: the files a compiled plan's directory would hold."""
        device = torch.device(self.device).index
        kernels = []
        for entry in sorted(set(self._entries.values())):
            kernel = bridge.compiled_kernel(device, entry)
            if kernel is None:
                raise RuntimeError(f"bridged kernel {entry} was not observed compiling")
            kernels.append(kernel)
        return {
            "kernel.cu": b"".join(source for source, _ in kernels),
            "kernel.ptx": b"".join(ptx for _, ptx in kernels),
        }

    def batch(self, repeat=1):
        if not isinstance(repeat, int) or not 1 <= repeat <= 10000:
            raise ValueError("invalid prepared batch repeat")
        if self._prepared is None:
            raise ValueError("prepare with a validated launch first")
        scalars, tensors = self._prepared
        for _ in range(repeat):
            self._launch(scalars, tensors)


class BridgedCompiled:
    """A bridged plan in ``CompiledKernel``'s place: there is no artifact
    directory, since nothing of ours is compiled."""

    def __init__(self, semantic, plan, contract, *, numerics):
        self.plan = plan
        self.semantic = semantic
        self.directory = None
        self.cache_hit = True
        self.kernel = BridgedKernel(semantic, plan, contract, numerics)
        self.entrypoint = None
        self.load_compile_ms = 0.0

    def __call__(self, scalars, *tensors):
        return self.kernel(scalars, *tensors)
