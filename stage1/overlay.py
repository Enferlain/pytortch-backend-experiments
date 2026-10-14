"""Reproducible isolated live-wheel Inductor copy, never edit shared torch files."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch

ROOT = Path(__file__).resolve().parent
EXPECTED = '5c4886908584029761b579af026dcfb627c84070'

def install(*, precision_casts=True):
    """Install the local overlay; disable precision fixes only for legacy probes."""
    if torch.version.git_version != EXPECTED:
        raise RuntimeError('Stage 1 source hooks require the recorded PyTorch wheel commit')
    if any(n == 'torch._inductor' or n.startswith('torch._inductor.') for n in sys.modules):
        raise RuntimeError('install Stage 1 overlay before importing Inductor')
    src = Path(torch.__file__).resolve().parent / '_inductor'
    dest = ROOT/'overlay/torch/_inductor'
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Requirements belong to the scheduling implementation selected by the
    # existing constructor registration (including its config-dependent factory).
    gate = '''        if not has_triton():
            if (
                device.type == "cuda"
                and (device_props := torch.cuda.get_device_properties(device)).major < 7
            ):
                raise GPUTooOldForTriton(device_props, inspect.currentframe())
            elif is_gpu(device.type) and not device.type == "mps":
                raise TritonMissing(inspect.currentframe())

        return device_scheduling(self)
'''
    edits = {
        'choices.py': [
            ('    def reduction_split_factor(\n', '''    def reduction_split_factor_with_extents(
        self, device, reduction_numel_hint, numel_hint, inner_reduction,
        *, reduction_numel, numel,
    ):
        """Choose a split with symbolic extents available for scheduling guards.

        The default preserves existing hint-only heuristics and overrides.
        """
        return self.reduction_split_factor(
            device, reduction_numel_hint, numel_hint, inner_reduction
        )

    def prepare_reduction_split(self, device, reduction_numel, numel, reduction_numel_hint, numel_hint):
        """Optional scheduling guards, including hint-only early-return paths."""

    def reduction_split_factor(
'''),
        ],
        'ir.py': [
            ('        numel_hint = V.graph.sizevars.optimization_hint(numel)\n',
             '        numel_hint = V.graph.sizevars.optimization_hint(numel)\n        V.choices.prepare_reduction_split(device, reduction_numel, numel, reduction_numel_hint, numel_hint)\n'),
            ('V.choices.reduction_split_factor, device, inner_reduction=True',
             'V.choices.reduction_split_factor_with_extents, device, inner_reduction=True, reduction_numel=reduction_numel, numel=numel'),
            ('V.choices.reduction_split_factor, device, inner_reduction=False',
             'V.choices.reduction_split_factor_with_extents, device, inner_reduction=False, reduction_numel=reduction_numel, numel=numel'),
        ],
        'scheduler.py': [
            (gate, '''        backend = device_scheduling(self)
        backend.check_backend_available(device)
        return backend
'''),
            ('import inspect\n', ''),
            ('from torch.utils._triton import has_triton\n', ''),
            ('from .exc import GPUTooOldForTriton, TritonMissing\n', ''),
            ('class BaseScheduling:  # noqa: docstring_linter\n', '''class BaseScheduling:  # noqa: docstring_linter
    def check_backend_available(self, device: torch.device) -> None:
        """Validate selected compiler dependencies; raise an actionable error.

        Called once per device by Scheduler.create_backend, before grouping.
        Constructors must remain safe for feature queries with scheduler=None.
        """

'''),
        ],
        'codegen/triton.py': [
            ('    kernel_type: type[Any] = TritonKernel\n', '''    def check_backend_available(self, device: torch.device) -> None:
        import inspect
        from torch.utils._triton import has_triton
        from ..exc import GPUTooOldForTriton, TritonMissing
        from ..utils import is_gpu

        if not has_triton():
            if (
                device.type == "cuda"
                and (device_props := torch.cuda.get_device_properties(device)).major < 7
            ):
                raise GPUTooOldForTriton(device_props, inspect.currentframe())
            elif is_gpu(device.type) and device.type != "mps":
                raise TritonMissing(inspect.currentframe())

    kernel_type: type[Any] = TritonKernel
'''),
        ],
    }
    for file in ['codegen/cuda_combined_scheduling.py', 'codegen/xpu/xpu_combined_scheduling.py']:
        marker = '    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:\n'
        edits[file] = [(marker, '''    def check_backend_available(self, device: torch.device) -> None:
        self._triton_scheduling.check_backend_available(device)

''' + marker)]
    precision_files = set()
    if precision_casts:
        from .precision import source_edits
        precision_edits = source_edits()
        precision_files.update(precision_edits)
        edits.update(precision_edits)
    patches = []
    hashes = {}
    for file, replacements in edits.items():
        path = dest/file
        before = path.read_text()
        after = before
        for old, new in replacements:
            if after.count(old) != 1:
                raise RuntimeError(f'source drift: {file}')
            after = after.replace(old, new)
        path.write_text(after)
        # New precision hunks use zero context so a blank unified-diff context
        # marker is not recorded as trailing whitespace in this tracked artifact.
        context = 0 if file in precision_files or file in {'choices.py', 'ir.py'} else 3
        patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile='a/torch/_inductor/'+file, tofile='b/torch/_inductor/'+file, n=context))
        hashes[file] = {'original': hashlib.sha256(before.encode()).hexdigest(), 'patched': hashlib.sha256(after.encode()).hexdigest()}
    artifacts = ROOT/'artifacts'
    if not precision_casts:
        # Legacy diagnostics must not replace normal, reviewable source evidence.
        artifacts /= 'legacy-overlay'
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts/'pytorch.patch').write_text(''.join(patches))
    (artifacts/'overlay.json').write_text(json.dumps(hashes, indent=2))
    torch.__path__.insert(0, str(dest.parent))
