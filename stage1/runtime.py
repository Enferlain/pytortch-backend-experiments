"""Stage 1 wrapper hook: build through Stage 0, launch through Stage 0."""
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import torch
from stage0.bridge import Kernel, native_path
from stage1.plan import bridge_description

ROOT = Path(__file__).resolve().parent
EVENTS = []
COMPILER_INVOCATIONS = 0
LAUNCHER_INVOCATIONS = 0
KERNEL_LAUNCHES = 0
LOCK = threading.RLock()
CACHE = {}

class CubeCLBackendUnavailable(RuntimeError):
    """Missing compiler/runtime dependency, distinct from unsupported semantics."""


def check_available(device):
    # Readiness is separate from compilation; never invoke a compiler as a probe.
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise CubeCLBackendUnavailable('CubeCL backend requires available PyTorch CUDA')
    for name in ('cubecl-stage0', 'libcubecl_stage0.so'):
        try:
            native_path(name)
        except RuntimeError as error:
            raise CubeCLBackendUnavailable(f'CubeCL compiler bridge missing: {name}; run bash stage0/setup.sh') from error
    if not (ROOT.parent/'stage0/libstage0_launcher.so').is_file():
        raise CubeCLBackendUnavailable('CubeCL native launcher missing; run bash stage0/setup.sh')


class CompiledKernel:
    def __init__(self, semantic, device, physical_plan):
        global COMPILER_INVOCATIONS
        description = bridge_description(semantic, physical_plan)
        raw = json.dumps(description, sort_keys=True)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        directory = ROOT/'artifacts/kernels'/digest[:16]
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'descriptor.json').write_text(raw)
        (directory/'semantic.json').write_text(json.dumps(semantic, sort_keys=True, indent=2))
        (directory/'physical-plan.json').write_text(json.dumps(physical_plan, sort_keys=True, indent=2))
        proc = subprocess.run([str(native_path('cubecl-stage0')), str(directory), 'cpp', str(directory/'descriptor.json')], capture_output=True, text=True)
        (directory/'compiler.log').write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        COMPILER_INVOCATIONS += 1
        self.kernel = Kernel(directory, device=device, compile_ptx=True)
        self.description = description
        self.directory = directory
        EVENTS.append({'event':'compile', 'descriptor_sha256':digest,
            'ptx_sha256':hashlib.sha256((directory/'kernel.ptx').read_bytes()).hexdigest(),
            'entrypoint':description['entrypoint'], 'artifact_dir':str(directory)})

    def __call__(self, *tensors):
        global LAUNCHER_INVOCATIONS, KERNEL_LAUNCHES
        # The unchanged bridge retrieves current_stream anew and records all buffers.
        with torch.profiler.record_function('stage1::CubeCL.launch'):
            result = self.kernel(*tensors)
        LAUNCHER_INVOCATIONS += 1
        KERNEL_LAUNCHES += int(tensors[0].numel() != 0)
        EVENTS.append({'event':'launch', 'entrypoint':self.description['entrypoint'],
            'n':tensors[0].numel(), 'pointers':[t.data_ptr() for t in tensors],
            'stream':torch.cuda.current_stream(self.kernel.device).cuda_stream})
        return result


def compile_kernel(description, device, physical_plan):
    # Process cache only; modules/metadata retain the Stage 0 lifetime contract.
    key = (json.dumps(description, sort_keys=True), json.dumps(physical_plan, sort_keys=True), device)
    with LOCK:
        if key not in CACHE:
            CACHE[key] = CompiledKernel(description, device, physical_plan)
        return CACHE[key]
