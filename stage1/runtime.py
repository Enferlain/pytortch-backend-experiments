"""Stage 1 wrapper hook: build through Stage 0, launch through Stage 0."""
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import torch
from stage0.bridge import Kernel, native_path

ROOT = Path(__file__).resolve().parent
EVENTS = []
COMPILER_INVOCATIONS = 0
LAUNCHER_INVOCATIONS = 0
KERNEL_LAUNCHES = 0
LOCK = threading.RLock()
CACHE = {}

class CompiledKernel:
    def __init__(self, description, device):
        global COMPILER_INVOCATIONS
        raw = json.dumps(description, sort_keys=True)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        directory = ROOT/'artifacts/kernels'/digest[:16]
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'descriptor.json').write_text(raw)
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


def compile_kernel(description, device):
    # Process cache only; modules/metadata retain the Stage 0 lifetime contract.
    key = (json.dumps(description, sort_keys=True), device)
    with LOCK:
        if key not in CACHE:
            CACHE[key] = CompiledKernel(description, device)
        return CACHE[key]
