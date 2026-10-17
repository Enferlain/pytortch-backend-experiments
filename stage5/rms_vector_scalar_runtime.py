"""Recompile unchanged 5nb scalar control into the NEW packet only."""

import hashlib
import json
import subprocess
from types import SimpleNamespace

from stage5.rms_trial_runtime import (
    BINARY,
    MECHANISM,
    PLAN,
    Kernel,
    admit,
    cpu_gate,
    sha,
)
from stage5.rms_vector_trial_data import OUT

CACHE = {}


def compile_kernel(semantic, device):
    admit(semantic)
    cpu_gate()
    key = (json.dumps(semantic, sort_keys=True), device)
    if key not in CACHE:
        identity = {
            "semantic": semantic,
            "binary": sha(BINARY),
            "control_adapter": sha(__file__),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        directory = OUT / "scalar-control" / digest
        directory.mkdir(parents=True, exist_ok=True)
        packet = {
            "schema": 5,
            "semantic": semantic,
            "physical_plan": PLAN,
            "mechanism": MECHANISM,
            "entrypoint": "cubecl5_rms_cubek_" + digest[:16],
        }
        descriptor = directory / "descriptor.json"
        descriptor.write_text(json.dumps(packet, indent=2))
        run = subprocess.run(
            [str(BINARY), str(directory), str(descriptor)],
            capture_output=True,
            text=True,
        )
        (directory / "compiler.txt").write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError(run.stderr)
        kernel = Kernel(directory, semantic, device)
        CACHE[key] = SimpleNamespace(
            kernel=kernel,
            semantic=semantic,
            directory=directory,
            entrypoint=packet["entrypoint"],
        )
    return CACHE[key].kernel
