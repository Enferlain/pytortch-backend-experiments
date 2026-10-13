"""All established gates, sequential CUDA work and overlay installation."""

import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PYTHON = str(PROJECT / "stage0/.venv/bin/python")
env = {**os.environ, "CUDA_PATH": str(PROJECT / "stage0/cuda13-toolkit/nvidia/cu13")}
commands = [("stage0_to_3", [PYTHON, "stage4/regressions.py"])]
commands += [
    (label, [PYTHON, path])
    for label, path in (
        ("stage4_main", "stage4/verify.py"),
        ("stage4_runtime", "stage4/runtime_checks.py"),
        ("stage4_components", "stage4/components.py"),
        ("stage4_bounds", "stage4/verify_bounds.py"),
        ("stage4_counter_guard", "stage4/counter_guard_probe.py"),
    )
]
rows = []
for label, cmd in commands:
    with (ROOT / (label + "-regression.log")).open("w") as log:
        p = subprocess.run(
            cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    rows.append({"name": label, "command": cmd, "exit_code": p.returncode})
    (ROOT / "artifacts/regressions.json").write_text(json.dumps(rows, indent=2))
    print(label, "PASS" if p.returncode == 0 else "FAIL", flush=True)
    if p.returncode:
        raise SystemExit(p.returncode)
paths = [
    "stage4/artifacts/regression-details.json",
    "stage4/artifacts/results.json",
    "stage4/artifacts/runtime-checks.json",
    "stage4/artifacts/components.json",
    "stage4/artifacts/bounds.json",
    "stage4/artifacts/counter-guard.json",
]
(ROOT / "artifacts/regression-details.json").write_text(
    json.dumps(
        {
            p: json.loads((PROJECT / p).read_text())
            for p in paths
            if (PROJECT / p).exists()
        },
        indent=2,
    )
)
