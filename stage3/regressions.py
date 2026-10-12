"""Sequential fresh-process baseline suites (overlay installation is not concurrent)."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PYTHON = str(PROJECT / "stage0/.venv/bin/python")
environment = {
    **os.environ,
    "CUDA_PATH": str(PROJECT / "stage0/cuda13-toolkit/nvidia/cu13"),
}
commands = [
    ("stage0_cpp", [PYTHON, "stage0/verify.py", "cpp"]),
    ("stage0_llvm", [PYTHON, "stage0/verify.py", "llvm"]),
    ("stage1", [PYTHON, "stage1/verify.py"]),
    ("stage1_5", [PYTHON, "stage1/verify_boundaries.py"]),
    ("stage2", [PYTHON, "stage2/verify.py"]),
    ("stage2_components", [PYTHON, "stage2/components.py"]),
    (
        "fp_ir",
        [
            "cargo",
            "test",
            "--manifest-path",
            "stage0/Cargo.toml",
            "--locked",
            "--features",
            "llvm,hip-probe",
            "--test",
            "fp_rewrites",
        ],
    ),
    ("fp_runtime", [PYTHON, "fp_hardening/verify.py"]),
    ("fp_inductor", [PYTHON, "fp_hardening/verify_inductor.py"]),
]
results = []
for name, command in commands:
    with (ROOT / (name + ".log")).open("w") as log:
        proc = subprocess.run(
            command, cwd=PROJECT, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
    results.append({"name": name, "command": command, "exit_code": proc.returncode})
    (ROOT / "artifacts/regressions.json").write_text(json.dumps(results, indent=2))
    print(name, "PASS" if proc.returncode == 0 else "FAIL", flush=True)
    if proc.returncode:
        raise SystemExit(proc.returncode)
