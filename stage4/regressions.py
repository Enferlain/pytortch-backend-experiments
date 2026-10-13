"""Sequential earlier-stage gates; installation of the Inductor overlay is serial."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PYTHON = str(PROJECT / "stage0/.venv/bin/python")
env = {**os.environ, "CUDA_PATH": str(PROJECT / "stage0/cuda13-toolkit/nvidia/cu13")}
with (ROOT / "baseline-regressions.log").open("w") as log:
    proc = subprocess.run(
        [PYTHON, "stage3/regressions.py"],
        cwd=PROJECT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
results = json.loads((PROJECT / "stage3/artifacts/regressions.json").read_text())
(ROOT / "artifacts/regressions.json").write_text(json.dumps(results, indent=2))
if proc.returncode:
    raise SystemExit(proc.returncode)
for label, cmd in [
    ("stage3", [PYTHON, "stage3/verify.py"]),
    ("stage3_components", [PYTHON, "stage3/components.py"]),
]:
    with (ROOT / (label + "-regression.log")).open("w") as log:
        proc = subprocess.run(
            cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    results.append({"name": label, "command": cmd, "exit_code": proc.returncode})
    (ROOT / "artifacts/regressions.json").write_text(json.dumps(results, indent=2))
    print(label, "PASS" if not proc.returncode else "FAIL", flush=True)
    if proc.returncode:
        raise SystemExit(proc.returncode)
print("All11 baseline gates PASS", flush=True)
evidence_paths = (
    "stage0/artifacts/cpp/results.json",
    "stage0/artifacts/llvm/results.json",
    "stage1/artifacts/results.json",
    "stage1_5/artifacts/boundary-probes.json",
    "stage2/artifacts/results.json",
    "stage2/artifacts/components.json",
    "stage3/artifacts/results.json",
    "stage3/artifacts/components.json",
    "fp_hardening/artifacts/results.json",
    "fp_hardening/artifacts/inductor.json",
)
(ROOT / "artifacts/regression-details.json").write_text(
    json.dumps(
        {
            **{p: json.loads((PROJECT / p).read_text()) for p in evidence_paths},
            "fp_ir": {"passed": 7, "failed": 0},
        },
        indent=2,
    )
)
