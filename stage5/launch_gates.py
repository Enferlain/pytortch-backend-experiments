"""Sequential follow-up gates; retain new evidence without replacing baselines."""

import json
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PYTHON = str(PROJECT / "stage0/.venv/bin/python")
OUTPUT = ROOT / "artifacts/launch"
OUTPUT.mkdir(parents=True, exist_ok=True)
env = {
    **os.environ,
    "STAGE0_CARGO_PROFILE": "release",
    "CUDA_PATH": str(PROJECT / "stage0/cuda13-toolkit/nvidia/cu13"),
    "CUBECL_STAGE5_TRACE": "1",
}
commands = [
    (name, [PYTHON, "-m", "stage5." + name])
    for name in ("components", "controls", "proof_controls", "descriptor_controls")
]
commands += [
    ("cache_init", [PYTHON, "-m", "stage5.cache_probe", "init"]),
    ("cache_reload", [PYTHON, "-m", "stage5.cache_probe", "reload"]),
    ("compiler_compare", [PYTHON, "-m", "stage5.compiler_compare"]),
    ("math", [PYTHON, "-m", "stage5.math_policy", "--impl", "cubecl"]),
    ("launch_controls", [PYTHON, "-m", "stage5.launch_controls"]),
    ("launch_hook_checks", [PYTHON, "-m", "stage5.launch_hook_checks"]),
    (
        "corpus",
        [
            PYTHON,
            "-m",
            "stage5.bench",
            "--impl",
            "cubecl_stage5",
            "--profile",
            "--kernel-samples",
            "3",
            "--output",
            str(OUTPUT / "corpus.json"),
        ],
    ),
    (
        "paired",
        [
            PYTHON,
            "-m",
            "stage5.launch_profile",
            "--perf",
            "--output",
            str(OUTPUT / "performance.json"),
        ],
    ),
]
rows = []
parser = argparse.ArgumentParser()
parser.add_argument("--start", choices=[n for n, _ in commands])
args = parser.parse_args()
if args.start:
    offset = [n for n, _ in commands].index(args.start)
    saved = json.loads((OUTPUT / "gates.json").read_text())
    history_path = OUTPUT / "gate-attempts.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(saved)
    history_path.write_text(json.dumps(history, indent=2))
    rows = saved[:offset]
    assert len(rows) == offset and all(r["exit_code"] == 0 for r in rows)
    commands = commands[offset:]
for name, command in commands:
    with (ROOT / ("launch-" + name + "-gate.log")).open("w") as log:
        result = subprocess.run(
            command,
            cwd=PROJECT,
            env={**env, "CUBECL_STAGE5_TRACE": "0"}
            if name in ("launch_controls", "paired")
            else env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    rows.append({"name": name, "command": command, "exit_code": result.returncode})
    (OUTPUT / "gates.json").write_text(json.dumps(rows, indent=2))
    print(name, "PASS" if result.returncode == 0 else "FAIL", flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
paths = [
    "components.json",
    "controls.json",
    "proof-controls.json",
    "descriptor-controls.json",
    "cache-probe-init.json",
    "cache-probe-reload.json",
    "compiler-comparison.json",
]
(OUTPUT / "stage5-controls.json").write_text(
    json.dumps(
        {
            p: json.loads((ROOT / "artifacts" / p).read_text())
            for p in paths
            if (ROOT / "artifacts" / p).is_file()
        },
        indent=2,
    )
)
