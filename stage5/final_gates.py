"""Sequential final Stage5 verification and fair fixed-corpus captures."""

import json
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PYTHON = str(PROJECT / "stage0/.venv/bin/python")
REFERENCE = "/home/imi/Projects/sd-scripts/.venv-wsl/bin/python"
env = {
    **os.environ,
    "CUBECL_STAGE5_TRACE": "1",
    "STAGE0_CARGO_PROFILE": "release",
    "CUDA_PATH": str(PROJECT / "stage0/cuda13-toolkit/nvidia/cu13"),
    "TRITON_LIBCUDA_PATH": "/usr/lib/wsl/lib",
    "TRITON_LIBDEVICE_PATH": str(
        PROJECT / "stage0/cuda13-toolkit/nvidia/cu13/nvvm/libdevice/libdevice.10.bc"
    ),
}
commands = [
    (name, [PYTHON, "-m", "stage5." + name])
    for name in (
        "components",
        "controls",
        "proof_controls",
        "descriptor_controls",
        "launch_controls",
    )
]
commands += [
    ("cache_init", [PYTHON, "-m", "stage5.cache_probe", "init"]),
    ("cache_reload", [PYTHON, "-m", "stage5.cache_probe", "reload"]),
    ("compiler_compare", [PYTHON, "-m", "stage5.compiler_compare"]),
]
for impl in (
    "cubecl_baseline",
    "cubecl_stage5",
    "aten",
    "triton_matched",
    "triton_default",
):
    python = REFERENCE if impl.startswith("triton") else PYTHON
    commands.append(
        ("bench_" + impl, [python, "-m", "stage5.bench", "--impl", impl, "--profile"])
    )
for impl in ("cubecl", "triton_matched", "triton_default"):
    python = REFERENCE if impl.startswith("triton") else PYTHON
    commands.append(
        ("math_" + impl, [python, "-m", "stage5.math_policy", "--impl", impl])
    )
parser = argparse.ArgumentParser()
parser.add_argument("--start")
args = parser.parse_args()
rows = []
if args.start:
    names = [name for name, _ in commands]
    offset = names.index(args.start)
    saved = json.loads((ROOT / "artifacts/final-gates.json").read_text())
    rows = [r for r in saved if r["name"] in names[:offset]]
    assert len(rows) == offset and all(r["returncode"] == 0 for r in rows)
    commands = commands[offset:]
for name, command in commands:
    with (ROOT / ("final-" + name + ".log")).open("w") as log:
        p = subprocess.run(
            command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    rows.append({"name": name, "command": command, "returncode": p.returncode})
    (ROOT / "artifacts/final-gates.json").write_text(json.dumps(rows, indent=2))
    print(name, "PASS" if not p.returncode else "FAIL", flush=True)
    if p.returncode:
        raise SystemExit(p.returncode)
