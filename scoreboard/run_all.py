"""Run every requested (workload, path) pair in its own process and summarize.

Each path runs in the interpreter that matches it: eager and CubeCL in the isolated
Triton-free 3.14t environment, Triton in the sd-scripts environment.

Example:
  stage0/.venv/bin/python scoreboard/run_all.py                      # everything, fail-fast CubeCL
  stage0/.venv/bin/python scoreboard/run_all.py --on-reject record   # CubeCL inventory run
  stage0/.venv/bin/python scoreboard/run_all.py --workloads sdxl --resolution 512
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scoreboard.report import render

PYTHON = {
    "eager": os.environ.get(
        "SCOREBOARD_ISOLATED_PYTHON", str(ROOT / "stage0/.venv/bin/python")
    ),
    "cubecl": os.environ.get(
        "SCOREBOARD_ISOLATED_PYTHON", str(ROOT / "stage0/.venv/bin/python")
    ),
    "triton": os.environ.get(
        "SCOREBOARD_TRITON_PYTHON", "/home/imi/Projects/sd-scripts/.venv-wsl/bin/python"
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument("--workloads", default="sdxl,anima,optimizer")
parser.add_argument("--paths", default="eager,triton,cubecl")
parser.add_argument("--out-dir")
parser.add_argument("--on-reject", choices=("fail", "record"), default="fail")
parser.add_argument("--timeout", type=int, default=3 * 3600, help="seconds per run")
parser.add_argument(
    "--passthrough",
    nargs=argparse.REMAINDER,
    default=[],
    help="extra run_one.py options, e.g. --passthrough --resolution 512 --steps 5",
)
args = parser.parse_args()

out = Path(
    args.out_dir
    or ROOT
    / "scoreboard/results"
    / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
)
if args.on_reject == "record" and not args.out_dir:
    out = out.with_name(out.name + "-inventory")
out.mkdir(parents=True, exist_ok=True)

for workload in args.workloads.split(","):
    for path in args.paths.split(","):
        stem = f"{workload}-{path}"
        command = [
            PYTHON[path],
            str(ROOT / "scoreboard/run_one.py"),
            "--path",
            path,
            "--workload",
            workload,
            "--out",
            str(out / f"{stem}.json"),
            *args.passthrough,
        ]
        if path == "cubecl":
            command += ["--on-reject", args.on_reject]
        print(f"[{datetime.now().astimezone():%H:%M:%S}] {stem} ...", flush=True)
        began = time.perf_counter()
        with open(out / f"{stem}.log", "w") as log:
            try:
                code = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                ).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
                (out / f"{stem}.json").write_text(
                    json.dumps(
                        {
                            "workload": workload,
                            "path": path,
                            "status": "timeout",
                            "environment": {"gil": {}},
                            "failure": {
                                "stage": "timeout",
                                "error": f"no result within {args.timeout}s",
                            },
                        }
                    )
                )
        print(f"    exit {code} after {time.perf_counter() - began:.0f}s", flush=True)

print(render(out))
(out / "summary.md").write_text(render(out))
print(f"Results: {out}")
