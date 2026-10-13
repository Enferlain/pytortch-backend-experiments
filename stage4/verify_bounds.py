"""Isolate each device-trap probe so a poisoned CUDA context cannot hide a failure."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = []
for mode in ("upper", "negative_gather", "negative_index", "large", "min_i64"):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "bounds_probe.py"), mode],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (ROOT / ("bounds-" + mode + ".log")).write_text(proc.stdout + proc.stderr)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    row = json.loads(proc.stdout.splitlines()[-1])
    assert row["launched"] and not row["gil"]
    rows.append(row)
    (ROOT / "artifacts/bounds.json").write_text(json.dumps(rows, indent=2))
    print(mode, "runtime bounds PASS", flush=True)
