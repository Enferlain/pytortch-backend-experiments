"""Focused Linux Nsight counters; raw CSV remains local, summaries are curated."""

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ncu_path = shutil.which("ncu")
if ncu_path is None:
    raise RuntimeError("WSL-native Nsight Compute (ncu) must be available on PATH")
NCU = Path(ncu_path).resolve()
result = {
    "tool": str(NCU),
    "version": subprocess.check_output([str(NCU), "--version"], text=True),
    "sha256": hashlib.sha256(NCU.read_bytes()).hexdigest(),
    "classification": "WSL-native Linux profiling of production CUDA13.0 NVRTC/driver kernels; host counter permissions enabled",
    "records": [],
}
for name in (
    "scalar",
    "vector",
    "row_tree",
    "row_plane",
    "row_packed",
    "column_tree",
    "column_coalesced",
):
    raw = ROOT / ("ncu-" + name + ".csv")
    cmd = [
        str(NCU),
        "--kernel-name",
        "regex:cubecl5_",
        "--launch-skip",
        "1",
        "--launch-count",
        "1",
        "--section",
        "LaunchStats",
        "--section",
        "Occupancy",
        "--section",
        "SpeedOfLight",
        "--section",
        "MemoryWorkloadAnalysis",
        "--csv",
        "--log-file",
        str(raw),
        str(PROJECT / "stage0/.venv/bin/python"),
        "-m",
        "stage5.ncu_probe",
        name,
    ]
    p = subprocess.run(
        cmd,
        cwd=PROJECT,
        env={**os.environ, "STAGE0_CARGO_PROFILE": "release"},
        capture_output=True,
        text=True,
    )
    (ROOT / ("ncu-" + name + ".log")).write_text(p.stdout + p.stderr)
    content = raw.read_text() if raw.exists() else ""
    if p.returncode or "ERR_" in content or '"ID"' not in content:
        raise RuntimeError(f"Nsight {name} failed: {p.returncode}: {content[-2000:]}")
    rows = list(csv.DictReader(content[content.index('"ID"') :].splitlines()))
    manifest = json.loads(
        (ROOT / "artifacts" / ("ncu-" + name + "-manifest.json")).read_text()
    )
    result["records"].append(
        {
            "case": name,
            **manifest,
            "command": cmd,
            "raw_csv_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "metrics": [
                {
                    k: row[k]
                    for k in (
                        "Section Name",
                        "Metric Name",
                        "Metric Unit",
                        "Metric Value",
                    )
                }
                for row in rows
                if row["Metric Name"]
            ],
        }
    )
    (ROOT / "artifacts/profiling.json").write_text(json.dumps(result, indent=2))
    print("Nsight", name, "PASS", flush=True)
