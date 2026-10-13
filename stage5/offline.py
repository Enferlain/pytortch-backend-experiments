"""Windows CUDA 13.1 offline diagnostics, never the production compiler route."""

import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
BIN = Path("/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin")


def winpath(path):
    return subprocess.check_output(
        ["wslpath", "-w", str(path.resolve())], text=True
    ).strip()


def run(directory):
    result = {
        "classification": "offline CUDA13.1 diagnostics; production uses PyTorch cu130/NVRTC CUDA13.0 and driver JIT",
        "production_ptx_sha256": hashlib.sha256(
            (directory / "kernel.ptx").read_bytes()
        ).hexdigest(),
        "tools": {},
    }
    for name in ("ptxas", "nvdisasm", "cuobjdump"):
        exe = BIN / (name + ".exe")
        if not exe.is_file():
            raise FileNotFoundError(exe)
        result["tools"][name] = {
            "path": str(exe),
            "version": subprocess.check_output([str(exe), "--version"], text=True),
            "sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
        }
    out = ROOT / "artifacts/offline" / directory.name
    out.mkdir(parents=True, exist_ok=True)
    cubin = out / "diagnostic.cubin"
    cmd = [
        str(BIN / "ptxas.exe"),
        "-arch=sm_86",
        "--fmad=false",
        "-v",
        winpath(directory / "kernel.ptx"),
        "-o",
        winpath(cubin),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    result["ptxas"] = {
        "command": cmd,
        "stdout": p.stdout,
        "stderr": p.stderr,
        "returncode": p.returncode,
    }
    (out / "ptxas.txt").write_text(p.stdout + p.stderr)
    if p.returncode == 0:
        for name, options in [
            ("nvdisasm", []),
            ("cuobjdump", ["--dump-resource-usage", "--dump-sass"]),
        ]:
            p = subprocess.run(
                [str(BIN / (name + ".exe")), *options, winpath(cubin)],
                capture_output=True,
                text=True,
            )
            (out / (name + ".txt")).write_text(p.stdout + p.stderr)
            result[name] = {
                "returncode": p.returncode,
                "output_sha256": hashlib.sha256(p.stdout.encode()).hexdigest(),
                "lines": len(p.stdout.splitlines()),
                "vector_memory_lines": [
                    line.strip()
                    for line in p.stdout.splitlines()
                    if re.search(r"\b(?:LDG|STG)[.\w]*\.(?:128|64)\b", line)
                ],
                "local_memory_lines": [
                    line.strip()
                    for line in p.stdout.splitlines()
                    if "LDL" in line or "STL" in line
                ],
            }
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.directory), indent=2))
