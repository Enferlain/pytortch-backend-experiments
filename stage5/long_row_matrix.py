"""Fresh-process matched matrix; historical captures are never overwritten."""

import os
from pathlib import Path
import subprocess
import sys
import json
import argparse

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/long-row"
CACHE = ROOT / "cache/long-row"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    reference_python = Path("/home/imi/Projects/sd-scripts/.venv-wsl/bin/python")
    environment = {
        **os.environ,
        "STAGE0_CARGO_PROFILE": "release",
        "CUDA_PATH": str(ROOT.parent / "stage0/cuda13-toolkit/nvidia/cu13"),
        "TRITON_LIBCUDA_PATH": "/usr/lib/wsl/lib",
        "TRITON_LIBDEVICE_PATH": str(
            ROOT.parent
            / "stage0/cuda13-toolkit/nvidia/cu13/nvvm/libdevice/libdevice.10.bc"
        ),
    }
    specs = [
        ("baseline-fixed", "cubecl", "large_first", "baseline", "inherited"),
        (
            "compensated-fixed",
            "cubecl",
            "large_first",
            "compensated_diagnostic",
            "inherited",
        ),
        ("combined-fixed", "cubecl", "large_first", "combined_diagnostic", "inherited"),
        (
            "combined-guarded-short",
            "cubecl",
            "short_first",
            "combined_diagnostic",
            "guarded",
        ),
        (
            "combined-guarded-large",
            "cubecl",
            "large_first",
            "combined_diagnostic",
            "guarded",
        ),
        ("production-guarded-short", "cubecl", "short_first", "first_use", "guarded"),
        ("production-guarded-large", "cubecl", "large_first", "first_use", "guarded"),
        ("triton-short", "triton", "short_first", "first_use", "inherited"),
        ("triton-large", "triton", "large_first", "first_use", "inherited"),
        ("triton-mid", "triton", "mid_first", "first_use", "inherited"),
        ("triton-medium", "triton", "medium_first", "first_use", "inherited"),
        ("triton-million", "triton", "million_first", "first_use", "inherited"),
    ]
    completed = []
    for name, impl, history, policy, splits in specs:
        if args.references_only and name not in (
            "triton-mid",
            "triton-medium",
            "triton-million",
        ):
            continue
        interpreter = reference_python if impl == "triton" else Path(sys.executable)
        command = [
            str(interpreter),
            "-m",
            "stage5.reference_reduction_probe",
            "--impl",
            impl,
            "--history",
            history,
            "--cubecl-policy",
            policy,
            "--cubecl-split-policy",
            splits,
            "--long-row",
            "--rounds",
            "3",
            "--output",
            str(OUTPUT / (name + ".json")),
        ]
        with (CACHE / (name + ".log")).open("w") as log:
            result = subprocess.run(
                command,
                env=environment,
                cwd=ROOT.parent,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        completed.append(
            {"name": name, "command": command, "exit_code": result.returncode}
        )
        manifest = (
            "extra-reference-runs.json" if args.references_only else "matrix-runs.json"
        )
        (OUTPUT / manifest).write_text(json.dumps(completed, indent=2))
        print(name, "PASS" if result.returncode == 0 else "FAIL", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
