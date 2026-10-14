"""Fresh-process regression gates with new evidence and preserved stage baselines."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUTPUT = ROOT / "artifacts/precision"
CACHE = ROOT / "cache/precision/gates"


def main(extra_commands=()):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    # Established reports reference these tracked captures. Collect new gate
    # results separately, then restore the historical bytes; generated caches
    # and overlay remain current. Source and Beads files are never restored.
    tracked = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "stage0/artifacts",
                "stage1/artifacts",
                "stage1_5/artifacts",
                "stage2/artifacts",
                "stage3/artifacts",
                "stage4/artifacts",
                "fp_hardening/artifacts",
                "stage5/artifacts",
            ],
            cwd=PROJECT,
        )
        .decode()
        .split("\0")
    )
    saved = {
        p: (PROJECT / p).read_bytes()
        for p in tracked
        if p
        and not p.startswith(str(OUTPUT.relative_to(PROJECT)) + "/")
        and (PROJECT / p).is_file()
    }
    for p in ("stage5/artifacts/math-cubecl.json",):
        if (PROJECT / p).is_file():
            saved[p] = (PROJECT / p).read_bytes()
    commands = [("stage0_to_4_fp", [sys.executable, "stage5/regressions.py"])]
    commands += [
        (name, [sys.executable, "-m", "stage5." + name])
        for name in (
            "components",
            "controls",
            "proof_controls",
            "descriptor_controls",
            "launch_controls",
            "launch_hook_checks",
        )
    ]
    commands += [
        ("cache_" + mode, [sys.executable, "-m", "stage5.cache_probe", mode])
        for mode in ("init", "reload")
    ]
    commands += [
        ("compiler_compare", [sys.executable, "-m", "stage5.compiler_compare"]),
        (
            "math_cubecl",
            [sys.executable, "-m", "stage5.math_policy", "--impl", "cubecl"],
        ),
        (
            "corpus",
            [
                sys.executable,
                "-m",
                "stage5.bench",
                "--impl",
                "cubecl_stage5",
                "--profile",
                "--samples",
                "3",
                "--kernel-samples",
                "3",
                "--output",
                str(OUTPUT / "corpus.json"),
            ],
        ),
    ]
    environment = {
        **os.environ,
        "STAGE0_CARGO_PROFILE": "release",
        "CUDA_PATH": str(PROJECT / "stage0/cuda13-toolkit/nvidia/cu13"),
        "CUBECL_STAGE5_TRACE": "1",
    }
    commands += list(extra_commands)
    rows = []
    try:
        for name, command in commands:
            with (CACHE / (name + ".log")).open("w") as log:
                p = subprocess.run(
                    command,
                    cwd=PROJECT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            rows.append({"name": name, "command": command, "exit_code": p.returncode})
            (OUTPUT / "gates.json").write_text(json.dumps(rows, indent=2))
            print(name, "PASS" if p.returncode == 0 else "FAIL", flush=True)
            if p.returncode:
                raise SystemExit(p.returncode)
    finally:
        # Snapshot every refreshed tracked result, including failed attempts.
        changed = []
        for relative, original in saved.items():
            path = PROJECT / relative
            if path.is_file() and path.read_bytes() != original:
                target = OUTPUT / "regressions" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                changed.append(relative)
                path.write_bytes(original)
        (OUTPUT / "refreshed-artifacts.json").write_text(json.dumps(changed, indent=2))


if __name__ == "__main__":
    main()
