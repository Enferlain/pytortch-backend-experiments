"""Sequential fresh-process experiments; versioned evidence, no baseline overwrite."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="")
    parser.add_argument("--only", choices=("history", "constants", "host", "systems"))
    parser.add_argument("--precision-probe", action="store_true")
    parser.add_argument("--all-api-probe", action="store_true")
    args = parser.parse_args()
    out = ROOT / "artifacts/attribution"
    cache = ROOT / "cache/attribution"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        STAGE0_CARGO_PROFILE="release",
        CUDA_PATH=str(PROJECT / "stage0/cuda13-toolkit/nvidia/cu13"),
        TRITON_LIBCUDA_PATH="/usr/lib/wsl/lib",
        TRITON_LIBDEVICE_PATH=str(
            PROJECT / "stage0/cuda13-toolkit/nvidia/cu13/nvvm/libdevice/libdevice.10.bc"
        ),
    )
    env.pop("PYTHON_GIL", None)
    # torch.cuda.nvtx uses string ranges rather than registered NVTX strings.
    env["NSYS_NVTX_PROFILER_REGISTER_ONLY"] = "0"
    cube = str(PROJECT / "stage0/.venv/bin/python")
    reference = "/home/imi/Projects/sd-scripts/.venv-wsl/bin/python"
    commands = []
    for impl, python in (("cubecl", cube), ("triton", reference)):
        for history in ("large_first", "short_first"):
            name = "history-" + impl + "-" + history
            commands.append(
                (
                    "history",
                    name,
                    [
                        python,
                        "-m",
                        "stage5.attribution",
                        "history",
                        "--impl",
                        impl,
                        "--history",
                        history,
                        "--output",
                        str(out / (name + ".json")),
                    ],
                )
            )
        name = "constants-" + impl
        commands.append(
            (
                "constants",
                name,
                [
                    python,
                    "-m",
                    "stage5.attribution",
                    "constants",
                    "--impl",
                    impl,
                    "--output",
                    str(out / (name + ".json")),
                ],
            )
        )
    for impl in ("cubecl", "reference", "eager", "triton"):
        for size in ("tiny", "larger"):
            name = "host-" + impl + "-" + size
            command = [
                cube if impl in ("cubecl", "reference") else reference,
                "-m",
                "stage5.system_profile",
                "--impl",
                impl,
                "--size",
                size,
            ]
            commands.append(
                ("host", name, [*command, "--output", str(out / (name + ".json"))])
            )
            for run in range(3 if impl in ("cubecl", "reference") else 1):
                name = f"systems-{impl}-{size}-{run}"
                nsys = shutil.which("nsys")
                if nsys is None:
                    raise RuntimeError("WSL-native Nsight Systems required on PATH")
                prefix = [
                    str(Path(nsys).resolve()),
                    "profile",
                    "--trace=cuda,nvtx,cublas",
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--capture-range=nvtx",
                    "--nvtx-capture=stage5_capture",
                    "--capture-range-end=stop",
                    "--force-overwrite=true",
                    "--output=" + str(cache / name),
                ]
                commands.append(
                    (
                        "systems",
                        name,
                        [
                            *prefix,
                            *command,
                            "--capture",
                            "--output",
                            str(out / (name + ".json")),
                        ],
                    )
                )
            if impl in ("cubecl", "reference") and size == "tiny":
                name = f"systems-{impl}-deep"
                prefix[-1] = "--output=" + str(cache / name)
                commands.append(
                    (
                        "systems",
                        name,
                        [
                            *prefix,
                            *command,
                            "--capture",
                            "--deep",
                            "--steps",
                            "20",
                            "--output",
                            str(out / (name + ".json")),
                        ],
                    )
                )
    if args.only:
        commands = [c for c in commands if c[0] == args.only]
    if args.all_api_probe:
        if args.only != "systems":
            raise ValueError("all API probe requires --only systems")
        category, name, command = next(
            c for c in commands if c[1] == "systems-cubecl-deep"
        )
        name += "-all-api"
        command = [command[0], command[1], "--cuda-trace-all-apis=true", *command[2:]]
        command = [
            "--output=" + str(cache / name) if item.startswith("--output=") else item
            for item in command
        ]
        command[-1] = str(out / (name + ".json"))
        commands = [(category, name, command)]
    if args.precision_probe:
        if args.only != "constants":
            raise ValueError("precision probe requires --only constants")
        commands = [
            (
                category,
                name + "-probe",
                [*command[:-1], str(out / (name + "-probe.json")), "--precision-probe"],
            )
            for category, name, command in commands
        ]
    if args.start:
        index = next(i for i, c in enumerate(commands) if c[1] == args.start)
        commands = commands[index:]
    ledger = out / "processes.json"
    rows = json.loads(ledger.read_text()) if ledger.exists() else []
    for category, name, command in commands:
        print("START", name, flush=True)
        report = cache / (name + ".nsys-rep")
        previous_report_mtime = report.stat().st_mtime_ns if report.exists() else None
        with (cache / (name + ".log")).open("w") as log:
            process = subprocess.run(
                command, env=env, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT
            )
        fresh_report = (
            report.exists() and report.stat().st_mtime_ns != previous_report_mtime
        )
        rows.append(
            {
                "name": name,
                "category": category,
                "command": command,
                "exit_code": process.returncode,
                "report_verified_fresh": fresh_report
                if category == "systems"
                else None,
                "environment": {
                    key: env.get(key)
                    for key in (
                        "PYTHON_GIL",
                        "NSYS_NVTX_PROFILER_REGISTER_ONLY",
                        "CUDA_PATH",
                        "TRITON_LIBCUDA_PATH",
                        "TRITON_LIBDEVICE_PATH",
                    )
                },
            }
        )
        ledger.write_text(json.dumps(rows, indent=2))
        if process.returncode:
            print((cache / (name + ".log")).read_text()[-8000:], flush=True)
            raise SystemExit(process.returncode)
        if category == "systems" and not fresh_report:
            raise RuntimeError("NVTX capture produced no report: " + name)
        print("PASS", name, flush=True)


if __name__ == "__main__":
    main()
