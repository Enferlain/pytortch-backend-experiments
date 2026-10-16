"""Execute the immutable, predeclared fresh-process pair schedule once."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    repo = root.parents[3]
    plan = json.loads((root / "predeclared-plan.json").read_text())
    assert (
        hashlib.sha256((root / "predeclared-plan.json").read_bytes()).hexdigest()
        == (root / "predeclared-plan.sha256").read_text().strip()
    )
    env = dict(
        os.environ,
        LD_LIBRARY_PATH="/usr/lib/wsl/lib:"
        + str(repo / "stage0/python-deps/nvidia/cu13/lib"),
    )
    for task in plan["schedule"][args.start : args.stop]:
        output = root / (task["id"] + f"-{task['round']:02d}")
        if output.exists():
            raise FileExistsError(output)
        command = [
            str(repo / "stage0/.venv/bin/python"),
            "-m",
            "stage5.reduction_memory_state",
            "--output",
            str(output),
            "--plan",
            str(root / "predeclared-plan.json"),
            "--contrast",
            task["id"],
            "--round-start",
            str(task["round"]),
            "--iterations",
            "65536",
            "--modes",
            *task["modes"],
            "--pm-metrics",
            ",".join(plan["pm_metrics"]),
            "--sampler-library",
            str(root / "sampler.so"),
            "--idle-library",
            str(root / "idle.so"),
        ]
        log_path = root / (output.name + ".log")
        with log_path.open("x") as log:
            process = subprocess.run(
                command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        with (root / "commands.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"command": command, "returncode": process.returncode})
                + "\n"
            )
        if process.returncode:
            raise RuntimeError(f"Failed {output.name}; preserve log and stop")
        result = json.loads((output / "result.json").read_text())
        print(
            output.name,
            {k: v["median_us"] for k, v in result["summary"].items()},
            flush=True,
        )


if __name__ == "__main__":
    main()
