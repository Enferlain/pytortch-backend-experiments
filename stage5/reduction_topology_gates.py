"""Rerun established gates without overwriting historical tracked evidence."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

from stage5.reduction_topology_model import OUTPUT, SOURCE


def main():
    project = Path(__file__).resolve().parents[1]
    cache = project / "stage5/cache/reduction-topology/gates"
    cache.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "STAGE0_CARGO_PROFILE": "release",
        "CUDA_PATH": str(project / "stage0/cuda13-toolkit/nvidia/cu13"),
    }
    names = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=project)
        .decode()
        .split("\0")
    )
    snapshot = {
        project / name: (project / name).read_bytes()
        for name in names
        if name and "/artifacts/" in name and (project / name).is_file()
    }
    # The prior experiment may still be untracked. Protect its curated packet
    # too, rather than treating git's index as the complete evidence inventory.
    snapshot.update(
        {
            p: p.read_bytes()
            for p in SOURCE.rglob("*")
            if p.is_file() and not p.is_relative_to(OUTPUT)
        }
    )
    commands = json.loads((SOURCE / "gates.json").read_text())
    results = []
    (OUTPUT / "gate-status.json").write_text(
        json.dumps(
            {"expected": len(commands), "finished": 0, "complete": False}, indent=2
        )
    )
    try:
        for gate in commands:
            command = [
                str(v).replace(str(SOURCE) + "/", str(OUTPUT) + "/")
                for v in gate["command"]
            ]
            log = cache / (gate["name"] + ".log")
            with log.open("w") as stream:
                run = subprocess.run(
                    command,
                    cwd=project,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            results.append(
                {
                    "name": gate["name"],
                    "command": command,
                    "exit_code": run.returncode,
                    "log": str(log),
                    "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                }
            )
            (OUTPUT / "gates.json").write_text(json.dumps(results, indent=2))
            (OUTPUT / "gate-status.json").write_text(
                json.dumps(
                    {
                        "expected": len(commands),
                        "finished": len(results),
                        "complete": len(results) == len(commands)
                        and all(r["exit_code"] == 0 for r in results),
                    },
                    indent=2,
                )
            )
            print(gate["name"], "PASS" if run.returncode == 0 else "FAIL", flush=True)
            if run.returncode:
                raise SystemExit(run.returncode)
    finally:
        changed = {}
        for path, before in snapshot.items():
            after = path.read_bytes() if path.exists() else b""
            if after != before:
                relative = str(path.relative_to(project))
                changed[relative] = {
                    "original_sha256": hashlib.sha256(before).hexdigest(),
                    "capture_sha256": hashlib.sha256(after).hexdigest(),
                }
                # Preserve compact gate outcomes, not another copy of every
                # representative kernel and compiler manifest.
                if (
                    path.suffix == ".json"
                    and len(after) < 250_000
                    and "representatives" not in path.parts
                ):
                    try:
                        changed[relative]["capture"] = json.loads(after)
                    except (ValueError, UnicodeDecodeError):
                        pass
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(before)
        (OUTPUT / "regression-captures.json").write_text(
            json.dumps(
                {"restored_historical_tracked_files": True, "changed": changed},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
