"""Run established gates and restore their historical tracked evidence."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import argparse

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "stage5/artifacts/reference512-admission/regressions"


def main(resume=None):
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = ROOT / "stage5/cache/reference512-gates.tar.gz"
    names = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    paths = [
        p
        for n in names
        if n
        and (p := ROOT / n).is_file()
        and ("/artifacts/" in n or (p.parent == ROOT and p.name.endswith("REPORT.md")))
    ]
    before = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    with tarfile.open(snapshot, "w:gz") as archive:
        for p in paths:
            archive.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    commands = json.loads(
        (ROOT / "stage5/artifacts/adoption/regressions/gates.json").read_text()
    )
    assert len(commands) == 18
    rows = []
    if resume:
        previous = json.loads(Path(resume).read_text())
        for row in previous:
            if row["exit_code"]:
                break
            row["log"] = str(Path(resume).parent / Path(row["log"]).name)
            assert (
                hashlib.sha256(Path(row["log"]).read_bytes()).hexdigest()
                == row["sha256"]
            )
            rows.append(row)
        assert [r["name"] for r in rows] == [r["name"] for r in commands[: len(rows)]]
    env = {**os.environ, "CUDA_PATH": str(ROOT / "stage0/cuda13-toolkit/nvidia/cu13")}
    try:
        for gate in commands[len(rows) :]:
            log = OUT / (gate["name"] + ".log")
            command = list(gate["command"])
            if "--output" in command:
                i = command.index("--output") + 1
                command[i] = str(OUT / Path(command[i]).name)
            with log.open("x") as handle:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            rows.append(
                {
                    "name": gate["name"],
                    "command": command,
                    "exit_code": result.returncode,
                    "log": str(log),
                    "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                }
            )
            (OUT / "gates.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(gate["name"], result.returncode, flush=True)
            if result.returncode:
                raise RuntimeError("gate failed: " + gate["name"])
    finally:
        changed = []
        with tarfile.open(snapshot) as archive:
            for member in archive.getmembers():
                dest = ROOT / member.name
                data = archive.extractfile(member).read()
                if not dest.exists() or dest.read_bytes() != data:
                    changed.append(member.name)
                    dest.write_bytes(data)
        assert all(
            hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h
            for n, h in before.items()
        )
        (OUT / "restoration.json").write_text(
            json.dumps(
                {
                    "tracked_files": len(before),
                    "restored": changed,
                    "all_preserved": True,
                },
                indent=2,
            )
            + "\n"
        )
    print("all18 gates pass; historical tracked evidence restored", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from")
    main(parser.parse_args().resume_from)
