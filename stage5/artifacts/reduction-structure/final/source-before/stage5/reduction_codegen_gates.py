"""Established regression gates with all earlier artifact packets preserved."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from stage5.reduction_codegen_mechanisms import OUT, ROOT


def artifact_files(roots):
    return {
        p
        for root in roots
        for p in root.rglob("*")
        if (p.is_file() or p.is_symlink()) and not p.is_relative_to(OUT)
    }


def snapshot_history(paths):
    return {
        p: ("symlink", os.readlink(p)) if p.is_symlink() else ("file", p.read_bytes())
        for p in paths
    }


def history_matches(p, kind, before):
    if kind == "symlink":
        return p.is_symlink() and os.readlink(p) == before
    return p.is_file() and not p.is_symlink() and p.read_bytes() == before


def restore_history(project, snapshot, roots, initial_paths):
    additions = sorted(artifact_files(roots) - initial_paths)
    saved = None
    if additions:
        # Keep new gate evidence, while removing it from historical discovery.
        saved = Path(tempfile.mkdtemp(prefix="historical-additions-", dir=OUT))
        for p in additions:
            target = saved / p.relative_to(project)
            target.parent.mkdir(parents=True, exist_ok=True)
            p.replace(target)
    restored = []
    for p, (kind, before) in snapshot.items():
        if not history_matches(p, kind, before):
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.exists():
                shutil.rmtree(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            if kind == "symlink":
                p.symlink_to(before)
            else:
                p.write_bytes(before)
            restored.append(str(p.relative_to(project)))
    return {
        "protected_files": len(snapshot),
        "restored": restored,
        "gate_created_files": [str(p.relative_to(project)) for p in additions],
        "gate_created_files_preserved_in": str(saved) if saved else None,
        "no_new_historical_files": not (artifact_files(roots) - initial_paths),
        "all_hashes_match": all(
            history_matches(p, kind, before) for p, (kind, before) in snapshot.items()
        ),
    }


def main():
    project = ROOT.parent
    cache = ROOT / "cache/codegen/gates"
    cache.mkdir(parents=True, exist_ok=True)
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=project)
        .decode()
        .split("\0")
    )
    scope = ROOT / "artifacts/reduction-structure"
    protected_roots = {
        project / name.partition("/artifacts/")[0] / "artifacts"
        for name in tracked
        if "/artifacts/" in name
    } | {scope}
    initial_paths = artifact_files(protected_roots)
    snapshot = snapshot_history(initial_paths)
    results = []
    commands = json.loads((scope / "gates.json").read_text())
    env = {
        **os.environ,
        "STAGE0_CARGO_PROFILE": "release",
        "CUDA_PATH": str(project / "stage0/cuda13-toolkit/nvidia/cu13"),
    }
    try:
        for gate in commands:
            cmd = [
                str(v).replace(str(scope) + "/", str(OUT) + "/")
                for v in gate["command"]
            ]
            log = cache / (gate["name"] + ".log")
            with log.open("w") as stream:
                run = subprocess.run(
                    cmd,
                    cwd=project,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            results.append(
                {
                    "name": gate["name"],
                    "command": cmd,
                    "exit_code": run.returncode,
                    "log": str(log),
                    "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                }
            )
            (OUT / "gates.json").write_text(json.dumps(results, indent=2))
            print(gate["name"], "PASS" if run.returncode == 0 else "FAIL", flush=True)
            if run.returncode:
                raise SystemExit(run.returncode)
    finally:
        preservation = restore_history(
            project, snapshot, protected_roots, initial_paths
        )
        (OUT / "historical-preservation.json").write_text(
            json.dumps(preservation, indent=2)
        )
        (OUT / "gate-status.json").write_text(
            json.dumps(
                {
                    "expected": len(commands),
                    "finished": len(results),
                    "complete": len(results) == len(commands)
                    and all(r["exit_code"] == 0 for r in results)
                    and preservation["all_hashes_match"]
                    and preservation["no_new_historical_files"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
