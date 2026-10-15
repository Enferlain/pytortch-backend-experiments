"""CPU regression fixtures for native-review provenance findings."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from unittest.mock import patch

from stage5 import reduction_codegen_gates as gates
from stage5.reduction_codegen_setup import extract_fresh


def main():
    checks = []
    with tempfile.TemporaryDirectory() as name:
        project = Path(name)
        root = project / "stage5"
        output = root / "artifacts/reduction-structure/codegen"
        output.mkdir(parents=True)
        history = project / "stage0/artifacts"
        history.mkdir(parents=True)
        originals = {
            "tracked": b"tracked",
            "untracked": b"untracked",
            "ignored": b"ignored",
            "regular": b"regular",
        }
        for filename, value in originals.items():
            (history / filename).write_bytes(value)
        external = project / "external"
        external.write_bytes(b"external")
        (history / "link").symlink_to(external)
        (history / "broken").symlink_to("missing target")
        subprocess.run(["git", "init", "-q", str(project)], check=True)
        subprocess.run(
            ["git", "-C", str(project), "add", str(history / "tracked")], check=True
        )
        body = """from pathlib import Path
import sys
project=Path(sys.argv[1])
h=project/'stage0/artifacts'
out=project/'stage5/artifacts/reduction-structure/codegen'
external=project/'external'
(h/'tracked').write_bytes(b'changed')
(h/'untracked').write_bytes(b'changed untracked')
(h/'ignored').unlink()
(h/'regular').unlink(); (h/'regular').symlink_to(external)
(h/'link').unlink(); (h/'link').write_bytes(b'replaced symlink')
(h/'broken').unlink(); (h/'broken').symlink_to('changed target')
(h/'new/nested').mkdir(parents=True, exist_ok=True)
(h/'new/nested/failure.txt').write_bytes(b'new evidence')
(out/'live').write_bytes(b'current output')
"""
        for exit_code in (0, 7):
            commands = [
                {
                    "name": "fixture",
                    "command": [
                        sys.executable,
                        "-c",
                        body + f"\nraise SystemExit({exit_code})\n",
                        str(project),
                    ],
                }
            ]
            (root / "artifacts/reduction-structure/gates.json").write_text(
                json.dumps(commands)
            )
            with patch.object(gates, "ROOT", root), patch.object(gates, "OUT", output):
                if exit_code:
                    try:
                        gates.main()
                    except SystemExit as error:
                        assert error.code == exit_code
                    else:
                        raise AssertionError("gate failure lost")
                else:
                    gates.main()
            for filename, value in originals.items():
                p = history / filename
                assert not p.is_symlink() and p.read_bytes() == value, filename
            assert (history / "link").is_symlink() and os.readlink(
                history / "link"
            ) == str(external)
            assert (history / "broken").is_symlink() and os.readlink(
                history / "broken"
            ) == "missing target"
            assert external.read_bytes() == b"external"
            assert not (history / "new/nested/failure.txt").exists()
            report = json.loads((output / "historical-preservation.json").read_text())
            saved = Path(report["gate_created_files_preserved_in"])
            assert (
                saved / "stage0/artifacts/new/nested/failure.txt"
            ).read_bytes() == b"new evidence"
            assert (output / "live").read_bytes() == b"current output"
            status = json.loads((output / "gate-status.json").read_text())
            assert status["complete"] == (exit_code == 0)
            assert report["all_hashes_match"] and report["no_new_historical_files"]
            assert report["protected_files"] == 7
            checks.append(
                {
                    "gate_exit_code": exit_code,
                    "all_regular_and_symlink_identities_restored": True,
                    "new_evidence_preserved_outside_history": True,
                    "completion_status_correct": True,
                }
            )
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as files:
            data = b'[package]\nname="pinned"\n'
            info = tarfile.TarInfo("crates/reduce/Cargo.toml")
            info.size = len(data)
            files.addfile(info, io.BytesIO(data))
        destination = project / "cache/cubek"
        (destination / "crates/reduce").mkdir(parents=True)
        (destination / "crates/reduce/build.rs").write_text("stale executable code")
        extract_fresh(archive.getvalue(), destination)
        assert {
            str(p.relative_to(destination))
            for p in destination.rglob("*")
            if p.is_file()
        } == {"crates/reduce/Cargo.toml"}
        target = project / "external-dir"
        target.mkdir()
        (target / "keep").write_text("safe")
        linked = project / "linked"
        linked.symlink_to(target, target_is_directory=True)
        extract_fresh(archive.getvalue(), linked)
        assert not linked.is_symlink() and (target / "keep").read_text() == "safe"
        checks.append(
            {
                "fresh_archive_removes_stale_build_rs": True,
                "cache_symlink_target_untouched": True,
            }
        )
    packet = Path(
        "stage5/artifacts/reduction-structure/codegen/review-provenance-checks.json"
    )
    previous = json.loads(packet.read_text())
    previous["cpu_checks"] = checks
    previous["checks"] = {
        "full_main_success_and_failure_paths": True,
        "tracked_untracked_ignored_and_symlink_identity": True,
        "fresh_archive_and_symlink_destination": True,
    }
    packet.write_text(json.dumps(previous, indent=2))
    print(json.dumps({"complete": True, "cpu_checks": checks}, indent=2))


if __name__ == "__main__":
    main()
