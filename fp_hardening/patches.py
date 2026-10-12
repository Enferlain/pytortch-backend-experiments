"""Apply only the known pinned patch states; preserve unrelated local changes."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".stage0-sources/cubecl"
PIN = "6bba5fdf04206d9c3710dfbc46aa9850450ab405"


def ensure():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(SOURCE), *args])

    if git("rev-parse", "HEAD").decode().strip() != PIN:
        raise RuntimeError("CubeCL pin differs from the audited version")
    if git("diff", "--cached") or git("ls-files", "--others", "--exclude-standard"):
        raise RuntimeError(
            "preserve and inspect unexpected staged/untracked CubeCL files"
        )
    current = git("diff")
    base = (ROOT / "stage2/cubecl-ieee.patch").read_bytes()
    full = (ROOT / "fp_hardening/cubecl-fp.patch").read_bytes()
    if current == full:
        return
    if current not in (b"", base):
        raise RuntimeError(
            "preserve and inspect CubeCL changes outside the exact audited patch stack"
        )
    patch = (
        ROOT
        / "fp_hardening"
        / ("cubecl-fp-upgrade.patch" if current else "cubecl-fp.patch")
    )
    git("apply", "--check", str(patch))
    git("apply", str(patch))
    if git("diff") != full:
        raise RuntimeError("applied CubeCL patch differs from retained evidence")


if __name__ == "__main__":
    ensure()
