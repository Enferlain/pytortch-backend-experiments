"""The audited CubeCL source state: pin plus the exact FP patch stack."""

import hashlib
import subprocess
from pathlib import Path

from cubecl_torch.runtime import native

DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / ".stage0-sources" / "cubecl"


class SourceMismatch(RuntimeError):
    """The CubeCL checkout is not the audited pin and patch state."""


def check(source=DEFAULT_SOURCE, *, pin=native.CUBECL_PIN, patch=native.FP_PATCH):
    """Verify ``source``; return its provenance fields. Never modifies it."""

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(source), *args], capture_output=True, check=True
        ).stdout

    try:
        head = git("rev-parse", "HEAD").decode().strip()
        staged = git("diff", "--cached")
        untracked = git("ls-files", "--others", "--exclude-standard")
        diff = git("diff", "HEAD")
    except (OSError, subprocess.CalledProcessError) as error:
        raise SourceMismatch(
            f"cannot inspect CubeCL checkout {source}: {error}"
        ) from error
    if head != pin:
        raise SourceMismatch(
            f"CubeCL checkout {source} is at {head}, audited pin is {pin}"
        )
    if staged or untracked:
        raise SourceMismatch(f"CubeCL checkout {source} has staged or untracked files")
    if diff != Path(patch).read_bytes():
        raise SourceMismatch(
            f"CubeCL checkout {source} differs from the audited FP patch stack {patch}"
        )
    return {"cubecl_commit": head, "fp_patch_sha256": hashlib.sha256(diff).hexdigest()}
