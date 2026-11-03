"""The audited source state: the CubeCL pin plus its exact patch stack, and the
CubeK pin plus its own.

``python -m cubecl_torch.compiler.source [cubecl] [cubek]`` runs the same checks
for ``native/build.sh``; it needs neither torch nor a build.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from cubecl_torch.runtime import native

DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / ".stage0-sources" / "cubecl"
DEFAULT_CUBEK = DEFAULT_SOURCE.parent / "cubek"


class SourceMismatch(RuntimeError):
    """A checkout is not in its audited state."""


def _git(source, what):
    """A git runner for ``source`` that ignores the caller's ``GIT_*`` state,
    takes no optional locks, and reports any failure as a mismatch."""
    base = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("GIT_") or k == "GIT_EXEC_PATH"
    }
    base["GIT_OPTIONAL_LOCKS"] = "0"

    def git(*args, env=None):
        try:
            return subprocess.run(
                ["git", "-C", str(source), *args],
                capture_output=True,
                check=True,
                env={**base, **(env or {})},
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            detail = (getattr(error, "stderr", None) or b"").decode(errors="replace")
            raise SourceMismatch(
                f"cannot inspect {what} checkout {source}: git {' '.join(args)}: "
                f"{detail.strip() or error}"
            ) from error

    return git


def _check_stack(source, what, pin, patches):
    """Verify that ``source`` is ``pin`` with each patch in ``patches/series``
    applied in order; return its head. Never modifies it.

    The expected tree is the pin with the stack applied to a temporary index;
    the actual tree is the working tree staged into a temporary copy of the
    checkout's index. Both write their objects to a temporary object directory,
    so the checkout's repository is untouched.
    """
    git = _git(source, what)
    stack = native.patch_stack(patches)
    missing = [str(patch) for patch in stack if not patch.is_file()]
    if missing:
        raise SourceMismatch(
            f"audited {what} patch stack {patches} lacks {', '.join(missing)}"
        )
    head = git("rev-parse", "HEAD").decode().strip()
    if head != pin:
        raise SourceMismatch(
            f"{what} checkout {source} is at {head}, audited pin is {pin}"
        )
    if git("diff", "--cached"):
        raise SourceMismatch(f"{what} checkout {source} has staged changes")
    # Entries flagged assume-unchanged (lowercase tag) or skip-worktree (S)
    # are never rehashed from a copy of the index, so they are refused.
    flagged = [
        line[2:]
        for line in git("ls-files", "-v").decode().splitlines()
        if line[:1].islower() or line[:1] == "S"
    ]
    if flagged:
        raise SourceMismatch(
            f"{what} checkout {source} hides changes from git: {', '.join(flagged)}"
        )
    objects = git("rev-parse", "--path-format=absolute", "--git-path", "objects")
    index = git("rev-parse", "--path-format=absolute", "--git-path", "index")
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        (scratch / "objects").mkdir()
        shared = {
            "GIT_OBJECT_DIRECTORY": str(scratch / "objects"),
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": objects.decode().strip(),
        }
        expected_env = {**shared, "GIT_INDEX_FILE": str(scratch / "expected")}
        actual_env = {**shared, "GIT_INDEX_FILE": str(scratch / "actual")}
        git("read-tree", pin, env=expected_env)
        for patch in stack:
            try:
                git("apply", "--cached", str(patch), env=expected_env)
            except SourceMismatch as error:
                raise SourceMismatch(
                    f"audited patch {patch} does not apply at the pin {pin}: {error}"
                ) from error
        expected = git("write-tree", env=expected_env).decode().strip()
        # A copy of the real index keeps its stat cache, so only changed
        # files are rehashed.
        (scratch / "actual").write_bytes(Path(index.decode().strip()).read_bytes())
        git("add", "-A", env=actual_env)
        actual = git("write-tree", env=actual_env).decode().strip()
        if expected != actual:
            paths = git(
                "diff-tree", "-r", "--name-only", expected, actual, env=actual_env
            )
            raise SourceMismatch(
                f"{what} checkout {source} differs from the audited patch stack "
                f"{patches} in: {', '.join(paths.decode().split())}"
            )
    return head


def check(source=DEFAULT_SOURCE, *, pin=native.CUBECL_PIN, patches=native.PATCHES):
    """Verify the CubeCL checkout; return its provenance fields."""
    return {
        "cubecl_commit": _check_stack(source, "CubeCL", pin, patches),
        "patch_stack_sha256": native.patch_stack_sha256(patches),
    }


def check_cubek(
    source=DEFAULT_CUBEK, *, pin=native.CUBEK_PIN, patches=native.CUBEK_PATCHES
):
    """Verify the CubeK checkout against its own stack; return its provenance
    fields."""
    return {
        "cubek_commit": _check_stack(source, "CubeK", pin, patches),
        "cubek_patch_stack_sha256": native.patch_stack_sha256(patches),
    }


def check_all(cubecl=DEFAULT_SOURCE, cubek=None):
    """Both checks; CubeK defaults to the sibling of the CubeCL checkout."""
    cubek = Path(cubecl).parent / "cubek" if cubek is None else cubek
    return {**check(cubecl), **check_cubek(cubek)}


if __name__ == "__main__":
    try:
        check_all(*(Path(a) for a in sys.argv[1:3]))
    except SourceMismatch as error:
        sys.exit(str(error))
