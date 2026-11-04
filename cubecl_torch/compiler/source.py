"""The source state of the CubeCL ecosystem we build: ``sources/cubecl`` and
``sources/cubek``, developed in this repository.

A source's state is the upstream revision it last merged (``sources/LOCK.toml``),
its committed tree, and a digest of its working changes (staged or not, and
untracked files), empty when clean: each exact source state is one value.
``native/build.sh`` records the state its binaries were built from; provenance is
that record (``runtime.native.source_state``), and a build whose sources have
changed since is stale (``check``).

``python -m cubecl_torch.compiler.source record PATH`` writes the record,
``upstream NAME`` prints a source's upstream revision and ``delta NAME`` our
changes from it; none needs torch or a build.
"""

import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from cubecl_torch.runtime import native

REPO = Path(__file__).resolve().parents[2]
NAMES = ("cubecl", "cubek")


class SourceMismatch(RuntimeError):
    """The sources are not in the state the native build recorded."""


def _git(repo):
    """A git runner for ``repo`` that ignores the caller's ``GIT_*`` state,
    takes no optional locks, and reports any failure as a mismatch."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("GIT_") or k == "GIT_EXEC_PATH"
    }
    env["GIT_OPTIONAL_LOCKS"] = "0"

    def git(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                check=True,
                env=env,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            detail = (getattr(error, "stderr", None) or b"").decode(errors="replace")
            raise SourceMismatch(
                f"cannot inspect {repo}: git {' '.join(args)}: {detail.strip() or error}"
            ) from error

    return git


def lock(repo=REPO):
    return tomllib.loads((Path(repo) / "sources" / "LOCK.toml").read_text())


def state(repo=REPO):
    """``{name: {"upstream", "tree", "changes"}}`` for each source in ``repo``."""
    git = _git(repo)
    locked = lock(repo)
    result = {}
    for name in NAMES:
        prefix = f"sources/{name}"
        # Entries flagged assume-unchanged (lowercase tag) or skip-worktree (S)
        # hide their edits from git, so they are refused.
        flagged = [
            line[2:]
            for line in git("ls-files", "-v", "--", prefix).decode().splitlines()
            if line[:1].islower() or line[:1] == "S"
        ]
        if flagged:
            raise SourceMismatch(
                f"{prefix} hides changes from git: {', '.join(flagged)}"
            )
        diff = git("diff", "--binary", "HEAD", "--", prefix)
        untracked = sorted(
            p
            for p in git(
                "ls-files", "--others", "--exclude-standard", "-z", "--", prefix
            ).split(b"\0")
            if p
        )
        changes = ""
        if diff or untracked:
            h = hashlib.sha256(diff)
            for path in untracked:
                data = (Path(repo) / path.decode()).read_bytes()
                h.update(path + b"\0" + str(len(data)).encode() + b"\0" + data)
            changes = h.hexdigest()
        result[name] = {
            "upstream": locked[name]["revision"],
            "tree": git("rev-parse", f"HEAD:{prefix}").decode().strip(),
            "changes": changes,
        }
    return result


def record(path, repo=REPO):
    """Write the sources' current state to ``path``, as ``build.sh`` does."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(state(repo), indent=1, sort_keys=True) + "\n")


def check(repo=REPO, path=None):
    """The state the native build recorded, verified to be the current one."""
    try:
        recorded = native.source_state(path)
    except native.NativeUnavailable as error:
        raise SourceMismatch(str(error)) from error
    current = state(repo)
    if recorded != current:
        changed = sorted(n for n in NAMES if recorded.get(n) != current[n])
        raise SourceMismatch(
            f"sources/{', sources/'.join(changed)} changed since the native build "
            f"recorded them; run cubecl_torch/native/build.sh"
        )
    return recorded


def delta(name, repo=REPO, *args):
    """Our changes to ``name`` from the upstream revision it last merged."""
    revision = lock(repo)[name]["revision"]
    return _git(repo)("diff", *args, f"{revision}^{{tree}}", f"HEAD:sources/{name}")


if __name__ == "__main__":
    command, *rest = sys.argv[1:] or ["check"]
    try:
        if command == "record":
            record(rest[0])
        elif command == "upstream":
            print(lock()[rest[0]]["revision"])
        elif command == "delta":
            sys.stdout.buffer.write(delta(rest[0], REPO, *rest[1:]))
        else:
            check()
    except SourceMismatch as error:
        sys.exit(str(error))
