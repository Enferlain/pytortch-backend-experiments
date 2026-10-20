"""Integrity-checked on-disk artifact cache with a configurable root.

Root resolution: ``configure(path)``, then ``CUBECL_TORCH_CACHE_DIR``, then a
per-user default (``%LOCALAPPDATA%`` on Windows, else ``$XDG_CACHE_HOME`` or
``~/.cache``). Keys cover producer inputs and compiler provenance only, never
front-end selection policy.

Each entry directory holds ``contract.json``, ``kernel.ptx``, any producer files
and ``integrity.json`` (SHA-256 of every file). An entry whose files do not match
is never loaded; ``get_or_build`` rebuilds it in place.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from cubecl_torch.runtime.artifact import KernelArtifact, check_provenance

_configured = None
CONTRACT = "contract.json"
PTX = "kernel.ptx"
INTEGRITY = "integrity.json"


def configure(path):
    """Set the cache root explicitly (``None`` restores env/default resolution)."""
    global _configured
    _configured = None if path is None else Path(path)


def default_root():
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    else:
        base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "cubecl_torch"


def cache_root():
    if _configured is not None:
        return _configured
    if env := os.environ.get("CUBECL_TORCH_CACHE_DIR"):
        return Path(env)
    return default_root()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def artifact_key(producer_inputs, provenance):
    """Key from producer inputs (JSON-serialisable) and compiler provenance."""
    return digest(
        canonical({"inputs": producer_inputs, "provenance": provenance}).encode()
    )


@contextmanager
def file_lock(path):
    """Exclusive lock on ``path`` across processes and threads (one open per caller)."""
    with path.open("a+b") as handle:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


class ArtifactCache:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else cache_root()
        self.builds = 0

    def directory(self, key):
        return self.root / "artifacts" / key

    def _intact(self, directory):
        integrity = directory / INTEGRITY
        if not integrity.is_file():
            return False
        try:
            saved = json.loads(integrity.read_text())
        except ValueError:
            return False
        files = saved.get("files", {})
        return (
            CONTRACT in files
            and PTX in files
            and all(
                (directory / name).is_file()
                and digest((directory / name).read_bytes()) == value
                for name, value in files.items()
            )
        )

    def load(self, key):
        """The cached artifact, or None if absent or failing its integrity check."""
        directory = self.directory(key)
        if not self._intact(directory):
            return None
        artifact = KernelArtifact.from_contract(
            json.loads((directory / CONTRACT).read_text()),
            (directory / PTX).read_bytes(),
        )
        check_provenance(artifact.provenance)
        return artifact

    def get_or_build(self, key, build):
        """Load ``key`` or call ``build(directory) -> (artifact, producer_files)``.

        ``build`` writes into a private staging directory; ``producer_files``
        names files it wrote there, which the integrity record then covers. The
        entry appears only once complete; a failed build leaves nothing behind.
        Returns ``(artifact, cache_hit)``.
        """
        locks = self.root / "locks"
        locks.mkdir(parents=True, exist_ok=True)
        with file_lock(locks / f"{key}.lock"):
            artifact = self.load(key)
            if artifact is not None:
                return artifact, True
            staging = Path(
                tempfile.mkdtemp(prefix=f"{key[:16]}.", dir=self._staging_root())
            )
            try:
                artifact, producer_files = build(staging)
                check_provenance(artifact.provenance)
                (staging / CONTRACT).write_text(artifact.contract_json())
                (staging / PTX).write_bytes(artifact.ptx)
                names = sorted({CONTRACT, PTX, *producer_files})
                payload = {
                    "key": key,
                    "files": {n: digest((staging / n).read_bytes()) for n in names},
                }
                (staging / INTEGRITY).write_text(json.dumps(payload, indent=2))
                directory = self.directory(key)
                if directory.exists():
                    shutil.rmtree(directory)
                directory.parent.mkdir(parents=True, exist_ok=True)
                staging.rename(directory)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            self.builds += 1
            return artifact, False

    def _staging_root(self):
        path = self.root / "staging"
        path.mkdir(parents=True, exist_ok=True)
        return path
