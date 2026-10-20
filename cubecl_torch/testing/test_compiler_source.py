"""Audited-source check against fake checkouts; never the real one (task 3.3)."""

import subprocess

import pytest
import torch

from cubecl_torch.compiler import source
from cubecl_torch.compiler.build import compile_descriptor
from cubecl_torch.compiler.plan import PhysicalCandidate
from cubecl_torch.runtime.cache import ArtifactCache
from cubecl_torch.testing.fixtures import sin_plus_one_descriptor


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    ).stdout


@pytest.fixture
def checkout(tmp_path):
    """A fake checkout: one commit, then the 'audited' patch applied."""
    repo = tmp_path / "cubecl"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "test")
    (repo / "math.rs").write_text("fn f() {}\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    pin = git(repo, "rev-parse", "HEAD").decode().strip()
    (repo / "math.rs").write_text("fn f() { /* audited */ }\n")
    patch = tmp_path / "audited.patch"
    patch.write_bytes(git(repo, "diff", "HEAD"))
    return repo, pin, patch


def test_audited_state_passes(checkout):
    repo, pin, patch = checkout
    provenance = source.check(repo, pin=pin, patch=patch)
    assert provenance["cubecl_commit"] == pin


@pytest.mark.parametrize(
    "change",
    ["unexpected_edit", "missing_patch", "staged", "untracked", "other_commit"],
)
def test_mismatch_is_refused(checkout, change):
    repo, pin, patch = checkout
    if change == "unexpected_edit":
        (repo / "math.rs").write_text("fn f() { /* audited */ fast_math(); }\n")
    elif change == "missing_patch":
        git(repo, "checkout", "--", "math.rs")
    elif change == "staged":
        (repo / "new.rs").write_text("\n")
        git(repo, "add", "new.rs")
    elif change == "untracked":
        (repo / "new.rs").write_text("\n")
    else:
        git(repo, "commit", "-qam", "drift")
    with pytest.raises(source.SourceMismatch):
        source.check(repo, pin=pin, patch=patch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compile_refused_and_nothing_written(checkout, tmp_path):
    repo, _, _ = checkout
    root = tmp_path / "cache"
    with pytest.raises(source.SourceMismatch):
        compile_descriptor(
            sin_plus_one_descriptor(),
            PhysicalCandidate("scalar_pointwise_baseline").to_dict(),
            0,
            cache=ArtifactCache(root),
            cubecl_source=repo,
        )
    assert not root.exists() or not any(p.is_file() for p in root.rglob("*"))
