"""Audited-source check against fake checkouts and clones; never the real one."""

import os
import subprocess

import pytest
import torch

from cubecl_torch.compiler import source
from cubecl_torch.compiler.build import compile_descriptor
from cubecl_torch.compiler.plan import PhysicalCandidate
from cubecl_torch.runtime import native
from cubecl_torch.runtime.cache import ArtifactCache
from cubecl_torch.testing.fixtures import audited_clones, sin_plus_one_descriptor


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    ).stdout


def stack(directory, *patches):
    """Write a patch stack: ``(name, bytes)`` pairs in application order."""
    directory.mkdir(exist_ok=True)
    for name, data in patches:
        (directory / name).write_bytes(data)
    (directory / "series").write_text("".join(f"{name}\n" for name, _ in patches))
    return directory


@pytest.fixture
def checkout(tmp_path):
    """A fake checkout: one commit, then a two-patch 'audited' stack applied.

    The second patch edits a line the first adds and creates a new file, so the
    checkout's diff is neither patch and the stack is not a concatenation.
    """
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
    first = git(repo, "diff", "HEAD")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "first")
    (repo / "math.rs").write_text("fn f() { /* audited twice */ }\n")
    (repo / "external.rs").write_text("fn g() {}\n")
    git(repo, "add", ".")
    second = git(repo, "diff", "--cached")
    git(repo, "reset", "-q", "--hard", pin)
    for data in (first, second):
        subprocess.run(["git", "-C", str(repo), "apply"], input=data, check=True)
    patches = stack(tmp_path / "patches", ("fp.patch", first), ("bridge.patch", second))
    return repo, pin, patches


def test_audited_state_passes(checkout):
    repo, pin, patches = checkout
    provenance = source.check(repo, pin=pin, patches=patches)
    assert provenance["cubecl_commit"] == pin
    assert provenance["patch_stack_sha256"] == native.patch_stack_sha256(patches)
    assert git(repo, "status", "--porcelain") == b" M math.rs\n?? external.rs\n"


def test_stack_digest_covers_order_and_contents(checkout, tmp_path):
    _, _, patches = checkout
    names = (patches / "series").read_text().split()
    data = {n: (patches / n).read_bytes() for n in names}
    reordered = stack(tmp_path / "reordered", *((n, data[n]) for n in reversed(names)))
    edited = stack(tmp_path / "edited", *((n, data[n] + b"\n") for n in names))
    digests = {native.patch_stack_sha256(d) for d in (patches, reordered, edited)}
    assert len(digests) == 3


@pytest.mark.parametrize(
    "change",
    [
        "unexpected_edit",
        "missing_patch",
        "stack_without_patch",
        "staged",
        "untracked",
        "other_commit",
        "assume_unchanged",
        "skip_worktree",
    ],
)
def test_mismatch_is_refused(checkout, change):
    repo, pin, patches = checkout
    if change == "unexpected_edit":
        (repo / "math.rs").write_text("fn f() { /* audited twice */ fast_math(); }\n")
    elif change == "missing_patch":
        # The checkout lacks the second patch.
        (repo / "external.rs").unlink()
        (repo / "math.rs").write_text("fn f() { /* audited */ }\n")
    elif change == "stack_without_patch":
        # The series names a patch file that is not there.
        (patches / "bridge.patch").unlink()
    elif change == "staged":
        (repo / "new.rs").write_text("\n")
        git(repo, "add", "new.rs")
    elif change == "untracked":
        (repo / "new.rs").write_text("\n")
    elif change in ("assume_unchanged", "skip_worktree"):
        # A flagged entry is never rehashed, so its edit would go unseen.
        git(repo, "update-index", f"--{change.replace('_', '-')}", "math.rs")
        (repo / "math.rs").write_text("fn f() { /* audited twice */ fast_math(); }\n")
    else:
        git(repo, "commit", "-qam", "drift")
    with pytest.raises(source.SourceMismatch):
        source.check(repo, pin=pin, patches=patches)


def test_caller_git_environment_is_ignored(checkout, tmp_path, monkeypatch):
    # As from inside a git hook of another repository.
    repo, pin, patches = checkout
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere" / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "elsewhere-index"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    assert source.check(repo, pin=pin, patches=patches)["cubecl_commit"] == pin


def test_series_blank_lines_and_no_final_newline(checkout):
    repo, pin, patches = checkout
    (patches / "series").write_text("\nfp.patch\n\n  bridge.patch")
    assert [p.name for p in native.patch_stack(patches)] == ["fp.patch", "bridge.patch"]
    source.check(repo, pin=pin, patches=patches)


@pytest.fixture
def clones(tmp_path):
    return audited_clones(tmp_path)


def build(cubecl, cubek):
    return subprocess.run(
        ["bash", str(native.NATIVE / "build.sh")],
        check=False,
        env={**os.environ, "CUBECL_SOURCE": str(cubecl), "CUBEK_SOURCE": str(cubek)},
        capture_output=True,
        text=True,
    )


def test_build_audit_on_real_stack(clones):
    # The real stack reproduces the audited state; build.sh then refuses an
    # extra edit and a missing patch hunk before building anything.
    cubecl, cubek = clones
    source.check_all(cubecl, cubek)
    target = cubecl / "crates" / "cubecl-ir" / "src" / "dialect" / "math.rs"
    target.write_bytes(target.read_bytes() + b"\n// unaudited\n")
    extra = build(cubecl, cubek)
    assert extra.returncode != 0
    assert "differs from the audited patch stack" in extra.stderr
    assert "crates/cubecl-ir/src/dialect/math.rs" in extra.stderr
    git(cubecl, "checkout", "--", "crates/cubecl-ir/src/dialect/math.rs")
    missing = build(cubecl, cubek)
    assert missing.returncode != 0
    assert "differs from the audited patch stack" in missing.stderr


@pytest.mark.parametrize("change", ["other_commit", "edit", "staged", "untracked"])
def test_unexpected_cubek_state_refuses_build(clones, change):
    cubecl, cubek = clones
    readme = next(p for p in cubek.iterdir() if p.is_file() and p.suffix == ".md")
    if change == "other_commit":
        git(cubek, "checkout", "-q", "HEAD~1")
    elif change == "edit":
        readme.write_text(readme.read_text() + "\nlocal\n")
    elif change == "staged":
        readme.write_text(readme.read_text() + "\nlocal\n")
        git(cubek, "add", readme.name)
    else:
        (cubek / "local.rs").write_text("\n")
    with pytest.raises(source.SourceMismatch, match="CubeK checkout"):
        source.check_cubek(cubek)
    refused = build(cubecl, cubek)
    assert refused.returncode != 0
    assert "CubeK checkout" in refused.stderr


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
