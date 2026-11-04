"""The sources' state against fake repositories; never the real one."""

import json
import subprocess

import pytest
import torch

from cubecl_torch.compiler import build, source
from cubecl_torch.compiler.plan import PhysicalCandidate
from cubecl_torch.runtime import native
from cubecl_torch.runtime.cache import ArtifactCache
from cubecl_torch.testing.fixtures import sin_plus_one_descriptor

LOCK = """[cubecl]
upstream = "https://example.invalid/cubecl"
revision = "1111111111111111111111111111111111111111"

[cubek]
upstream = "https://example.invalid/cubek"
revision = "2222222222222222222222222222222222222222"
"""


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    """A repository with both sources committed, an ignored build directory
    and an unrelated file outside them."""
    root = tmp_path / "repo"
    for name in source.NAMES:
        (root / "sources" / name / "target").mkdir(parents=True)
        (root / "sources" / name / "math.rs").write_text("fn f() {}\n")
        (root / "sources" / name / ".gitignore").write_text("/target\n")
    (root / "sources" / "LOCK.toml").write_text(LOCK)
    (root / "other.py").write_text("x = 1\n")
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "test")
    git(root, "add", ".")
    git(root, "commit", "-qm", "base")
    return root


def cubecl(repo):
    return source.state(repo)["cubecl"]


def test_clean_state(repo):
    state = source.state(repo)
    for name in source.NAMES:
        assert (
            state[name]["tree"]
            == git(repo, "rev-parse", f"HEAD:sources/{name}").decode().strip()
        )
        assert state[name]["changes"] == ""
    assert state["cubecl"]["upstream"] == "1" * 40
    assert state["cubek"]["upstream"] == "2" * 40


def test_each_working_state_is_its_own_value(repo):
    """Two different uncommitted edits never share a value (a dirty flag alone
    would), and committing an edit moves it from changes to the tree."""
    clean = cubecl(repo)
    math = repo / "sources" / "cubecl" / "math.rs"
    math.write_text("fn f() { a(); }\n")
    first = cubecl(repo)
    math.write_text("fn f() { b(); }\n")
    second = cubecl(repo)
    assert len({clean["changes"], first["changes"], second["changes"]}) == 3
    assert clean["tree"] == first["tree"] == second["tree"]
    git(repo, "commit", "-qam", "b")
    committed = cubecl(repo)
    assert committed["changes"] == "" and committed["tree"] != clean["tree"]


@pytest.mark.parametrize("change", ["staged", "untracked", "deleted"])
def test_working_changes_are_seen(repo, change):
    clean = cubecl(repo)
    path = repo / "sources" / "cubecl" / "new.rs"
    if change == "deleted":
        (repo / "sources" / "cubecl" / "math.rs").unlink()
    else:
        path.write_text("fn g() {}\n")
        if change == "staged":
            git(repo, "add", str(path))
    assert cubecl(repo)["changes"] not in ("", clean["changes"])


def test_ignored_and_unrelated_files_are_not_sources(repo):
    clean = source.state(repo)
    (repo / "sources" / "cubecl" / "target" / "build.o").write_bytes(b"\0")
    (repo / "other.py").write_text("x = 2\n")
    (repo / "notes.txt").write_text("unrelated\n")
    assert source.state(repo) == clean


def test_untracked_contents_are_part_of_the_value(repo):
    path = repo / "sources" / "cubek" / "new.rs"
    path.write_text("fn a() {}\n")
    first = source.state(repo)["cubek"]["changes"]
    path.write_text("fn b() {}\n")
    assert source.state(repo)["cubek"]["changes"] != first


@pytest.mark.parametrize("flag", ["assume-unchanged", "skip-worktree"])
def test_hidden_changes_are_refused(repo, flag):
    git(repo, "update-index", f"--{flag}", "sources/cubecl/math.rs")
    (repo / "sources" / "cubecl" / "math.rs").write_text("fn f() { hidden(); }\n")
    with pytest.raises(source.SourceMismatch, match="hides changes"):
        source.state(repo)


def test_caller_git_environment_is_ignored(repo, tmp_path, monkeypatch):
    # As from inside a git hook of another repository.
    clean = source.state(repo)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere" / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "elsewhere-index"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    assert source.state(repo) == clean


def test_recorded_state_must_be_current(repo, tmp_path):
    record = tmp_path / "state" / "source-state.json"
    source.record(record, repo)
    assert source.check(repo, record) == source.state(repo)
    (repo / "sources" / "cubek" / "math.rs").write_text("fn f() { edited(); }\n")
    with pytest.raises(source.SourceMismatch, match="sources/cubek changed since"):
        source.check(repo, record)
    with pytest.raises(source.SourceMismatch, match=r"build\.sh"):
        source.check(repo, tmp_path / "never-built.json")


def test_delta_is_against_the_locked_upstream_revision(repo):
    """``delta`` compares a source with LOCK.toml's revision. An upstream
    commit's root tree is the source's folder, as after ``git subtree add``:
    here one made from the base commit's sources/cubecl, so the delta is
    exactly a later edit."""
    tree = git(repo, "rev-parse", "HEAD:sources/cubecl").decode().strip()
    upstream = git(repo, "commit-tree", tree, "-m", "upstream").decode().strip()
    (repo / "sources" / "LOCK.toml").write_text(LOCK.replace("1" * 40, upstream))
    (repo / "sources" / "cubecl" / "math.rs").write_text("fn f() { ours(); }\n")
    (repo / "sources" / "cubecl" / "new.rs").write_text("fn g() {}\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "ours")
    diff = source.delta("cubecl", repo, "--name-only").decode().split()
    assert diff == ["math.rs", "new.rs"]


def test_real_build_matches_its_sources():
    """The native build in this repository records the sources it was built
    from, and they have not changed since."""
    assert source.check() == native.source_state()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_stale_build_refuses_compilation(tmp_path, monkeypatch):
    stale = json.loads(json.dumps(native.source_state()))
    stale["cubecl"]["changes"] = "0" * 64
    record = tmp_path / "source-state.json"
    record.write_text(json.dumps(stale))
    monkeypatch.setenv("CUBECL_TORCH_SOURCE_STATE", str(record))
    build.identity.cache_clear()
    root = tmp_path / "cache"
    try:
        with pytest.raises(source.SourceMismatch, match="changed since"):
            build.compile_descriptor(
                sin_plus_one_descriptor(),
                PhysicalCandidate("scalar_pointwise_baseline").to_dict(),
                0,
                cache=ArtifactCache(root),
            )
    finally:
        build.identity.cache_clear()
    assert not root.exists() or not any(p.is_file() for p in root.rglob("*"))
