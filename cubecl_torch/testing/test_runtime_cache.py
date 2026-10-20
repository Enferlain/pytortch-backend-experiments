"""Integrity-checked artifact cache and its location (task 2.3)."""

import pytest

from cubecl_torch.runtime import cache
from cubecl_torch.runtime.artifact import ProvenanceError
from cubecl_torch.testing.fixtures import stage0_sin_plus_one

KEY_INPUTS = {"producer": "stage0-programmatic"}


def builder(calls, artifact=None):
    def build(directory):
        calls.append(directory)
        (directory / "kernel.cu").write_text("// producer file\n")
        return artifact or stage0_sin_plus_one(), ["kernel.cu"]

    return build


def key():
    return cache.artifact_key(KEY_INPUTS, stage0_sin_plus_one().provenance)


def test_root_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("CUBECL_TORCH_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(cache, "_configured", None)
    assert cache.cache_root() == tmp_path / "xdg" / "cubecl_torch"
    monkeypatch.setenv("CUBECL_TORCH_CACHE_DIR", str(tmp_path / "env"))
    assert cache.cache_root() == tmp_path / "env"
    cache.configure(tmp_path / "explicit")
    try:
        assert cache.cache_root() == tmp_path / "explicit"
    finally:
        cache.configure(None)


def test_custom_directory_used_exclusively(tmp_path, monkeypatch):
    other = tmp_path / "other"
    monkeypatch.setenv("CUBECL_TORCH_CACHE_DIR", str(other))
    monkeypatch.setenv("XDG_CACHE_HOME", str(other))
    root = tmp_path / "chosen"
    calls = []
    cache.ArtifactCache(root).get_or_build(key(), builder(calls))
    written = sorted(
        p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()
    )
    assert written and all(p.parts[0] == "chosen" for p in written)
    assert not other.exists()


def test_warm_reuse_builds_nothing(tmp_path):
    calls = []
    store = cache.ArtifactCache(tmp_path)
    first, hit = store.get_or_build(key(), builder(calls))
    assert not hit
    again, hit = cache.ArtifactCache(tmp_path).get_or_build(key(), builder(calls))
    assert hit and len(calls) == 1 and again == first


@pytest.mark.parametrize("name", ["kernel.ptx", "contract.json", "kernel.cu"])
def test_corrupted_entry_is_rebuilt(tmp_path, name):
    calls = []
    store = cache.ArtifactCache(tmp_path)
    store.get_or_build(key(), builder(calls))
    path = store.directory(key()) / name
    path.write_bytes(path.read_bytes() + b" ")
    assert store.load(key()) is None
    _, hit = store.get_or_build(key(), builder(calls))
    assert not hit and len(calls) == 2
    assert store.load(key()) is not None


def test_build_with_unaudited_provenance_is_not_cached(tmp_path):
    bad = stage0_sin_plus_one(provenance={"cubecl_commit": "0" * 40})
    store = cache.ArtifactCache(tmp_path)
    with pytest.raises(ProvenanceError):
        store.get_or_build(key(), builder([], bad))
    assert store.load(key()) is None


def test_key_covers_inputs_and_provenance_only():
    provenance = stage0_sin_plus_one().provenance
    assert cache.artifact_key(KEY_INPUTS, provenance) == key()
    assert cache.artifact_key({"producer": "x"}, provenance) != key()
    assert cache.artifact_key(KEY_INPUTS, {**provenance, "compiler": "y"}) != key()
