"""Refusal of artifacts not produced from the audited CubeCL state (task 2.4)."""

import json

import pytest
import torch

from cubecl_torch.runtime import cache
from cubecl_torch.runtime.artifact import ProvenanceError, audited_provenance
from cubecl_torch.runtime.launch import LoadedKernel
from cubecl_torch.testing.fixtures import stage0_sin_plus_one

ALTERED = [
    {"cubecl_commit": "0" * 40},
    {"fp_patch_sha256": "0" * 64},
    {"cubecl_commit": None},
]


@pytest.mark.parametrize("change", ALTERED)
def test_load_refuses_altered_provenance(change):
    provenance = {**audited_provenance(), **change}
    with pytest.raises(ProvenanceError, match="provenance mismatch"):
        LoadedKernel(stage0_sin_plus_one(provenance=provenance))


def test_cache_refuses_altered_provenance_on_disk(tmp_path):
    store = cache.ArtifactCache(tmp_path)
    key = cache.artifact_key({}, audited_provenance())
    store.get_or_build(key, lambda d: (stage0_sin_plus_one(), []))
    directory = store.directory(key)
    contract = json.loads((directory / "contract.json").read_text())
    contract["provenance"]["fp_patch_sha256"] = "0" * 64
    (directory / "contract.json").write_text(json.dumps(contract))
    # Re-seal the integrity record so only provenance is wrong.
    integrity = json.loads((directory / "integrity.json").read_text())
    integrity["files"]["contract.json"] = cache.digest(
        (directory / "contract.json").read_bytes()
    )
    (directory / "integrity.json").write_text(json.dumps(integrity))
    with pytest.raises(ProvenanceError):
        store.load(key)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_audited_provenance_loads():
    LoadedKernel(stage0_sin_plus_one())
