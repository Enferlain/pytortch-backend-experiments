"""The manifest diff names changed cases, regions and files (5.5)."""

import copy

import pytest

from cubecl_torch.testing.manifest_check import diff, record

DIGESTS = {"cubecl.ir": "a" * 64, "kernel.cu": "b" * 64, "kernel.ptx": "c" * 64}
REGION = {
    "role": "pointwise",
    "descriptor": "d" * 64,
    "policy": "first_use",
    "plan": "p1",
    "artifact": {"plan": "p1", **DIGESTS},
    "compiled": [{"plan": "p0", **DIGESTS}, {"plan": "p1", **DIGESTS}],
    "candidates": ["p0", "p1"],
    "rejected": [],
    "selections": [],
}
MANIFEST = {"add|first_use": [copy.deepcopy(REGION), copy.deepcopy(REGION)]}


def test_identical_records_pass():
    assert diff(MANIFEST, copy.deepcopy(MANIFEST)) == {}


def test_changed_candidate_file_is_named():
    current = copy.deepcopy(MANIFEST)
    current["add|first_use"][1]["compiled"][0]["kernel.ptx"] = "e" * 64
    assert diff(MANIFEST, current) == {
        "add|first_use": ["region 1 candidate p0: kernel.ptx"]
    }


def test_changed_chosen_plan_and_region_count():
    current = copy.deepcopy(MANIFEST)
    current["add|first_use"][0]["plan"] = "p0"
    current["add|first_use"].append(copy.deepcopy(REGION))
    report = diff(MANIFEST, current)["add|first_use"]
    assert "region count 2 -> 3" in report and "region 0 plan changed" in report


def test_record_refuses_errored_cases(tmp_path):
    failed = {"case": "add", "policy": "baseline", "impl": "package"}
    failed.update(regions=[], error="RuntimeError: broken")
    with pytest.raises(ValueError, match=r"add|baseline"):
        record([failed], tmp_path / "manifest.json")
    assert not (tmp_path / "manifest.json").exists()
