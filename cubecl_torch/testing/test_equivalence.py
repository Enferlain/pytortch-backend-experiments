"""The equivalence harness reports per case and detects divergence (5.1, 5.2)."""

import copy
import json

import pytest
import torch

from cubecl_torch.testing import equivalence

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CASE = "rms_backward-bfloat16-64x1024"


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    work = tmp_path_factory.mktemp("equivalence")
    return equivalence.run([CASE], ["baseline", "first_use"], jobs=4, work=work)


def by(results, impl, policy):
    return next(r for r in results if r["impl"] == impl and r["policy"] == policy)


def test_subset_identical(results):
    report = equivalence.compare(results)
    assert [(r["case"], r["policy"], r["identical"]) for r in report] == [
        (CASE, "baseline", True),
        (CASE, "first_use", True),
    ]
    assert all(r["regions"] == 3 for r in report)


def test_detects_altered_artifact(results):
    altered = copy.deepcopy(results)
    region = by(altered, "package", "baseline")["regions"][1]
    region["artifact"]["kernel.ptx"] = "0" * 64
    problems = equivalence.compare_case(
        by(altered, "stage5", "baseline"),
        by(altered, "package", "baseline"),
        outputs=True,
    )
    assert problems == ["region 1 artifact differs"]


def test_detects_duplicated_region(results):
    altered = copy.deepcopy(results)
    regions = by(altered, "package", "baseline")["regions"]
    # Same artifacts, one more region mapped onto them: a set comparison would pass.
    regions.insert(1, copy.deepcopy(regions[0]))
    problems = equivalence.compare_case(
        by(altered, "stage5", "baseline"),
        by(altered, "package", "baseline"),
        outputs=True,
    )
    assert "region count 3 vs 4" in problems


def test_detects_perturbed_timing(results, tmp_path):
    selection = json.loads(
        by(results, "stage5", "first_use")["regions"][0]["selections"][0]
    )
    winner = selection["winner"]
    perturbed = equivalence.run(
        [CASE], ["first_use"], jobs=1, work=tmp_path, perturb=winner, impls=("package",)
    )
    problems = equivalence.compare_case(
        by(results, "stage5", "first_use"), perturbed[0], outputs=False
    )
    assert any("selections differs" in p for p in problems)
    assert any("plan differs" in p for p in problems)
