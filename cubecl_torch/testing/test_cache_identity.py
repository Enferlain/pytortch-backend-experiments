"""Policy edits leave runtime artifact keys unchanged (tasks 2.3 and 5 of design)."""

import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

from cubecl_torch.compiler import build
from cubecl_torch.runtime import native
from cubecl_torch.testing.fixtures import REPO

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# The copy is outside any git repository, so the check that the sources are
# still the recorded ones (git) is left out: the key takes the recorded state.
KEY_SCRIPT = """
from cubecl_torch.compiler import build, descriptor, source
from cubecl_torch.runtime import native
source.check = lambda *args, **kwargs: native.source_state()
from cubecl_torch.compiler.plan import PhysicalCandidate
from cubecl_torch.runtime.cache import artifact_key
from cubecl_torch.testing.fixtures import sin_plus_one_descriptor
packet, _ = descriptor.packet(
    sin_plus_one_descriptor(), PhysicalCandidate("scalar_pointwise_baseline").to_dict()
)
print(artifact_key(packet, build.provenance(0)))
"""


@pytest.fixture
def package_copy(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(
        REPO / "cubecl_torch",
        root / "cubecl_torch",
        ignore=shutil.ignore_patterns("target", "__pycache__", "*.so"),
    )
    return root


def key(root, compiler=None, state=None):
    env = {
        **os.environ,
        "CUBECL_TORCH_SOURCE_STATE": str(state or native.source_state_path()),
        "CUBECL_TORCH_COMPILER": str(compiler or build.compiler_path()),
        "CUBECL_TORCH_LAUNCHER": str(native.launcher_path()),
        "CUBECL_TORCH_METADATA_LIB": str(native.metadata_library_path()),
    }
    return subprocess.run(
        [sys.executable, "-c", KEY_SCRIPT],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def touch(paths):
    for path in paths:
        path.write_text(path.read_text() + "\n# policy edit\n")


def test_policy_edit_keeps_artifact_key(package_copy):
    before = key(package_copy)
    policy = [*(package_copy / "cubecl_torch" / "inductor").rglob("*.py")]
    policy += list((package_copy / "cubecl_torch" / "inductor").rglob("*.json"))
    assert policy
    touch(policy)
    assert key(package_copy) == before


def test_producer_edit_changes_artifact_key(package_copy):
    before = key(package_copy)
    touch([package_copy / "cubecl_torch" / "compiler" / "binding.py"])
    assert key(package_copy) != before


def test_compiler_binary_change_changes_artifact_key(package_copy, tmp_path):
    """The binary's digest reaches the key, so code linked into it reaches the key."""
    compiler = tmp_path / "compiler"
    shutil.copy2(build.compiler_path(), compiler)
    before = key(package_copy, compiler)
    with compiler.open("ab") as binary:
        binary.write(b"\0")
    assert key(package_copy, compiler) != before


def test_source_state_change_changes_artifact_key(package_copy, tmp_path):
    """Another recorded source state (an edit to sources/ since built) gets new
    keys."""
    before = key(package_copy)
    state = json.loads(native.source_state_path().read_text())
    state["cubek"]["changes"] = "0" * 64
    record = tmp_path / "source-state.json"
    record.write_text(json.dumps(state))
    assert key(package_copy, state=record) != before
