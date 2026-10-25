"""Deterministic compilation and malformed-descriptor rejection (task 3.4)."""

import copy

import pytest
import torch

from cubecl_torch.compiler.build import compile_descriptor
from cubecl_torch.compiler.errors import MalformedDescriptor
from cubecl_torch.compiler.plan import PhysicalCandidate
from cubecl_torch.runtime.cache import ArtifactCache
from cubecl_torch.testing.fixtures import descriptor, sin_plus_one_descriptor

BASELINE = PhysicalCandidate("scalar_pointwise_baseline").to_dict()
REDUCTION = PhysicalCandidate("shared_tree_reduction_baseline").to_dict()
# Three shared arrays (Welford's mean, m2, weight): their slice order once came
# from a per-process hash seed in CubeCL (bead g57).
WELFORD = PhysicalCandidate("plane_shared", plane_width=32).to_dict()
FILES = ("cubecl.ir", "kernel.cu", "kernel.ptx")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    ("semantic", "plan"),
    [(sin_plus_one_descriptor(), BASELINE), (descriptor("reduction"), REDUCTION)],
    ids=["pointwise", "reduction"],
)
def test_recompile_from_empty_cache_is_identical(tmp_path, semantic, plan):
    first = compile_descriptor(semantic, plan, 0, cache=ArtifactCache(tmp_path / "a"))
    second = compile_descriptor(semantic, plan, 0, cache=ArtifactCache(tmp_path / "b"))
    assert not first.cache_hit and not second.cache_hit
    for name in FILES:
        assert (first.directory / name).read_bytes() == (
            second.directory / name
        ).read_bytes()
    assert first.artifact == second.artifact


def broken(edit):
    semantic = copy.deepcopy(sin_plus_one_descriptor())
    edit(semantic)
    return semantic


MALFORMED = {
    "missing numel": (lambda d: d.pop("numel"), "numel"),
    "missing buffer stride": (lambda d: d["buffers"][0].pop("stride"), "stride"),
    "rank mismatch": (
        lambda d: d["buffers"][0]["size"].append({"op": "constant", "value": 2}),
        "rank",
    ),
    "scalar slot out of range": (
        lambda d: d["buffers"][1].update(size=[{"op": "scalar", "slot": 3}]),
        "slot",
    ),
    "two outputs": (lambda d: d["buffers"][0].update(access="write"), "output buffer"),
    "store to undeclared buffer": (lambda d: d["ops"][-1].update(buffer=5), "buffer 5"),
    "unknown dtype": (lambda d: d["buffers"][0].update(dtype="float8"), "dtype"),
    "scalar ids": (lambda d: d["scalars"][0].update(id=4), "scalars"),
    "no buffers": (lambda d: d.update(buffers=[]), "buffers"),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_multi_array_kernel_is_identical_across_compiler_processes(tmp_path):
    # Each compile runs the native compiler in a fresh process; with three
    # arrays, an order left to the hash seed survives three compiles 1 in 36.
    semantic = descriptor("welford")
    results = [
        compile_descriptor(semantic, WELFORD, 0, cache=ArtifactCache(tmp_path / str(i)))
        for i in range(3)
    ]
    for name in FILES:
        assert len({(r.directory / name).read_bytes() for r in results}) == 1, name


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_malformed_descriptor_rejected(tmp_path, name):
    edit, words = MALFORMED[name]
    root = tmp_path / "cache"
    with pytest.raises(MalformedDescriptor, match=words):
        compile_descriptor(broken(edit), BASELINE, 0, cache=ArtifactCache(root))
    assert not root.exists() or not any(p.is_file() for p in root.rglob("*"))


def test_malformed_reduction_domain_rejected(tmp_path):
    semantic = copy.deepcopy(descriptor("reduction"))
    for op in semantic["ops"]:
        if op["op"] == "store":
            op["iteration_scope"] = "both"
    with pytest.raises(MalformedDescriptor, match="logical domain"):
        compile_descriptor(semantic, REDUCTION, 0, cache=ArtifactCache(tmp_path))
    assert not any(p.is_file() for p in tmp_path.rglob("*"))
