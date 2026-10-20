"""Descriptor -> CubeCL IR -> CUDA -> PTX -> ``KernelArtifact``, cached.

The Rust ``cubecl-stage0`` binary replays the schema-5 packet into CubeCL IR and
emits CUDA through ``CppCompiler<Cuda>``; NVRTC produces PTX. The cache key
covers the packet and the compiler provenance only (no selection policy).
"""

import copy
import json
import os
import subprocess
import threading
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch

from cubecl_torch.compiler import (
    binding,
    column_plan,
    descriptor,
    final_plan,
    nvrtc,
    source,
)
from cubecl_torch.compiler.plan import PhysicalCandidate
from cubecl_torch.runtime import native
from cubecl_torch.runtime.artifact import KernelArtifact
from cubecl_torch.runtime.cache import ArtifactCache, artifact_key, digest

PACKAGE = Path(__file__).resolve().parent
PRODUCER_FILES = (
    "descriptor.json",
    "semantic.json",
    "physical-plan.json",
    "manifest.json",
    "kernel.cu",
    "cubecl.ir",
    "nvrtc.json",
)
_LOCK = threading.Lock()
COMPILER_INVOCATIONS = 0


def compiler_path():
    return Path(
        os.environ.get(
            "CUBECL_TORCH_COMPILER",
            native.NATIVE / "target" / "release" / "cubecl-stage0",
        )
    )


@cache
def _producer_source_digest():
    """Code that shapes artifacts or contracts: this package and the contract format."""
    files = sorted(PACKAGE.glob("*.py")) + [
        PACKAGE / "column_templates.json",
        PACKAGE.parent / "runtime" / "artifact.py",
    ]
    return digest(b"".join(p.name.encode() + p.read_bytes() for p in files))


@cache
def identity(device=0, cubecl_source=source.DEFAULT_SOURCE):
    """Compiler provenance: audited source, binaries, toolchain and target device."""
    library = nvrtc.library_path()
    props = torch.cuda.get_device_properties(device)
    headers = nvrtc.include_dirs()[0]
    header_names = (
        "cuda_fp16.h",
        "cuda_fp16.hpp",
        "cuda_bf16.h",
        "cuda_bf16.hpp",
        "vector_types.h",
        "vector_functions.h",
    )
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    return {
        **source.check(cubecl_source),
        "compiler": digest(compiler_path().read_bytes()),
        "metadata_library": digest(native.metadata_library_path().read_bytes()),
        "launcher": digest(native.launcher_path().read_bytes()),
        "producer_source": _producer_source_digest(),
        "nvrtc": {"path": library, "sha256": digest(Path(library).read_bytes())},
        "nvrtc_builtins": {
            str(p.resolve()): digest(p.read_bytes())
            for p in Path(library).parent.glob("libnvrtc-builtins.so*")
            if p.is_file()
        },
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "torch_commit": torch.version.git_version,
        "cuda_header_sha256": {
            n: digest((headers / n).read_bytes()) for n in header_names
        },
        "driver_snapshot": driver,
        "device": str(props.uuid),
        "capability": [props.major, props.minor],
        "math": ["--fmad=false", "precision_cast_barriers", "conservative_fp_patch"],
    }


def _check_manifest(manifest, semantic, plan):
    if (
        manifest.get("schema") != 1
        or manifest.get("address_bits") != 64
        or manifest.get("descriptor_schema") != 5
        or manifest.get("grid_constants")
        or manifest.get("physical_plan") != plan
        or len(manifest.get("arguments", ())) != len(semantic["buffers"]) + 1
    ):
        raise RuntimeError("compiler output does not match the pointer ABI")
    if plan["strategy"] == column_plan.STRATEGY and (
        manifest.get("mechanism") != "cubecl-column-plain-fast64-v1"
        or manifest.get("layout_rank") != 2
        or manifest.get("metadata", {}).get("scalar_count") != 7
        or manifest.get("metadata", {}).get("bytes") != 80
    ):
        raise RuntimeError("column metadata ABI mismatch")


@dataclass(frozen=True)
class Compiled:
    artifact: KernelArtifact
    directory: Path
    cache_hit: bool
    plan: PhysicalCandidate


def compile_descriptor(
    semantic, plan, device=0, *, cache=None, cubecl_source=source.DEFAULT_SOURCE
):
    """Compile ``semantic`` under physical ``plan`` for ``device``.

    ``cubecl_source`` is the CubeCL checkout the compiler binary was built from;
    compilation is refused unless it is the audited pin and patch state.

    Raises MalformedDescriptor (nothing cached) for schema violations, and the
    Stage 5 ValueErrors for plans whose mechanism contract the descriptor misses.
    """
    semantic = copy.deepcopy(semantic)
    descriptor.validate(semantic)
    candidate = PhysicalCandidate.from_dict(plan)
    plan = candidate.to_dict()
    if candidate.strategy == column_plan.STRATEGY and (
        why := column_plan.semantic_reason(semantic)
    ):
        raise ValueError("unsupported column physical contract: " + why)
    if candidate.strategy == final_plan.STRATEGY and (
        why := final_plan.semantic_reason(semantic)
    ):
        raise ValueError("unsupported final physical contract: " + why)
    packet, entry = descriptor.packet(semantic, plan)
    if isinstance(device, torch.device):
        device = device.index or 0
    provenance = identity(device, Path(cubecl_source))
    key = artifact_key(packet, provenance)
    store = cache or ArtifactCache()

    def build(directory):
        global COMPILER_INVOCATIONS
        for name, value in (
            ("descriptor", packet),
            ("semantic", semantic),
            ("physical-plan", plan),
        ):
            (directory / f"{name}.json").write_text(
                json.dumps(value, indent=2, sort_keys=True)
            )
        proc = subprocess.run(
            [
                str(compiler_path()),
                str(directory),
                "cpp",
                str(directory / "descriptor.json"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        (directory / "compiler.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            raise RuntimeError("CubeCL physical compiler failure: " + proc.stderr)
        with _LOCK:
            COMPILER_INVOCATIONS += 1
        manifest = json.loads((directory / "manifest.json").read_text())
        _check_manifest(manifest, semantic, plan)
        fields = binding.bind(semantic, plan, manifest)
        ptx = nvrtc.compile_ptx(
            (directory / "kernel.cu").read_text(), directory, provenance["capability"]
        )
        artifact = KernelArtifact(
            ptx=ptx, entry=entry, provenance=copy.deepcopy(provenance), **fields
        )
        return artifact, PRODUCER_FILES

    artifact, hit = store.get_or_build(key, build)
    return Compiled(artifact, store.directory(key), hit, candidate)
