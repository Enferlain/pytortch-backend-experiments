"""The kernel artifact contract: everything needed to cache, validate and launch.

Deliberately minimal and shaped by one producer so far (the descriptor
compiler). Expect it to change when the first CubeK-produced kernel arrives.

Launch ABI: one ``u64`` device pointer per tensor parameter, in order, then one
pointer to CubeCL ``InfoBuilder`` metadata built from the tensors' element spans
and ``info_scalars``.
"""

import json
from dataclasses import dataclass, field

from cubecl_torch.runtime import native

DTYPES = ("float32", "float16", "bfloat16", "bool", "int64")


class ProvenanceError(ValueError):
    """The artifact was not produced from the audited CubeCL source state."""


@dataclass(frozen=True)
class TensorParam:
    """A tensor pointer parameter and its guard: dtype plus exact layout rules."""

    dtype: str
    size: tuple
    stride: tuple
    offset: dict
    access: str = "read"

    def __post_init__(self):
        if self.dtype not in DTYPES or self.access not in ("read", "write"):
            raise ValueError(f"unsupported tensor parameter {self.dtype}/{self.access}")
        if len(self.size) != len(self.stride):
            raise ValueError("tensor parameter size and stride ranks differ")


@dataclass(frozen=True)
class ScalarParam:
    """A signed-i64 launch scalar; ``positive`` scalars must be > 0."""

    positive: bool = False


@dataclass(frozen=True)
class Guard:
    """A launch-scalar guard, reported as an extent mismatch.

    ``range``: ``minimum <= rules[0] <= maximum`` (either bound may be None).
    ``member``: the tuple of rule values is one of ``allowed``.
    ``output_numel``: the last tensor's numel equals ``rules[0]``.
    """

    kind: str
    rules: tuple
    message: str
    minimum: int | None = None
    maximum: int | None = None
    allowed: tuple = ()

    def __post_init__(self):
        if self.kind not in ("range", "member", "output_numel"):
            raise ValueError(f"unknown guard kind {self.kind!r}")


@dataclass(frozen=True)
class LaunchShape:
    grid: dict
    block: tuple
    shared_bytes: int = 0


@dataclass(frozen=True)
class KernelArtifact:
    ptx: bytes
    entry: str
    tensors: tuple
    scalars: tuple
    # Logical extent; zero means "no launch" and skips the alias check.
    extent: dict
    info_scalars: tuple
    metadata_words: int
    launch: LaunchShape
    guards: tuple = ()
    # Device requirements checked against driver queries at load time.
    threads: int = 256
    plane_width: int = 0
    # Pure kernels (no RNG or mutation) may be launched repeatedly for timing.
    repeatable: bool = True
    provenance: dict = field(default_factory=dict)

    def contract(self):
        """JSON-serialisable contract without the PTX bytes."""
        return {
            "entry": self.entry,
            "tensors": [
                {
                    "dtype": t.dtype,
                    "size": list(t.size),
                    "stride": list(t.stride),
                    "offset": t.offset,
                    "access": t.access,
                }
                for t in self.tensors
            ],
            "scalars": [{"positive": s.positive} for s in self.scalars],
            "extent": self.extent,
            "info_scalars": list(self.info_scalars),
            "metadata_words": self.metadata_words,
            "launch": {
                "grid": self.launch.grid,
                "block": list(self.launch.block),
                "shared_bytes": self.launch.shared_bytes,
            },
            "guards": [
                {
                    "kind": g.kind,
                    "rules": list(g.rules),
                    "message": g.message,
                    "minimum": g.minimum,
                    "maximum": g.maximum,
                    "allowed": [list(a) for a in g.allowed],
                }
                for g in self.guards
            ],
            "threads": self.threads,
            "plane_width": self.plane_width,
            "repeatable": self.repeatable,
            "provenance": self.provenance,
        }

    @classmethod
    def from_contract(cls, contract, ptx):
        c = contract
        return cls(
            ptx=ptx,
            entry=c["entry"],
            tensors=tuple(
                TensorParam(
                    t["dtype"],
                    tuple(t["size"]),
                    tuple(t["stride"]),
                    t["offset"],
                    t["access"],
                )
                for t in c["tensors"]
            ),
            scalars=tuple(ScalarParam(s["positive"]) for s in c["scalars"]),
            extent=c["extent"],
            info_scalars=tuple(c["info_scalars"]),
            metadata_words=c["metadata_words"],
            launch=LaunchShape(
                c["launch"]["grid"],
                tuple(c["launch"]["block"]),
                c["launch"]["shared_bytes"],
            ),
            guards=tuple(
                Guard(
                    g["kind"],
                    tuple(g["rules"]),
                    g["message"],
                    g["minimum"],
                    g["maximum"],
                    tuple(tuple(a) for a in g["allowed"]),
                )
                for g in c["guards"]
            ),
            threads=c["threads"],
            plane_width=c["plane_width"],
            repeatable=c["repeatable"],
            provenance=c["provenance"],
        )

    def contract_json(self):
        return json.dumps(self.contract(), indent=2, sort_keys=True)


def audited_provenance():
    """The provenance fields every loadable artifact must carry unchanged."""
    return {
        "cubecl_commit": native.CUBECL_PIN,
        "fp_patch_sha256": native.audited_patch_sha256(),
    }


def check_provenance(provenance):
    for name, expected in audited_provenance().items():
        actual = provenance.get(name)
        if actual != expected:
            raise ProvenanceError(
                f"artifact provenance mismatch: {name} is {actual!r}, audited state is {expected!r}"
            )
