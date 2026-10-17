"""Compile-time physical choices; logical SSA/index descriptions stay scalar."""

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class PhysicalCandidate:
    strategy: str
    workgroup: tuple[int, int, int] = (256, 1, 1)
    vector_width: int = 1
    elements_per_unit: int = 1
    outputs_per_workgroup: int = 1
    plane_width: int = 0
    version: int = 2
    private_accumulators: int = 1
    private_merge: str = "halving"

    @property
    def identity(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def units(self):
        x, y, z = self.workgroup
        return x * y * z

    def to_dict(self):
        result = {**asdict(self), "workgroup": list(self.workgroup)}
        # Preserve the frozen schema-2 plans and their artifact identities.
        if self.private_accumulators == 1:
            del result["private_accumulators"]
        if self.private_merge == "halving":
            del result["private_merge"]
        return result

    @classmethod
    def from_dict(cls, value):
        if type(value.get("private_accumulators", 1)) is not int:
            raise ValueError("private accumulator count must be an integer")
        result = cls(**{**value, "workgroup": tuple(value["workgroup"])})
        if result.version != 2 or result not in catalog():
            raise ValueError("unsupported Stage5 physical plan")
        return result


def catalog():
    result = [
        PhysicalCandidate("scalar_pointwise_baseline"),
        PhysicalCandidate("shared_tree_reduction_baseline"),
        PhysicalCandidate("shared_tree_compensated"),
        PhysicalCandidate(
            "independent_planes_final9",
            (32, 4, 1),
            outputs_per_workgroup=4,
            plane_width=32,
        ),
    ]
    result.extend(
        PhysicalCandidate("shared_tree_multi_accumulator", private_accumulators=n)
        for n in (2, 4)
    )
    result.append(
        PhysicalCandidate("plane_shared", plane_width=32, private_accumulators=4)
    )
    result.append(
        PhysicalCandidate(
            "plane_shared_load_batch", plane_width=32, private_accumulators=4
        )
    )
    result.append(
        PhysicalCandidate(
            "plane_shared_load_batch",
            plane_width=32,
            private_accumulators=4,
            private_merge="adjacent",
        )
    )
    result.append(
        PhysicalCandidate(
            "plane_shared_reference512",
            (512, 1, 1),
            plane_width=32,
            private_accumulators=4,
            private_merge="adjacent",
        )
    )
    for size in (64, 128, 256):
        if size != 256:
            result.append(PhysicalCandidate("scalar_pointwise", (size, 1, 1)))
            result.append(PhysicalCandidate("shared_tree", (size, 1, 1)))
        result.append(PhysicalCandidate("plane_shared", (size, 1, 1), plane_width=32))
        for width in (2, 4):
            result.append(PhysicalCandidate("vector_pointwise", (size, 1, 1), width))
            if size != 64:
                result.append(
                    PhysicalCandidate(
                        "plane_shared", (size, 1, 1), width, plane_width=32
                    )
                )
    result.append(
        PhysicalCandidate("scalar_pointwise", (128, 1, 1), elements_per_unit=4)
    )
    for size in (128, 256):
        result.append(PhysicalCandidate("vector_pointwise", (size, 1, 1), 8))
        result.append(
            PhysicalCandidate(
                "independent_planes",
                (32, size // 32, 1),
                outputs_per_workgroup=size // 32,
                plane_width=32,
            )
        )
        result.append(
            PhysicalCandidate(
                "coalesced_columns", (32, size // 32, 1), outputs_per_workgroup=32
            )
        )
    return tuple(result)


def bucket(value, boundaries):
    for i, end in enumerate(boundaries):
        if value <= end:
            return i
    return len(boundaries)


def shape_bucket(semantic, scalars):
    from stage2.indexing import evaluate

    if semantic["schema"] == 3:
        return [
            bucket(evaluate(semantic["outer"], scalars), (0, 1, 31, 255)),
            bucket(
                evaluate(semantic["reduction_extent"], scalars),
                (0, 1, 32, 256, 4096, 65536, 1048576),
            ),
        ]
    return [
        bucket(evaluate(semantic["numel"], scalars), (0, 1, 256, 4096, 65536, 1048576))
    ]
