"""Physical plan representations: frozen Stage 1/3 plans and Stage 5 candidates.

Physical execution choices stay separate from logical scalar semantics. The
catalog is the set of plans the native compiler implements.
"""

import json
from dataclasses import asdict, dataclass

import sympy

from cubecl_torch.compiler.errors import UnsupportedCubeCL


@dataclass(frozen=True)
class PhysicalKernelPlan:
    iteration_mapping: str = "absolute_linear_unit"
    workgroup: tuple[int, int, int] = (256, 1, 1)
    elements_per_unit: int = 1
    local_vector_width: int = 1
    bounds_strategy: str = "explicit_less_than_numel"
    reduction_mapping: str = "none"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        fields = dict(value)
        fields["workgroup"] = tuple(fields["workgroup"])
        plan = cls(**fields)
        if plan != cls():
            raise NotImplementedError(
                "CubeCL unsupported physical schedule; only scalar_256 is implemented"
            )
        return plan


def plan_pointwise(numel, reduction_numel):
    if not isinstance(numel, sympy.Integer) or numel < 0 or reduction_numel != 1:
        raise NotImplementedError(
            "CubeCL physical schedule requires a static pointwise range"
        )
    return PhysicalKernelPlan()


@dataclass(frozen=True)
class ReductionPlan:
    iteration_mapping: str = "one_workgroup_per_outer"
    workgroup: tuple = (256, 1, 1)
    elements_per_unit: str = "strided_loop_step_workgroup"
    local_vector_width: int = 1
    reduction_mapping: str = "shared_binary_tree"
    accumulator_dtype: str = "float32"
    shared_elements: int = 256
    barriers: str = "uniform_after_publish_and_each_tree_level"
    result_owner: str = "unit_zero_store_all_units_read"
    bounds_strategy: str = "loop_reduction_coordinate_less_than_extent"
    phases: int = 1

    def to_dict(self):
        result = asdict(self)
        result["workgroup"] = list(self.workgroup)
        return result

    @classmethod
    def from_dict(cls, value):
        if value != cls().to_dict():
            raise UnsupportedCubeCL("unsupported physical reduction topology")
        return cls()


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
        if result.version != 2 or result not in (*catalog(), *welford_catalog()):
            raise ValueError("unsupported Stage5 physical plan")
        return result


def welford_catalog():
    """Plans offered to Welford kernels only, after the shared catalog.

    Eight-wide plane lanes (16-bit storage, 16-byte loads) reach Triton's lane
    count for Welford rows (welford-reductions 5). Independent planes with
    vector chains (several whole-vector accumulators per lane, merged as a
    tree) give one warp per row the same lane counts: 512, 1024 and 2048 at
    vector width 4 or 8 (welford-reductions 7). Kept out of ``catalog()`` so
    every other kernel's candidate and rejected sets, and so the Stage 5
    equivalence and manifest records, stay unchanged.
    """
    planes = tuple(
        PhysicalCandidate("plane_shared", (size, 1, 1), 8, plane_width=32)
        for size in (128, 256)
    )
    chains = tuple(
        PhysicalCandidate(
            "independent_planes",
            (32, 4, 1),
            width,
            outputs_per_workgroup=4,
            plane_width=32,
            private_accumulators=lanes // (32 * width),
        )
        for width in (4, 8)
        for lanes in (512, 1024, 2048)
    )
    return planes + chains


def catalog():
    result = [
        PhysicalCandidate("scalar_pointwise_baseline"),
        PhysicalCandidate("bounded_column_pointwise"),
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
    from cubecl_torch.compiler.indexing import evaluate

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
