"""An execution plan, separate from reduction meaning and index expressions."""

from dataclasses import asdict, dataclass
from stage1.capabilities import UnsupportedCubeCL


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
