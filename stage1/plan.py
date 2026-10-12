"""Physical execution choices, separate from Inductor logical scalar semantics."""
from dataclasses import asdict, dataclass
import hashlib
import json
import sympy


@dataclass(frozen=True)
class PhysicalKernelPlan:
    iteration_mapping: str = 'absolute_linear_unit'
    workgroup: tuple[int, int, int] = (256, 1, 1)
    elements_per_unit: int = 1
    local_vector_width: int = 1
    bounds_strategy: str = 'explicit_less_than_numel'
    reduction_mapping: str = 'none'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        fields = dict(value)
        fields['workgroup'] = tuple(fields['workgroup'])
        plan = cls(**fields)
        if plan != cls():
            raise NotImplementedError('CubeCL unsupported physical schedule; only scalar_256 is implemented')
        return plan


def plan_pointwise(numel, reduction_numel):
    if not isinstance(numel, sympy.Integer) or numel < 0 or reduction_numel != 1:
        raise NotImplementedError('CubeCL physical schedule requires a static pointwise range')
    return PhysicalKernelPlan()


def bridge_description(semantic, physical):
    """Adapt semantics + checked plan to the unchanged Stage 1 Rust wire schema.

    The only implemented plan matches the launcher's existing fixed 256 ABI.
    Future physical plans need their own adapter/builder, not new replay ops.
    """
    PhysicalKernelPlan.from_dict(physical)
    if semantic['schema'] != 1 or semantic['dtype'] != 'float32':
        raise NotImplementedError('CubeCL unsupported semantic descriptor')
    ops = []
    for operation in semantic['ops']:
        op = dict(operation)
        if op['op'] in ('load', 'store'):
            if op['index'] != 'linear':
                raise NotImplementedError('CubeCL physical plan requires linear accesses')
            op['index'] = 'i'
        ops.append(op)
    descriptor = {**semantic, 'physical': 'scalar_256', 'ops': ops}
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[:16]
    descriptor['entrypoint'] = 'cubecl_' + digest
    return descriptor
