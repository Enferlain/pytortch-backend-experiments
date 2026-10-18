"""Explicit column-only wrapper substitutions; ordinary scheduling is retained."""

from functools import partial

from torch._inductor.virtualized import V
from stage4 import backend as semantic
from stage5.backend import CubeCLScheduling
from stage5.choices import CubeCLChoices
from stage5.column_layout_runtime import admit
from stage5.layout_runtime import admit as generic_admit


class Scheduling(CubeCLScheduling):
    def __init__(self, *args, frozen_generic=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_generic = frozen_generic

    def define_kernel(self, src_code, node_schedule, kernel):
        family, _ = generic_admit(kernel.descriptor)
        if family == "bias_column":
            admit(kernel.descriptor)
        wrapper = V.graph.wrapper_code
        name = "cubecl_column_trial_" + str(wrapper.next_kernel_suffix())
        method = "generic_compile" if self.frozen_generic else "qualified_dispatch"
        wrapper.add_import_once(
            f"from stage5.column_layout_runtime import {method} as column_compile"
        )
        wrapper.define_kernel(
            name,
            f"column_compile({kernel.descriptor!r}, {V.graph.get_current_device_or_throw().index})",
        )
        return name


def compile_backend(gm, inputs, *, frozen_generic=False):
    return semantic.compile_backend(
        gm,
        inputs,
        scheduling=partial(
            Scheduling, tuning="baseline", frozen_generic=frozen_generic
        ),
        choices_class=CubeCLChoices,
        backend_name="cubecl_stage5_column_trial",
    )
