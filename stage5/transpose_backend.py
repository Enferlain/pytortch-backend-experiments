"""One diagnostic substitution in an otherwise unchanged generated wrapper."""

from functools import partial

from torch._inductor.virtualized import V
from stage4 import backend as semantic
from stage5.backend import CubeCLScheduling
from stage5.choices import CubeCLChoices
from stage5.transpose_runtime import admit


class Scheduling(CubeCLScheduling):
    def define_kernel(self, src_code, node_schedule, kernel):
        admit(kernel.descriptor)
        wrapper = V.graph.wrapper_code
        name = "cubecl_transpose_trial_" + str(wrapper.next_kernel_suffix())
        wrapper.add_import_once(
            "from stage5.transpose_runtime import compile_kernel as transpose_compile"
        )
        wrapper.define_kernel(
            name,
            f"transpose_compile({kernel.descriptor!r}, {V.graph.get_current_device_or_throw().index})",
        )
        return name


def compile_backend(gm, inputs):
    return semantic.compile_backend(
        gm,
        inputs,
        scheduling=partial(Scheduling, tuning="baseline"),
        choices_class=CubeCLChoices,
        backend_name="cubecl_stage5_transpose_trial",
    )
