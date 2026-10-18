"""One explicit mixed-family generated wrapper; scheduling/fusion unchanged."""

from functools import partial

from torch._inductor.virtualized import V
from stage4 import backend as semantic
from stage5.backend import CubeCLScheduling
from stage5.choices import CubeCLChoices
from stage5.composition_trial import role


class Scheduling(CubeCLScheduling):
    def __init__(self, *args, mode, **kw):
        super().__init__(*args, **kw)
        self.mode = mode

    def define_kernel(self, src_code, node_schedule, kernel):
        if role(kernel.descriptor) == "unadmitted":
            raise ValueError("mixed graph requires a new descriptor adapter; stop this trial")
        wrapper = V.graph.wrapper_code
        name = "cubecl_composition_" + str(wrapper.next_kernel_suffix())
        wrapper.add_import_once("from stage5.composition_runtime import compile_kernel as composition_compile")
        wrapper.define_kernel(name, f"composition_compile({kernel.descriptor!r}, {V.graph.get_current_device_or_throw().index}, {self.mode!r})")
        return name


def compile_backend(gm, inputs, *, mode):
    return semantic.compile_backend(
        gm, inputs, scheduling=partial(Scheduling, tuning="first_use", mode=mode),
        choices_class=CubeCLChoices, backend_name="cubecl_stage5_composition_" + mode,
    )
