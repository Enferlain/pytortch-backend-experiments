"""Experimental cubecl_stage5; semantic replay/capability/fusion stay Stage4."""

from functools import partial
import ctypes as C
import time
from torch._inductor.virtualized import V
from torch._inductor import config
from stage4 import backend as semantic
from stage0.bridge import ROOT as BRIDGE_ROOT

TRACE = semantic.TRACE


class CubeCLKernel(semantic.CubeCLKernel):
    def codegen_kernel(self):
        began = time.perf_counter()
        result = super().codegen_kernel()
        TRACE.append(
            {
                "event": "descriptor_construction",
                "elapsed_ms": (time.perf_counter() - began) * 1000,
                "scope": "layout/symbol binding, descriptor rewrite/finalization/serialization; semantic replay already ran",
            }
        )
        return result


class CubeCLScheduling(semantic.CubeCLScheduling):
    kernel_type = CubeCLKernel

    def __init__(self, scheduler, *, tuning):
        self.tuning = tuning
        super().__init__(scheduler)

    def check_backend_available(self, device):
        super().check_backend_available(device)
        native = C.CDLL(str(BRIDGE_ROOT / "libstage0_launcher.so"))
        if not hasattr(native, "stage5_launch") or not hasattr(
            native, "stage5_resources"
        ):
            raise RuntimeError("Stage5 explicit launch/resource ABI not built")

    def define_kernel(self, src_code, node_schedule, kernel):
        wrapper = V.graph.wrapper_code
        name = "cubecl_stage5_" + str(wrapper.next_kernel_suffix())
        wrapper.add_import_once(
            "from stage5.runtime import compile_kernel as cubecl_stage5_compile"
        )
        from stage5.reference512_plan import role, scheduler_context, SHAPES

        if self.tuning == "first_use" and role(kernel.descriptor) is not None:
            from torch._inductor.utils import DelayReplaceLine

            if not hasattr(self, "_reference512_records"):
                self._reference512_records = []
            nodes = [n for n in node_schedule if hasattr(n, "get_outputs")]
            record = {
                "semantic": kernel.descriptor,
                "nodes": [n.get_name() for n in nodes],
                "outputs": [o for n in nodes for o in n.get_outputs()],
                "input_names": kernel.input_names,
                "output_names": kernel.output_names,
            }
            if role(kernel.descriptor) == "producer":
                import sympy

                n, b = kernel.runtime_symbols
                predicate = sympy.Or(
                    *(
                        sympy.And(sympy.Eq(b, bb), sympy.Eq(n, nn))
                        for bb, nn in sorted(SHAPES)
                    )
                )
                # Existing ShapeEnv guards bind the final's restricted choices
                # to the same finite scope, without changing a split decision.
                record["shape_admitted"] = bool(
                    V.graph.sizevars.evaluate_expr(predicate)
                )
            self._reference512_records.append(record)
            device = V.graph.get_current_device_or_throw().index

            # Existing Inductor deferred lines resolve after both kernels exist.
            # The emitted factory is ordinary Python: no lazy warm-call wrapper.
            def factory():
                ctx = scheduler_context(record, self._reference512_records)
                return f"cubecl_stage5_compile({kernel.descriptor!r}, {device}, policy='first_use', reduction_context={ctx!r})"

            wrapper.header.writeline(
                DelayReplaceLine(
                    "__PAIR_FACTORY__", factory, f"\n{name} = __PAIR_FACTORY__"
                )
            )
            return name
        wrapper.define_kernel(
            name,
            f"cubecl_stage5_compile({kernel.descriptor!r}, {V.graph.get_current_device_or_throw().index}, policy={self.tuning!r})",
        )
        return name


def compile_backend(
    gm, inputs, *, tuning="first_use", rng_policy="generated", split_policy="guarded"
):
    if tuning not in (
        "first_use",
        "baseline",
        "combined_diagnostic",
        "compensated_diagnostic",
        "load_batch_diagnostic",
        "adjacent_merge_diagnostic",
        "reference512_diagnostic",
    ):
        raise ValueError(
            "experimental Stage5 tuning: first_use, baseline, combined_diagnostic, compensated_diagnostic, load_batch_diagnostic, adjacent_merge_diagnostic or reference512_diagnostic"
        )
    if split_policy not in ("guarded", "inherited"):
        raise ValueError("experimental Stage5 split policy: guarded or inherited")
    from .choices import CubeCLChoices
    from .index_masks import widen_mask_indices

    previous_pass = config._pre_fusion_custom_pass

    def prepare(nodes):
        return widen_mask_indices(previous_pass(nodes) if previous_pass else nodes)

    with config.patch({"_pre_fusion_custom_pass": prepare}):
        return semantic.compile_backend(
            gm,
            inputs,
            rng_policy=rng_policy,
            scheduling=partial(CubeCLScheduling, tuning=tuning),
            backend_name="cubecl_stage5_" + tuning,
            choices_class=CubeCLChoices if split_policy == "guarded" else None,
        )
