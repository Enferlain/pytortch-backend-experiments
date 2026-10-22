"""Physical-plan level of the class chain; semantic replay/capability/fusion stay in effects."""

import time
from functools import partial

from torch._inductor import config
from torch._inductor.virtualized import V

from cubecl_torch.compiler.nvrtc import Numerics
from cubecl_torch.inductor import effects as semantic

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

    def define_kernel(self, src_code, node_schedule, kernel):
        wrapper = V.graph.wrapper_code
        name = "cubecl_kernel_" + str(wrapper.next_kernel_suffix())
        # The same config switches Triton's fp fusion and division rounding.
        numerics = Numerics(
            fp_contract=not config.emulate_precision_casts,
            exact_division=config.eager_numerics.division_rounding,
        )
        wrapper.add_import_once(
            "from cubecl_torch.inductor.tuning import compile_kernel as cubecl_compile"
        )
        wrapper.add_import_once("from cubecl_torch.compiler.nvrtc import Numerics")
        from cubecl_torch.inductor.reference512_plan import (
            SHAPES,
            role,
            scheduler_context,
        )

        if self.tuning == "first_use" and role(kernel.descriptor) is not None:
            from torch._inductor.utils import DelayReplaceLine

            graph = V.graph
            if getattr(self, "_reference512_graph", None) is not graph:
                self._reference512_graph = graph
                self._reference512_records = []
            records = self._reference512_records
            nodes = [n for n in node_schedule if hasattr(n, "get_outputs")]
            record = {
                "semantic": kernel.descriptor,
                "nodes": [n.get_name() for n in nodes],
                "outputs": [o for n in nodes for o in n.get_outputs()],
                "input_names": kernel.input_names,
                "output_names": kernel.output_names,
            }
            record["symbols"] = list(kernel.runtime_symbols)
            records.append(record)
            device = V.graph.get_current_device_or_throw().index

            # Existing Inductor deferred lines resolve after both kernels exist.
            # The emitted factory is ordinary Python: no lazy warm-call wrapper.
            def factory():
                import sympy

                # Both descriptors must already exist before adding guards:
                # earlier equalities would specialize the final's scalar ABI.
                for producer in records:
                    if (
                        role(producer["semantic"]) != "producer"
                        or "shape_admitted" in producer
                    ):
                        continue
                    # Stage2 symbol_slot order: template slot0=N, slot1=B.
                    assert (
                        len(producer["symbols"])
                        == len(producer["semantic"]["scalars"])
                        == 2
                    )
                    n, b = producer["symbols"]
                    sizes = graph.sizevars
                    selected_n = next(
                        (
                            v
                            for v in sorted({nn for _, nn in SHAPES})
                            if sizes.evaluate_expr(sympy.Eq(n, v))
                        ),
                        None,
                    )
                    selected_b = next(
                        (
                            v
                            for v in sorted({bb for bb, _ in SHAPES})
                            if sizes.evaluate_expr(sympy.Eq(b, v))
                        ),
                        None,
                    )
                    producer["shape_admitted"] = (selected_b, selected_n) in SHAPES
                ctx = scheduler_context(record, records)
                return f"cubecl_compile({kernel.descriptor!r}, {device}, policy='first_use', reduction_context={ctx!r}, numerics={numerics!r})"

            wrapper.header.writeline(
                DelayReplaceLine(
                    "__PAIR_FACTORY__", factory, f"\n{name} = __PAIR_FACTORY__"
                )
            )
            return name
        wrapper.define_kernel(
            name,
            f"cubecl_compile({kernel.descriptor!r}, {V.graph.get_current_device_or_throw().index}, policy={self.tuning!r}, numerics={numerics!r})",
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
    from cubecl_torch.inductor.choices import CubeCLChoices
    from cubecl_torch.inductor.index_masks import widen_mask_indices

    previous_pass = config._pre_fusion_custom_pass

    def prepare(nodes):
        return widen_mask_indices(previous_pass(nodes) if previous_pass else nodes)

    with config.patch({"_pre_fusion_custom_pass": prepare}):
        return semantic.compile_backend(
            gm,
            inputs,
            rng_policy=rng_policy,
            scheduling=partial(CubeCLScheduling, tuning=tuning),
            backend_name="cubecl_torch_" + tuning,
            choices_class=CubeCLChoices if split_policy == "guarded" else None,
        )
