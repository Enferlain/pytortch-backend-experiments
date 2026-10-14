"""Experimental guards for split/no-split regimes, not a new split planner."""

import sympy
import hashlib
from pathlib import Path
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch._inductor.choices import InductorChoices
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.virtualized import V
from torch._inductor import config

EVENTS = []


class CubeCLChoices(InductorChoices):
    @staticmethod
    def uuid():
        return (
            "cubecl-split-regimes-v1:"
            + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        )

    def prepare_reduction_split(
        self, device, reduction_numel, numel, reduction_numel_hint, numel_hint
    ):
        # Guard original single-axis extents, not Inductor-generated partial
        # extents. Partials retain the captured logical split within this regime.
        sizes = V.graph.sizevars
        guarded = []
        if not hasattr(self, "planning_hints"):
            self.planning_hints = {}
        if (
            isinstance(reduction_numel, sympy.Symbol)
            and device.type == "cuda"
            and not free_unbacked_symbols((reduction_numel, numel))
        ):
            props = DeviceProperties.create(device)
            enough_outputs = sympy.Ge(numel, 2 * props.multi_processor_count)
            has_outputs = bool(sizes.evaluate_expr(enough_outputs))
            guarded.append([str(enough_outputs), has_outputs])
            if not has_outputs:
                # Only SM count affects the inherited threshold on major >=10.
                if props.major is not None and props.major >= 10:
                    predicate = sympy.Lt(numel, props.multi_processor_count)
                    guarded.append(
                        [str(predicate), bool(sizes.evaluate_expr(predicate))]
                    )
                threshold = self._inner_reduction_no_split_threshold(
                    props, numel_hint, props.multi_processor_count
                )
                predicate = sympy.Le(reduction_numel, threshold)
                short = bool(sizes.evaluate_expr(predicate))
                guarded.append([str(predicate), short])
                # Candidate buckets reuse existing Stage5 outer buckets and
                # the default heuristic's private-work quantum. They are not
                # universal optimal thresholds. Representative planning hints
                # prevent the first size inside a bucket changing its topology.
                small_outer = sympy.Le(numel, 31)
                is_small = bool(sizes.evaluate_expr(small_outer))
                guarded.append([str(small_outer), is_small])
                if isinstance(numel, sympy.Symbol):
                    self.planning_hints[numel] = 8 if is_small else 64
                anchor = threshold if short else None
                if not short:
                    quantum = props.warp_size_or_default**2 * 8
                    # Nine is the measured 65K reference's split count; it is
                    # an experimental representative, not a universal minimum.
                    first = max(9, config.min_num_split) * quantum
                    middle = 1 << (first - 1).bit_length()
                    for boundary in sorted({first, middle, 1048576}):
                        predicate = sympy.Le(reduction_numel, boundary)
                        fits = bool(sizes.evaluate_expr(predicate))
                        guarded.append([str(predicate), fits])
                        if fits:
                            anchor = boundary
                            break
                if anchor is not None:
                    self.planning_hints[reduction_numel] = anchor
                else:
                    self.planning_hints.pop(reduction_numel, None)
                    self.planning_hints.pop(numel, None)
        EVENTS.append(
            {
                "event": "split_regime_guard",
                "outer": str(numel),
                "reduction_extent": str(reduction_numel),
                "outer_hint": numel_hint,
                "reduction_hint": reduction_numel_hint,
                "guards": guarded,
                "planning_anchor": self.planning_hints.get(reduction_numel),
                "policy": self.uuid(),
            }
        )

    def reduction_split_factor_with_extents(
        self,
        device,
        reduction_numel_hint,
        numel_hint,
        inner_reduction,
        *,
        reduction_numel,
        numel,
    ):
        props = DeviceProperties.create(device)
        planning = (
            getattr(self, "planning_hints", {})
            if inner_reduction and numel_hint < 2 * props.multi_processor_count
            else {}
        )
        planned_r = int(
            V.graph.sizevars.optimization_hint(
                sympy.sympify(reduction_numel).xreplace(planning)
            )
        )
        planned_x = int(
            V.graph.sizevars.optimization_hint(sympy.sympify(numel).xreplace(planning))
        )
        split = super().reduction_split_factor_with_extents(
            device,
            planned_r,
            planned_x,
            inner_reduction,
            reduction_numel=reduction_numel,
            numel=numel,
        )
        EVENTS.append(
            {
                "event": "split_regime",
                "outer": str(numel),
                "reduction_extent": str(reduction_numel),
                "outer_hint": numel_hint,
                "reduction_hint": reduction_numel_hint,
                "planning_reduction_hint": planned_r,
                "planning_outer_hint": planned_x,
                "inner": inner_reduction,
                "split": split,
                "policy": self.uuid(),
            }
        )
        return split
