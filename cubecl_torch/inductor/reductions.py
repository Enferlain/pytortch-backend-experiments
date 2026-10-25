"""Experimental reduction replay; physical execution belongs to the Rust adapter."""

import contextlib

import sympy
import torch
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.simd import SIMDKernel, SIMDScheduling
from torch._inductor.virtualized import V
from torch._prims_common import get_computation_dtype

from cubecl_torch.compiler.indexing import encode
from cubecl_torch.compiler.plan import PhysicalKernelPlan, ReductionPlan
from cubecl_torch.inductor.capabilities import UnsupportedCubeCL
from cubecl_torch.inductor.diagnostics import TraceLog
from cubecl_torch.inductor.reduction_checks import fusion_allowed
from cubecl_torch.inductor.semantic_checks import FLOATS, WELFORD
from cubecl_torch.inductor.semantics import (
    CubeCLKernel as PointwiseKernel,
)
from cubecl_torch.inductor.semantics import (
    CubeCLOps as PointwiseOps,
)
from cubecl_torch.inductor.semantics import (
    CubeCLScheduling as PointwiseScheduling,
)

TRACE = TraceLog()


class CubeCLOps(PointwiseOps):
    @staticmethod
    def index_expr(expr, dtype):
        # Mean emits a floating value of an integer shape expression, not a
        # floating runtime scalar ABI. Preserve exact i64 until explicit cast.
        value = V.kernel.emit(
            "index",
            dtype=torch.int64,
            semantic=torch.int64,
            index=V.kernel.check_index(expr),
        )
        return V.kernel.convert(value, get_computation_dtype(dtype), semantic=dtype)

    value_expr = index_expr


class CubeCLKernel(PointwiseKernel):
    overrides = CubeCLOps
    supported_input_counts = (0, 1, 2, 3, 4)
    trace = TRACE

    def __init__(self, *args, **kwargs):
        self.operations = []
        self.replay = []
        self.replay_stack = []
        self.semantic_cache = {}
        self.runtime_symbols = []
        SIMDKernel.__init__(self, *args, **kwargs)
        self._index_dtype = torch.int64
        self.reduction_kernel = self.features.is_reduction()
        expected = {"x", "r0_"} if self.reduction_kernel else {"x"}
        if set(self.numels) != expected:
            raise UnsupportedCubeCL(
                "unsupported physical schedule: one outer and one reduction range"
            )
        self.physical_plan = (
            ReductionPlan() if self.reduction_kernel else PhysicalKernelPlan()
        )

    @contextlib.contextmanager
    def disable_reduction(self):
        prior = self.inside_reduction
        self.inside_reduction = False
        try:
            yield
        finally:
            self.inside_reduction = prior

    def check_index(self, index):
        expanded = self.simplify_indexing(index)
        substitutions = {s: e.expr for s, e in self.range_tree_nodes.items()}
        for _ in range(len(substitutions) + 1):
            new = expanded.xreplace(substitutions)
            if new == expanded:
                break
            expanded = new
        if not self.reduction_kernel:
            return encode(
                sympy.simplify(expanded),
                self.symbol_slot,
                self.range_trees[0].index_sym(),
            )
        outer, reduction = self.range_trees
        marker = sympy.Symbol("cubecl_reduction_coordinate", integer=True)
        tree = encode(
            sympy.simplify(expanded.xreplace({reduction.index_sym(): marker})),
            lambda s: -1 if s == marker else self.symbol_slot(s),
            outer.index_sym(),
        )

        def replace(v):
            if v.get("op") == "scalar" and v.get("slot") == -1:
                return {"op": "reduction"}
            return {
                k: [replace(x) for x in a] if k == "args" else a for k, a in v.items()
            }

        return replace(tree)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        if (
            reduction_type == "welford_reduce"
            and dtype == src_dtype == torch.float32
            and InductorChoices.should_use_persistent_reduction(self.features, False)
        ):
            # Where Triton keeps the row in registers it computes the two-pass
            # mean and sum of squared deviations (welford_reduce_fallback),
            # clearly more accurate than Welford on short rows with an offset.
            self.welford_two_pass = True
            return self.welford_reduce_fallback(dtype, value)
        if reduction_type in WELFORD and dtype == src_dtype == torch.float32:
            # (mean, m2, weight); welford_combine reduces such triples.
            values = value if reduction_type == "welford_combine" else (value,)
            _, components = self.emit_reduction(
                dtype, src_dtype, reduction_type, values, components=3
            )
            return components
        if reduction_type not in ("sum", "max", "min") or dtype not in FLOATS:
            raise UnsupportedCubeCL(
                f"unsupported reduction {reduction_type} over {dtype}: "
                "sum/max/min of float32/16/bf16, Welford of float32 only"
            )
        result, _ = self.emit_reduction(dtype, src_dtype, reduction_type, (value,))
        # The logical reduction result is rounded to its semantic dtype before
        # a fused consumer can reuse it, just as a materialized lowp sum is.
        if dtype != torch.float32:
            result = self.convert(result, dtype, semantic=dtype)
            result = self.convert(result, torch.float32, semantic=dtype)
        return result

    def emit_reduction(self, dtype, src_dtype, reduction_type, values, components=1):
        """The reduction op over float32 operands, and its result components.

        A single-value reduction is its own result; a multi-value one is read
        through ``reduction_component`` ops, one per component.
        """
        values = [self.convert(v, torch.float32, semantic=src_dtype) for v in values]
        result = self.emit(
            "reduction",
            *values,
            dtype=torch.float32,
            semantic=dtype,
            reduction_type=reduction_type,
            source_dtype=str(src_dtype).removeprefix("torch."),
            accumulator_dtype="float32",
        )
        self.replay.append(
            {
                "op": "reduction",
                "dtype": str(dtype),
                "src_dtype": str(src_dtype),
                "reduction_type": reduction_type,
                "input": values[0].ssa if len(values) == 1 else [v.ssa for v in values],
                "result": result.ssa,
            }
        )
        if components == 1:
            return result, (result,)
        return result, tuple(
            self.emit("reduction_component", result, dtype=torch.float32, index=k)
            for k in range(components)
        )

    def store(self, name, index, value, mode=None):
        super().store(name, index, value, mode)
        if self.reduction_kernel:
            self.operations[-1]["iteration_scope"] = (
                "reduction" if self.inside_reduction else "outer"
            )

    def store_reduction(self, name, index, value):
        with self.disable_reduction():
            self.store(name, index, value)

    def finalize_descriptor(self):
        # Reuse the complete Stage 2 buffer/SSA/CSE serialization, then add
        # semantic reduction domains separately from the physical plan.
        if self.reduction_kernel:
            self.descriptor.update(
                schema=3,
                outer=self.descriptor["numel"],
                reduction_extent=encode(self.numels["r0_"], self.symbol_slot),
            )
            self.descriptor["scalars"] = self.scalar_declarations()


class CubeCLScheduling(PointwiseScheduling):
    kernel_type = CubeCLKernel

    def can_fuse(self, a, b):
        result = fusion_allowed(a, b, TRACE) and SIMDScheduling.can_fuse(self, a, b)
        TRACE.append(
            {
                "event": "fusion_delegate",
                "nodes": [a.get_name(), b.get_name()],
                "accepted": bool(result),
            }
        )
        return result

    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse

    @classmethod
    def get_tiling_and_scores(
        cls, node_schedule, numel, reduction_numel, coalesce_analysis=None
    ):
        cls.kernel_type.trace.append(
            {
                "event": "schedule",
                "nodes": [
                    n.get_name() if hasattr(n, "get_name") else str(n)
                    for n in node_schedule
                ],
                "numel": str(numel),
                "reduction_numel": str(reduction_numel),
            }
        )
        return (
            {"x": numel, "r0_": reduction_numel}
            if any(getattr(n, "is_reduction", lambda: False)() for n in node_schedule)
            else {"x": numel}
        ), {}
