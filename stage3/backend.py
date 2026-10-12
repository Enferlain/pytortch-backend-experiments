"""Experimental reduction replay; physical execution belongs to the Rust adapter."""

import contextlib
import sympy
import torch
from torch._inductor import config
from torch._inductor.codegen.common import device_codegens, register_backend_for_device
from torch._inductor.codegen.simd import SIMDKernel, SIMDScheduling
from torch._inductor.virtualized import V
from torch._prims_common import get_computation_dtype
from stage1.capabilities import UnsupportedCubeCL
from stage1.plan import PhysicalKernelPlan
from stage2.backend import (
    CubeCLKernel as PointwiseKernel,
    CubeCLScheduling as PointwiseScheduling,
    CubeCLOps as PointwiseOps,
)
from stage2.indexing import encode
from stage3.capabilities import pre_fusion, fusion_allowed
from stage2.capabilities import check_inputs
from stage3.plan import ReductionPlan

TRACE = []


class CubeCLOps(PointwiseOps):
    def _default(self, name, args, kwargs):
        if name == "rsqrt":
            return V.kernel.operation(name, *args)
        return super()._default(name, args, kwargs)

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
        if reduction_type != "sum" or dtype not in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ):
            raise UnsupportedCubeCL("unsupported reduction operation/dtype")
        value = self.convert(value, torch.float32, semantic=src_dtype)
        result = self.emit(
            "reduction",
            value,
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
                "input": value.ssa,
                "result": result.ssa,
            }
        )
        # The logical reduction result is rounded to its semantic dtype before
        # a fused consumer can reuse it, just as a materialized lowp sum is.
        if dtype != torch.float32:
            result = self.convert(result, dtype, semantic=dtype)
            result = self.convert(result, torch.float32, semantic=dtype)
        return result

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
            self.descriptor["scalars"] = [
                {"dtype": "int64", "id": i, "positive": s.is_positive is True}
                for i, s in enumerate(self.runtime_symbols)
            ]


class CubeCLScheduling(PointwiseScheduling):
    kernel_type = CubeCLKernel

    def check_backend_available(self, device):
        super().check_backend_available(device)
        import ctypes as C
        from stage0.bridge import ROOT as STAGE0_ROOT
        from stage1.runtime import CubeCLBackendUnavailable
        from fp_hardening.patches import SOURCE, ROOT as PROJECT_ROOT, PIN
        import subprocess

        launcher = C.CDLL(str(STAGE0_ROOT / "libstage0_launcher.so"))
        if not hasattr(launcher, "stage3_launch") or not hasattr(
            launcher, "stage0_kernel_attributes"
        ):
            raise CubeCLBackendUnavailable(
                "CubeCL Stage 3 launch ABI missing; run bash stage3/setup.sh"
            )
        if (
            subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"])
            .decode()
            .strip()
            != PIN
            or subprocess.check_output(["git", "-C", str(SOURCE), "diff", "HEAD"])
            != (PROJECT_ROOT / "fp_hardening/cubecl-fp.patch").read_bytes()
        ):
            raise CubeCLBackendUnavailable(
                "CubeCL Stage 3 requires the exact audited FP source stack; run bash stage3/setup.sh"
            )
        TRACE.append(
            {
                "event": "availability",
                "backend": type(self).__name__,
                "device": str(device),
                "descriptor_schemas": [2, 3],
            }
        )

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
        TRACE.append(
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

    def define_kernel(self, src_code, node_schedule, kernel):
        wrapper = V.graph.wrapper_code
        name = "cubecl_stage3_" + str(wrapper.next_kernel_suffix())
        wrapper.add_import_once(
            "from stage3.runtime import compile_kernel as cubecl_stage3_compile"
        )
        wrapper.define_kernel(
            name,
            f"cubecl_stage3_compile({kernel.descriptor!r}, {V.graph.get_current_device_or_throw().index}, {kernel.physical_plan.to_dict()!r})",
        )
        return name


_registered = False


def register():
    global _registered
    if _registered:
        return
    previous = device_codegens["cuda"]

    def select(scheduler):
        return (
            CubeCLScheduling(scheduler)
            if config.cuda_backend == "cubecl_stage3"
            else previous.scheduling(scheduler)
        )

    register_backend_for_device(
        "cuda",
        select,
        previous.wrapper_codegen,
        previous.cpp_wrapper_codegen,
        previous.fx_wrapper_codegen,
    )
    _registered = True


def compile_backend(gm, example_inputs):
    check_inputs(example_inputs, TRACE)
    if config.numerics == "strict":
        TRACE.append(
            {
                "event": "capability",
                "boundary": "backend_selection",
                "decision": "reject",
                "reason": "exact eager reduction ordering mode unsupported",
            }
        )
        raise UnsupportedCubeCL(
            "unsupported input contract: exact eager reduction ordering mode"
        )
    TRACE.append({"event": "graph", "graph": str(gm.graph)})
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor.codegen.common import init_backend_registration

    init_backend_registration()
    register()
    previous = config._pre_fusion_custom_pass

    def validate(nodes):
        return pre_fusion(previous(nodes) if previous else nodes, TRACE)

    return compile_fx(
        gm,
        example_inputs,
        config_patches={
            "cuda_backend": "cubecl_stage3",
            "compile_threads": 1,
            "fx_graph_cache": False,
            "_pre_fusion_custom_pass": validate,
            "triton.cudagraphs": False,
            "inplace_buffers": False,
            "benchmark_kernel": False,
            "combo_kernels": False,
            "triton.coalesce_tiling_analysis": False,
            "emulate_precision_casts": True,
            "joint_graph_constant_folding": False,
        },
    )
