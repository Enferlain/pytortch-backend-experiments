"""Experimental Stage 2: typed scalar replay on the existing SIMD seam."""

import json
import math
import traceback

import sympy
import torch
from torch._inductor.codegen.simd import SIMDKernel, SIMDScheduling
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.virtualized import V
from torch._prims_common import get_computation_dtype

from cubecl_torch.compiler.indexing import encode
from cubecl_torch.compiler.plan import PhysicalKernelPlan
from cubecl_torch.inductor.capabilities import UnsupportedCubeCL
from cubecl_torch.inductor.diagnostics import TraceLog
from cubecl_torch.inductor.pointwise import (
    CubeCLKernel as BaselineKernel,
)
from cubecl_torch.inductor.pointwise import (
    CubeCLScheduling as BaselineScheduling,
)
from cubecl_torch.inductor.pointwise import (
    CubeCLValue as BaselineValue,
)
from cubecl_torch.inductor.semantic_checks import (
    FLOATS,
    INTEGERS,
    VALUES,
    fusion_allowed,
    scalar_input,
)

TRACE = TraceLog()
# float64 names runtime scalar values only; it is never storage or arithmetic.
DTYPES = {d: str(d).removeprefix("torch.") for d in (*FLOATS, *INTEGERS, torch.float64)}


class CubeCLValue(BaselineValue):
    def __init__(self, name, bounds, dtype, shape=()):
        super().__init__(name, bounds, dtype, shape)
        self.semantic_dtype = dtype


def json_float(value):
    """A float literal for the descriptor: non-finite values as strings."""
    value = float(value)
    if math.isfinite(value):
        return value
    return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")


class CubeCLOps(DefaultHandler):
    def _default(self, name, args, kwargs):
        if name in (
            "sin",
            "cos",
            "neg",
            "exp",
            "sqrt",
            "reciprocal",
            "add",
            "sub",
            "mul",
            "truediv",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "where",
            "maximum",
            "minimum",
            "isnan",
        ):
            return V.kernel.operation(name, *args)
        raise UnsupportedCubeCL(f"CubeCL Stage 2 semantic operation {name} unsupported")

    @staticmethod
    def constant(value, dtype):
        if dtype not in VALUES:
            raise UnsupportedCubeCL(f"constant dtype {dtype}")
        if dtype == torch.int64:
            if not isinstance(value, int) or not -(1 << 63) < value < (1 << 63):
                raise UnsupportedCubeCL("int64 constant out of range")
        elif dtype == torch.bool:
            value = bool(value)
        else:
            value = json_float(value)
        # A scalar literal is not a load from low-precision tensor storage.
        # ATen CUDA's opmath scalar functors retain it in compute precision.
        # Explicit materialization/cast/store barriers still round tensor values.
        return V.kernel.emit(
            "constant", dtype=get_computation_dtype(dtype), semantic=dtype, value=value
        )

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None, use_compute_types=True):
        if dtype not in VALUES:
            raise UnsupportedCubeCL(f"cast dtype {dtype}")
        target = get_computation_dtype(dtype) if use_compute_types else dtype
        return V.kernel.convert(x, target, semantic=dtype)

    @staticmethod
    def index_expr(expr, dtype):
        if dtype != torch.int64:
            raise UnsupportedCubeCL("index values require signed int64")
        return V.kernel.emit(
            "index",
            dtype=torch.int64,
            semantic=torch.int64,
            index=V.kernel.check_index(expr),
        )

    value_expr = index_expr

    @staticmethod
    def masked(mask, body, other):
        return V.kernel.masked(mask, body, other)


class CubeCLKernel(BaselineKernel):
    overrides = CubeCLOps
    supported_input_counts = (1, 2, 3, 4)
    trace = TRACE
    dtype_names = DTYPES
    value_dtypes = VALUES

    def __init__(self, *args, **kwargs):
        self.operations = []
        self.replay = []
        self.replay_stack = []
        self.semantic_cache = {}
        self.runtime_symbols = []
        SIMDKernel.__init__(self, *args, **kwargs)
        # Descriptor/address arithmetic always uses signed i64, independently of
        # the common SIMD feature probe's conservative 32-bit specialization guards.
        self._index_dtype = torch.int64
        if (
            self.features.is_reduction()
            or self.features.reduction_numel != 1
            or set(self.numels) != {"x"}
        ):
            raise UnsupportedCubeCL(
                "unsupported physical plan: scalar flattened pointwise only"
            )
        self.physical_plan = PhysicalKernelPlan()

    def create_cse_var(self, name, bounds, dtype=None, shape=None):
        if dtype not in (*self.value_dtypes, torch.float64) or shape not in (None, ()):
            raise UnsupportedCubeCL(f"CSE dtype/shape {dtype}/{shape}")
        return CubeCLValue(name, bounds, dtype, ())

    def emit(self, op, *args, dtype=torch.float32, semantic=None, **fields):
        if any(not isinstance(x, CubeCLValue) for x in args):
            raise UnsupportedCubeCL(f"untyped operands for {op}")
        semantic = semantic or dtype
        record = {
            "op": op,
            "args": [x.ssa for x in args],
            "dtype": self.dtype_names[dtype],
            "semantic_dtype": self.dtype_names[semantic],
            **fields,
        }
        key = json.dumps(record, sort_keys=True)
        if key in self.semantic_cache:
            value = self.semantic_cache[key]
            value.use_count += 1
            return value
        value = self.cse.newvar(dtype=dtype, shape=())
        value.semantic_dtype = semantic
        self.operations.append({"id": value.ssa, **record})
        self.semantic_cache[key] = value
        return value

    def convert(self, x, dtype, semantic=None):
        semantic = semantic or dtype
        if dtype == x.dtype and semantic == x.semantic_dtype:
            return x
        # Even same-compute-type semantic transitions have explicit typed records.
        return self.emit("cast", x, dtype=dtype, semantic=semantic)

    def operation(self, name, *args):
        if name == "where":
            mask, x, y = args
            if mask.dtype != torch.bool:
                raise UnsupportedCubeCL("where predicate must be bool")
            semantic = torch.promote_types(x.semantic_dtype, y.semantic_dtype)
            dtype = get_computation_dtype(semantic)
            return self.emit(
                name,
                mask,
                self.convert(x, dtype),
                self.convert(y, dtype),
                dtype=dtype,
                semantic=semantic,
            )
        semantic = args[0].semantic_dtype
        for x in args[1:]:
            semantic = torch.promote_types(semantic, x.semantic_dtype)
        if semantic not in FLOATS and name not in ("eq", "ne", "lt", "le", "gt", "ge"):
            raise UnsupportedCubeCL("integer/bool tensor arithmetic is outside Stage 2")
        dtype = get_computation_dtype(semantic)
        args = [self.convert(x, dtype, semantic=semantic) for x in args]
        if name in ("eq", "ne", "lt", "le", "gt", "ge", "isnan"):
            return self.emit(name, *args, dtype=torch.bool, semantic=torch.bool)
        return self.emit(name, *args, dtype=dtype, semantic=semantic)

    def symbol_slot(self, symbol):
        if symbol not in self.runtime_symbols:
            self.runtime_symbols.append(symbol)
            self.args.size(symbol)
        return self.runtime_symbols.index(symbol)

    def scalar_value(self, name):
        """A 0-d CPU float64 input read as a runtime scalar (Inductor's unspec arg).

        Its slot holds the buffer name; the wrapper passes ``name.item()``.
        """
        if name not in self.runtime_symbols:
            self.runtime_symbols.append(name)
        return self.emit(
            "scalar_value",
            dtype=torch.float64,
            semantic=torch.float64,
            slot=self.runtime_symbols.index(name),
        )

    def scalar_declarations(self):
        return [
            {"dtype": "float64", "id": i, "positive": False}
            if isinstance(s, str)
            else {"dtype": "int64", "id": i, "positive": s.is_positive is True}
            for i, s in enumerate(self.runtime_symbols)
        ]

    def check_index(self, index):
        expanded = self.simplify_indexing(index)
        substitutions = {s: e.expr for s, e in self.range_tree_nodes.items()}
        # Range entries can reference other entries. Resolve until no range symbol remains.
        for _ in range(len(substitutions) + 1):
            new = expanded.xreplace(substitutions)
            if new == expanded:
                break
            expanded = new
        expanded = sympy.simplify(expanded)
        return encode(expanded, self.symbol_slot, self.range_trees[0].index_sym())

    def check_buffer(self, name):
        dtype = V.graph.get_dtype(name)
        if dtype not in (*FLOATS, torch.bool):
            raise UnsupportedCubeCL(f"buffer storage {dtype}")
        return dtype

    def load(self, name, index):
        if not self.replay_stack:
            self.replay_stack = [
                {"file": f.filename, "line": f.lineno, "function": f.name}
                for f in traceback.extract_stack()
            ]
        if scalar_input(name):
            return self.scalar_value(name)
        dtype = self.check_buffer(name)
        self.args.input(name)
        raw = self.emit(
            "load",
            dtype=dtype,
            semantic=dtype,
            buffer_name=name,
            index=self.check_index(index),
        )
        return self.convert(raw, get_computation_dtype(dtype), semantic=dtype)

    def store(self, name, index, value, mode=None):
        if mode is not None:
            raise UnsupportedCubeCL(f"store mode {mode}")
        dtype = self.check_buffer(name)
        self.args.output(name)
        value = self.convert(value, dtype, semantic=dtype)
        self.operations.append(
            {
                "op": "store",
                "args": [value.ssa],
                "dtype": self.dtype_names[dtype],
                "semantic_dtype": self.dtype_names[dtype],
                "buffer_name": name,
                "index": self.check_index(index),
            }
        )

    def masked(self, mask, body, other):
        if mask.dtype != torch.bool:
            raise UnsupportedCubeCL("masked predicate must be bool")
        outer_ops, outer_cache, outer_cse = (
            self.operations,
            self.semantic_cache,
            self.cse,
        )
        self.operations = []
        self.semantic_cache = dict(outer_cache)
        self.cse = outer_cse.scoped_copy()
        try:
            result = body()
            if isinstance(other, CubeCLValue):
                raise UnsupportedCubeCL("masked fill must be a scalar literal")
            nested = self.operations
        finally:
            self.operations, self.semantic_cache, self.cse = (
                outer_ops,
                outer_cache,
                outer_cse,
            )
        # Fill constants belong outside the branch (their ids are currently in nested).
        # Keep fill construction as a separate else region, never speculate body loads.
        return self.emit(
            "masked",
            mask,
            dtype=result.dtype,
            semantic=result.semantic_dtype,
            body=nested,
            result=result.ssa,
            other=json_float(other) if isinstance(other, float) else other,
        )

    def codegen_kernel(self):
        self.input_names = [
            n for n in self.args.input_buffers if n not in self.removed_buffers
        ]
        self.output_names = [
            n for n in self.args.output_buffers if n not in self.removed_buffers
        ]
        if (
            len(self.input_names) not in self.supported_input_counts
            or len(self.output_names) != 1
            or self.args.inplace_buffers
        ):
            raise UnsupportedCubeCL(
                f"supported input counts {self.supported_input_counts} and one disjoint output required: inputs={self.input_names}, outputs={self.output_names}, inplace={self.args.inplace_buffers}"
            )
        names = self.input_names + self.output_names

        def rewrite(ops):
            result = []
            for raw in ops:
                op = dict(raw)
                if "buffer_name" in op:
                    name = op.pop("buffer_name")
                    if name in self.removed_buffers:
                        if op["op"] == "store":
                            continue
                        raise UnsupportedCubeCL("unresolved fused temporary load")
                    op["buffer"] = names.index(name)
                if "body" in op:
                    op["body"] = rewrite(op["body"])
                result.append(op)
            return result

        # Capture metadata first so every runtime symbol has a stable slot.
        numel = encode(self.numels["x"], self.symbol_slot)
        buffers = []
        for i, n in enumerate(names):
            l = V.graph.get_buffer(n).get_layout()
            buffers.append(
                {
                    "access": "read" if i < len(self.input_names) else "write",
                    "dtype": self.dtype_names[self.check_buffer(n)],
                    "size": [encode(s, self.symbol_slot) for s in l.size],
                    "stride": [encode(s, self.symbol_slot) for s in l.stride],
                    "offset": encode(l.offset, self.symbol_slot),
                }
            )
        self.descriptor = {
            "schema": 2,
            "index_dtype": "int64",
            "numel": numel,
            "scalars": self.scalar_declarations(),
            "buffers": buffers,
            "ops": rewrite(self.operations),
        }
        self.finalize_descriptor()
        self.trace.append(
            {
                "event": "kernel",
                "nodes": [n.get_name() for n in self.features.scheduler_nodes()],
                "descriptor": self.descriptor,
                "physical_plan": self.physical_plan.to_dict(),
                "symbols": [str(s) for s in self.runtime_symbols],
                "replay": self.replay,
                "actual_replay_stack": self.replay_stack,
            }
        )
        return json.dumps(self.descriptor, sort_keys=True)

    def finalize_descriptor(self):
        """Allow a derived semantic frontend to add domains before serialization."""

    def call_kernel(self, name, node=None, deallocate_ws=True):
        scalars = ", ".join(
            f"{s}.item()"
            if isinstance(s, str)
            else V.graph.wrapper_code.codegen_sizevar(s)
            for s in self.runtime_symbols
        )
        V.graph.wrapper_code.writeline(
            f"{name}(({scalars}{',' if len(self.runtime_symbols) == 1 else ''}), {', '.join(self.input_names + self.output_names)})"
        )


class CubeCLScheduling(BaselineScheduling):
    kernel_type = CubeCLKernel

    def can_fuse(self, a, b):
        accepted = fusion_allowed(a, b, TRACE)
        result = accepted and SIMDScheduling.can_fuse(self, a, b)
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
        if reduction_numel != 1:
            raise UnsupportedCubeCL("reductions unsupported")
        TRACE.append(
            {
                "event": "schedule",
                "nodes": [n.get_name() for n in node_schedule],
                "numel": str(numel),
            }
        )
        return {"x": numel}, {}
