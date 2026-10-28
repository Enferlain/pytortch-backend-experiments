"""Inductor kernel and scheduling: replay fused nodes into a typed descriptor.

The kernel records one scalar SSA value per CUDA unit; ``codegen_kernel``
serializes the records with buffer layouts and runtime scalars into the
descriptor that ``cubecl_compile`` (``tuning.compile_kernel``) builds. What may
reach this code is decided earlier, in ``admission``.
"""

import contextlib
import functools
import json
import math
import time
import traceback

import sympy
import torch
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.codegen.simd import SIMDKernel, SIMDScheduling
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.virtualized import V
from torch._prims_common import get_computation_dtype
from torch.utils._ordered_set import OrderedSet

from cubecl_torch.compiler.descriptor import MAX_BUFFERS
from cubecl_torch.compiler.errors import UnsupportedCubeCL
from cubecl_torch.compiler.indexing import encode
from cubecl_torch.compiler.nvrtc import Numerics
from cubecl_torch.compiler.plan import PhysicalKernelPlan, ReductionPlan
from cubecl_torch.inductor import diagnostics
from cubecl_torch.inductor.admission import (
    ALIASES,
    ARITHMETIC,
    FLOATS,
    INTEGERS,
    OP_RULES,
    VALUES,
    WELFORD,
    fusion_allowed,
    scalar_input,
)
from cubecl_torch.inductor.diagnostics import TraceLog

TRACE = TraceLog()
# float64 names runtime scalar values only; it is never storage or arithmetic.
DTYPES = {d: str(d).removeprefix("torch.") for d in (*FLOATS, *INTEGERS, torch.float64)}


class CubeCLValue(CSEVariable):
    """One scalar per CUDA unit, its SSA identity the CSE name. ``dtype`` is the
    compute dtype, ``semantic_dtype`` the dtype the value stands for."""

    def __init__(self, name, bounds, dtype, shape=()):
        super().__init__(name, bounds, dtype=dtype, shape=shape)
        self.ssa = name
        self.semantic_dtype = dtype


def json_float(value):
    """A float literal for the descriptor: non-finite values as strings."""
    value = float(value)
    if math.isfinite(value):
        return value
    return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")


def fill(other, dtype):
    """A masked region's literal fill in the descriptor form of ``dtype``."""
    if dtype == torch.bool:
        return bool(other)
    if dtype in INTEGERS:
        if isinstance(other, float) and not other.is_integer():
            raise UnsupportedCubeCL(f"masked fill {other} is not a {dtype} value")
        return int(other)
    return json_float(other) if isinstance(other, float) else other


# The signed integer of each float format's width: bit-level float ops.
FLOAT_BITS = {
    torch.float32: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
}


def float_bits(x):
    """x's bits in its own format (a float16/bfloat16 value rounded to it first)."""
    return CubeCLOps.to_dtype_bitcast(x, FLOAT_BITS[x.semantic_dtype], x.semantic_dtype)


def compute_bits(x):
    """The bits of x's float32 compute value. Its sign is x's: rounding to a
    16-bit format never changes a sign, but CUDA's conversion makes every NaN
    the positive canonical one, so sign tests skip the round trip, as Triton."""
    return CubeCLOps.to_dtype_bitcast(
        V.kernel.convert(x, torch.float32), torch.int32, torch.float32
    )


def from_bits(bits, dtype):
    """The ``dtype`` value with these bits, in its compute dtype."""
    return CubeCLOps.to_dtype_bitcast(bits, dtype, FLOAT_BITS[dtype])


def promoted(*args):
    """The operands in their common dtype, as a binary op sees them."""
    semantic = functools.reduce(torch.promote_types, (x.semantic_dtype for x in args))
    dtype = get_computation_dtype(semantic)
    return [V.kernel.convert(x, dtype, semantic=semantic) for x in args]


def integral(name):
    """A rounding op: integer operands are already integral, as eager."""

    def op(x):
        return x if x.dtype in INTEGERS else V.kernel.operation(name, x)

    return staticmethod(op)


def integral_to_int(name):
    """Inductor's generic ``*_to_int``: the rounding op, then the cast."""

    def op(x, dtype):
        return CubeCLOps.to_dtype(getattr(CubeCLOps, name)(x), dtype)

    return staticmethod(op)


def canonical_region(record):
    """``record`` with the values its body defines renamed by position.

    Each fused node builds its masked regions under a scoped CSE, so identical
    regions differ only in those names; this key lets them share one value
    (and the reductions over them one reduction, as Triton's reduction cache).
    Only names defined inside the body are renamed: operands from outside keep
    theirs."""
    names = {}

    def collect(ops):
        for op in ops:
            if "id" in op:
                names[op["id"]] = f"%{len(names)}"
            if "body" in op:
                collect(op["body"])

    collect(record["body"])

    def rename(value, key=None):
        if isinstance(value, dict):
            return {k: rename(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [rename(v, key) for v in value]
        if key in ("id", "args", "result") and isinstance(value, str):
            return names.get(value, value)
        return value

    return rename(record)


class CubeCLOps(DefaultHandler):
    floor, ceil, trunc, round = map(integral, ("floor", "ceil", "trunc", "round"))
    floor_to_int, ceil_to_int, trunc_to_int, round_to_int = map(
        integral_to_int, ("floor", "ceil", "trunc", "round")
    )

    def _default(self, name, args, kwargs):
        name = ALIASES.get(name, name)
        if name in OP_RULES or name == "where":
            return V.kernel.operation(name, *args)
        raise UnsupportedCubeCL(f"CubeCL semantic operation {name} unsupported")

    @staticmethod
    def int_truediv(a, b):
        # Inductor's default: true division of the operands as float32, under
        # the float division policy (div.full, or division_rounding).
        return V.kernel.operation(
            "truediv",
            V.kernel.convert(a, torch.float32),
            V.kernel.convert(b, torch.float32),
        )

    @staticmethod
    def to_dtype_bitcast(x, dtype, src_dtype):
        # As Triton: round a float16/bfloat16 value held in float32 compute to
        # its source dtype first, reinterpret, then compute in float32 again.
        # Never to or from bool: a reinterpreted byte need not be 0 or 1.
        if (
            dtype.itemsize != src_dtype.itemsize
            or dtype not in VALUES
            or torch.bool in (dtype, src_dtype)
        ):
            raise UnsupportedCubeCL(f"to_dtype_bitcast {src_dtype} to {dtype}")
        x = V.kernel.convert(x, src_dtype)
        bits = V.kernel.emit("bitcast", x, dtype=dtype, semantic=dtype)
        return V.kernel.convert(bits, get_computation_dtype(dtype), semantic=dtype)

    @staticmethod
    def remainder(a, b):
        # Python's remainder (the divisor's sign) over the C remainder, as
        # eager CUDA: for floats the exact fmod plus the divisor when the signs
        # differ. Inductor's generic form tests signbit over Triton's inexact
        # frem; with a nonzero r, < 0 decides the same, and a NaN r stays NaN.
        r = V.kernel.operation("mod", a, b)
        if r.dtype in INTEGERS and not INTEGERS[r.dtype][0]:
            return r  # unsigned: the remainders agree
        zero = CubeCLOps.constant(0 if r.dtype in INTEGERS else 0.0, r.semantic_dtype)
        signs = V.kernel.operation(
            "ne", V.kernel.operation("lt", r, zero), V.kernel.operation("lt", b, zero)
        )
        adjust = V.kernel.operation(
            "bitwise_and", V.kernel.operation("ne", r, zero), signs
        )
        return V.kernel.operation("where", adjust, V.kernel.operation("add", r, b), r)

    @staticmethod
    def identity(x):
        # x as is (Inductor uses it to trigger CSE, and for sign of a bool).
        return x

    @staticmethod
    def sign(x):
        # Triton's form: int8(x > 0) - int8(x < 0) in x's dtype, so NaN and
        # -0.0 give +0; integers -1, 0 or 1.
        zero = 0 if x.semantic_dtype in INTEGERS else 0.0
        above, below = (
            CubeCLOps.to_dtype(V.kernel.operation(op, x, zero), torch.int8)
            for op in ("gt", "lt")
        )
        return CubeCLOps.to_dtype(
            V.kernel.operation("sub", above, below), x.semantic_dtype
        )

    @staticmethod
    def signbit(x):
        # The sign bit: set for -0.0 and a negative NaN. Integers test < 0
        # (Inductor's lowering answers unsigned and bool itself).
        if x.semantic_dtype not in FLOAT_BITS:
            return V.kernel.operation("lt", x, 0)
        return V.kernel.operation("lt", compute_bits(x), 0)

    @staticmethod
    def copysign(x, y):
        # x's magnitude bits with y's sign bit, as libdevice copysign, on the
        # float32 compute values (float16/bfloat16 round at the store).
        x, y = promoted(x, y)
        dtype = x.semantic_dtype
        width = torch.iinfo(torch.int32)
        bits = V.kernel.operation(
            "bitwise_or",
            V.kernel.operation("bitwise_and", compute_bits(x), width.max),
            V.kernel.operation("bitwise_and", compute_bits(y), width.min),
        )
        result = CubeCLOps.to_dtype_bitcast(bits, torch.float32, torch.int32)
        return V.kernel.convert(result, get_computation_dtype(dtype), semantic=dtype)

    @staticmethod
    def nextafter(x, y):
        # As eager: one step of the operands' own format (float16/bfloat16 in
        # 16 bits, never float32), subnormals kept; x == y gives y, NaN gives NaN.
        x, y = promoted(x, y)
        dtype = x.semantic_dtype
        bx, by = float_bits(x), float_bits(y)
        x, y = from_bits(bx, dtype), from_bits(by, dtype)
        width = torch.iinfo(bx.semantic_dtype)
        # Bits grow away from zero: up when y lies beyond x on x's side.
        away = V.kernel.operation(
            "eq", V.kernel.operation("gt", y, x), V.kernel.operation("gt", x, 0.0)
        )
        step = V.kernel.operation(
            "where",
            away,
            V.kernel.operation("add", bx, 1),
            V.kernel.operation("sub", bx, 1),
        )
        # From zero: the smallest subnormal with y's sign.
        smallest = V.kernel.operation(
            "bitwise_or", V.kernel.operation("bitwise_and", by, width.min), 1
        )
        bits = V.kernel.operation(
            "where", V.kernel.operation("eq", x, 0.0), smallest, step
        )
        result = V.kernel.operation(
            "where", V.kernel.operation("eq", x, y), y, from_bits(bits, dtype)
        )
        unordered = V.kernel.operation(
            "bitwise_or",
            V.kernel.operation("isnan", x),
            V.kernel.operation("isnan", y),
        )
        return V.kernel.operation(
            "where", unordered, V.kernel.operation("add", x, y), result
        )

    @staticmethod
    def ldexp(x, n):
        # musl's scalbnf in float32, selects for its branches: x * 2**n with
        # one rounding, n wrapped to int32 as eager and Triton. Factors of
        # 2**127 and 2**-102 bring n into the normal exponent range first.
        dtype = x.semantic_dtype
        y = V.kernel.convert(x, torch.float32)
        n = CubeCLOps.to_dtype(n, torch.int32)
        for factor, bound, op, shift in (
            (2.0**127, 127, "gt", -127),
            (2.0**127, 127, "gt", -127),
            (2.0**-102, -126, "lt", 102),
            (2.0**-102, -126, "lt", 102),
        ):
            far = V.kernel.operation(op, n, bound)
            y = V.kernel.operation(
                "where", far, V.kernel.operation("mul", y, factor), y
            )
            n = V.kernel.operation("where", far, V.kernel.operation("add", n, shift), n)
        n = V.kernel.operation("maximum", V.kernel.operation("minimum", n, 127), -126)
        exponent = V.kernel.operation(
            "bitwise_left_shift", V.kernel.operation("add", n, 127), 23
        )
        scale = CubeCLOps.to_dtype_bitcast(exponent, torch.float32, torch.int32)
        result = V.kernel.operation("mul", y, scale)
        return V.kernel.convert(result, get_computation_dtype(dtype), semantic=dtype)

    @staticmethod
    def frexp(x):
        # musl's frexpf on the float32 bits (float16/bfloat16 values are exact
        # there): mantissa in [0.5, 1) with x's sign, int32 exponent; a
        # subnormal is scaled by 2**64 first; zeros, inf and NaN give (x, 0).
        dtype = x.semantic_dtype
        # A float16/bfloat16 value computed in the kernel is rounded to its
        # format first, as eager stores it (a float32 0.99999994 is 1.0 there).
        x = V.kernel.convert(from_bits(float_bits(x), dtype), torch.float32)
        subnormal = V.kernel.operation(
            "bitwise_and",
            V.kernel.operation(
                "lt", V.kernel.operation("abs", x), torch.finfo(torch.float32).tiny
            ),
            V.kernel.operation("ne", x, 0.0),
        )
        scaled = V.kernel.operation(
            "where", subnormal, V.kernel.operation("mul", x, 2.0**64), x
        )
        bits = CubeCLOps.to_dtype_bitcast(scaled, torch.int32, torch.float32)
        biased = V.kernel.operation(
            "bitwise_and", V.kernel.operation("bitwise_right_shift", bits, 23), 0xFF
        )
        exponent = V.kernel.operation("sub", biased, 126)
        exponent = V.kernel.operation(
            "where", subnormal, V.kernel.operation("sub", exponent, 64), exponent
        )
        # Keep the sign and fraction (0x807FFFFF), set the exponent of 0.5.
        mantissa = CubeCLOps.to_dtype_bitcast(
            V.kernel.operation(
                "bitwise_or",
                V.kernel.operation("bitwise_and", bits, -0x7F800001),
                0x3F000000,
            ),
            torch.float32,
            torch.int32,
        )
        special = V.kernel.operation(
            "bitwise_or",
            V.kernel.operation("eq", x, 0.0),
            V.kernel.operation(
                "bitwise_or",
                V.kernel.operation("isinf", x),
                V.kernel.operation("isnan", x),
            ),
        )
        mantissa = V.kernel.operation("where", special, x, mantissa)
        zero = CubeCLOps.constant(0, torch.int32)
        exponent = V.kernel.operation("where", special, zero, exponent)
        return (
            V.kernel.convert(mantissa, get_computation_dtype(dtype), semantic=dtype),
            exponent,
        )

    @staticmethod
    def square(x):
        # Inductor's generic form, as Triton: one multiply.
        return V.kernel.operation("mul", x, x)

    @staticmethod
    def relu(x):
        # Inductor's maximum(x, 0), as Triton, with the zero in x's dtype rather
        # than int32 (no promotion of int8/int16): NaN propagates, -0.0 gives +0.0.
        zero = 0 if x.semantic_dtype in INTEGERS else 0.0
        return V.kernel.operation(
            "maximum", x, CubeCLOps.constant(zero, x.semantic_dtype)
        )

    @staticmethod
    def logical_not(x):
        # As Triton: a zero test, which on bool is negation.
        if x.dtype == torch.bool:
            return V.kernel.operation("bitwise_not", x)
        zero = CubeCLOps.constant(0 if x.dtype in INTEGERS else 0.0, x.semantic_dtype)
        return V.kernel.operation("eq", x, zero)

    @staticmethod
    def constant(value, dtype):
        if dtype not in VALUES:
            raise UnsupportedCubeCL(f"constant dtype {dtype}")
        if dtype == torch.bool:
            value = bool(value)
        elif dtype in INTEGERS:
            signed, width = INTEGERS[dtype]
            low, high = (
                (-(1 << (width - 1)), 1 << (width - 1)) if signed else (0, 1 << width)
            )
            if not isinstance(value, int) or not low <= value < high:
                raise UnsupportedCubeCL(f"{dtype} constant out of range")
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
        if dtype == torch.bool:
            if x.dtype == torch.bool:
                return x
            # As Triton: a nonzero test, so NaN is True by definition.
            zero = CubeCLOps.constant(
                0 if x.dtype in INTEGERS else 0.0, x.semantic_dtype
            )
            return V.kernel.operation("ne", x, zero)
        if dtype == torch.uint8 and x.dtype in FLOATS:
            # Through int64 as eager's cast (c10 static_cast_with_inter_type):
            # values wrap instead of saturating at 0 or 255. Triton goes
            # through int16, which saturates beyond +-32767.
            x = V.kernel.convert(x, torch.int64)
        target = get_computation_dtype(dtype) if use_compute_types else dtype
        return V.kernel.convert(x, target, semantic=dtype)

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

    @staticmethod
    def masked(mask, body, other):
        return V.kernel.masked(mask, body, other)

    @staticmethod
    def rand(seed, counter):
        if seed.dtype != torch.int64 or counter.dtype != torch.int32:
            raise UnsupportedCubeCL("unsupported RNG dtype/algorithm contract")
        return V.kernel.emit(
            "rand",
            seed,
            counter,
            dtype=torch.float32,
            semantic=torch.float32,
            algorithm="philox4x32_10_top24",
        )

    @staticmethod
    def indirect_indexing(value, size, check=True):
        return V.kernel.indirect_indexing(value, size, check)

    @staticmethod
    def load_seed(name, offset):
        return V.kernel.load(name, offset)


class CubeCLKernel(SIMDKernel):
    overrides = CubeCLOps
    # Inputs beside at least one output, within descriptor.MAX_BUFFERS.
    supported_input_counts = tuple(range(MAX_BUFFERS))

    def __init__(self, *args, **kwargs):
        self.indirect_values = {}
        self.operations = []
        self.replay = []
        self.replay_stack = []
        self.semantic_cache = {}
        self.runtime_symbols = []
        super().__init__(*args, **kwargs)
        # Descriptor/address arithmetic always uses signed i64, independently of
        # the common SIMD feature probe's conservative 32-bit specialization guards.
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

    def _reject_advanced(self, *args, **kwargs):
        raise UnsupportedCubeCL(
            "scan, sort, bucketize and partial reduction kernels unsupported"
        )

    partial_accumulate = _reject_advanced
    scan = _reject_advanced
    sort = _reject_advanced
    bucketize = _reject_advanced

    def dtype_to_str(self, dtype):
        if dtype in (torch.int32, torch.int64):
            return "index"
        if dtype == torch.float32:
            return "float32"
        raise UnsupportedCubeCL(f"dtype {dtype}")

    def record_op_trace(self, name, args, kwargs, result=None):
        # Inductor calls this after every op; the replay is part of the
        # "kernel" trace event, recorded only while tracing.
        if not diagnostics.tracing():
            return

        def display(v):
            return (
                {
                    "ssa": v.ssa,
                    "dtype": str(v.dtype),
                    "shape": (),
                    "mask": "logical_in_bounds",
                }
                if isinstance(v, CubeCLValue)
                else str(v)
            )

        self.replay.append(
            {
                "op": name,
                "args": [display(v) for v in args],
                "kwargs": {k: display(v) for k, v in kwargs.items()},
                "result": display(result),
            }
        )

    def create_cse_var(self, name, bounds, dtype=None, shape=None):
        if dtype not in (*VALUES, torch.float64) or shape not in (None, ()):
            raise UnsupportedCubeCL(f"CSE dtype/shape {dtype}/{shape}")
        return CubeCLValue(name, bounds, dtype, ())

    def emit(self, op, *args, dtype=torch.float32, semantic=None, **fields):
        if any(not isinstance(x, CubeCLValue) for x in args):
            raise UnsupportedCubeCL(f"untyped operands for {op}")
        semantic = semantic or dtype
        record = {
            "op": op,
            "args": [x.ssa for x in args],
            "dtype": DTYPES[dtype],
            "semantic_dtype": DTYPES[semantic],
            **fields,
        }
        key = json.dumps(
            canonical_region(record) if "body" in record else record, sort_keys=True
        )
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
        # A Python scalar operand takes the first typed operand's dtype.
        typed = next((x for x in args if hasattr(x, "ssa")), None)
        if typed is None:
            raise UnsupportedCubeCL("operation requires a typed operand")
        args = tuple(
            x if hasattr(x, "ssa") else CubeCLOps.constant(x, typed.semantic_dtype)
            for x in args
        )
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
        allowed, result, reason = OP_RULES.get(name, ARITHMETIC)
        semantic = functools.reduce(
            torch.promote_types, (x.semantic_dtype for x in args)
        )
        if semantic not in allowed:
            raise UnsupportedCubeCL(f"{name} over {semantic}: {reason}")
        dtype = get_computation_dtype(semantic)
        args = [self.convert(x, dtype, semantic=semantic) for x in args]
        if result != "same":
            return self.emit(name, *args, dtype=result, semantic=result)
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
        if self.indirect_values:
            return self.indirect_index(expanded)
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

    def indirect_index(self, expanded):
        """An index over tensor-derived values: they read as their SSA ids."""
        names = list(self.indirect_values)
        tree = encode(
            sympy.simplify(expanded),
            lambda s: -1 - names.index(s) if s in names else self.symbol_slot(s),
            self.range_trees[0].index_sym(),
        )

        def rewrite(v):
            if v.get("op") == "scalar" and v["slot"] < 0:
                return {
                    "op": "value",
                    "id": self.indirect_values[names[-1 - v["slot"]]],
                }
            return {
                k: [rewrite(x) for x in a] if k == "args" else a for k, a in v.items()
            }

        return rewrite(tree)

    def indirect_indexing(self, value, size, check=True):
        if value.dtype != torch.int64:
            raise UnsupportedCubeCL("unsupported indirect dtype/unchecked bounds")
        value = self.emit(
            "indirect",
            value,
            dtype=torch.int64,
            semantic=torch.int64,
            size=self.check_index(size),
            bounds="require_in_range",
        )
        symbol = sympy.Symbol(value.ssa, integer=True)
        self.indirect_values[symbol] = value.ssa
        return symbol

    def indirect_load(self, name, index):
        return self.load(name, index)

    def check_bounds(self, expr, size, lower, upper):
        # The explicit indirect op checks both bounds before the address is used.
        return None

    def check_buffer(self, name):
        dtype = V.graph.get_dtype(name)
        if dtype not in (*FLOATS, torch.bool, torch.int64):
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
            return self.atomic_store(name, index, value, mode)
        dtype = self.check_buffer(name)
        self.args.output(name)
        value = self.convert(value, dtype, semantic=dtype)
        record = {
            "op": "store",
            "args": [value.ssa],
            "dtype": DTYPES[dtype],
            "semantic_dtype": DTYPES[dtype],
            "buffer_name": name,
            "index": self.check_index(index),
        }
        if self.reduction_kernel:
            record["iteration_scope"] = (
                "reduction" if self.inside_reduction else "outer"
            )
        self.operations.append(record)
        return None

    def atomic_store(self, name, index, value, mode):
        if (
            mode != "atomic_add"
            or self.reduction_kernel
            or self.check_buffer(name) != torch.float32
        ):
            raise UnsupportedCubeCL(
                "unsupported atomic operation/dtype/physical topology"
            )
        self.args.output(name)
        value = self.convert(value, torch.float32, semantic=torch.float32)
        self.operations.append(
            {
                "op": "atomic_add",
                "args": [value.ssa],
                "dtype": "float32",
                "semantic_dtype": "float32",
                "buffer_name": name,
                "index": self.check_index(index),
                "effect": "atomic_rmw",
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
            other=fill(other, result.semantic_dtype),
        )

    @contextlib.contextmanager
    def disable_reduction(self):
        prior = self.inside_reduction
        self.inside_reduction = False
        try:
            yield
        finally:
            self.inside_reduction = prior

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
        # A fused consumer sees the float32 result, as in Triton; a stored
        # result rounds at its store. Eager's rounding before reuse comes back
        # with Triton's switch, emulate_precision_casts (multi-output-kernels
        # group 5: the rounded bf16 row mean of a fused normalization was
        # measurably less accurate than Triton's).
        if dtype != torch.float32 and config.emulate_precision_casts:
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

    def store_reduction(self, name, index, value):
        with self.disable_reduction():
            self.store(name, index, value)

    def codegen_kernel(self):
        began = time.perf_counter()

        # KernelArgs canonicalizes mutation versions to physical storage names.
        # Match its existing mapping in serialization; CSE retains logical names.
        def canonicalize(ops):
            for op in ops:
                if "buffer_name" in op:
                    name = op["buffer_name"]
                    op["buffer_name"] = V.graph.scheduler.mutation_real_name.get(
                        name, name
                    )
                if "body" in op:
                    canonicalize(op["body"])

        canonicalize(self.operations)
        if diagnostics.tracing():
            TRACE.append(
                {
                    "event": "argument_bookkeeping",
                    "inputs": {
                        k: v if isinstance(v, str) else type(v).__name__
                        for k, v in self.args.input_buffers.items()
                    },
                    "outputs": {
                        k: v if isinstance(v, str) else type(v).__name__
                        for k, v in self.args.output_buffers.items()
                    },
                    "removed": list(self.removed_buffers),
                    "operations": self.operations.copy(),
                }
            )
        self.input_names = [
            n for n in self.args.input_buffers if n not in self.removed_buffers
        ]
        self.output_names = [
            n for n in self.args.output_buffers if n not in self.removed_buffers
        ]
        if self.args.inplace_buffers:
            # backend.py sets inplace_buffers=False, so Inductor never makes an
            # output take over an input's storage (bead 4z8.31).
            raise UnsupportedCubeCL(
                f"in-place buffers unexpected with inplace_buffers=False: {self.args.inplace_buffers}"
            )
        if (
            len(self.input_names) not in self.supported_input_counts
            or not self.output_names
            or len(self.input_names) + len(self.output_names) > MAX_BUFFERS
        ):
            raise UnsupportedCubeCL(
                f"at most {MAX_BUFFERS} buffers with at least one output required: inputs={self.input_names}, outputs={self.output_names}"
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
                    "dtype": DTYPES[self.check_buffer(n)],
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
        if self.reduction_kernel:
            # Schema 3: the semantic reduction domain, apart from the physical plan.
            self.descriptor.update(
                schema=3,
                outer=self.descriptor["numel"],
                reduction_extent=encode(self.numels["r0_"], self.symbol_slot),
            )
            self.descriptor["scalars"] = self.scalar_declarations()
        else:
            # Schema 4: schema 2 plus the kernel's effects.
            self.descriptor["schema"] = 4
            self.descriptor["effects"] = {
                "rng": "explicit_seed_counter"
                if any(op["op"] == "rand" for op in self.operations)
                else None,
                "mutation": any(op["op"] == "atomic_add" for op in self.operations),
            }
        TRACE.append(
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
        result = json.dumps(self.descriptor, sort_keys=True)
        TRACE.append(
            {
                "event": "descriptor_construction",
                "elapsed_ms": (time.perf_counter() - began) * 1000,
                "scope": "layout/symbol binding, descriptor rewrite/finalization/serialization; semantic replay already ran",
            }
        )
        return result

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


class CubeCLScheduling(SIMDScheduling):
    kernel_type = CubeCLKernel

    def __init__(self, scheduler, *, tuning):
        self.tuning = tuning
        super().__init__(scheduler)

    def check_backend_available(self, device):
        from cubecl_torch.inductor.availability import check_available

        check_available(device)
        TRACE.append(
            {
                "event": "availability",
                "backend": type(self).__name__,
                "device": str(device),
            }
        )

    @classmethod
    def get_backend_features(cls, device):
        return OrderedSet()

    def can_fuse(self, a, b):
        return fusion_allowed(a, b) and super().can_fuse(a, b)

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

    def codegen_combo_kernel(self, *args, **kwargs):
        raise UnsupportedCubeCL("combo kernels unsupported")

    def codegen_template(self, *args, **kwargs):
        raise UnsupportedCubeCL("template kernels unsupported")

    def define_kernel(self, src_code, node_schedule, kernel):
        wrapper = V.graph.wrapper_code
        name = "cubecl_kernel_" + str(wrapper.next_kernel_suffix())
        # The same config switches Triton's fp fusion and division rounding.
        # A two-pass Welford kernel divides once per row: its mean must be
        # correctly rounded so constant rows give variance exactly 0 (a
        # reciprocal multiply is off by an ulp, then contracted into x - mean).
        numerics = Numerics(
            fp_contract=not config.emulate_precision_casts,
            exact_division=config.eager_numerics.division_rounding
            or getattr(kernel, "welford_two_pass", False),
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
                # Both descriptors must already exist before adding guards:
                # earlier equalities would specialize the final's scalar ABI.
                for producer in records:
                    if (
                        role(producer["semantic"]) != "producer"
                        or "shape_admitted" in producer
                    ):
                        continue
                    # symbol_slot order: template slot0=N, slot1=B.
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
