"""Inductor backend contract coverage: every item a full backend must handle.

The checklist is derived from the pinned Inductor source, not from workloads:
every ``OpsHandler`` method, every ``ReductionType`` and every ``BackendFeature``
is assigned to a family with its current status, plus the surfaces Inductor does
not enumerate in one place (dtypes, indexing, scheduling, runtime). Families
name their prerequisites; that order is the work order (tracked in beads).

``python -m cubecl_torch.testing.coverage_inventory`` regenerates
``coverage_inventory.md``; ``test_coverage_inventory.py`` fails when the pinned
Inductor has an item this module does not classify, or when a status claims
support the backend's accepted-op sets do not show.
"""

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MARKDOWN = HERE / "coverage_inventory.md"
INDUCTOR = REPO / "stage0" / "python-deps" / "torch" / "_inductor"

STATUS_MEANING = {
    "implemented": "works for the declared domain in its note (dtypes, modes, "
    "exclusions); says nothing about the rest of Inductor's contract for the item",
    "complete": "every dtype, mode and argument form Inductor can request for the "
    "item, under the numerical policy (family policy), with tests; none yet",
    "missing": "rejected by the capability checks",
    "n/a": "not applicable to this backend",
}

# family: (title, prerequisite families, beads, scope note)
FAMILIES = {
    "policy": (
        "Numerical policy: Inductor+Triton as a floor, per-family contracts",
        (),
        ("4z8.17",),
        (
            "Never measurably worse than Triton; keep proven improvements. Fused math "
            "rounds once to storage, graph casts round, FMA contraction, div.full "
            "division; Inductor's emulation/division switches restore eager's."
        ),
    ),
    "core": (
        "Memory, control flow and conversions",
        (),
        ("4z8.23", "4z8.24", "4z8.25", "4z8.30"),
        "Loads/stores, masks, where, constants, index values, casts, indirect indexing.",
    ),
    "int": (
        "Integer, bool and bitwise value arithmetic",
        ("policy",),
        ("4x9.1",),
        "Values carry their width and signedness (bool, int8/16/32/64, uint8); storage beyond bool/int64 is the dtypes family.",
    ),
    "basic": (
        "Basic float arithmetic",
        ("policy",),
        ("4x9.2",),
        "The common pointwise float ops.",
    ),
    "extrema": (
        "Extrema, rounding and remainder",
        ("policy",),
        ("x3c",),
        "NaN/signed-zero/rounding contracts must be stated before these ops.",
    ),
    "transcendental": (
        "Transcendental float math",
        ("policy",),
        ("4x9.3",),
        "Each op needs a stated accuracy contract against Triton/libdevice.",
    ),
    "special": (
        "Special functions",
        ("policy", "transcendental"),
        ("4x9.4",),
        "Bessel, gamma-family, error-function variants, orthogonal polynomials.",
    ),
    "rng": (
        "Random number generation",
        ("int",),
        ("4x9.5",),
        "rand and load_seed exist; randn and randint64 do not.",
    ),
    "reduce": (
        "Reduction framework",
        ("policy",),
        ("4x9.6", "4z8.21", "4x9.18"),
        (
            "Reduction types as combiners, multi-value accumulators, combine "
            "functions for split plans."
        ),
    ),
    "reduce-welford": (
        "Welford mean/variance reductions",
        ("reduce",),
        ("4z8.15",),
        "var, std, var_mean, every normalization.",
    ),
    "reduce-extrema": (
        "Max/min reductions",
        ("reduce", "extrema"),
        ("4x9.7",),
        "softmax, cross-entropy, amax scaling.",
    ),
    "reduce-arg": (
        "Argmax/argmin reductions",
        ("reduce", "reduce-extrema", "int"),
        ("4x9.8",),
        "Value plus index accumulators.",
    ),
    "reduce-other": (
        "any, prod and xor_sum reductions",
        ("reduce", "int"),
        ("4x9.9",),
        "any covers GradScaler/isfinite checks.",
    ),
    "reduce-softmax": (
        "Online softmax reductions",
        ("reduce", "reduce-extrema", "transcendental"),
        ("4x9.10",),
        "Inductor's fused max+sum softmax lowering.",
    ),
    "index": (
        "Index-expression and reduction-shape completeness",
        (),
        ("4z8.18", "4z8.8", "4z8.7", "4z8.22", "4z8.26"),
        "Max/Min and Abs in layouts and indices, provable divisor positivity, float constants in index math, multi-dim reductions.",
    ),
    "dtypes": (
        "Storage dtypes beyond float32/16/bf16, bool and int64",
        ("int", "policy"),
        ("4x9.11",),
        "float64, fp8, int8/16/32, uint8 storage.",
    ),
    "effects": (
        "Mutation, in-place buffers and scatter",
        (),
        ("4z8.9", "fm5", "4x9.12"),
        "User-visible mutation, Inductor buffer reuse, masked scatter, determinism.",
    ),
    "scan": (
        "Scan, sort and bucketize kernels",
        ("reduce", "int"),
        ("4x9.13",),
        "New kernel shapes with their own combine semantics.",
    ),
    "templates": (
        "Matmul/convolution templates with fused epilogues",
        ("reduce",),
        ("4x9.14",),
        "The CubeK path; needs the artifact contract extended.",
    ),
    "scheduling": (
        "Scheduling features: foreach/combo kernels, tiling, store loop order",
        (),
        ("4x9.15",),
        "Mostly performance: correct results exist without them.",
    ),
    "runtime": (
        "Runtime: CUDA graphs, fallback, platforms, compile-time behaviour",
        (),
        ("4z8.10", "4z8.3", "s2m", "9zp", "dm4"),
        "Cross-cutting; fallback has no prerequisite and decides how gaps surface.",
    ),
    "n/a": (
        "Not applicable to this backend",
        (),
        (),
        "Backend-specific hooks of other code generators, or FX plumbing.",
    ),
}

# Every OpsHandler method -> (family, status, note). Statuses: STATUS_MEANING;
# implemented items must state their domain in the note.
OPS = {
    # core
    "constant": (
        "core",
        "implemented",
        "float (finite, nan, inf), bool and integer literals (int8/16/32/64, uint8, range-checked per width)",
    ),
    "load": (
        "core",
        "implemented",
        "float32/16/bf16, bool, int64 storage; 0-d CPU float64 inputs as scalar values",
    ),
    "store": (
        "core",
        "implemented",
        "plain stores of float32/16/bf16, bool, int64; atomic_add float32 only (nondeterministic); no other store modes",
    ),
    "masked": (
        "core",
        "implemented",
        "bool mask, scalar-literal fill (float, bool or integer, range-checked) only; not around reductions or stores",
    ),
    "where": (
        "core",
        "implemented",
        "bool predicate; float, integer or bool operands promoted as torch.promote_types, floats computed in float32",
    ),
    "index_expr": (
        "core",
        "implemented",
        "index values: int64 arithmetic, then cast to the requested integer width or floating dtype",
    ),
    "value_expr": (
        "core",
        "implemented",
        "index values: int64 arithmetic, then cast to the requested integer width or floating dtype",
    ),
    "to_dtype": (
        "core",
        "implemented",
        "between float32/16/bf16, bool and int8/16/32/64/uint8: float to integer truncates and wraps as eager (through int32, uint8 through int64; NaN and out-of-range included, where Triton saturates), narrowing wraps, to bool is != 0; float64 scalar values to float only",
    ),
    "identity": ("core", "missing", ""),
    "indirect_indexing": (
        "core",
        "implemented",
        "int64 indices, bounds checked on device; not in reductions or masked regions; no indirect quotient/modulo",
    ),
    "check_bounds": (
        "core",
        "implemented",
        "emits nothing: indirect indexing checks its own bounds, and only check_bounds without requested bounds is admitted (as Triton); requested bounds rejected (4z8.30)",
    ),
    "store_reduction": ("core", "implemented", "results of the implemented reductions"),
    "partial_accumulate": ("core", "missing", ""),
    "device_assert_async": ("core", "missing", ""),
    "load_seed": ("rng", "implemented", "int64 seed buffer"),
    "rand": (
        "rng",
        "implemented",
        "philox4x32-10 top-24-bit float32; extent below 2^31; not in reductions or masked regions",
    ),
    "rand_eager": ("rng", "missing", ""),
    "randn": ("rng", "missing", ""),
    "randint64": ("rng", "missing", ""),
    # conversions to integer
    "trunc_to_int": ("extrema", "missing", ""),
    "ceil_to_int": ("extrema", "missing", ""),
    "floor_to_int": ("extrema", "missing", ""),
    "round_to_int": ("extrema", "missing", ""),
    "to_dtype_bitcast": (
        "int",
        "implemented",
        "same-size reinterpretation; float16/bfloat16 values rounded to the source dtype first, as Triton",
    ),
    # reductions and kernel classes
    "reduction": (
        "reduce",
        "implemented",
        (
            "sum, max, min: float32/16/bf16 inputs, float32 accumulation; Welford: "
            "float32; one reduction dimension, at most 4 per kernel"
        ),
    ),
    "scan": ("scan", "missing", ""),
    "sort": ("scan", "missing", ""),
    "bucketize": ("scan", "missing", ""),
    "dot": ("templates", "missing", "native matmul lowering"),
    # basic float arithmetic
    "add": (
        "basic",
        "implemented",
        "float32/float16/bfloat16 values computed in float32; no float64; may contract with a product (FMA) unless precision emulation; integers (int8/16/32/64, uint8) at their own width, wrapping modulo 2**width; not bool",
    ),
    "sub": (
        "basic",
        "implemented",
        "float32/float16/bfloat16 values computed in float32; no float64; may contract with a product (FMA) unless precision emulation; integers (int8/16/32/64, uint8) at their own width, wrapping modulo 2**width; not bool",
    ),
    "mul": (
        "basic",
        "implemented",
        "float32/float16/bfloat16 values computed in float32; no float64; may contract with an add (FMA) unless precision emulation; integers (int8/16/32/64, uint8) at their own width, wrapping modulo 2**width; not bool",
    ),
    "truediv": (
        "basic",
        "implemented",
        "float32/float16/bfloat16 values computed in float32; no float64; div.full (up to 2 ulp) as Triton; correctly rounded under eager_numerics.division_rounding",
    ),
    "neg": (
        "basic",
        "implemented",
        "float32/float16/bfloat16 values computed in float32; no float64; per the numerical policy; integers (int8/16/32/64, uint8) at their own width, wrapping modulo 2**width; not bool",
    ),
    "reciprocal": (
        "basic",
        "implemented",
        "float32/float16/bfloat16 values computed in float32; no float64; div.full (up to 2 ulp) as Triton; correctly rounded under eager_numerics.division_rounding",
    ),
    "sqrt": (
        "basic",
        "implemented",
        "float32/float16/bfloat16 values computed in float32; no float64; matched Triton bit for bit in the 4z8.17 probe",
    ),
    "rsqrt": (
        "basic",
        "implemented",
        "float32/float16/bfloat16 values computed in float32; no float64; per the numerical policy",
    ),
    "exp": (
        "basic",
        "implemented",
        "float32/float16/bfloat16 values computed in float32; no float64; matched Triton bit for bit in the 4z8.17 probe",
    ),
    "sin": (
        "basic",
        "implemented",
        "float32/float16/bfloat16 values computed in float32; no float64; matched Triton bit for bit in the 4z8.17 probe",
    ),
    "cos": (
        "basic",
        "implemented",
        "float32/float16/bfloat16 values computed in float32; no float64; matched Triton bit for bit in the 4z8.17 probe",
    ),
    "abs": (
        "basic",
        "implemented",
        "float32/float16/bfloat16 values computed in float32: sign bit cleared (abs.f32; NaN stays NaN, payload open); integers (int8/16/32/64) wrap at the minimum as eager, uint8 unchanged; not bool",
    ),
    "square": (
        "basic",
        "implemented",
        "x * x as Triton (contracts with a following add like any multiply); float32/float16/bfloat16 computed in float32, integers wrapping modulo 2**width; not bool",
    ),
    "relu": (
        "basic",
        "implemented",
        "maximum(x, 0) with the zero in x's dtype, as Triton: NaN propagates (payload open), -0.0 gives +0.0 (eager gives -0.0); float32/16/bf16 and integers; not bool",
    ),
    "fma": (
        "basic",
        "implemented",
        "float32/float16/bfloat16 values computed in float32, rounded once (CUDA fma) whatever the contraction setting; constant folding rounds once too; not integers",
    ),
    "div_rn": (
        "basic",
        "implemented",
        "float32/float16/bfloat16 values computed in float32: IEEE division (__fdiv_rn) under every division policy; not integers",
    ),
    "mul_rn": (
        "basic",
        "implemented",
        "float32/float16/bfloat16 values computed in float32: a product rounded on its own (__fmul_rn), never contracted into an add, subnormals kept as eager (Triton flushes them); not integers",
    ),
    "pow": ("transcendental", "missing", ""),
    # comparisons and predicates
    "eq": (
        "core",
        "implemented",
        "float32/16/bf16 (compared in float32), integer (signed or unsigned by dtype) or bool (False < True) operands; bool result",
    ),
    "ne": (
        "core",
        "implemented",
        "float32/16/bf16 (compared in float32), integer (signed or unsigned by dtype) or bool (False < True) operands; bool result",
    ),
    "lt": (
        "core",
        "implemented",
        "float32/16/bf16 (compared in float32), integer (signed or unsigned by dtype) or bool (False < True) operands; bool result",
    ),
    "le": (
        "core",
        "implemented",
        "float32/16/bf16 (compared in float32), integer (signed or unsigned by dtype) or bool (False < True) operands; bool result",
    ),
    "gt": (
        "core",
        "implemented",
        "float32/16/bf16 (compared in float32), integer (signed or unsigned by dtype) or bool (False < True) operands; bool result",
    ),
    "ge": (
        "core",
        "implemented",
        "float32/16/bf16 (compared in float32), integer (signed or unsigned by dtype) or bool (False < True) operands; bool result",
    ),
    "isinf": ("extrema", "missing", ""),
    "isnan": ("extrema", "implemented", "float32/16/bf16 operands, bool result"),
    "signbit": ("extrema", "missing", ""),
    "logical_and": (
        "int",
        "implemented",
        "bool operands (Inductor converts the aten op's inputs); bool result",
    ),
    "logical_or": (
        "int",
        "implemented",
        "bool operands (Inductor converts the aten op's inputs); bool result",
    ),
    "logical_not": (
        "int",
        "implemented",
        "bool operands negate; other values test == 0, as Triton",
    ),
    "logical_xor": (
        "int",
        "implemented",
        "bool operands (Inductor converts the aten op's inputs); bool result",
    ),
    # extrema, rounding, remainder
    "minimum": (
        "extrema",
        "implemented",
        "float32/16/bf16 compute: NaN-propagating, -0 < +0 on sm_80+ (PTX min.NaN); integers signed or unsigned by dtype; bool as or/and",
    ),
    "maximum": (
        "extrema",
        "implemented",
        "float32/16/bf16 compute: NaN-propagating, -0 < +0 on sm_80+ (PTX max.NaN); integers signed or unsigned by dtype; bool as or/and",
    ),
    "fmaximum": (
        "reduce-softmax",
        "missing",
        "combine of the fmax reduction; only producer is the two-pass softmax fallback",
    ),
    "floor": ("extrema", "missing", ""),
    "ceil": ("extrema", "missing", ""),
    "trunc": ("extrema", "missing", ""),
    "round": ("extrema", "missing", ""),
    "sign": ("extrema", "missing", ""),
    "copysign": ("extrema", "missing", ""),
    "fmod": (
        "extrema",
        "implemented",
        (
            "float32/16/bf16: exact truncating remainder (CUDA fmod), subnormals kept, as eager; "
            "Triton flushes float32/bf16 subnormal operands"
        ),
    ),
    "remainder": (
        "extrema",
        "implemented",
        (
            "floats as eager: exact fmod plus the divisor when the signs differ (Triton's frem "
            "remainder is inexact for large quotients and loses -0.0); integers: the divisor's "
            "sign, as eager"
        ),
    ),
    "mod": (
        "extrema",
        "implemented",
        "float32/16/bf16: exact fmod; integers: C remainder, zero divisor unspecified",
    ),
    "nextafter": ("extrema", "missing", ""),
    "frexp": ("extrema", "missing", ""),
    "ldexp": ("extrema", "missing", ""),
    # integer and bitwise
    "bitwise_and": (
        "int",
        "implemented",
        "bool and int8/16/32/64/uint8 operands; xor on bool is !=",
    ),
    "bitwise_or": (
        "int",
        "implemented",
        "bool and int8/16/32/64/uint8 operands; xor on bool is !=",
    ),
    "bitwise_xor": (
        "int",
        "implemented",
        "bool and int8/16/32/64/uint8 operands; xor on bool is !=",
    ),
    "bitwise_not": (
        "int",
        "implemented",
        "int8/16/32/64/uint8 complement; on bool, logical negation (never C++ ~)",
    ),
    "bitwise_left_shift": (
        "int",
        "implemented",
        "int8/16/32/64/uint8; amounts below 0 or at least the width give 0, or the sign fill shifting right, as eager",
    ),
    "bitwise_right_shift": (
        "int",
        "implemented",
        "int8/16/32/64/uint8; amounts below 0 or at least the width give 0, or the sign fill shifting right, as eager",
    ),
    "and_": (
        "int",
        "implemented",
        "Inductor's operator spelling of bitwise_and: bool and integer operands",
    ),
    "or_": (
        "int",
        "implemented",
        "Inductor's operator spelling of bitwise_or: bool and integer operands",
    ),
    "xor": (
        "int",
        "implemented",
        "Inductor's operator spelling of bitwise_xor: bool and integer operands",
    ),
    "lshift": (
        "int",
        "implemented",
        "Inductor's operator spelling of bitwise_left_shift",
    ),
    "rshift": (
        "int",
        "implemented",
        "Inductor's operator spelling of bitwise_right_shift",
    ),
    "truncdiv": (
        "int",
        "implemented",
        "int8/16/32/64/uint8 C division; a zero divisor gives an unspecified value, never a trap",
    ),
    "floordiv": (
        "int",
        "implemented",
        "int8/16/32/64/uint8 floor division as c10::div_floor_integer; a zero divisor gives an unspecified value, never a trap",
    ),
    "int_truediv": (
        "int",
        "implemented",
        "integer operands divided as float32 under the float division policy; no aten lowering of the pinned Inductor emits it",
    ),
    # transcendental
    "exp2": ("transcendental", "missing", ""),
    "expm1": ("transcendental", "missing", ""),
    "log": ("transcendental", "missing", ""),
    "log2": ("transcendental", "missing", ""),
    "log10": ("transcendental", "missing", ""),
    "log1p": ("transcendental", "missing", ""),
    "tan": ("transcendental", "missing", ""),
    "tanh": ("transcendental", "missing", ""),
    "sigmoid": ("transcendental", "missing", ""),
    "sinh": ("transcendental", "missing", ""),
    "cosh": ("transcendental", "missing", ""),
    "asin": ("transcendental", "missing", ""),
    "acos": ("transcendental", "missing", ""),
    "atan": ("transcendental", "missing", ""),
    "atan2": ("transcendental", "missing", ""),
    "asinh": ("transcendental", "missing", ""),
    "acosh": ("transcendental", "missing", ""),
    "atanh": ("transcendental", "missing", ""),
    "erf": ("transcendental", "missing", ""),
    "erfc": ("transcendental", "missing", ""),
    "erfinv": ("transcendental", "missing", ""),
    "lgamma": ("transcendental", "missing", ""),
    "hypot": ("transcendental", "missing", ""),
    # special functions
    **dict.fromkeys(
        [
            "airy_ai",
            "bessel_j0",
            "bessel_j1",
            "bessel_y0",
            "bessel_y1",
            "digamma",
            "erfcx",
            "igamma",
            "igammac",
            "gammainc",
            "gammaincc",
            "i0",
            "i0e",
            "i1",
            "i1e",
            "log_ndtr",
            "modified_bessel_i0",
            "modified_bessel_i1",
            "modified_bessel_k0",
            "modified_bessel_k1",
            "ndtr",
            "ndtri",
            "polygamma",
            "scaled_modified_bessel_k0",
            "scaled_modified_bessel_k1",
            "spherical_bessel_j0",
            "zeta",
            "chebyshev_polynomial_t",
            "chebyshev_polynomial_u",
            "chebyshev_polynomial_v",
            "chebyshev_polynomial_w",
            "legendre_polynomial_p",
            "shifted_chebyshev_polynomial_t",
            "shifted_chebyshev_polynomial_u",
            "shifted_chebyshev_polynomial_v",
            "shifted_chebyshev_polynomial_w",
            "hermite_polynomial_h",
            "hermite_polynomial_he",
            "laguerre_polynomial_l",
        ],
        ("special", "missing", ""),
    ),
    # not applicable
    "halide_clamp": ("n/a", "n/a", "Halide backend only"),
    "inline_asm_elementwise": ("n/a", "n/a", "Triton inline assembly"),
    "output": ("n/a", "n/a", "FX graph plumbing"),
    "placeholder": ("n/a", "n/a", "FX graph plumbing"),
}

REDUCTIONS = {
    "sum": (
        "reduce",
        "implemented",
        "float32/16/bf16 inputs, float32 accumulation; extents above 1048576 via guarded split",
    ),
    "prod": ("reduce-other", "missing", ""),
    "any": ("reduce-other", "missing", ""),
    "xor_sum": ("reduce-other", "missing", ""),
    "max": (
        "reduce-extrema",
        "implemented",
        (
            "float32/16/bf16 inputs, float32 accumulation; NaN-propagating, -0 < +0 on "
            "sm_80+; bitwise equal to Triton; every non-sum-only plan, split, dynamic"
        ),
    ),
    "min": (
        "reduce-extrema",
        "implemented",
        (
            "float32/16/bf16 inputs, float32 accumulation; NaN-propagating, -0 < +0 on "
            "sm_80+; bitwise equal to Triton; every non-sum-only plan, split, dynamic"
        ),
    ),
    "fmax": (
        "reduce-softmax",
        "missing",
        "only producer is Inductor's two-pass softmax fallback",
    ),
    "argmax": ("reduce-arg", "missing", ""),
    "argmin": ("reduce-arg", "missing", ""),
    "argmax_value": ("reduce-arg", "missing", ""),
    "argmin_value": ("reduce-arg", "missing", ""),
    "argmax_with_value": ("reduce-arg", "missing", ""),
    "argmin_with_value": ("reduce-arg", "missing", ""),
    "welford_reduce": (
        "reduce-welford",
        "implemented",
        (
            "float32 (Inductor upcasts inputs); one reduction dimension, extent <= "
            "1048576 per kernel; Triton's two-pass on persistent-size rows; every "
            "non-sum-only plan, split, dynamic; one external output per kernel "
            "(4z8.21); dynamic GroupNorm indexing pending (4z8.22)"
        ),
    ),
    "welford_combine": (
        "reduce-welford",
        "implemented",
        (
            "float32 (mean, m2, weight) partials: split second stages and masked "
            "first stages; every non-sum-only plan, split, dynamic"
        ),
    ),
    "online_softmax_reduce": ("reduce-softmax", "missing", ""),
    "dot": ("templates", "missing", ""),
}

FEATURES = {
    "FOREACH": ("scheduling", "missing", "combo kernels are disabled"),
    "BUCKETIZE": ("scan", "missing", ""),
    "INPLACE_BUFFERS": ("effects", "missing", "inplace_buffers is disabled"),
    "MASKED_SCATTER_WITH_INDEX": ("effects", "missing", ""),
    "SCAN": ("scan", "missing", ""),
    "SORT": ("scan", "missing", ""),
    "TUPLE_REDUCTION": (
        "scan",
        "missing",
        "only gates multi-output scans in the pinned Inductor",
    ),
    "PREFER_STORE_LOOP_ORDER": ("scheduling", "missing", ""),
    "TRITON_TEMPLATES": ("templates", "missing", ""),
    "REDUCE_TO_SINGLE_ELEMENT": (
        "reduce",
        "n/a",
        (
            "not advertised; Triton only advertises it under config.triton."
            "cooperative_reductions (default off); Inductor splits instead (4x9.16)"
        ),
    ),
}

# Surfaces Inductor does not enumerate in one interface.
SURFACES = {
    "float32/float16/bfloat16 storage": (
        "core",
        "implemented",
        "direct strided, nonnegative strides",
    ),
    "bool storage": (
        "core",
        "implemented",
        "load/store and predicates; no bool arithmetic",
    ),
    "int64 storage": (
        "core",
        "implemented",
        "load/store; values feed index arithmetic only",
    ),
    "float64 storage": ("dtypes", "missing", "float64 runtime scalar values exist"),
    "fp8 storage (e4m3fn, e5m2)": ("dtypes", "missing", ""),
    "int8/int16/int32/uint8 storage": ("dtypes", "missing", ""),
    "complex dtypes": ("n/a", "n/a", "Inductor falls back to ATen"),
    "complex graph inputs feeding ATen fallbacks": (
        "core",
        "missing",
        "rejected at backend selection (Wan 2.2, Qwen-Image 2.1 RoPE; 4z8.24)",
    ),
    "rank 0..25 tensors": ("core", "implemented", "MAX_RANK sanity bound"),
    "dynamic shapes, int64 shape scalars": (
        "core",
        "implemented",
        "sizes/strides as int64 launch scalars; no Max/Min (4z8.18)",
    ),
    "lifted Python floats (0-d CPU float64)": (
        "core",
        "implemented",
        "float64 scalar values feeding a cast to float compute",
    ),
    "Max/Min in layout expressions": ("index", "missing", "4z8.18"),
    "Abs in index expressions (reflect padding)": ("index", "missing", "4z8.26"),
    "float constants in index arithmetic": ("index", "missing", "4z8.8"),
    "multi-dimension reductions": ("index", "missing", "4z8.7"),
    "split/cooperative long reductions": (
        "reduce",
        "implemented",
        (
            "guarded split for float sum, max and min, static and dynamic shapes, "
            "including extents the split does not divide"
        ),
    ),
    "indirect indexing inside reductions": (
        "core",
        "missing",
        "gather-then-reduce: LM cross-entropy, per-sample modulation in packed batches (4z8.25)",
    ),
    "more than 4-5 direct inputs per kernel": ("core", "missing", "4z8.23"),
    "multi-output kernels": ("reduce", "missing", "4z8.21"),
    "CPU-only graphs": (
        "runtime",
        "missing",
        "host-side bookkeeping graphs (Wan 2.2); 4z8.3",
    ),
    "user-visible mutation in graphs": ("effects", "missing", "4z8.9"),
    "deterministic scatter backward": ("effects", "missing", "fm5"),
    "2-D tiling": ("scheduling", "missing", "every kernel is one flattened range"),
    "CUDA graph capture (reduce-overhead)": ("runtime", "missing", "4z8.10"),
    "graceful fallback for unsupported kernels": ("runtime", "missing", "4z8.3"),
    "native Windows": ("runtime", "missing", "s2m"),
    "compilation during training": ("runtime", "missing", "9zp"),
    "bounded metadata retention": ("runtime", "missing", "dm4"),
}


def _source():
    tree = ast.parse((INDUCTOR / "ops_handler.py").read_text())
    ops, reductions = [], []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "OpsHandler":
            ops = [
                n.name
                for n in node.body
                if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
            ]
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "ReductionType"
        ):
            reductions = [e.value for e in node.value.slice.elts]
    tree = ast.parse((INDUCTOR / "codegen" / "common.py").read_text())
    features = next(
        [
            t.targets[0].id
            for t in n.body
            if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name)
        ]
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "BackendFeature"
    )
    return ops, reductions, features


def pinned_items():
    """The pinned Inductor's ops, reduction types and backend features."""
    return _source()


def render():
    sections = (
        ("Ops (`OpsHandler`)", OPS),
        ("Reduction types (`ReductionType`)", REDUCTIONS),
        ("Backend features (`BackendFeature`)", FEATURES),
        ("Other surfaces", SURFACES),
    )
    tables = [
        *OPS.values(),
        *REDUCTIONS.values(),
        *FEATURES.values(),
        *SURFACES.values(),
    ]
    lines = [
        "# Inductor backend contract coverage",
        "",
        "Generated by `python -m cubecl_torch.testing.coverage_inventory`; do not edit by hand.",
        "",
        (
            "Every op of Inductor's `OpsHandler`, every `ReductionType` and every "
            "`BackendFeature` in the pinned PyTorch, plus surfaces Inductor does not "
            "enumerate, assigned to a family with its status. Families list their "
            "prerequisites; that order is the work order, tracked in beads."
        ),
        "",
        "Statuses:",
        "",
        *(f"- **{k}**: {v}." for k, v in STATUS_MEANING.items()),
        "",
        "## Families",
        "",
        "| Family | Scope | Prerequisites | Beads | Implemented (declared domain) | Complete | Missing |",
        "|---|---|---|---|---|---|---|",
    ]
    for family, (title, prerequisites, beads, note) in FAMILIES.items():
        counts = {
            s: sum(1 for f, st, _ in tables if f == family and st == s)
            for s in ("implemented", "complete", "missing")
        }
        lines.append(
            f"| `{family}` | {title}. {note} | "
            f"{', '.join(f'`{p}`' for p in prerequisites) or '-'} | "
            f"{', '.join(beads) or '-'} | {counts['implemented']} | "
            f"{counts['complete']} | {counts['missing']} |"
        )
    for heading, table in sections:
        lines += [
            "",
            f"## {heading}",
            "",
            "| Item | Family | Status | Note |",
            "|---|---|---|---|",
        ]
        for name, (family, status, note) in table.items():
            lines.append(f"| `{name}` | `{family}` | {status} | {note} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    MARKDOWN.write_text(render())
    sys.stdout.write(f"wrote {MARKDOWN}\n")
