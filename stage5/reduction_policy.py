"""Versioned per-element qualification; reference arithmetic is not an oracle."""

import math

VERSION = "reference-or-better-v1"
SANITY_VERSION = "qualified-reduction-corruption-guard-v1"
TOLERANCES = {
    "torch.float32": (2e-5, 2e-5),
    "torch.float16": (0.002, 0.001),
    "torch.bfloat16": (0.016, 0.008),
}


def sanity_limit(dtype, expected_magnitude):
    # Coarse corruption protection, not an accuracy oracle or universal
    # reassociation bound. The absolute allowance preserves the executed
    # lost-unit diagnostic (accurate 1 vs old-tree 0). New overlap domains need
    # input-aware qualification; currently compensation tunes alone.
    return 2.0 + max(0.01, 2 * TOLERANCES[dtype][0]) * expected_magnitude


def qualify_case(actual, reference):
    for field in ("case", "dtype", "shape", "input_sha256", "aten", "fp64"):
        if actual[field] != reference[field]:
            raise ValueError("unmatched qualification field: " + field)
    rtol, atol = TOLERANCES[actual["dtype"]]
    elements = []
    for i, (a, t, truth) in enumerate(
        zip(actual["values"], reference["values"], reference["fp64"], strict=True)
    ):
        if not math.isfinite(a) or not math.isfinite(t):
            same = (math.isnan(a) and math.isnan(t)) or a == t
            clause = "special_class" if same else "special_mismatch"
        else:
            near = abs(a - t) <= atol + rtol * abs(t)
            better = abs(a - truth) <= abs(t - truth)
            same = near or better
            clause = (
                "reference_tolerance"
                if near
                else "no_greater_fp64_error"
                if better
                else "failure"
            )
        elements.append(
            {
                "element": i,
                "actual": a,
                "reference": t,
                "fp64": truth,
                "clause": clause,
                "passes": same,
            }
        )
    return {
        "case": actual["case"],
        "passes": all(x["passes"] for x in elements),
        "elements": elements,
        "policy": VERSION,
    }


def qualified_for_runtime(semantic, plan):
    # The combined candidate failed a matched held-out element even with the
    # efficient split graph. It remains explicit diagnostic-only. Existing
    # compensated sum/epilogue behavior retains its executed regression scope.
    return semantic["schema"] == 3 and plan.strategy == "shared_tree_compensated"
