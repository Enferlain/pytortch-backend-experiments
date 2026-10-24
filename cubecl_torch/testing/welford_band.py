"""Welford accuracy against Inductor+Triton (welford-reductions 4.3, 7.1).

For every non-special case of ``welford_reference``, CubeCL's error against
the stored fp64 statistics is compared with Triton's on the same inputs, row
by row: each reduced row (each normalized row for a norm's output) of every
input is one pair. Rows where both errors are equal are dropped. A statistic
is worse than Triton when CubeCL loses significantly more rows than it wins
(one-sided sign test), when its mean per-row error is significantly larger
(paired, so rare large losses count by their size), both at
p < ``ALPHA``, or when CubeCL's error is non-finite on a row where Triton's is
finite. There is no tolerance margin (bd memory
triton-floor-band: compare at matched settings; the test only separates a
consistent difference from rounding noise). Record a run from the repository
root::

    stage0/.venv/bin/python -m cubecl_torch.testing.welford_band \\
        --out cubecl_torch/testing/results/welford-band-3090.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

ALPHA = 0.01


def mean_abs(value, reference):

    finite = reference.isfinite()
    return float((value.double() - reference)[finite].abs().mean())


def binomial_tail(losses, n):
    """P(X >= losses) for X ~ Binomial(n, 1/2)."""
    if losses <= 0:
        return 1.0
    terms = [
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        for k in range(losses, n + 1)
    ]
    top = max(terms)
    return min(1.0, math.exp(top - n * math.log(2)) * sum(math.exp(t - top) for t in terms))


def sign_test(cubecl, triton):
    """Paired per-row comparison of two error sequences (finite rows only).

    Returns wins, losses, ties, the one-sided p-value of CubeCL losing more
    rows, and the rows where only CubeCL's error is non-finite.
    """
    wins = losses = ties = nonfinite = 0
    for c, t in zip(cubecl, triton, strict=True):
        c_ok, t_ok = math.isfinite(c), math.isfinite(t)
        if not c_ok and t_ok:
            nonfinite += 1
        elif c_ok and not t_ok:
            wins += 1
        elif not c_ok or c == t:
            ties += 1
        elif c < t:
            wins += 1
        else:
            losses += 1
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "nonfinite": nonfinite,
        "p": binomial_tail(losses, wins + losses),
    }


def mean_difference(cubecl, triton):
    """One-sided p-value of CubeCL's mean per-row error exceeding Triton's
    (paired, normal approximation over the finite rows): the magnitude check
    the sign test lacks, so rare large losses are not outvoted."""
    differences = [
        c - t
        for c, t in zip(cubecl, triton, strict=True)
        if math.isfinite(c) and math.isfinite(t)
    ]
    n = len(differences)
    if n < 2:
        return 1.0
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
    if variance == 0:
        return 1.0 if mean <= 0 else 0.0
    z = mean / math.sqrt(variance / n)
    return 0.5 * math.erfc(z / math.sqrt(2))


def outside(test):
    return test["nonfinite"] > 0 or test["p"] < ALPHA or test["p_mean"] < ALPHA


def statistics_of(case, value):
    if case["kind"] == "var_mean":
        return {"var": value[0], "mean": value[1]}
    return {"mean": value[1], "rstd": value[2], "out": value[0]}


def measure(names=None):
    """One row per case: per-input errors of both backends and the verdict."""
    import torch

    import cubecl_torch.inductor as ci
    from cubecl_torch.inductor import tuning
    from cubecl_torch.testing import welford_reference as ref

    data = torch.load(ref.DATA, weights_only=False)["cases"]
    rows = {}
    for case in ref.cases():
        if case["kind"] == "special" or (names and case["name"] not in names):
            continue
        record = data[case["name"]]
        inputs = ref.case_inputs(case)
        if [[ref.digest(a) for a in args] for args in inputs] != record["input_sha256"]:
            rows[case["name"]] = {"error": "input digest"}
            continue
        torch._dynamo.reset()
        tuning.CACHE.clear()
        compiled = torch.compile(
            ref.FUNCTIONS[case["kind"]],
            backend=ci.compile_backend,
            dynamic=case["dynamic"],
        )
        try:
            values = [
                [v.cpu() for v in compiled(*(a.cuda() for a in args))]
                for args in inputs
            ]
        except Exception as error:  # noqa: BLE001  a rejection is recorded
            rows[case["name"]] = {"error": str(error).splitlines()[0][:400]}
            continue
        errors = {"cubecl": {}, "triton": {}}
        rows_of = {"cubecl": {}, "triton": {}}

        def add(backend, name, value, reference, count):
            errors[backend].setdefault(name, []).append(mean_abs(value, reference))
            rows_of[backend].setdefault(name, []).extend(
                ref.row_errors(value, reference, count).tolist()
            )

        for args, triton, value in zip(inputs, record["results"], values):
            fp64 = triton["fp64"]
            ours = statistics_of(case, value)
            for name, reference in fp64.items():
                add("cubecl", name, ours[name], reference, reference.numel())
                add("triton", name, triton[name], reference, reference.numel())
            if case["kind"] != "var_mean":
                out = ref.fp64_reference(case, args)["out"]
                add("cubecl", "out", ours["out"], out, fp64["mean"].numel())
                errors["triton"].setdefault("out", []).append(
                    triton["out_error"]["mean_abs"]
                )
                rows_of["triton"].setdefault("out", []).extend(
                    triton["out_row_error"].tolist()
                )
        tests = {
            name: {
                **sign_test(rows_of["cubecl"][name], rows_of["triton"][name]),
                "p_mean": mean_difference(
                    rows_of["cubecl"][name], rows_of["triton"][name]
                ),
            }
            for name in rows_of["cubecl"]
        }
        kernels = [k for k in tuning.CACHE.values() if k.semantic["schema"] == 3]
        rows[case["name"]] = {
            "triton_split": record["split"],
            "triton_algorithms": [
                k["algorithm"]
                for k in record["kernels"]
                if not k["name"].startswith("triton_poi")
            ],
            "cubecl_reduction_kernels": len(kernels),
            "cubecl_selected": sorted(
                {
                    f"{p.strategy}/{p.workgroup[0]}x{p.private_accumulators}"
                    f"v{p.vector_width}/{p.private_merge}"
                    for k in kernels
                    for p in k.selections.values()
                }
            ),
            "errors": errors,
            "sign_test": tests,
            "outside": sorted(name for name, test in tests.items() if outside(test)),
        }
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("names", nargs="*")
    options = parser.parse_args(argv)
    from cubecl_torch.runtime import guard

    guard.install()
    import torch

    import cubecl_torch.inductor as ci

    ci.install()
    rows = measure(set(options.names) or None)
    payload = {
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(),
        "cases": rows,
    }
    if options.out:
        options.out.write_text(json.dumps(payload, indent=1) + "\n")
    for name, row in rows.items():
        if row.get("outside") or row.get("error"):
            sys.stdout.write(f"{name}: {row.get('outside') or row.get('error')}\n")
    sys.stdout.write(
        f"{sum(bool(r.get('outside')) for r in rows.values())} of {len(rows)} cases outside\n"
    )


if __name__ == "__main__":
    main()
