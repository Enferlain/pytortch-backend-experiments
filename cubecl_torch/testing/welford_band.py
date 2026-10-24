"""Welford accuracy against Inductor+Triton's band (welford-reductions 4.3).

For every non-special case of ``welford_reference``, CubeCL's error against
the stored fp64 statistics is compared with Triton's error on the same
inputs. A statistic is outside the band when CubeCL's mean absolute error
exceeds Triton's by more than ``MARGIN`` in both seed halves (bd memory
triton-floor-band: compare at matched settings, check seed-subset stability
before calling a gap). Record a run from the repository root::

    stage0/.venv/bin/python -m cubecl_torch.testing.welford_band \\
        --out cubecl_torch/testing/results/welford-band.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

MARGIN = 0.10


def mean_abs(value, reference):

    finite = reference.isfinite()
    return float((value.double() - reference)[finite].abs().mean())


def halves(errors):
    """Mean error of the first and second half of the inputs."""
    if len(errors) < 2:
        raise ValueError("the band compares two seed halves")
    n = len(errors) // 2
    return [sum(errors[:n]) / n, sum(errors[n:]) / (len(errors) - n)]


def outside(cubecl, triton):
    """CubeCL worse than Triton by more than MARGIN in both seed halves.

    A non-finite CubeCL error (NaN where the reference is finite) is outside.
    """
    if not all(math.isfinite(c) for c in cubecl):
        return True
    return all(c > t * (1 + MARGIN) for c, t in zip(halves(cubecl), halves(triton)))


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
        for args, triton, value in zip(inputs, record["results"], values):
            fp64 = triton["fp64"]
            ours = statistics_of(case, value)
            for name, reference in fp64.items():
                errors["cubecl"].setdefault(name, []).append(
                    mean_abs(ours[name], reference)
                )
                errors["triton"].setdefault(name, []).append(
                    mean_abs(triton[name], reference)
                )
            if case["kind"] != "var_mean":
                out = ref.fp64_reference(case, args)["out"]
                errors["cubecl"].setdefault("out", []).append(
                    mean_abs(ours["out"], out)
                )
                errors["triton"].setdefault("out", []).append(
                    triton["out_error"]["mean_abs"]
                )
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
            "outside": sorted(
                name
                for name in errors["cubecl"]
                if outside(errors["cubecl"][name], errors["triton"][name])
            ),
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
        "margin": MARGIN,
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
