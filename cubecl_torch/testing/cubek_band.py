"""Bridged ``cubek_reduce`` plans against Triton's accuracy band
(torch-runtime-bridge task 4.2).

Runs ``welford_band.measure`` over the stored ``row_sum`` references with one
bridged plan pinned on every reduction its structural rule admits, and its
numerical rule lifted, so the verdict is the plan's and not first-use
selection's. A kernel the rule refuses keeps its ordinary candidates, and its
case has no verdict for the plan. From the repository root::

    stage0/.venv/bin/python -m cubecl_torch.testing.cubek_band plane --parallel \\
        --out cubecl_torch/testing/results/cubek-band-plane-parallel.json

It runs on ``cuda:0``, the RTX 3090, and exits non-zero if a case has no verdict from
the pinned plan.
"""

import argparse
import json
from pathlib import Path


def pin(routine, parallel, planes=0, mapping=""):
    """Offer reduction kernels the bridged plan only, where its structural
    rule admits it, with the numerical policy lifted for it. ``planes`` forces
    a cube routine's blueprint (a diagnostic plan, not in the catalog)."""
    from cubecl_torch.compiler.plan import CUBEK_REDUCE, PhysicalCandidate
    from cubecl_torch.inductor import legality, tuning

    plan = PhysicalCandidate(
        CUBEK_REDUCE,
        routine=routine,
        parallel_output=parallel,
        planes=planes,
        mapping=mapping,
    )
    candidates = tuning.candidates

    def forced(semantic, scalars, tensors, target, **kwargs):
        accepted, rejected = candidates(semantic, scalars, tensors, target, **kwargs)
        if semantic["schema"] != 3 or legality.structural_reason(
            plan, semantic, scalars, tensors, target
        ):
            return accepted, rejected
        return [plan], rejected

    tuning.candidates = forced
    numerical = legality.numerical_reason
    legality.numerical_reason = lambda p, *a: None if p == plan else numerical(p, *a)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("routine", choices=("unit", "plane", "cube", "cube_unit"))
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument(
        "--planes",
        type=int,
        default=0,
        help="force a cube routine's planes per cube (cube: plane accumulators "
        "merged by a plane reduction; cube_unit: unit accumulators merged as a "
        "tree in shared memory)",
    )
    parser.add_argument(
        "--mapping",
        choices=("inductor",),
        default="",
        help="derive the cube's blueprint per shape (inductor: as Inductor's "
        "reduction config maps the row)",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("names", nargs="*")
    options = parser.parse_args(argv)
    from cubecl_torch.runtime import guard

    guard.install()
    import torch

    import cubecl_torch.inductor as ci

    ci.install()
    from cubecl_torch.testing import welford_band
    from cubecl_torch.testing import welford_reference as ref

    plan = pin(options.routine, options.parallel, options.planes, options.mapping)
    names = options.names or [c["name"] for c in ref.cases() if c["kind"] == "row_sum"]
    torch.cuda.set_device(0)
    rows = welford_band.measure(set(names))
    failed = sorted(set(names) - rows.keys())
    for name, row in rows.items():
        if "error" in row:
            print(f"{name}: {row['error'][:110]}")
            failed.append(name)
            continue
        # The verdict is the pinned plan's, or it is no verdict.
        if row["cubecl_selected"] != [welford_band.plan_label(plan)]:
            print(f"{name}: ran {row['cubecl_selected']}")
            failed.append(name)
            continue
        tests = row["sign_test"]
        print(
            f"{name}: outside {row['outside']} "
            + " ".join(f"{k} {t['p']:.3g}/{t['p_mean']:.3g}" for k, t in tests.items())
        )
    if options.out:
        options.out.write_text(json.dumps(rows, indent=1, sort_keys=True))
    if failed:
        raise SystemExit(f"no verdict for {sorted(failed)}")


if __name__ == "__main__":
    main()
