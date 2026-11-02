"""Tile plans against Triton's accuracy band (cubecl-tile-stage-1 task 4.2).

Runs ``welford_band.measure`` with one tile plan pinned on every reduction, so the
verdict is the plan's and not first-use selection's. From the repository root::

    stage0/.venv/bin/python -m cubecl_torch.testing.tile_band 4 \\
        --out cubecl_torch/testing/results/tile-band-width4.json

It runs on ``cuda:0``, the RTX 3090, and exits non-zero if a case has no verdict from
the pinned plan.
"""

import argparse
import json
from pathlib import Path


def pin(width, validate=False):
    """Offer reduction kernels only the tile plan of ``width``, with the numerical
    policy lifted for it; ``validate`` reports accesses past their buffers."""
    from cubecl_torch.compiler.plan import tile_catalog
    from cubecl_torch.inductor import legality, tuning

    plan = next(p for p in tile_catalog() if p.vector_width == width)
    candidates = tuning.candidates

    def forced(semantic, *args, **kwargs):
        accepted, rejected = candidates(semantic, *args, **kwargs)
        if semantic["schema"] != 3:
            return accepted, rejected
        if validate:
            semantic["validate_reads"] = True
        return [plan], rejected

    tuning.candidates = forced
    numerical = legality.numerical_reason
    legality.numerical_reason = lambda p, *a: None if p == plan else numerical(p, *a)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("width", type=int)
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

    pin(options.width)
    names = options.names or [
        c["name"] for c in ref.cases() if c["kind"] in ("row_sum", "sum_consumer")
    ]
    torch.cuda.set_device(0)
    rows = welford_band.measure(set(names))
    failed = sorted(set(names) - rows.keys())
    for name, row in rows.items():
        if "error" in row:
            print(f"{name}: {row['error'][:110]}")
            failed.append(name)
            continue
        # The verdict is the pinned plan's, or it is no verdict.
        selected = {s.split("/")[0:2].__str__() for s in row["cubecl_selected"]}
        if selected != {str(["tile_row_sum", f"32x1v{options.width}"])}:
            print(f"{name}: ran {row['cubecl_selected']}")
            failed.append(name)
            continue
        tests = row["sign_test"]
        print(
            f"{name}: outside {row['outside']} "
            + " ".join(f"{k} {t['p']:.3g}/{t['p_mean']:.3g}" for k, t in tests.items()),
            row["cubecl_selected"],
        )
    if options.out:
        options.out.write_text(json.dumps(rows, indent=1, sort_keys=True))
    if failed:
        raise SystemExit(f"no verdict for {sorted(failed)}")


if __name__ == "__main__":
    main()
