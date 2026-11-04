"""Offline qualification of the default path's plans against Triton's band
(bead zah.11, step 1).

The band (``welford_band.measure``) gives a verdict for whichever plan first-use
selection picks. This pins each plan instead: for every reference case, each
reduction kernel in turn is pinned to each plan the default path admits for
it, the case's other kernels held at the plan selection chose, and the band is
measured. The result is one verdict per (case, kernel, plan), the record the
runtime output check stands in for today. From the repository root::

    stage0/.venv/bin/python -m cubecl_torch.testing.qualification --gpu 3090 \\
        --out cubecl_torch/testing/results/qualification-3090.json

``names`` limits the cases. Run with an empty ``CUBECL_TORCH_CACHE_DIR`` so
selection is measured, not read from an earlier run's cache.
"""

import argparse
import json
from pathlib import Path

from cubecl_torch.testing.extrema_timing import DEVICES


def run(names=None):
    import torch

    from cubecl_torch.inductor import tuning
    from cubecl_torch.testing import welford_band
    from cubecl_torch.testing import welford_reference as ref
    from cubecl_torch.testing.welford_band import plan_label

    select = tuning.AutotunedKernel.select
    seen = {}
    pins = {}

    def recording_select(self, key, key_data, plans, rejected, *args):
        if key in pins:
            return pins[key]
        winner = select(self, key, key_data, plans, rejected, *args)
        seen.setdefault(key, {"plans": list(plans), "default": winner})
        return winner

    tuning.AutotunedKernel.select = recording_select
    wanted = [
        c["name"]
        for c in ref.cases()
        if c["kind"] != "special" and (not names or c["name"] in names)
    ]
    result = {}
    for name in wanted:
        seen.clear()
        pins.clear()
        # Selection as the default path makes it; it also lists each kernel's
        # admitted plans (a kernel with one admitted plan never selects).
        default = welford_band.measure({name})[name]
        kernels = dict(seen)
        entry = {"default": default, "kernels": []}
        for index, (key, record) in enumerate(kernels.items()):
            verdicts = []
            for plan in record["plans"]:
                pins.clear()
                pins.update({k: r["default"] for k, r in kernels.items()})
                pins[key] = plan
                torch.cuda.synchronize()
                row = welford_band.measure({name})[name]
                verdicts.append(
                    {
                        "plan": plan.to_dict(),
                        "label": plan_label(plan),
                        "outside": row.get("outside"),
                        "error": row.get("error"),
                        "errors": row.get("errors"),
                    }
                )
            entry["kernels"].append(
                {
                    "index": index,
                    "key": key,
                    "default": plan_label(record["default"]),
                    "plans": verdicts,
                }
            )
        result[name] = entry
        outside = [
            f"{k['index']}:{i}:{v['label']}"
            for k in entry["kernels"]
            for i, v in enumerate(k["plans"])
            if v["outside"] or v["error"]
        ]
        admitted = sum(len(k["plans"]) for k in entry["kernels"])
        print(f"{name}: {admitted} pinned, outside {outside}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=DEVICES, default="3090")
    parser.add_argument("--out", type=Path)
    parser.add_argument("names", nargs="*")
    options = parser.parse_args(argv)
    from cubecl_torch.runtime import guard

    guard.install()
    import torch

    import cubecl_torch.inductor as ci

    ci.install()
    from cubecl_torch.testing.cubek_timing import _device

    _device(torch, options.gpu)
    result = run(set(options.names) or None)
    if options.out:
        options.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
