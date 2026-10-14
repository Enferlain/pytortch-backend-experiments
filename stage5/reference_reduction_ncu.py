"""One existing physical kernel for Nsight, without nested CUPTI autotuning."""

import argparse
import json
import os
from pathlib import Path
import traceback


def run(args, result):
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.artifact import CompiledKernel
    from stage5.fixtures import captured
    from stage5.plan import PhysicalCandidate
    from stage5.reference_reduction_probe import tensor_sha, sha

    plans = {
        "baseline": PhysicalCandidate("shared_tree_reduction_baseline"),
        "compensated": PhysicalCandidate("shared_tree_compensated"),
        "plane": PhysicalCandidate("plane_shared", plane_width=32),
    }
    plan = plans[args.plan]
    torch.manual_seed(7301)
    x = torch.randn((8, 65539), device="cuda")
    y = torch.empty(8, device="cuda")
    k = CompiledKernel(captured("row_sum", "float32"), 0, plan.to_dict())
    # Nsight selects the third matching physical launch. No torch.profiler or
    # first-use autotuner runs in this process; counters replay this pure kernel.
    for _ in range(3):
        k((65539, 8), x, y)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x.sum(-1), rtol=2e-5, atol=2e-5)
    evidence.state("after_reference_ncu")
    result.update(
        qualification_complete=True,
        component_reference_check=True,
        shape=list(x.shape),
        input_sha256=tensor_sha(x),
        values=y.tolist(),
        directory=str(k.directory),
        plan=plan.to_dict(),
        attributes=k.kernel.attributes,
        resources=k.kernel.resources,
        independence=evidence.inventory(),
        harness_sha256=sha(__file__),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", choices=("baseline", "compensated", "plane"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"resources_only": True, "qualification_complete": False}
    try:
        run(args, result)
    except Exception as error:
        result.update(failure=repr(error), traceback=traceback.format_exc())
        args.output = args.output.with_suffix(".failed.json")
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
