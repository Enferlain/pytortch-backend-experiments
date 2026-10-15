"""Collect executed SASS for qualified isolated mechanisms, in recorded order."""

from contextlib import ExitStack
import json
import os

from stage5.reduction_codegen_mechanisms import Component, OUT, FROZEN


def main():
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.reduction_codegen_compare import reference
    from stage5.attribution import inventory
    from stage5.reference_reduction_probe import tensor_sha

    torch.set_num_threads(1)
    names = ("uniform", "owner", "narrow", "uniform_narrow")
    for name in names:
        q = json.loads((OUT / f"{name}-qualification.json").read_text())
        assert q["complete"] and len(q["checks"]) == 152
    components = {
        "frozen": Component(FROZEN / "kernels/float32/candidate"),
        **{n: Component(OUT / n / "float32") for n in names},
    }
    actions = []
    for b, n in ((8, 1048576), (64, 65539)):
        torch.manual_seed(7301)
        x = torch.randn((b, n), device="cuda")
        buffers = {
            name: torch.empty((b, 9), device="cuda") for name in (*components, "triton")
        }
        for name, k in components.items():
            k((n, b), x, buffers[name])
        ref = reference(b, n, x, buffers["triton"])
        ref.batch(1)
        assert all(
            torch.equal(buffers["frozen"].view(torch.int32), y.view(torch.int32))
            for y in buffers.values()
        )
        with ExitStack() as stack:
            for name, k in components.items():
                stack.enter_context(k.kernel.prepared((n, b), x, buffers[name]))
            paths = {
                **{name: k.kernel for name, k in components.items()},
                "triton": ref,
            }
            for k in paths.values():
                for _ in range(20):
                    k.batch(1)
            torch.cuda._sleep(50_000_000)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for name, k in paths.items():
                k.batch(1)
                actions.append(
                    {
                        "id": len(actions),
                        "name": name,
                        "shape": [b, n],
                        "entrypoint": k.manifest["entrypoint"],
                        "attributes": k.attributes,
                        "input_sha256": None,
                    }
                )
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            input_hash = tensor_sha(x)
            for a in actions[-len(paths) :]:
                a["input_sha256"] = input_hash
    (OUT / "mechanism-capture.json").write_text(
        json.dumps(
            {
                "complete": True,
                "actions": actions,
                "environment": inventory(torch, "cubecl"),
            },
            indent=2,
        )
    )
    print("Captured", len(actions), "qualified mechanism/reference actions", flush=True)


if __name__ == "__main__":
    main()
