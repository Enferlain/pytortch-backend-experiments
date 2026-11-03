"""Record the Triton config Inductor chooses for plain row sums
(``x.sum(-1)`` on contiguous float32 rows), for ``bridged.inductor_blueprint``.
Run in the Triton environment, from the repository root::

    PYTHON_GIL=0 /home/imi/Projects/sd-scripts/.venv-wsl/bin/python -m \\
        cubecl_torch.testing.inductor_configs cubecl_torch/testing/data/inductor_row_configs.json

It compiles each shape with caches off on every GPU and records ``R0_BLOCK``
and ``num_warps`` of the kernel's launcher with the device's capability, or
the kernel count where Inductor splits the reduction.
"""

import json
import sys

SHAPES = (
    (2048, 8192),
    (256, 32768),
    (4096, 2048),
    (2048, 4096),
    (2048, 4099),
    (300, 4096),
    (7, 2049),
    (5, 1025),
    (64, 1500),
    (1, 3000),
    (16, 100000),
)


def main(out):
    import os

    os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    import torch
    from torch._inductor.runtime import triton_heuristics

    seen = []
    run = triton_heuristics.CachingAutotuner.run

    def record(self, *args, **kwargs):
        result = run(self, *args, **kwargs)
        seen.append(self.launchers[0].config)
        return result

    triton_heuristics.CachingAutotuner.run = record
    rows = []
    for index in range(torch.cuda.device_count()):
        device = torch.device("cuda", index)
        for shape in SHAPES:
            torch._dynamo.reset()
            seen.clear()
            x = torch.randn(*shape, device=device)
            torch.compile(lambda t: t.sum(-1))(x)
            torch.cuda.synchronize(device)
            row = {
                "shape": list(shape),
                "capability": list(torch.cuda.get_device_capability(device)),
                "device": torch.cuda.get_device_name(device),
            }
            if len(seen) == 1:
                (config,) = seen
                row |= {
                    "R0_BLOCK": config.kwargs.get("R0_BLOCK"),
                    "num_warps": config.num_warps,
                }
            else:
                # A split reduction (a partials kernel and its merge), not a
                # looped row: bridged plans do not take it.
                row["split"] = len(seen)
            rows.append(row)
    with open(out, "w") as file:
        json.dump({"torch": torch.__version__, "configs": rows}, file, indent=1)
        file.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
