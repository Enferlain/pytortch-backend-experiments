"""Real captured descriptors for focused WSL-native Nsight diagnostics."""

import json
from pathlib import Path
import sys
from stage1 import evidence

evidence.guard()
import torch
from stage5.fixtures import captured
from stage5.artifact import CompiledKernel
from stage5.plan import PhysicalCandidate

name = sys.argv[1] if len(sys.argv) > 1 else "vector"
plans = {
    "scalar": PhysicalCandidate("scalar_pointwise_baseline"),
    "vector": PhysicalCandidate("vector_pointwise", (128, 1, 1), 4),
    "row_tree": PhysicalCandidate("shared_tree_reduction_baseline"),
    "row_plane": PhysicalCandidate("plane_shared", (128, 1, 1), plane_width=32),
    "row_packed": PhysicalCandidate(
        "independent_planes", (32, 4, 1), outputs_per_workgroup=4, plane_width=32
    ),
    "column_tree": PhysicalCandidate("shared_tree_reduction_baseline"),
    "column_coalesced": PhysicalCandidate(
        "coalesced_columns", (32, 8, 1), outputs_per_workgroup=32
    ),
}
if name.startswith("column"):
    artifacts = json.loads(
        (Path(__file__).parent / "artifacts/baseline-cubecl_baseline.json").read_text()
    )["artifacts"]
    semantic = next(
        a["semantic"]
        for a in artifacts
        if a["semantic"]["schema"] == 3
        and a["semantic"]["buffers"][0]["dtype"] == "float32"
        and len(a["semantic"]["buffers"]) == 2
        and a["semantic"]["outer"] == {"op": "scalar", "slot": 0}
        and all(
            o["op"] in ("load", "cast", "reduction", "store")
            for o in a["semantic"]["ops"]
        )
    )
    x = torch.randn(1024, 256, device="cuda")
    scalars, expected = (256, 1024), x.sum(0)
elif name.startswith("row"):
    semantic = captured("row_sum", "float32")
    x = torch.randn(256, 1024, device="cuda")
    scalars, expected = (1024, 256), x.sum(-1)
else:
    semantic = captured("add", "float32")
    x = torch.randn(1048583, device="cuda")
    scalars, expected = (x.numel(),), x + 1
k = CompiledKernel(semantic, 0, plans[name].to_dict())
y = torch.empty_like(expected)
for _ in range(3):
    k(scalars, x, y)
torch.cuda.synchronize()
torch.testing.assert_close(y, expected, rtol=2e-5, atol=2e-5)
evidence.state("after_nsight_probe")
(Path(__file__).parent / "artifacts" / ("ncu-" + name + "-manifest.json")).write_text(
    json.dumps(
        {
            "directory": str(k.directory),
            "plan": plans[name].to_dict(),
            "manifest": k.kernel.manifest,
            "attributes": k.kernel.attributes,
            "resources": k.kernel.resources,
            "independence": evidence.inventory(),
        },
        indent=2,
    )
)
