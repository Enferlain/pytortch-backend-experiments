"""Collect actual lowered nodes before selecting new Stage 2 representations."""

import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(__file__).parent / "cache")
from stage1 import evidence

evidence.guard()
import torch
from stage1.overlay import install

install()
from stage1.backend import register
from torch._inductor.codegen.common import init_backend_registration
from torch._inductor.compile_fx import compile_fx
from torch._inductor.virtualized import V

init_backend_registration()
register()


class Observed(Exception):
    pass


records = []


def backend(gm, inputs):
    def capture(nodes):
        record = {
            "graph": str(gm.graph),
            "inputs": [
                str(v.shape) if isinstance(v, torch.Tensor) else str(v) for v in inputs
            ],
            "nodes": [],
        }
        for n in nodes:
            if not hasattr(n, "_body"):
                continue
            record["nodes"].append(
                {
                    "name": n.get_name(),
                    "ranges": str(n.get_ranges()),
                    "indices": str(n._body.indexing_exprs),
                    "graph": str(n._body.root_block.graph),
                    "subblocks": {
                        k: str(v.graph) for k, v in n._body.subblocks.items()
                    },
                    "buffers": {
                        d.name: str(V.graph.get_buffer(d.name).get_layout())
                        for d in n.read_writes.reads | n.read_writes.writes
                        if hasattr(d, "name")
                    },
                }
            )
        records.append(record)
        raise Observed()

    return compile_fx(
        gm,
        inputs,
        config_patches={
            "cuda_backend": "cubecl",
            "compile_threads": 1,
            "fx_graph_cache": False,
            "_pre_fusion_custom_pass": capture,
            "triton.cudagraphs": False,
        },
    )


def fusion_long(x):
    a = x
    for _ in range(20):
        a = torch.sin(a) + 0.01
    return a * a + a


fixtures = [
    ("fusion_long", fusion_long, [torch.randn(257, device="cuda")], False),
    (
        "half",
        lambda x: torch.sin(x) + 1,
        [torch.randn(257, device="cuda", dtype=torch.float16)],
        False,
    ),
    (
        "bf16_cast",
        lambda x: (torch.exp(x).float() / 2).to(torch.bfloat16),
        [torch.randn(257, device="cuda", dtype=torch.bfloat16)],
        False,
    ),
    (
        "ops",
        lambda x: torch.where(x <= 0, -x, torch.sqrt(torch.exp(x) - 1) / 2),
        [torch.randn(257, device="cuda")],
        False,
    ),
    (
        "bias",
        lambda x, b: x + b,
        [torch.randn(3, 257, device="cuda"), torch.randn(257, device="cuda")],
        True,
    ),
    (
        "column",
        lambda x, b: x + b,
        [torch.randn(3, 257, device="cuda"), torch.randn(3, 1, device="cuda")],
        False,
    ),
    (
        "transpose",
        lambda x: torch.sin(x) + 1,
        [torch.randn(3, 257, device="cuda").T],
        True,
    ),
    (
        "slice",
        lambda x: torch.sin(x) + 1,
        [torch.randn(515, device="cuda")[1::2]],
        True,
    ),
    (
        "mask",
        lambda x: torch.cat([x, x[:1]]) + 1,
        [torch.randn(257, device="cuda")],
        True,
    ),
    (
        "fusion",
        lambda x: torch.sin(x.clone()) + 1,
        [torch.randn(257, device="cuda")],
        False,
    ),
]
for name, f, args, dynamic in fixtures:
    start = len(records)
    try:
        torch.compile(f, backend=backend, fullgraph=True, dynamic=dynamic)(*args)
    except Exception as e:
        if len(records) == start:
            records.append({"error": str(e)})
    records[-1]["fixture"] = name
Path("stage2/artifacts/lowering-probe.json").write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
