"""Capture real reduction lowering before backend fusion/code generation."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache")
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
records = []


class Observed(Exception):
    pass


def backend(gm, inputs):
    def capture(nodes):
        records.append(
            {
                "graph": str(gm.graph),
                "nodes": [
                    {
                        "name": n.get_name(),
                        "group": str(n.group),
                        "ranges": str(n.get_ranges()),
                        "indices": str(n._body.indexing_exprs),
                        "graph": str(n._body.root_block.graph),
                        "subblocks": {
                            k: str(v.graph) for k, v in n._body.subblocks.items()
                        },
                        "hint": str(n.node.data.reduction_hint)
                        if n.is_reduction()
                        else None,
                        "buffers": {
                            d.name: str(V.graph.get_buffer(d.name).get_layout())
                            for d in n.read_writes.reads | n.read_writes.writes
                            if hasattr(d, "name")
                        },
                    }
                    for n in nodes
                    if hasattr(n, "_body")
                ],
            }
        )
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
            "emulate_precision_casts": True,
            "joint_graph_constant_folding": False,
        },
    )


def rms(x):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5)


fixtures = [
    ("row", lambda x: x.sum(-1)),
    ("column", lambda x: x.sum(0)),
    ("mean", lambda x: x.mean(-1) + 1),
    ("rms", rms),
]
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    for name, function in fixtures:
        before = len(records)
        try:
            torch.compile(function, backend=backend, fullgraph=True, dynamic=True)(
                torch.randn(3, 257, dtype=dtype, device="cuda")
            )
        except Exception as error:
            if len(records) == before:
                raise
            records[-1]["termination"] = type(error).__name__
        records[-1].update(fixture=name, dtype=str(dtype))
ROOT.joinpath("artifacts").mkdir(exist_ok=True)
ROOT.joinpath("artifacts/lowering.json").write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
