"""Capture actual stateful lowering before accepting or implementing semantics."""

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

init_backend_registration()
register()
records = []


class Observed(Exception):
    pass


def backend(gm, inputs):
    def capture(nodes):
        entries = []
        for n in nodes:
            entry = {
                "name": n.get_name(),
                "type": type(n).__name__,
                "ir_type": type(n.node).__name__,
                "group": str(getattr(n, "group", None)),
                "dependencies": str(n.read_writes),
            }
            if hasattr(n, "_body"):
                entry.update(
                    ranges=str(n.get_ranges()),
                    indices=str(n._body.indexing_exprs),
                    indirect=str(n._body.indirect_vars),
                    graph=str(n._body.root_block.graph),
                    subblocks={k: str(v.graph) for k, v in n._body.subblocks.items()},
                )
            else:
                entry.update(
                    kernel=getattr(n.node, "python_kernel_name", None),
                    target=str(getattr(n.node, "op_overload", None)),
                )
            entries.append(entry)
        records.append({"graph": str(gm.graph), "nodes": entries})
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


x = torch.randn(3, 257, device="cuda")
i = torch.tensor([0, 2, 2, 1], device="cuda", dtype=torch.int64)
w = torch.randn(7, 33, device="cuda", requires_grad=True)
fixtures = [
    ("rand", lambda x: torch.rand_like(x) + x, (x,)),
    ("two_rand", lambda x: torch.rand_like(x) + torch.rand_like(x), (x,)),
    ("dropout", lambda x: torch.nn.functional.dropout(x, 0.25, True), (x,)),
    ("gather", lambda x, i: torch.gather(x, 0, i), (x.flatten(), i)),
    ("index", lambda x, i: x[i], (x, i)),
    (
        "scatter_add",
        lambda x, i: torch.zeros(7, device=x.device).scatter_add(0, i, x[:4]),
        (x.flatten(), i),
    ),
    ("embedding", lambda w, i: torch.nn.functional.embedding(i, w), (w, i)),
    ("mm", lambda x: x @ x.t(), (x,)),
]
for name, fn, args in fixtures:
    state = torch.cuda.get_rng_state().clone()
    before = len(records)
    try:
        torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)(*args)
    except Exception as error:
        if len(records) == before:
            records.append({"error": str(error), "error_type": type(error).__name__})
    records[-1].update(
        fixture=name,
        compile_preserved_rng_state=torch.equal(state, torch.cuda.get_rng_state()),
    )
evidence.state("after_probe")
(ROOT / "artifacts/lowering.json").write_text(
    json.dumps({"fixtures": records, "independence": evidence.inventory()}, indent=2)
)
print(json.dumps(records, indent=2))
