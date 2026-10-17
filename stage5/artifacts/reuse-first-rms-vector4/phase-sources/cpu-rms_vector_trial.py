"""Phased wqc trial; CPU qualification precedes any vector4 GPU source."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

from stage5.rms_trial import qualify, special
from stage5.rms_vector_trial_data import (
    OLD,
    OUT,
    ROOT,
    ROWS,
    admitted,
    model,
    tensor_sha,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def reference():
    import torch
    from stage1.overlay import install

    install()
    from stage5.corpus import rms
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx
    from stage5.reduction_policy import TOLERANCES
    from stage5.attribution import inventory

    torch.set_num_threads(1)
    torch._dynamo.config.recompile_limit = 256
    patches = {
        "triton.cudagraphs": False,
        "compile_threads": 1,
        "fx_graph_cache": False,
        "emulate_precision_casts": True,
        "joint_graph_constant_folding": False,
        "eager_numerics.division_rounding": True,
        "eager_numerics.disable_ftz": True,
        "eager_numerics.use_pytorch_libdevice": True,
    }

    def backend(gm, inputs):
        return compile_fx(gm, inputs, config_patches=patches)

    old_meta = json.loads((OLD / "reference.json").read_text())
    assert old_meta["complete"] and old_meta["data_sha256"] == sha(OLD / "reference.pt")
    old = torch.load(OLD / "reference.pt", weights_only=True)
    values = {label: d for label, d in old.items() if admitted(d["input"].shape)}
    rejected = [label for label, d in old.items() if not admitted(d["input"].shape)]
    assert len(values) == 15 and len(rejected) == 51
    compiled = torch.compile(rms, backend=backend, fullgraph=True, dynamic=True)
    with config.patch(patches):
        for dtype in ("float32", "float16", "bfloat16"):
            td = getattr(torch, dtype)
            for b in ROWS:
                torch.manual_seed(19340 + b)
                cpu_x = torch.randn((b, 1024), dtype=torch.float32).to(td)
                x = cpu_x.cuda()
                actual, aten = compiled(x), rms(x)
                rtol, atol = TOLERANCES[str(td)]
                torch.testing.assert_close(
                    actual, aten, rtol=rtol, atol=atol, equal_nan=True
                )
                special(torch, actual, aten)
                d = cpu_x.double()
                label = f"vector_row_tail_{dtype}_{b}_1024"
                values[label] = {
                    "input": cpu_x,
                    "reference": actual.cpu(),
                    "aten": aten.cpu(),
                    "fp64": d * torch.rsqrt((d * d).mean(-1, keepdim=True) + 1e-5),
                }
                print("reference", label, "PASS", flush=True)
    torch.save(values, OUT / "reference.pt")
    code = OUT / "reference-code"
    for p in (ROOT / "cache/reuse-first-rms-vector4/reference").rglob("*"):
        if p.is_file() and p.suffix in (
            ".py",
            ".ptx",
            ".cubin",
            ".ttir",
            ".ttgir",
            ".llir",
            ".json",
        ):
            target = code / p.relative_to(
                ROOT / "cache/reuse-first-rms-vector4/reference"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
    write(
        "reference.json",
        {
            "complete": True,
            "cases": len(values),
            "data_sha256": sha(OUT / "reference.pt"),
            "parent_reference_sha256": sha(OLD / "reference.json"),
            "parent_data_sha256": sha(OLD / "reference.pt"),
            "rejected_old_cases": rejected,
            "input_sha256": {k: tensor_sha(d["input"]) for k, d in values.items()},
            "source_sha256": {
                "rms_vector_trial.py": sha(__file__),
                "rms_vector_trial_data.py": sha(ROOT / "rms_vector_trial_data.py"),
            },
            "configuration": patches,
            "inventory": inventory(torch, "triton"),
            "code": {
                str(p.relative_to(OUT)): sha(p) for p in code.rglob("*") if p.is_file()
            },
        },
    )


def cpu():
    from stage1 import evidence

    evidence.guard()
    import torch

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    meta = json.loads((OUT / "reference.json").read_text())
    assert meta["complete"] and meta["data_sha256"] == sha(OUT / "reference.pt")
    data = torch.load(OUT / "reference.pt", weights_only=True)
    assert len(data) == 36 and not (ROOT / "rms_vector_probe/src/main.rs").exists()
    rows, models = [], {}
    for label, d in data.items():
        sums, output = model(d["input"])
        special(torch, sums, sums[:, :1].expand_as(sums))
        torch.testing.assert_close(
            sums, sums[:, :1].expand_as(sums), rtol=0, atol=0, equal_nan=True
        )
        result = qualify(torch, label, d["input"], output, d)
        rows.append(result)
        models[label] = {"sums": sums, "output": output}
        write("cpu.json", {"complete": False, "rows": rows})
        assert result["passes"], result
        print("cpu", label, "PASS", flush=True)
    torch.save(models, OUT / "cpu-model.pt")
    write(
        "cpu.json",
        {
            "complete": True,
            "rows": rows,
            "reference_sha256": sha(OUT / "reference.json"),
            "model_sha256": sha(OUT / "cpu-model.pt"),
            "gpu_candidate_source_existed": False,
            "order": "FP32 square; per lane four component chains at128*k+4*lane+c; ascending XOR1,2,4,8,16 per component; +0 then ascending component fold; original epilogue/all-lane output",
            "source_sha256": {
                "rms_vector_trial.py": sha(__file__),
                "rms_vector_trial_data.py": sha(ROOT / "rms_vector_trial_data.py"),
            },
            "independence": evidence.inventory(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("reference", "cpu"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "cache/reuse-first-rms-vector4" / args.phase
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    try:
        globals()[args.phase]()
    except Exception:
        write(args.phase + "-failure.json", {"error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
