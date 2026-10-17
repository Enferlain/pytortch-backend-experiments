"""Phased reference/CPU gate for the single reuse-first RMS mechanism."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

from stage5.rms_trial_data import OUT, ROOT, fixtures, model, tensor_sha


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def special(torch, actual, reference):
    assert torch.equal(torch.isnan(actual), torch.isnan(reference))
    assert torch.equal(torch.isinf(actual), torch.isinf(reference))
    signs = (torch.isinf(actual) & torch.isinf(reference)) | (
        (actual == 0) & (reference == 0)
    )
    assert torch.equal(torch.signbit(actual[signs]), torch.signbit(reference[signs]))


def reference():
    import torch
    from stage1.overlay import install

    install()
    from stage5.corpus import rms
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx
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

    compiled = torch.compile(rms, backend=backend, fullgraph=True, dynamic=True)
    rows, values = [], {}
    from stage5.reduction_policy import TOLERANCES

    with config.patch(patches):
        for label, x in fixtures(torch):
            actual, aten = compiled(x), rms(x)
            torch.cuda.synchronize()
            rtol, atol = TOLERANCES[str(x.dtype)]
            torch.testing.assert_close(
                actual, aten, rtol=rtol, atol=atol, equal_nan=True
            )
            special(torch, actual, aten)
            cpu = x.cpu()
            d = cpu.double()
            truth = d * torch.rsqrt((d * d).mean(-1, keepdim=True) + 1e-5)
            values[label] = {
                "input": cpu,
                "reference": actual.cpu(),
                "aten": aten.cpu(),
                "fp64": truth,
            }
            rows.append(
                {
                    "case": label,
                    "shape": list(x.shape),
                    "dtype": str(x.dtype),
                    "input_sha256": tensor_sha(x),
                    "reference_sha256": tensor_sha(actual),
                }
            )
            print("reference", label, "PASS", flush=True)
    torch.save(values, OUT / "reference.pt")
    cache = ROOT / "cache/reuse-first-rms/reference"
    retained = OUT / "reference-code"
    for p in cache.rglob("*"):
        if p.is_file() and p.suffix in (
            ".py",
            ".ptx",
            ".cubin",
            ".ttir",
            ".ttgir",
            ".llir",
            ".json",
        ):
            target = retained / p.relative_to(cache)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
    write(
        "reference.json",
        {
            "complete": True,
            "rows": rows,
            "config": patches,
            "data_sha256": sha(OUT / "reference.pt"),
            "source_sha256": {
                "rms_trial.py": sha(__file__),
                "rms_trial_data.py": sha(ROOT / "rms_trial_data.py"),
            },
            "environment": inventory(torch, "triton"),
            "code": {
                str(p.relative_to(OUT)): sha(p)
                for p in retained.rglob("*")
                if p.is_file()
            },
        },
    )


def qualify(torch, label, x, actual, data):
    from stage5.reduction_policy import qualify_case, TOLERANCES

    ref = {
        "case": label,
        "dtype": str(x.dtype),
        "shape": list(x.shape),
        "input_sha256": tensor_sha(x),
        "aten": data["aten"].flatten().tolist(),
        "fp64": data["fp64"].flatten().tolist(),
        "values": data["reference"].flatten().tolist(),
    }
    row = {**ref, "values": actual.flatten().tolist()}
    result = qualify_case(row, ref)
    rtol, atol = TOLERANCES[str(x.dtype)]
    torch.testing.assert_close(
        actual, data["aten"], rtol=rtol, atol=atol, equal_nan=True
    )
    special(torch, actual, data["reference"])
    failures = [e for e in result["elements"] if not e["passes"]]
    return {
        "case": label,
        "passes": result["passes"],
        "elements": len(result["elements"]),
        "policy": result["policy"],
        "failures": failures[:32],
        "input_sha256": ref["input_sha256"],
    }


def cpu():
    from stage1 import evidence

    evidence.guard()
    import torch

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    reference_meta = json.loads((OUT / "reference.json").read_text())
    assert reference_meta["complete"] and reference_meta["data_sha256"] == sha(
        OUT / "reference.pt"
    )
    values = torch.load(OUT / "reference.pt", weights_only=True)
    rows, models = [], {}
    for label, data in values.items():
        x = data["input"]
        sums, output = model(x)
        special(torch, sums, sums[:, :1].expand_as(sums))
        torch.testing.assert_close(
            sums, sums[:, :1].expand_as(sums), rtol=0, atol=0, equal_nan=True
        )
        result = qualify(torch, label, x, output, data)
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
            "order": "FP32 chain by lane, ascending k; XOR1,2,4,8,16; all32 lanes",
            "source_sha256": {
                "rms_trial.py": sha(__file__),
                "rms_trial_data.py": sha(ROOT / "rms_trial_data.py"),
            },
            "independence": evidence.inventory(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("reference", "cpu"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "cache/reuse-first-rms" / args.phase
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    try:
        globals()[args.phase]()
    except Exception:
        write(
            args.phase + "-failure.json",
            {"error": traceback.format_exc(), "phase": args.phase},
        )
        raise


if __name__ == "__main__":
    main()
