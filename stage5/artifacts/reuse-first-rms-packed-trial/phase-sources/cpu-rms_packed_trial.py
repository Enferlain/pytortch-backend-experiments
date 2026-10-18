"""Single packed-storage-load trial; immutable wqc references and components."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

from stage5.rms_trial import qualify, special
from stage5.rms_vector_trial_data import model, tensor_sha

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "artifacts/reuse-first-rms-vector4"
OUT = ROOT / "artifacts/reuse-first-rms-packed-trial"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def cpu():
    from stage1 import evidence

    evidence.guard()
    import torch

    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    assert not (OUT / "cpu.json").exists(), "preserve the chronological gate"
    write("cpu.json", {"complete": False})
    assert not (ROOT / "rms_packed_probe/src/main.rs").exists()
    meta = json.loads((OLD / "reference.json").read_text())
    assert meta["complete"] and meta["data_sha256"] == sha(OLD / "reference.pt")
    data = torch.load(OLD / "reference.pt", weights_only=True)
    assert len(data) == 36
    rows = []
    for label, packet in data.items():
        sums, output = model(packet["input"])
        special(torch, sums, sums[:, :1].expand_as(sums))
        torch.testing.assert_close(sums, sums[:, :1].expand_as(sums), rtol=0, atol=0, equal_nan=True)
        row = qualify(torch, label, packet["input"], output, packet)
        assert row["passes"], row
        rows.append(row)
        print("cpu", label, "PASS", flush=True)
    snapshot = OUT / "phase-sources"
    snapshot.mkdir(exist_ok=True)
    (snapshot / "cpu-rms_packed_trial.py").write_bytes(Path(__file__).read_bytes())
    write("cpu.json", {
        "complete": True, "rows": rows, "gpu_candidate_source_existed": False,
        "reference_sha256": sha(OLD / "reference.json"), "data_sha256": sha(OLD / "reference.pt"),
        "model_source_sha256": sha(ROOT / "rms_vector_trial_data.py"),
        "producer_sha256": sha(snapshot / "cpu-rms_packed_trial.py"),
        "independence": evidence.inventory(),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("cpu",))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "cache/reuse-first-rms-packed-trial" / args.phase
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
