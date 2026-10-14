"""Retain focused new precision evidence and identify failed/preliminary attempts."""

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUTPUT = ROOT / "artifacts/precision"
RAW = ROOT / "cache/precision/attempts"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    attempts_path = OUTPUT / "attempts.json"
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    for stem in (
        "cubecl-attempt1",
        "cubecl-attempt2",
        "cubecl-attempt3",
        "cubecl-attempt4",
        "cubecl-final",
        "cubecl-cold-final",
        "cubecl-cold-final2",
        "cpu-attempt1",
        "cpu-attempt2",
        "cpu-attempt3",
        "reduction-order-probe",
    ):
        path = OUTPUT / (stem + ".json")
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        attempts.append(
            {
                "attempt": stem,
                "accepted": data.get("accepted"),
                "failure": data.get("failure"),
                "sha256": sha(path),
                "raw_retained": str(RAW / path.name),
                "qualification": (
                    "superseded passing capture; canonical final matrix is separate"
                    if data.get("accepted") is True
                    else "failed attempt; not acceptance evidence"
                    if data.get("accepted") is False
                    else "exploratory component; not the final acceptance matrix"
                ),
            }
        )
        shutil.move(str(path), RAW / path.name)
    attempts_path.write_text(json.dumps(attempts, indent=2))
    cube = json.loads((OUTPUT / "cubecl.json").read_text())
    assert cube["accepted"] and len(cube["checks"]) == 78 and len(cube["backward"]) == 9
    for name in ("triton", "cpu"):
        data = json.loads((OUTPUT / (name + ".json")).read_text())
        assert (
            data["accepted"]
            and len(data["checks"]) == 81
            and len(data["backward"]) == 9
        )
    for source in ("pytorch.patch", "overlay.json"):
        shutil.copyfile(PROJECT / "stage1/artifacts" / source, OUTPUT / source)
    # These fixed runners refreshed preexisting untracked control summaries in
    # the first full gate run. Save the new data; restore the available snapshot.
    for relative in (
        "stage5/artifacts/launch/controls.json",
        "stage5/artifacts/launch/hook-checks.json",
        "stage5/artifacts/regressions.json",
        "stage5/artifacts/math-cubecl.json",
    ):
        target = OUTPUT / "regressions" / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / relative, target)
    shutil.copyfile(
        ROOT / "artifacts/attribution/launch-controls.json",
        ROOT / "artifacts/launch/controls.json",
    )
    packets = []
    names = (
        "semantic.json",
        "physical-plan.json",
        "descriptor.json",
        "cubecl.ir",
        "kernel.cu",
        "kernel.ptx",
        "manifest.json",
        "nvrtc.json",
        "ptx-signature.txt",
        "resources.json",
        "integrity.json",
    )
    for p in cube["profiles"]:
        artifact = p["artifact"]
        cold = next(
            c
            for c in cube["cold_compiler_checks"]
            if c["case"] == p["case"] and c["dtype"] == p["dtype"]
        )
        directory = Path(cold["directory"])
        assert sha(directory / "kernel.ptx") == sha(
            Path(artifact["directory"]) / "kernel.ptx"
        )
        label = p["case"] + "-" + p["dtype"].removeprefix("torch.")
        target = OUTPUT / "representatives" / label
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            assert (directory / name).is_file(), directory / name
            shutil.copyfile(directory / name, target / name)
        packets.append(
            {
                "case": p["case"],
                "dtype": p["dtype"],
                "directory": str(target.relative_to(PROJECT)),
                "profile": p["kernels"],
                "sha256": {name: sha(target / name) for name in names},
            }
        )
    paths = (
        "stage1/overlay.py",
        "stage1/precision.py",
        "stage5/precision_checks.py",
        "stage5/precision_gates.py",
        "stage5/precision_harness_checks.py",
        "stage5/precision_probe.py",
        "stage5/attribution.py",
        "stage5/reduction_order_checks.py",
        "stage5/precision_curate.py",
        "stage0/src/descriptor3.rs",
        "stage0/src/descriptor5.rs",
        "fp_hardening/cubecl-fp.patch",
    )
    (OUTPUT / "manifest.json").write_text(
        json.dumps(
            {
                "representatives": packets,
                "source_sha256": {p: sha(PROJECT / p) for p in paths},
                "overlay": json.loads((OUTPUT / "overlay.json").read_text()),
                "production_cuda": "PyTorch cu130 / CUDA13.0 NVRTC; no 13.1 diagnostic substitution",
            },
            indent=2,
        )
    )
    print("Precision evidence curation PASS", flush=True)


if __name__ == "__main__":
    main()
