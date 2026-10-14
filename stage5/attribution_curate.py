"""Validate and preserve a bounded evidence packet without raw binary traces."""

import json
from pathlib import Path
import shutil
import subprocess

from stage5.attribution import sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/attribution"


def packet(source, target):
    target.mkdir(parents=True, exist_ok=True)
    names = (
        "semantic.json",
        "physical-plan.json",
        "descriptor.json",
        "cubecl.ir",
        "kernel.cu",
        "kernel.ptx",
        "manifest.json",
        "nvrtc.json",
        "resources.json",
        "integrity.json",
    )
    for name in names:
        shutil.copyfile(source / name, target / name)
    return {
        "path": str(target.relative_to(ROOT.parent)),
        "source": str(source),
        "sha256": {name: sha(target / name) for name in names},
    }


def main():
    records = []
    launches = []
    for impl in ("cubecl", "triton"):
        for history in ("large_first", "short_first"):
            path = OUT / f"history-{impl}-{history}.json"
            data = json.loads(path.read_text())
            last = data["records"][-1]
            assert last["shape"] == [8, 65539]
            assert last["new_graphs"] == (history == "large_first")
            count = 2 if history == "large_first" else 1
            assert all(len(p["kernels"]) == count for p in last["profiles"])
            trace = Path(last["profiles"][0]["raw_trace"])
            launches.append(
                {
                    "name": path.stem,
                    "trace_sha256": sha(trace),
                    "kernels": [
                        e
                        for e in json.loads(trace.read_text())["traceEvents"]
                        if e.get("cat") == "kernel"
                    ],
                }
            )
            if impl == "cubecl":
                names = {k["name"] for k in last["profiles"][0]["kernels"]}
                for item in data["artifacts"]:
                    source = Path(item["directory"])
                    manifest = json.loads((source / "manifest.json").read_text())
                    if manifest["entrypoint"] in names:
                        records.append(
                            packet(
                                source,
                                OUT
                                / "representatives"
                                / f"{path.stem}-{manifest['entrypoint']}",
                            )
                        )
            else:
                target = OUT / "representatives" / path.stem
                target.mkdir(parents=True, exist_ok=True)
                for item in data["wrapper_files"]:
                    source = Path(item["path"])
                    if "triton_red" in source.read_text():
                        shutil.copyfile(source, target / source.name)
    for suffix in ("cubecl", "triton", "cubecl-probe", "triton-probe"):
        data = json.loads((OUT / f"constants-{suffix}.json").read_text())
        assert len(data["records"]) == 18
        for row in data["records"]:
            for value in row["outputs"]:
                assert (
                    value["classification_equal"]
                    and value["zero_sign_equal"]
                    and value["dtype_equal"]
                )
                if (
                    suffix.endswith("probe")
                    or row["dtype"] == "torch.float32"
                    or row["variant"]
                    not in (
                        "fused_constructor",
                        "returned_constructor",
                        "explicit_roundtrip",
                    )
                ):
                    assert value["numeric_equal"]
        if suffix.startswith("cubecl"):
            row = next(
                r
                for r in data["records"]
                if r["dtype"] == "torch.float16" and r["variant"] == "fused_constructor"
            )
            names = {k["name"] for k in row["profile"]["kernels"]}
            for item in data["artifacts"]:
                source = Path(item["directory"])
                manifest = json.loads((source / "manifest.json").read_text())
                if manifest["entrypoint"] in names:
                    records.append(
                        packet(
                            source, OUT / "representatives" / ("constructor-" + suffix)
                        )
                    )
    systems = json.loads((OUT / "systems-analysis.json").read_text())
    assert len(systems) == 19
    for record in systems:
        data = record["capture"]
        assert len(record["steps"]) == data["steps"]
        if data["implementation"] in ("cubecl", "reference"):
            assert data["warm_compiles"] == data["warm_tunes"] == 0
            assert data["environment"]["gil"] is False
            independence = data["environment"]["independence"]
            assert (
                not independence["loaded_modules"]
                and not independence["native_mappings"]
            )
            counts = {
                (s["kernel_count"], s["generated_count"]) for s in record["steps"]
            }
            assert counts == ({(20, 8)} if data["size"] == "tiny" else {(22, 9)})
        assert not any("Synchronize" in name for name in record["api"]), (
            "Unexpected synchronization inside step"
        )
    launch = json.loads((ROOT / "artifacts/launch/performance.json").read_text())
    source_hashes = {
        name: sha(ROOT / name)
        for name in ("artifact.py", "runtime.py", "legality.py", "diagnostics.py")
    }
    assert all(
        launch["source_sha256"][name] == value for name, value in source_hashes.items()
    )
    patch = ROOT.parent / "fp_hardening/cubecl-fp.patch"
    assert (
        sha(patch) == "9ae155efb01454be6cc61766db3564fce49bcf8495af753c60afab5df07c97d2"
    )
    nsys = Path(shutil.which("nsys")).resolve()
    manifest = {
        "production_matches_launch_capture": source_hashes,
        "fp_patch_sha256": sha(patch),
        "nsys": {
            "path": str(nsys),
            "sha256": sha(nsys),
            "version": subprocess.check_output(
                [str(nsys), "--version"], text=True
            ).strip(),
        },
        "packets": records,
        "validation": "four symmetric histories; 18 variants per backend/mode; 19 Systems captures; native/GIL/Triton/synchronization checks",
    }
    (OUT / "evidence-manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "reduction-launches.json").write_text(json.dumps(launches, indent=2))
    shutil.copyfile(
        ROOT / "artifacts/launch/controls.json", OUT / "launch-controls.json"
    )
    print("Attribution evidence checks PASS", flush=True)


if __name__ == "__main__":
    main()
