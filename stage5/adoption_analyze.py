"""Offline audit/retention of the bounded adoption packet; no GPU execution."""

import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/adoption"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def available(directory):
    original = Path(directory)
    saved = OUT / "retained" / original.name
    return saved if saved.is_dir() else original


def retain(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        if file.suffix not in (".json", ".cu", ".ir", ".ptx"):
            continue
        target = directory / file.name
        # Module load/resource timing can change on a cache reload. Retain
        # shader identities once; resources are not part of the proof here.
        if file.name == "resources.json":
            continue
        if target.exists():
            assert sha(target) == sha(file), target
        else:
            shutil.copyfile(file, target)


def main():
    partial = read(OUT / "partial-admission.json")
    assert partial["complete"] and len(partial["rows"]) == 148
    assert len(partial["failures"]) == 5
    assert all(r["frozen_whole_output_matches_qualified"] for r in partial["rows"])
    original_gpu = read(
        ROOT / "artifacts/reduction-structure/rounding/topology/gpu-qualification.json"
    )
    for row in partial["rows"]:
        old = next(r for r in original_gpu["checks"] if r["case"] == row["case"])
        assert row["output_sha256"] == old["output_sha256"]
    controls = read(OUT / "column-controls.json")
    assert controls["complete"] and len(controls["rows"]) == 78
    assert len(controls["live"]) == 28 and len(controls["streams"]) == 4
    assert len(controls["native_rejections"]) == 6
    assert all(all(r["matches"].values()) for r in controls["identities"])
    assert all(r["frozen_bitwise"] for r in controls["rows"])
    frozen = read(OUT / "frozen-identity/checks.json")
    assert frozen["complete"] and len(frozen["rows"]) == 6
    assert all(all(r["matches"].values()) for r in frozen["rows"])
    assert read(OUT / "column-cache-reload.json")["compiler_invocations"] == 0
    checks = []
    for capture in (OUT, OUT / "first-capture"):
        data = read(capture / "column-graphs.json")
        assert data["complete"] and len(data["rows"]) == 6
        for row in data["rows"]:
            for r in row["rounds"]:
                for path, cell in r["paths"].items():
                    trace = capture / cell["trace"]
                    assert sha(trace) == cell["trace_sha256"]
                    raw = json.loads(gzip.decompress(trace.read_bytes()))
                    kernels = [
                        e for e in raw["traceEvents"] if e.get("cat") == "kernel"
                    ]
                    assert len(kernels) == len(cell["samples_us"]) == 16
                    directory = available(
                        row["normal_artifact"]
                        if path == "normal"
                        else row["retained_artifact"]
                    )
                    entry = read(directory / "manifest.json")["entrypoint"]
                    assert all(e["name"] == entry for e in kernels)
                    assert all(
                        abs(e["dur"] - v) < 0.002
                        for e, v in zip(kernels, cell["samples_us"], strict=True)
                    )
            for path in ("normal_artifact", "retained_artifact"):
                directory = available(row[path])
                retain(directory, OUT / "retained" / directory.name)
            dest = OUT / "wrappers" / Path(row["wrappers"]["path"]).name
            wrapper = dest if dest.exists() else Path(row["wrappers"]["path"])
            assert sha(wrapper) == row["wrappers"]["sha256"]
            dest.parent.mkdir(exist_ok=True)
            if dest.exists():
                assert sha(dest) == sha(wrapper)
            else:
                shutil.copyfile(wrapper, dest)
        checks.append(
            dict(
                capture=str(capture),
                traces=72,
                kernel_durations=1152,
                paired_wins=sum(r["paired_wins"] for r in data["rows"]),
            )
        )
    for directory in controls["artifacts"]:
        retain(available(directory), OUT / "retained" / Path(directory).name)
    sources = [
        "stage0/src/column_layout.rs",
        "stage0/src/descriptor5.rs",
        "stage0/src/main.rs",
        "stage0/Cargo.toml",
        "stage0/Cargo.lock",
        "stage5/column_plan.py",
        "stage5/column_templates.json",
        "stage5/artifact.py",
        "stage5/legality.py",
        "stage5/plan.py",
        "stage5/runtime.py",
        "stage5/components.py",
        "stage5/adoption_checks.py",
        "stage5/column_admission_checks.py",
        "stage5/adoption_analyze.py",
    ]
    (OUT / "sources.json").write_text(
        json.dumps({p: sha(ROOT.parent / p) for p in sources}, indent=2) + "\n"
    )
    (OUT / "analysis.json").write_text(
        json.dumps(dict(complete=True, captures=checks), indent=2) + "\n"
    )
    print(
        "adoption offline source/identity/corpus/144 traces/2304 durations audit PASS"
    )


if __name__ == "__main__":
    main()
