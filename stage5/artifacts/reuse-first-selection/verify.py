"""Offline verification of the selection packet, source snapshot and corpus policy."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(PROJECT), *args])


def main():
    capture = json.loads((OUT / "baseline-capture.json").read_text())
    snapshot = json.loads((OUT / "source-evidence.json").read_text())
    assert capture["harness_sha256"] == sha(OUT / "capture.py")
    assert len(capture["records"]) == 9
    expected = {
        f"rms_inference-{d}-{b}x{n}"
        for d in ("float32", "float16", "bfloat16")
        for b, n in ((64, 1024), (64, 255), (16, 4099))
    }
    assert {r["case"] for r in capture["records"]} == expected

    # bench.py owns a CLI and cannot be imported without running measurements.
    # Compile only its exact pure tolerance function to catch policy drift.
    tree = ast.parse((PROJECT / "stage5/bench.py").read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "tolerances"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "bench.tolerances", "exec"
        ),
        namespace,
    )
    tolerances = namespace["tolerances"]
    unique = set()
    for row in capture["records"]:
        assert row["correctness"] and row["policy"] == "baseline"
        policy = tolerances(SimpleNamespace(dtype=row["dtype"], family="rms_inference"))
        assert {k: row[k] for k in policy} == policy
        assert len(row["launches"]) == len(row["artifacts"]) == 1
        for artifact in row["artifacts"]:
            directory = OUT / artifact["directory"]
            unique.add(artifact["directory"])
            for name, value in artifact["files"].items():
                assert sha(directory / name) == value, (directory, name)
            integrity = json.loads((directory / "integrity.json").read_text())
            for name, value in integrity["files"].items():
                assert sha(directory / name) == value, ("integrity", directory, name)
            packet = json.loads((directory / "descriptor.json").read_text())
            raw = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
            assert hashlib.sha256(raw).hexdigest() == integrity["descriptor"]
            semantic = packet["semantic"]
            assert semantic == json.loads((directory / "semantic.json").read_text())
            assert packet["entrypoint"] == row["launches"][0]["entrypoint"]
            assert semantic["schema"] == 3 and semantic["index_dtype"] == "int64"
            assert len(semantic["buffers"]) == 2
            assert all(b["dtype"] == row["dtype"] for b in semantic["buffers"])
            sums = [o for o in semantic["ops"] if o["op"] == "reduction"]
            stores = [o for o in semantic["ops"] if o["op"] == "store"]
            assert len(sums) == len(stores) == 1
            assert sums[0]["reduction_type"] == "sum"
            assert sums[0]["dtype"] == sums[0]["accumulator_dtype"] == "float32"
            assert stores[0]["iteration_scope"] == "reduction"
    assert len(unique) == 3
    independence = capture["independence"]
    assert not independence["loaded_modules"] and not independence["native_mappings"]
    assert independence["gil_states"] and not any(independence["gil_states"].values())
    for source in snapshot["sources"]:
        path = PROJECT / source["path"]
        assert sha(path) == source["sha256"], source["path"]
        lines = path.read_text().splitlines()
        for anchor in source["anchors"]:
            assert lines[anchor["line"] - 1] == anchor["text"]
    for name, value in snapshot["frozen_source_files"].items():
        assert sha(PROJECT / name) == value
        assert hashlib.sha256(git("show", "HEAD:" + name)).hexdigest() == value
    for name, upstream in snapshot["upstream"].items():
        directory = PROJECT / ".stage0-sources" / name
        assert (
            git("-C", str(directory), "rev-parse", "HEAD").decode().strip()
            == upstream["revision"]
        )
        assert (
            hashlib.sha256(git("-C", str(directory), "diff", "HEAD")).hexdigest()
            == upstream["diff_sha256"]
        )
    history = snapshot["historical_measurements"]
    assert sha(PROJECT / history["source"]) == history["sha256"]
    result = {
        "complete": True,
        "scope": "selection and unmodified baseline only; no candidate qualification/timing",
        "baseline_cases": 9,
        "baseline_artifact_sets": len(unique),
        "source_files_checked": len(snapshot["sources"]),
        "protected_sources_equal_HEAD": True,
        "upstream_diffs_unchanged": True,
        "historical_measurements_unchanged": True,
        "triton_unloaded_and_GIL_disabled": True,
        "exact_corpus_tolerances_checked": True,
        "sum_store_and_integrity_checked": True,
        "harness_sha256": sha(OUT / "capture.py"),
        "verifier_sha256": sha(Path(__file__)),
        "baseline_capture_sha256": sha(OUT / "baseline-capture.json"),
        "source_evidence_sha256": sha(OUT / "source-evidence.json"),
    }
    (OUT / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "PASS:9 baseline cases,3 artifact sets,35 sources,corpus policy,integrity,frozen sources"
    )


if __name__ == "__main__":
    main()
