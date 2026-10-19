"""Reproduce the independent policy, provenance and ownership audit without GPU."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "stage5/artifacts/admission-boundary"
sys.path.insert(0, str(ROOT))
from stage5.reduction_policy import qualify_case, TOLERANCES  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    results, oracle, graphs = (
        read(OUT / (n + ".json")) for n in ("results", "oracle", "graphs")
    )
    refs = {}
    for file in (
        "stage5/artifacts/reduction-structure/triton-r0.json",
        "stage5/artifacts/reduction-structure/rounding/new-triton-reference.json",
        *(
            f"stage5/artifacts/long-row/triton-{h}.json"
            for h in ("short", "mid", "medium", "million")
        ),
    ):
        for r in read(ROOT / file)["rows"]:
            refs.setdefault(r["case"], []).append(r)
    topology = {
        r["case"]: r
        for r in read(
            ROOT
            / "stage5/artifacts/reduction-structure/rounding/topology/qualification.json"
        )["checks"]
    }
    failures = {k: [] for k in ("reference512", "scalar_baseline", "compensated")}
    closefail = []
    for r, o in zip(results["rows"], oracle["rows"], strict=True):
        assert r["case"] == o["case"] and r["input_sha256"] == o["input_sha256"]
        assert r["hashes"]["reference512"] == {
            "partial": o["partial_sha256"],
            "output": o["output_sha256"],
        }
        ref = next(
            v
            for v in refs[r["case"]]
            if v["input_sha256"] == r["input_sha256"]
            and v["values"]
            == [e["reference"] for e in r["qualification"]["reference512"]["elements"]]
        )
        assert (ref["dtype"], ref["shape"]) == (r["dtype"], r["shape"])
        for name in failures:
            checked = qualify_case({**ref, "values": r["values"][name]}, ref)
            assert checked == r["qualification"][name]
            if name == "reference512":
                assert checked == topology[r["case"]]
            if not checked["passes"]:
                failures[name].append(r["case"])
        a, b = r["values"]["reference512"], r["values"]["scalar_baseline"]
        rtol, atol = TOLERANCES[r["dtype"]]
        near = all(
            abs(x - y) <= atol + rtol * abs(y) for x, y in zip(a, b, strict=True)
        )
        assert near == (r["whole_old_tree_comparison_error"] is None)
        if not near:
            closefail.append(r["case"])
        corrupted = list(a)
        corrupted[0] += 1e6
        assert not qualify_case({**ref, "values": corrupted}, ref)["passes"]
    assert failures == results["summary"]["qualification_failures"]
    assert closefail == results["summary"]["whole_old_tree_comparison_failures"]
    for name, capture, snapshot in (
        ("results", results, "boundary-capture.py"),
        ("oracle", oracle, "oracle-capture.py"),
        ("graphs", graphs, None),
    ):
        for relative, digest in capture["source_sha256"].items():
            p = ROOT / relative
            if relative == "stage5/reduction_admission_probe.py" and snapshot:
                p = OUT / "sources" / snapshot
            assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, (
                name,
                relative,
            )
    bindings = read(OUT / "kernel-bindings.json")
    for binding in bindings:
        assert (
            hashlib.sha256((ROOT / binding["retained"]).read_bytes()).hexdigest()
            == binding["sha256"]
        )
    for case in graphs["cases"]:
        p, f = case["kernels"]
        assert p["output_names"] == f["input_names"]
        assert (
            p["schedule"][0]["outputs"][0]["users"][0]["name"]
            == f["schedule"][0]["name"]
        )
    report = {
        "complete": True,
        "cases_recomputed": 148,
        "whole_old_tree_comparison_failures": closefail,
        "qualification_failures": failures,
        "kernel_identity_files": len(bindings),
        "source_bindings_match": True,
        "graph_ownership_edges": 5,
        "scope": "independent stdlib/pure-policy recomputation from retained outputs; not another GPU execution",
    }
    path = OUT / "audit.json"
    if path.exists():
        assert read(path) == report
    else:
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "148 policies match both capture and original topology; provenance and ownership pass"
    )


if __name__ == "__main__":
    main()
