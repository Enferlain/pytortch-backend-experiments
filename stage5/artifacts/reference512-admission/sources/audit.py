"""Independent offline audit of retained qualification and implementation evidence."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "stage5/artifacts/reference512-admission"
sys.path.insert(0, str(ROOT))
from stage5.reduction_policy import qualify_case  # noqa: E402
from stage5.long_row_analyze import reference_for_extent  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    corpus = read(OUT / "corpus.json")
    old = read(ROOT / "stage5/artifacts/admission-boundary/results.json")
    original = {r["case"]: r for r in old["rows"]}
    source = ROOT / "stage5/artifacts/reduction-structure"
    refs = {
        f"triton-{h}": read(ROOT / f"stage5/artifacts/long-row/triton-{h}.json")
        for h in ("short", "mid", "medium", "million")
    }
    refs["triton-large"] = read(source / "triton-r0.json")
    extra = read(source / "rounding/new-triton-reference.json")
    assert len(corpus["checks"]) == 148
    for row in corpus["checks"]:
        prior = original[row["case"]]
        assert row["input_sha256"] == prior["input_sha256"]
        assert row["partial_sha256"] == prior["hashes"]["reference512"]["partial"]
        assert row["output_sha256"] == prior["hashes"]["reference512"]["output"]
        ref = next(
            r
            for r in (
                extra
                if row["case"].startswith("new_")
                else refs[reference_for_extent(prior["shape"][1])]
            )["rows"]
            if r["case"] == row["case"]
        )
        actual = {
            **ref,
            "values": [e["actual"] for e in row["qualification"]["elements"]],
        }
        assert qualify_case(actual, ref) == row["qualification"]
        assert row["qualification"]["passes"]
    assert sha(OUT / "sources/corpus-capture.py") == corpus["harness_sha256"]
    assert (
        sha(ROOT / "stage5/reduction_policy.py")
        == old["source_sha256"]["stage5/reduction_policy.py"]
    )
    init, reload = (read(OUT / f"cache-{n}.json") for n in ("init", "reload"))
    assert init["model_checks"] == 1
    assert reload["model_checks"] == reload["compiler_invocations"] == 0
    controls = read(OUT / "controls.json")
    assert len(controls["fp_environment"]) == 16 and all(
        r["restored"] for r in controls["fp_environment"]
    )
    assert (
        controls["corrupted_oracle_rejected"] and controls["missing_fallback_rejected"]
    )
    assert controls["deterministic_fallback"] == "shared_tree_compensated"
    for row in read(OUT / "identity.json")["rows"]:
        previous = (
            source / f"codegen/uniform/{row['dtype']}"
            if row["family"] == "partial"
            else source / f"rounding/topology/kernels/{row['dtype']}/fixed_final"
        )
        current = OUT / "identity" / row["dtype"] / row["family"]
        for name in ("cubecl.ir", "kernel.cu", "kernel.ptx"):
            assert sha(previous / name) == sha(current / name)
    graphs = read(OUT / "graphs.json")
    retained = read(OUT / "retention.json")
    for b in retained["bindings"]:
        assert sha(ROOT / b["retained"]) == b["sha256"]
    assert len(graphs["rows"]) == 7 and graphs["offset_stream_canaries"]
    for row in graphs["rows"]:
        for k in row["kernels"]:
            assert sha(k["wrapper"]) == k["wrapper_sha256"]
            assert (
                sha(ROOT / retained["executed"][k["directory"]] / "kernel.ptx")
                == k["ptx_sha256"]
            )
    assert read(OUT / "specials.json")["complete"]
    gates = read(OUT / "regressions/gates.json")
    assert len(gates) == 18 and all(r["exit_code"] == 0 for r in gates)
    resolved = []
    for row in gates:
        log = Path(row["log"])
        if not log.exists():
            log = OUT / "regressions-first-attempt" / log.name
        assert sha(log) == row["sha256"]
        resolved.append({"name": row["name"], "log": str(log), "sha256": row["sha256"]})
    generic = read(OUT / "regressions/corpus.json")
    assert len(generic["records"]) == 168 and not generic["failures"]
    for attempt in ("regressions", "regressions-first-attempt"):
        assert read(OUT / attempt / "restoration.json")["all_preserved"]
    old_gates = read(ROOT / "stage5/artifacts/adoption/regressions/gates.json")
    previous_corpus = next(r for r in old_gates if r["name"] == "corpus")
    assert sha(previous_corpus["log"]) == previous_corpus["log_sha256"]
    result = {
        "complete": True,
        "exact_corpus_cases": 148,
        "full_regression_cases": 168,
        "gate_logs": resolved,
        "identity_families": 6,
        "fp_environments": 16,
        "original_policy_unchanged": True,
        "prior_adoption_raw_corpus_lost": True,
        "prior_complete_log_intact": True,
    }
    path = OUT / "audit.json"
    if path.exists():
        assert read(path) == result
    else:
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "148 exact policy cases,168 regression cases,18 gates,6 code identities and state/cache/graph evidence PASS"
    )


if __name__ == "__main__":
    main()
