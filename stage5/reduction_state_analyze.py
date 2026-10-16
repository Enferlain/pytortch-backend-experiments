"""Validate raw CUPTI primer captures without trimming or pooling scopes."""

import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/reduction-structure/state-primers"
RUNS = ("matrix-r0", "matrix-r1", "read-write-r2", "selector-r3", "footprint-r4")
TARGET = "cubecl5_42026f8357481503"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    mid = statistics.median(values)
    return {
        "n": len(values),
        "median_us": mid,
        "mad_us": statistics.median(abs(v - mid) for v in values),
        "min_us": min(values),
        "max_us": max(values),
        "samples_us": values,
    }


def main():
    result = {"complete": False, "runs": {}, "validated_raw_traces": 0}
    identities = None
    selector_primer_hashes = set()
    for name in RUNS:
        directory = OUT / name
        d = json.loads((directory / "result.json").read_text())
        assert d["complete"] and d["shape"] == [64, 65539] and d["split"] == 9
        assert (
            d["input_sha256"]
            == "199358562e389b69090676348ef457ed0141f285d0609c9dd4ff6bfe3b44e218"
        )
        assert d["harness_sha256"] == sha(directory / "harness.py")
        assert d["primer_ptx_sha256"] == sha(directory / "primers/kernel.ptx")
        if name in ("read-write-r2", "selector-r3", "footprint-r4"):
            selector_primer_hashes.add(d["primer_ptx_sha256"])
        current = (d["artifact_sha256"], d["partial_sha256"], d["output_sha256"])
        if identities is None:
            identities = current
        assert current == identities
        for p, digest in d["artifact_sha256"].items():
            assert sha(p) == digest
        rows = {}
        for c in d["captures"]:
            trace = Path(c["trace"])
            assert sha(trace) == c["trace_sha256"] and c["bitwise_partial_and_output"]
            events = json.loads(trace.read_text())["traceEvents"]
            kernels = sorted(
                (e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"]
            )
            assert [{"name": e["name"], "us": e["dur"]} for e in kernels] == c[
                "kernels"
            ]
            assert (
                kernels[-1]["name"] == TARGET and kernels[-1]["dur"] == c["target_us"]
            )
            assert kernels[-1]["args"]["block"] == [512, 1, 1]
            assert kernels[-1]["args"]["grid"] == [576, 1, 1]
            assert kernels[-1]["args"]["registers per thread"] == 40
            assert kernels[-1]["args"]["shared memory"] == 64
            assert "spin_kernel" in kernels[0]["name"]
            apis = [
                e
                for e in events
                if "LaunchKernel" in e.get("name", "")
                and e.get("cat") in ("cuda_driver", "cuda_runtime")
            ]
            assert len(apis) == len(kernels)
            lead = (
                kernels[0]["ts"]
                + kernels[0]["dur"]
                - max(e["ts"] + e["dur"] for e in apis)
            )
            assert lead == c["prequeue_lead_us"] and lead > 0
            row = rows.setdefault(
                c["mode"], {"target": [], "primer": [], "gap": [], "lead": []}
            )
            row["target"].append(c["target_us"])
            row["primer"].append(sum(e["dur"] for e in kernels[2:-1]))
            row["gap"].append(c["target_gap_us"])
            row["lead"].append(lead)
            result["validated_raw_traces"] += 1
        result["runs"][name] = {
            mode: {k: stats(v) for k, v in row.items()} for mode, row in rows.items()
        }
    result["identity"] = {
        "artifact_sha256": identities[0],
        "partial_sha256": identities[1],
        "output_sha256": identities[2],
    }
    result["selectors"] = {}
    assert len(selector_primer_hashes) == 1
    result["selector_primer_ptx_sha256"] = next(iter(selector_primer_hashes))
    for name in ("read-write-r2", "selector-r3", "footprint-r4"):
        fast = result["runs"][name]["write_read"]["target"]
        slow = result["runs"][name]["read_write"]["target"]
        result["selectors"][name] = {
            "fast": fast,
            "slow": slow,
            "all_samples_separated": fast["max_us"] < slow["min_us"],
            "observed_gap_us": slow["min_us"] - fast["max_us"],
        }
    result["complete"] = True
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2))
    print(
        "Validated",
        result["validated_raw_traces"],
        "traces; selector separation:",
        {k: v["all_samples_separated"] for k, v in result["selectors"].items()},
    )


if __name__ == "__main__":
    main()
