"""Audit the final-only experiment against retained bytes and raw CUPTI data."""

import csv
import hashlib
import json
from pathlib import Path
import re

from stage5.reduction_state_analyze import stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/reduction-structure/final"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kernels(capture):
    trace = Path(capture["trace"])
    assert sha(trace) == capture["trace_sha256"]
    events = json.loads(trace.read_text())["traceEvents"]
    result = sorted(
        (e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"]
    )
    assert "spin_kernel" in result[0]["name"]
    apis = [
        e
        for e in events
        if "LaunchKernel" in e.get("name", "")
        and e.get("cat") in ("cuda_driver", "cuda_runtime")
    ]
    assert len(apis) == len(result)
    lead = result[0]["ts"] + result[0]["dur"] - max(e["ts"] + e["dur"] for e in apis)
    assert lead > 0
    return result[1:], lead


def main():
    packet = OUT / "validated-plane"
    q = json.loads((packet / "gpu-qualification.json").read_text())
    assert q["complete"] and len(q["checks"]) == 152 and len(q["direct_checks"]) == 3
    assert all(c["bitwise_final"] and c["canaries"] for c in q["checks"])
    assert all(c["bitwise"] for c in q["direct_checks"])
    assert all(sha(p) == digest for p, digest in q["frozen_files"].items())
    cpu = json.loads((OUT / "cpu-qualification.json").read_text())
    confirmation = json.loads(
        (OUT / "model-confirmation/cpu-qualification.json").read_text()
    )
    assert (
        cpu["complete"]
        and cpu == confirmation
        and len(cpu["checks"]) == 148
        and not cpu["failures"]
    )
    identity = json.loads((OUT / "code-identity/checks.json").read_text())
    assert identity["complete"] and len(identity["checks"]) == 6
    assert all(all(c["matches"].values()) for c in identity["checks"])
    guarded = json.loads((OUT / "code-identity-after-guard/checks.json").read_text())
    controls = json.loads((OUT / "guard-controls.json").read_text())
    assert guarded["complete"] and len(guarded["checks"]) == 9
    assert all(all(c["matches"].values()) for c in guarded["checks"])
    assert controls["complete"] and len(controls["checks"]) == 4
    assert all(c["exit_code"] != 0 and c["no_target_code"] for c in controls["checks"])
    result = {
        "complete": False,
        "cpu_cases": 148,
        "gpu_cases": 152,
        "final_only_dtype_controls": 3,
        "ordinary_code_identity_files": 18,
        "post_guard_identity_files": 27,
        "unsupported_probe_controls": 4,
        "raw_traces": 0,
        "timing": [],
        "cross_producer": [],
        "sass": {},
    }
    for dtype in ("float32", "float16", "bfloat16"):
        original = OUT / "candidate" / dtype / "kernel.ptx"
        accepted = packet / "candidate" / dtype / "kernel.ptx"
        uniform = OUT / "uniform-load/candidate" / dtype / "kernel.ptx"
        assert original.read_bytes() == accepted.read_bytes() == uniform.read_bytes()
    timing = json.loads((packet / "timing.json").read_text())
    assert timing["complete"] and timing["harness_sha256"] == sha(
        packet / "harness-gpu.py"
    )
    for case in timing["cases"]:
        groups = {}
        for c in case["captures"]:
            raw, _ = kernels(c)
            values = [e["dur"] for e in raw]
            if c["scope"] == "final_only":
                assert values == c["final_us"] and len(values) == 16
            else:
                assert values[::2] == c["partial_us"] and values[1::2] == c["final_us"]
                assert len(values) == 32
            group = groups.setdefault(c["scope"] + "/" + c["impl"], [])
            group.extend(c["final_us"])
            result["raw_traces"] += 1
        result["timing"].append(
            {"shape": case["shape"], "groups": {k: stats(v) for k, v in groups.items()}}
        )
    cross = json.loads((OUT / "cross-producer.json").read_text())
    assert cross["complete"]
    for case in cross["cases"]:
        assert all(sha(p) == digest for p, digest in case["artifacts"].items())
        groups = {}
        for c in case["captures"]:
            raw, lead = kernels(c)
            assert c["bitwise"] and lead == c["prequeue_lead_us"]
            assert [e["dur"] for e in raw[::2]] == c["partial_us"]
            assert [e["dur"] for e in raw[1::2]] == c["final_us"]
            group = groups.setdefault(
                c["producer"] + "/" + c["consumer"],
                {"partial": [], "final": [], "sum": []},
            )
            group["partial"].extend(c["partial_us"])
            group["final"].extend(c["final_us"])
            group["sum"].extend(
                a + b for a, b in zip(c["partial_us"], c["final_us"], strict=True)
            )
            result["raw_traces"] += 1
        result["cross_producer"].append(
            {
                "shape": case["shape"],
                "groups": {
                    k: {phase: stats(v) for phase, v in g.items()}
                    for k, g in groups.items()
                },
            }
        )
    sass = (OUT / "executed-sass.txt").read_text()
    for part in sass.split("Kernel Name")[1:]:
        name = part.splitlines()[0].strip()
        ops = []
        for line in part.splitlines()[1:]:
            match = re.match(r"0x[0-9a-f]+\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\s", line)
            if match:
                ops.append(match[1])
        result["sass"][name] = {
            op: ops.count(op) for op in sorted(set(ops)) if op != "NOP"
        }
    with (OUT / "ncu-raw.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    result["counter_code_structure_only"] = {
        row["Kernel Name"]: {
            k: row[k]
            for k in (
                "launch__registers_per_thread",
                "launch__registers_per_thread_allocated",
                "launch__shared_mem_per_block_dynamic",
                "sass__inst_executed_shared_loads",
                "sass__inst_executed_shared_stores",
            )
        }
        for row in rows[1:]
    }
    aten_cpu = json.loads((OUT / "aten-model/cpu-qualification.json").read_text())
    aten = json.loads((OUT / "aten-reference/reference.json").read_text())
    assert (
        aten_cpu["complete"]
        and len(aten_cpu["checks"]) == 148
        and not aten_cpu["failures"]
    )
    assert aten["complete"] and len(aten["checks"]) == 148
    for c in aten["checks"]:
        old = next(r for r in q["checks"] if r["case"] == c["case"])
        assert c["bitwise"] and c["input_sha256"] == old["input_sha256"]
        assert (
            c["partials_sha256"] == old["partial_sha256"]
            and c["output_sha256"] == old["output_sha256"]
        )
    aten_groups = {}
    for c in aten["captures"]:
        raw, _ = kernels(c)
        measured = raw if c["scope"] == "final_only" else raw[1::2]
        assert len(measured) == 16 and [e["dur"] for e in measured] == c["us"]
        assert all(e["name"] == c["kernel"] for e in measured)
        assert c["launch"]["grid"] == [1, 1, 1]
        assert c["launch"]["block"] == [8, c["b"], 1]
        assert c["launch"]["registers per thread"] == 26
        aten_groups.setdefault(str(c["b"]) + "/" + c["scope"], []).extend(c["us"])
        result["raw_traces"] += 1
    result["aten_reference"] = {k: stats(v) for k, v in aten_groups.items()}
    result["complete"] = True
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2))
    print(
        "Validated",
        result["raw_traces"],
        "raw traces, CPU/GPU gates, frozen code identity and executed SASS",
    )


if __name__ == "__main__":
    main()
