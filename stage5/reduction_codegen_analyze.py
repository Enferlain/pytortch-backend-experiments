"""Bind executed SASS actions to raw NCU rows and warm measurements."""

from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import re
import statistics

from stage5.reduction_codegen_mechanisms import OUT, ROOT, FROZEN


def actions(path):
    result = []
    for line in path.read_text().splitlines():
        if line.startswith("Kernel Name"):
            result.append(
                {
                    "entrypoint": line.split()[-1],
                    "warp": Counter(),
                    "active_threads": Counter(),
                    "scalar_input_load_threads": 0,
                }
            )
        m = re.match(
            r"^0x[0-9a-f]+\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+).*?\s+(\d+)\s+(\d+)\s*$",
            line,
        )
        if m:
            a = result[-1]
            op = m[1].split(".")[0]
            a["warp"][op] += int(m[2])
            a["active_threads"][op] += int(m[3])
            if re.match(r"LDG\.E(?:\.CONSTANT|\.EF)?$", m[1]):
                a["scalar_input_load_threads"] += int(m[3])
    assert result
    return result


def capture(label, identities=None):
    path = OUT / f"{label}-raw.csv"
    rows = list(csv.DictReader(path.open()))
    units, rows = rows[0], rows[1:]
    code = actions(OUT / f"{label}-sass.txt")
    assert len(code) == len(rows)
    if identities is not None:
        assert len(identities) == len(rows)
    result = []
    for i, (a, r) in enumerate(zip(code, rows, strict=True)):
        assert a["entrypoint"] == r["Kernel Name"]
        assert sum(a["warp"].values()) == float(
            r["smsp__inst_executed.sum"].replace(",", "")
        )
        if identities is not None:
            attributes = identities[i]["attributes"]
            expected = attributes.get(
                "registers_per_thread", attributes.get("registers")
            )
            assert int(r["launch__registers_per_thread"].replace(",", "")) == expected
        metrics = {
            k: {"value": v, "unit": units[k]}
            for k, v in r.items()
            if k
            in (
                "launch__registers_per_thread",
                "launch__occupancy_limit_registers",
                "launch__occupancy_per_register_count",
                "gpu__time_duration.sum",
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "smsp__inst_executed.sum",
                "smsp__sass_thread_inst_executed_op_global_ld_pred_on.sum",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
                "derived__local_spilling_requests",
                "derived__shared_spilling_requests",
            )
        }
        result.append(
            {"action": identities[i] if identities else i, **a, "metrics": metrics}
        )
    return result


def timings(path, field):
    data = json.loads(path.read_text())
    result = []
    for c in data["cases"]:
        groups = {}
        for r in c[field]:
            groups.setdefault(r["name"], []).append(r)
        record = {"shape": c["shape"], "implementations": {}}
        for name, rounds in groups.items():
            values = [v for r in rounds for v in r["samples_us"]]
            median = statistics.median(values)
            record["implementations"][name] = {
                "median_us": median,
                "mad_us": statistics.median(abs(v - median) for v in values),
                "round_medians": [r["median_us"] for r in rounds],
            }
        result.append(record)
    return result


def graphs():
    from stage5.reduction_structure_analyze import fingerprint, verify_environment
    from stage5.long_row_analyze import reference_for_extent
    from stage5.reduction_policy import qualify_case
    from stage5.reference_reduction_probe import sha

    captures = {
        impl: [
            json.loads((OUT / f"fixed-graph-{impl}-r{r}.json").read_text())
            for r in range(3)
        ]
        for impl in ("cubecl", "triton")
    }
    old = json.loads((FROZEN / "topology-cubecl-r0.json").read_text())
    efficient = {
        f"triton-{h}": json.loads(
            (ROOT / f"artifacts/long-row/triton-{h}.json").read_text()
        )
        for h in ("short", "mid", "medium", "million")
    }
    qualification = []
    for impl, rounds in captures.items():
        for r, c in enumerate(rounds):
            assert c["qualification_complete"] and len(c["rows"]) == 178
            verify_environment(c["environment"], impl == "cubecl")
            if impl == "cubecl":
                for a, b in zip(c["rows"], old["rows"], strict=True):
                    for k in (
                        "case",
                        "input_sha256",
                        "shape",
                        "dtype",
                        "values",
                        "aten",
                        "fp64",
                    ):
                        assert json.dumps(a[k]) == json.dumps(b[k]), (r, a["case"], k)
                assert c["numerical_failures"] == old["numerical_failures"]
    for r, c in enumerate(captures["cubecl"]):
        checks = []
        for row in c["rows"]:
            reference = (
                captures["triton"][r]
                if 8192 < row["shape"][-1] <= 73728
                else efficient[reference_for_extent(row["shape"][-1])]
            )
            check = qualify_case(
                row, next(x for x in reference["rows"] if x["case"] == row["case"])
            )
            assert check["passes"], check
            checks.append(check)
        qualification.append({"round": r, "checks": checks, "failures": []})
    cases = []
    for case in ("float32-8-1048576", "float32-64-65539"):
        record = {"case": case, "implementations": {}}
        input_hashes = set()
        for impl, rounds in captures.items():
            records = []
            identities = []
            for c in rounds:
                timed = next(t for t in c["timings"] if t["case"] == case)
                input_hashes.add(timed["input_sha256"])
                ids = [
                    fingerprint(k, impl == "cubecl") for k in timed["selected_kernels"]
                ]
                assert len(ids) == 2
                identities.append(ids)
                samples = [[], []]
                sums = []
                for profile in timed["profiles"]:
                    path = Path(profile["raw_trace"])
                    assert sha(path) == profile["sha256"]
                    events = sorted(
                        [
                            e
                            for e in json.loads(path.read_text())["traceEvents"]
                            if e.get("cat") == "kernel"
                        ],
                        key=lambda e: e["ts"],
                    )
                    assert (
                        len(events) == 2
                        and [e["name"] for e in events] == profile["names"]
                    )
                    for i, e in enumerate(events):
                        samples[i].append(e["dur"])
                    sums.append(sum(e["dur"] for e in events))
                assert len(samples[0]) == 7
                records.append(
                    {
                        "partial_median_us": statistics.median(samples[0]),
                        "final_median_us": statistics.median(samples[1]),
                        "graph_median_us": statistics.median(sums),
                        "samples_us": samples,
                    }
                )
            assert all(i == identities[0] for i in identities)
            record["implementations"][impl] = {
                "identity": identities[0],
                "rounds": records,
            }
        assert len(input_hashes) == 1
        record["input_sha256"] = input_hashes.pop()
        cases.append(record)
    return {
        "complete": True,
        "scope": "three C/T,T/C,C/T fresh graph rounds; seven CUPTI captures each; fixed split9/final; primary and secondary partial/final separated",
        "cases": cases,
        "qualification": qualification,
        "frozen_exact_recorded_rows_per_round": 178,
    }


def main():
    identities = json.loads((OUT / "mechanism-capture.json").read_text())["actions"]
    ncu = capture("mechanisms", identities)
    for a in ncu:
        b, n = a["action"]["shape"]
        assert a["scalar_input_load_threads"] == b * n, (
            a["action"],
            a["scalar_input_load_threads"],
        )
    graph = graphs()
    (OUT / "graph-analysis.json").write_text(json.dumps(graph, indent=2))
    result = {
        "complete": True,
        "graphs": graph,
        "ncu_mechanisms": ncu,
        "ncu_controls": {
            case: capture("comparison-" + case) for case in ("million", "64")
        },
        "warm_mechanisms": {
            m: timings(OUT / f"{m}-timing.json", "rounds")
            for m in ("uniform", "owner", "narrow", "uniform_narrow")
        },
        "warm_controls": timings(OUT / "comparisons.json", "timings"),
        "qualification_counts": {
            m: len(json.loads((OUT / f"{m}-qualification.json").read_text())["checks"])
            for m in ("uniform", "owner", "narrow", "uniform_narrow")
        },
    }
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2))
    for a in ncu:
        print(
            a["action"]["shape"],
            a["action"]["name"],
            {
                op: a["warp"][op]
                for op in ("BSSY", "BSYNC", "BAR", "SHFL", "IADD3", "IMAD", "ISETP")
            },
            "loads",
            a["scalar_input_load_threads"],
        )
    files = [
        p
        for p in OUT.rglob("*")
        if p.is_file()
        and p.suffix in (".json", ".cu", ".ptx", ".ir", ".patch", ".txt", ".csv", ".md")
        and p.name != "source-and-artifact-sha256.json"
    ]
    project = ROOT.parent
    files += [
        project / p
        for p in (
            "STAGE5_REPORT.md",
            "stage0/src/descriptor3.rs",
            "stage0/src/main.rs",
            "stage0/Cargo.lock",
            "stage5/codegen_probe/Cargo.lock",
            "stage5/codegen_probe/src/main.rs",
        )
    ]
    files += list(ROOT.glob("reduction_codegen_*.py"))
    (OUT / "source-and-artifact-sha256.json").write_text(
        json.dumps(
            {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
