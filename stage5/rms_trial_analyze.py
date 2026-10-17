"""Verify retained RMS trial evidence, preserved files and actual timing/code."""

import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/reuse-first-rms"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    integrity = read(OUT / "evidence-integrity.json")
    for name, h in integrity["artifacts"].items():
        assert sha(OUT / name) == h, name
    for name, h in integrity["live_files"].items():
        assert sha(Path(name)) == h, name
    components = read(OUT / "component-sources.json")
    assert len(components["actual_component_bodies"]) == 6
    for row in components["actual_component_bodies"]:
        assert row["unchanged"]
        for name in ("source", "exercised"):
            assert sha(ROOT.parent / row[name]) == row["sha256"], row[name]
    locks = [
        tomllib.loads(p.read_text())
        for p in (ROOT / "rms_probe/Cargo.lock", ROOT / "codegen_probe/Cargo.lock")
    ]
    assert [p for p in locks[0]["package"] if "source" in p] == [
        p for p in locks[1]["package"] if "source" in p
    ]
    ncu = read(OUT / "ncu-provenance.json")
    assert sha(Path(ncu["executable"])) == ncu["sha256"]
    assert shlex.split((OUT / "counter-command.txt").read_text()) == ncu["command"]
    environment = read(OUT / "environment.json")
    assert (
        sha(ROOT / "cache/reuse-first-rms/target/release/stage5-rms-probe")
        == (environment["isolated_compiler_binary_sha256"])
    )
    assert (
        sha(ROOT / "rms_probe/Cargo.lock") == environment["isolated_cargo_lock_sha256"]
    )
    assert sha(Path(environment["nvrtc"]["path"])) == environment["nvrtc"]["sha256"]
    for name, h in environment["nvrtc_builtins"].items():
        assert sha(Path(name)) == h
    reference, cpu, gpu, graphs = [
        read(OUT / (n + ".json")) for n in ("reference", "cpu", "gpu", "graphs")
    ]
    assert all(x["complete"] for x in (reference, cpu, gpu, graphs))
    assert len(reference["rows"]) == len(cpu["rows"]) == len(gpu["rows"]) == 66
    assert len(graphs["rows"]) == 9 and len(gpu["rejected"]) == 21
    assert sha(OUT / "reference.pt") == reference["data_sha256"]
    assert sha(OUT / "reference.json") == cpu["reference_sha256"]
    assert sha(OUT / "cpu-model.pt") == cpu["model_sha256"]
    assert sha(ROOT / "rms_trial_data.py") == cpu["source_sha256"]["rms_trial_data.py"]
    assert (
        sha(OUT / "phase-sources/reference-rms_trial.py")
        == reference["source_sha256"]["rms_trial.py"]
    )
    assert (
        sha(OUT / "phase-sources/cpu-rms_trial.py")
        == cpu["source_sha256"]["rms_trial.py"]
    )
    assert all(r["passes"] and not r["failures"] for r in cpu["rows"] + gpu["rows"])
    assert [r["input_sha256"] for r in cpu["rows"]] == [
        r["input_sha256"] for r in reference["rows"]
    ]
    assert [r["input_sha256"] for r in gpu["rows"]] == [
        r["input_sha256"] for r in reference["rows"]
    ]
    for p, h in reference["code"].items():
        assert sha(OUT / p) == h
    for p, h in graphs["wrappers"].items():
        assert sha(OUT / p) == h

    preservation = read(OUT / "preservation.json")
    for name, h in preservation["files"].items():
        assert sha(ROOT.parent / name) == h, name
    for name, saved in preservation["upstream"].items():
        path = ROOT.parent / ".stage0-sources" / name
        assert (
            subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            == saved["revision"]
        )
        assert (
            hashlib.sha256(
                subprocess.check_output(["git", "-C", str(path), "diff", "HEAD"])
            ).hexdigest()
            == saved["diff_sha256"]
        )

    cases, traces = [], 0
    for row in graphs["rows"]:
        scopes = {}
        for scope in ("rounds", "isolated"):
            samples = {"ordinary": [], "component": []}
            rounds = {"ordinary": [], "component": []}
            for r in row[scope]:
                p = OUT / r["trace"]
                assert sha(p) == r["trace_sha256"]
                events = [e for e in read(p)["traceEvents"] if e.get("cat") == "kernel"]
                assert len(events) == 16 and all(
                    e["name"] == r["kernel"] for e in events
                )
                durations = [e["dur"] for e in events]
                assert all(v > 0 for v in durations)
                assert all(
                    abs(a - b) <= 0.001
                    for a, b in zip(
                        sorted(durations), sorted(r["samples_us"]), strict=True
                    )
                )
                samples[r["name"]].extend(r["samples_us"])
                rounds[r["name"]].append(statistics.median(r["samples_us"]))
                traces += 1
            if row[scope]:
                assert len(rounds["ordinary"]) == len(rounds["component"]) == 6
                scopes[scope] = {
                    n: {
                        "median_us": statistics.median(v),
                        "mad_us": statistics.median(
                            abs(x - statistics.median(v)) for x in v
                        ),
                        "round_medians_us": rounds[n],
                    }
                    for n, v in samples.items()
                }
                scopes[scope]["component_winning_rounds"] = sum(
                    a < b
                    for a, b in zip(
                        rounds["component"], rounds["ordinary"], strict=True
                    )
                )
        cases.append(
            {
                "case": row["case"],
                "shape": row["shape"],
                "ordinary_plan": row["ordinary_plan"],
                **scopes,
            }
        )
    assert traces == 120

    # Guard-only runtime corrections after timing must preserve emitted GPU code.
    identity = []
    for row in graphs["rows"]:
        dtype = row["case"].split("_")[1]
        old = Path(row["component_directory"])
        new = Path(gpu["kernels"][dtype]["directory"])
        old_entry, new_entry = [
            read(p / "manifest.json")["entrypoint"] for p in (old, new)
        ]
        for name in ("cubecl.ir", "kernel.cu", "kernel.ptx"):
            assert (old / name).read_text().replace(old_entry, "ENTRY") == (
                new / name
            ).read_text().replace(new_entry, "ENTRY"), (dtype, name)
        identity.append(
            {
                "case": row["case"],
                "timed_and_final_qualified_code_equal_after_entrypoint_renaming": True,
            }
        )

    # Actual driver-JIT SASS counters; missing raw barrier metrics remain absent.
    parsed, name, columns = {}, None, None
    for line in (OUT / "executed-sass.txt").read_text().splitlines():
        if line.startswith("Kernel Name "):
            name, columns = line.split()[2], None
            parsed[name] = {"warp_instructions": 0, "opcodes": {}}
        elif line.startswith("Address "):
            columns = (
                line.index("Source"),
                line.index("Warp S"),
                line.index("Instru"),
                line.index("Thread"),
            )
            assert 0 < columns[0] < columns[1] < columns[2] < columns[3]
        elif name and re.match(r"^0x[0-9a-f]+ ", line):
            assert columns is not None
            a, b, c, d = columns
            inst, count = line[a:b].strip(), int(line[c:d].strip().replace(",", ""))
            opcode = re.match(r"(?:@!?U?P\d+\s+)?([A-Z0-9.]+)", inst)
            assert opcode, inst
            op = opcode[1].split(".")[0]
            parsed[name]["warp_instructions"] += count
            parsed[name]["opcodes"][op] = parsed[name]["opcodes"].get(op, 0) + count
    raw = [
        r
        for r in csv.DictReader((OUT / "ncu-raw-import.csv").read_text().splitlines())
        if r["ID"].isdigit()
    ]
    assert len(raw) == 2
    counters = []
    for r in raw:
        p = parsed[r["Kernel Name"]]
        assert (
            Decimal(r["smsp__inst_executed.sum"].replace(",", ""))
            == p["warp_instructions"]
        )
        counters.append(
            {
                "name": r["Kernel Name"],
                "grid": r["Grid Size"],
                "block": r["Block Size"],
                "registers": r["launch__registers_per_thread"],
                "shared_loads": r["smsp__sass_inst_executed_op_shared_ld.sum"],
                "shared_stores": r["smsp__sass_inst_executed_op_shared_st.sum"],
                "raw_barrier_metric": r.get("smsp__sass_inst_executed_op_barrier.sum"),
                **p,
            }
        )
    assert set(parsed) == {
        k["entrypoint"] for k in read(OUT / "counter-kernels.json")["kernels"]
    }
    admission = read(OUT / "admission.json")
    assert admission["complete"] and len(admission["negative_cases"]) == 12
    assert all(r["rejected_before_IR"] for r in admission["negative_cases"])
    result = {
        "complete": True,
        "policy": "reference-or-better-v1 plus unchanged eager and special-value gates",
        "cpu_cases": 66,
        "gpu_cases": 66,
        "output_elements": sum(r["elements"] for r in cpu["rows"]),
        "preserved_files": len(preservation["files"]),
        "raw_timing_traces": traces,
        "cases": cases,
        "code_identity": identity,
        "executed_sass": counters,
        "decision": "Qualified negative result; performance stop condition reached; no alternative mechanism or automatic eligibility",
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                ROOT / "rms_trial.py",
                ROOT / "rms_trial_data.py",
                ROOT / "rms_trial_runtime.py",
                ROOT / "rms_trial_backend.py",
                ROOT / "rms_probe/src/main.rs",
                ROOT / "rms_probe/Cargo.toml",
                ROOT / "rms_probe/Cargo.lock",
                Path(__file__),
            ]
        },
    }
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "PASS:",
        traces,
        "traces,",
        len(preservation["files"]),
        "preserved files,66 CPU/GPU cases and actual SASS",
    )
    for r in cases:
        print(
            r["case"],
            {n: r["rounds"][n]["median_us"] for n in ("ordinary", "component")},
        )


if __name__ == "__main__":
    main()
