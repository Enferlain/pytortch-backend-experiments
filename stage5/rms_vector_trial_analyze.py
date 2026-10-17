"""Offline evidence verification; no GPU work or recapture."""

import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import tomllib

from stage5.rms_vector_trial_data import OLD, OUT, ROOT


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    (OUT / "analysis.json").write_text('{"complete":false}\n')
    integrity = read(OUT / "evidence-integrity.json")
    for name, h in integrity["artifacts"].items():
        assert sha(OUT / name) == h, name
    for name, h in integrity["live_files"].items():
        assert sha(Path(name)) == h, name
    provenance = read(OUT / "ncu-provenance.json")
    assert sha(Path(provenance["executable"])) == provenance["sha256"]
    assert (
        shlex.split((OUT / "counter-command.txt").read_text()) == provenance["command"]
    )
    environment = read(OUT / "environment.json")
    assert environment["device"] == "510ddc27-95ac-c62f-d49b-242cc5183c72"
    assert sha(Path(environment["nvrtc"]["path"])) == environment["nvrtc"]["sha256"]
    for name, h in environment["nvrtc_builtins"].items():
        assert sha(Path(name)) == h
    for r in read(OLD / "component-sources.json")["actual_component_bodies"]:
        for key in ("source", "exercised"):
            assert sha(ROOT.parent / r[key]) == r["sha256"]
    ref, cpu, gpu, graphs, scalar, admission = [
        read(OUT / (n + ".json"))
        for n in (
            "reference",
            "cpu",
            "gpu",
            "graphs",
            "scalar-qualification",
            "admission",
        )
    ]
    assert all(d["complete"] for d in (ref, cpu, gpu, graphs, scalar, admission))
    assert len(cpu["rows"]) == len(gpu["rows"]) == len(scalar["rows"]) == 36
    assert (
        len(ref["rejected_old_cases"]) == 51 and not cpu["gpu_candidate_source_existed"]
    )
    assert len(gpu["rejected"]) == 27 and gpu["stream_offset_controls"] == 3
    assert len(admission["negative_cases"]) == 12 and all(
        r["rejected_before_IR"] for r in admission["negative_cases"]
    )
    assert all(
        r["passes"] and not r["failures"]
        for r in cpu["rows"] + gpu["rows"] + scalar["rows"]
    )
    assert sha(OUT / "reference.pt") == ref["data_sha256"]
    assert sha(OLD / "reference.json") == ref["parent_reference_sha256"]
    assert sha(OLD / "reference.pt") == ref["parent_data_sha256"]
    assert sha(OUT / "reference.json") == cpu["reference_sha256"]
    assert sha(OUT / "cpu-model.pt") == cpu["model_sha256"]
    for phase, record in (("reference", ref), ("cpu", cpu)):
        assert (
            sha(OUT / f"phase-sources/{phase}-rms_vector_trial.py")
            == record["source_sha256"]["rms_vector_trial.py"]
        )
        assert (
            sha(ROOT / "rms_vector_trial_data.py")
            == record["source_sha256"]["rms_vector_trial_data.py"]
        )
    recheck = read(OUT / "cpu-recheck.json")
    assert recheck["complete"] and recheck["gpu_candidate_source_existed"]
    assert len(recheck["rows"]) == 36 and all(r["passes"] for r in recheck["rows"])
    assert sha(OUT / "cpu-recheck-model.pt") == recheck["model_sha256"]
    for name, h in ref["code"].items():
        assert sha(OUT / name) == h
    for name, h in graphs["wrappers"].items():
        assert sha(OUT / name) == h
    qualified = {r["case"]: r for r in gpu["rows"]}
    scalars = {r["case"]: r for r in scalar["rows"]}
    assert set(qualified) == set(scalars) == set(ref["input_sha256"])
    for r in cpu["rows"] + gpu["rows"] + scalar["rows"]:
        assert r["input_sha256"] == ref["input_sha256"][r["case"]]

    cases, trace_count = [], 0
    assert len(graphs["rows"]) == 9
    for row in graphs["rows"]:
        assert row["input_sha256"] == ref["input_sha256"][row["case"]]
        assert (
            row["output_sha256"]["component"] == qualified[row["case"]]["output_sha256"]
        )
        assert row["output_sha256"]["scalar"] == scalars[row["case"]]["output_sha256"]
        scopes = {}
        for scope in ("isolated", "rounds"):
            samples = {n: [] for n in ("ordinary", "scalar", "component")}
            rounds = {n: [] for n in samples}
            clocks = set()
            for r in row[scope]:
                path = OUT / r["trace"]
                assert sha(path) == r["trace_sha256"]
                events = [
                    e for e in read(path)["traceEvents"] if e.get("cat") == "kernel"
                ]
                assert len(events) == 16 and all(
                    e["name"] == r["kernel"] and e["dur"] > 0 for e in events
                )
                assert all(
                    abs(a - b) <= 0.001
                    for a, b in zip(
                        sorted(e["dur"] for e in events),
                        sorted(r["samples_us"]),
                        strict=True,
                    )
                )
                samples[r["name"]] += r["samples_us"]
                rounds[r["name"]].append(statistics.median(r["samples_us"]))
                for key in ("activity_before", "activity_after"):
                    activity = r[key]
                    assert (
                        activity["gpu"]["returncode"]
                        == activity["processes"]["returncode"]
                        == 0
                    )
                    assert not list(
                        csv.DictReader(
                            activity["processes"]["stdout"].splitlines(),
                            skipinitialspace=True,
                        )
                    ), "unexpected compute process rows in retained capture"
                    devices = list(
                        csv.DictReader(
                            activity["gpu"]["stdout"].splitlines(),
                            skipinitialspace=True,
                        )
                    )
                    device = next(
                        d
                        for d in devices
                        if d["uuid"] == "GPU-510ddc27-95ac-c62f-d49b-242cc5183c72"
                    )
                    assert (
                        device["display_active"] == "Disabled"
                        and "3080" in device["name"]
                    )
                    clocks.add(
                        (
                            device["clocks.current.sm [MHz]"],
                            device["clocks.current.memory [MHz]"],
                        )
                    )
                trace_count += 1
            if row[scope]:
                assert all(len(v) == 6 for v in rounds.values())
                scopes[scope] = {
                    n: {
                        "median_us": statistics.median(v),
                        "round_medians_us": rounds[n],
                    }
                    for n, v in samples.items()
                }
                scopes[scope]["vector_winning_rounds"] = {
                    n: sum(
                        a < b
                        for a, b in zip(rounds["component"], rounds[n], strict=True)
                    )
                    for n in ("ordinary", "scalar")
                }
                scopes[scope]["clock_snapshots"] = sorted(clocks)
        dtype = (
            row["case"].split("_")[-3]
            if row["case"].startswith("vector_row_tail_")
            else row["case"].split("_")[1]
        )
        final = Path(gpu["kernels"][dtype]["directory"])
        timed = Path(row["component_directory"])
        assert final == timed
        identity = read(final / "identity.json")
        assert identity["adapter"] == sha(ROOT / "rms_vector_trial_runtime.py")
        assert identity["compiler_binary"] == sha(
            ROOT / "cache/reuse-first-rms/target/release/stage5-rms-vector-probe"
        )
        assert identity["cpu_gate"] == sha(OUT / "cpu.json")
        assert read(final / "manifest.json")["operand_vector_width"] == 4
        cases.append(
            {"case": row["case"], "ordinary_plan": row["ordinary_plan"], **scopes}
        )
    assert trace_count == 180

    # Ordinary/scalar/vector counters refer to exactly the three timed primary bodies.
    parsed, kernel, columns = {}, None, None
    for line in (OUT / "executed-sass.txt").read_text().splitlines():
        if line.startswith("Kernel Name "):
            kernel = line.split()[2]
            parsed[kernel] = {
                "warp_instructions": 0,
                "opcodes": {},
                "packed_128bit_ldg_static_entries": 0,
            }
        elif line.startswith("Address "):
            columns = tuple(
                line.index(k) for k in ("Source", "Warp S", "Instru", "Thread")
            )
        elif kernel and line.startswith("0x"):
            a, b, c, d = columns
            inst = line[a:b].strip()
            count = int(line[c:d].strip().replace(",", ""))
            opcode = re.match(r"(?:@!?U?P\d+\s+)?([A-Z0-9.]+)", inst)
            assert opcode, inst
            op = opcode[1].split(".")[0]
            if op == "LDG" and "128" in opcode[1].split("."):
                parsed[kernel]["packed_128bit_ldg_static_entries"] += 1
            parsed[kernel]["warp_instructions"] += count
            parsed[kernel]["opcodes"][op] = parsed[kernel]["opcodes"].get(op, 0) + count
    raw = [
        r
        for r in csv.DictReader((OUT / "ncu-raw-import.csv").read_text().splitlines())
        if r["ID"].isdigit()
    ]
    counter = read(OUT / "counter-kernels.json")
    assert len(raw) == len(counter["kernels"]) == 3
    resources = []
    for r, k in zip(raw, counter["kernels"], strict=True):
        assert r["Kernel Name"] == k["entrypoint"]
        p = parsed[k["entrypoint"]]
        if k["name"] == "component":
            assert p["packed_128bit_ldg_static_entries"] == 0
        assert (
            Decimal(r["smsp__inst_executed.sum"].replace(",", ""))
            == p["warp_instructions"]
        )
        directory = Path(graphs["rows"][0][k["name"] + "_directory"])
        assert sha(directory / "kernel.ptx") == k["ptx_sha256"]
        resources.append(
            {
                "name": k["name"],
                "registers": k["attributes"]["registers_per_thread"],
                **p,
            }
        )

    old_gpu = read(OLD / "gpu.json")
    for row in graphs["rows"]:
        dtype = (
            row["case"].split("_")[-3]
            if row["case"].startswith("vector_row_tail_")
            else row["case"].split("_")[1]
        )
        new = Path(row["scalar_directory"])
        old = Path(old_gpu["kernels"][dtype]["directory"])
        ne, oe = [read(p / "manifest.json")["entrypoint"] for p in (new, old)]
        for name in ("cubecl.ir", "kernel.cu", "kernel.ptx"):
            assert (new / name).read_text().replace(ne, "ENTRY") == (
                old / name
            ).read_text().replace(oe, "ENTRY")

    locks = [
        tomllib.loads((ROOT / p).read_text())
        for p in ("rms_vector_probe/Cargo.lock", "rms_probe/Cargo.lock")
    ]
    assert [p for p in locks[0]["package"] if "source" in p] == [
        p for p in locks[1]["package"] if "source" in p
    ]
    preservation = read(OUT / "preservation.json")
    for name, h in preservation["files"].items():
        assert sha(ROOT.parent / name) == h, name
    sources = [
        ROOT / n
        for n in (
            "rms_vector_trial.py",
            "rms_vector_trial_data.py",
            "rms_vector_trial_runtime.py",
            "rms_vector_trial_backend.py",
            "rms_vector_scalar_runtime.py",
            "rms_vector_probe/src/main.rs",
            "rms_vector_probe/Cargo.toml",
            "rms_vector_probe/Cargo.lock",
        )
    ] + [Path(__file__)]
    result = {
        "complete": True,
        "cpu_cases": 36,
        "gpu_cases": 36,
        "output_elements": sum(r["elements"] for r in cpu["rows"]),
        "historical_odd_rejections": 51,
        "live_negatives": 27,
        "traces": trace_count,
        "preserved_files": len(preservation["files"]),
        "cases": cases,
        "executed_sass": resources,
        "decision": "No graph win against ordinary in any paired round; stop, diagnostic only",
        "isolated_limit": "Primary isolated rounds cross clock states; pooled median is not a stable kernel-cost estimate",
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
    }
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "PASS:",
        36,
        "CPU/GPU cases,",
        trace_count,
        "traces,",
        len(preservation["files"]),
        "preserved files",
    )
    for case in cases:
        print(
            case["case"],
            {
                n: case["rounds"][n]["median_us"]
                for n in ("ordinary", "scalar", "component")
            },
        )


if __name__ == "__main__":
    main()
