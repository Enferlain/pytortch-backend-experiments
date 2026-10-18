"""Offline evidence verification for the single packed-storage candidate."""

import csv
from decimal import Decimal
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics
import tomllib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/reuse-first-rms-packed-trial"
OLD = ROOT / "artifacts/reuse-first-rms-vector4"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ledger():
    records = []
    name = columns = None
    for line in (OUT / "executed-sass.txt").read_text().splitlines():
        if line.startswith("Kernel Name "):
            name = line.split()[2]
        elif line.startswith("Address "):
            columns = tuple(
                line.index(k) for k in ("Source", "Warp S", "Instru", "Thread")
            )
        elif name and line.startswith("0x"):
            a, b, c, d = columns
            records.append(
                {
                    "kernel": name,
                    "address": line[:a].strip(),
                    "instruction": line[a:b].strip(),
                    "executed_warp_instructions": int(
                        line[c:d].strip().replace(",", "")
                    ),
                }
            )
    return records


def packing(records, gpu):
    rows = []
    for dtype, info in gpu["kernels"].items():
        directory = Path(info["directory"])
        manifest = read(directory / "manifest.json")
        entry = manifest["entrypoint"]
        rs = [r for r in records if r["kernel"] == entry]
        assert rs, entry
        ptx = (directory / "kernel.ptx").read_text()
        expected = "f32" if dtype == "float32" else "u16"
        loads = [
            s.strip()
            for s in ptx.splitlines()
            if re.search(r"ld\.global\.nc\.v4\." + expected, s)
        ]
        assert len(loads) == 5
        first_shuffle = next(i for i, r in enumerate(rs) if "SHFL." in r["instruction"])
        prefix = rs[:first_shuffle]
        metadata = [r for r in prefix if "LDG." in r["instruction"]][:2]
        assert len(metadata) == 2 and all(
            "LDG.E.64." in r["instruction"] for r in metadata
        )
        width = 128 if dtype == "float32" else 64
        body = prefix[prefix.index(metadata[-1]) + 1 :]
        input_loads = [r for r in body if f"LDG.E.{width}." in r["instruction"]]
        assert all(
            f"LDG.E.{width}." in r["instruction"]
            for r in body
            if "LDG." in r["instruction"] and r["executed_warp_instructions"] > 0
        )
        count = sum(r["executed_warp_instructions"] for r in input_loads)
        assert count == 512  # 64*1024/(4 storage elements *32 lanes)
        rows.append(
            {
                "dtype": dtype,
                "entrypoint": entry,
                "bits": width,
                "ptx_vector_loads": loads,
                "metadata_excluded": metadata,
                "input_loads": input_loads,
                "executed_input_load_warp_instructions": count,
                "executed_warp_instructions": sum(
                    r["executed_warp_instructions"] for r in rs
                ),
                "ptx_sha256": sha(directory / "kernel.ptx"),
            }
        )
    raw = list(csv.DictReader((OUT / "ncu-import.csv").open()))
    actual = [r for r in raw if r["Kernel Name"] in {x["entrypoint"] for x in rows}]
    assert len(actual) == 3
    for row in rows:
        metrics = next(r for r in actual if r["Kernel Name"] == row["entrypoint"])
        assert (
            Decimal(metrics["smsp__inst_executed.sum"].replace(",", ""))
            == row["executed_warp_instructions"]
        )
    return rows


def main():
    (OUT / "analysis.json").write_text('{"complete": false}\n')
    preservation = read(OUT / "preservation.json")
    for name, digest in preservation.items():
        assert sha(ROOT.parent / name) == digest, name
    for name, digest in read(OLD / "evidence-integrity.json")["artifacts"].items():
        assert sha(OLD / name) == digest, name
    for name, record in read(
        ROOT / "artifacts/reuse-first-rms-packed-load/sources.json"
    ).items():
        assert sha(ROOT.parent / name) == record["sha256"], name
        if name.startswith(".stage0-sources/cubek/"):
            assert (
                sha(
                    ROOT.parent
                    / name.replace(
                        ".stage0-sources/cubek/", "stage5/cache/codegen/cubek/", 1
                    )
                )
                == record["sha256"]
            )
    locks = [
        tomllib.loads((ROOT / name / "Cargo.lock").read_text())
        for name in ("rms_vector_probe", "rms_packed_probe", "rms_packed_observer")
    ]
    registry = lambda lock: [p for p in lock["package"] if "source" in p]
    assert registry(locks[0]) == registry(locks[1]) == registry(locks[2])
    cpu = read(OUT / "cpu.json")
    assert (
        cpu["complete"]
        and len(cpu["rows"]) == 36
        and all(r["passes"] for r in cpu["rows"])
    )
    assert not cpu["gpu_candidate_source_existed"]
    assert cpu["reference_sha256"] == sha(OLD / "reference.json")
    assert cpu["data_sha256"] == sha(OLD / "reference.pt")
    assert cpu["model_source_sha256"] == sha(ROOT / "rms_vector_trial_data.py")
    assert cpu["producer_sha256"] == sha(OUT / "phase-sources/cpu-rms_packed_trial.py")
    gpu = read(OUT / "gpu.json")
    assert gpu["complete"] and len(gpu["rows"]) == 36
    assert all(
        r["passes"]
        and r["frozen_bitwise"]
        and r["frozen_policy_passes"]
        and r["canaries"]
        for r in gpu["rows"]
    )
    assert len(gpu["rejected"]) == 27 and gpu["stream_offset_controls"] == 3
    assert len(read(OUT / "admission.json")["negative_cases"]) == 12
    environment = read(OUT / "environment.json")
    previous = read(OLD / "environment.json")
    for key in (
        "launcher",
        "fp_patch",
        "nvrtc",
        "cuda",
        "torch",
        "cuda_header_sha256",
        "nvrtc_builtins",
        "torch_commit",
        "device",
        "capability",
        "math",
    ):
        assert environment[key] == previous[key], key
    for key, filename in (
        ("compiler", "cubecl-stage0"),
        ("metadata_library", "libcubecl_stage0.so"),
    ):
        assert environment[key] == sha(ROOT.parent / "stage0/target/release" / filename)
        assert previous[key] == sha(ROOT.parent / "stage0/target/debug" / filename)
    assert environment["candidate_binary_sha256"] == sha(
        ROOT / "cache/reuse-first-rms/target/release/stage5-rms-packed-probe"
    )
    parity = read(OUT / "parity.json")
    assert parity["complete"] and len(parity["rows"]) == 36
    assert parity["values_sha256"] == sha(OUT / "observer-values.pt")
    assert parity["source_sha256"] == sha(ROOT / "rms_packed_observe.py")
    assert parity["observer_binary_sha256"] == sha(
        ROOT / "cache/reuse-first-rms/target/release/stage5-rms-packed-observer"
    )
    for mode, row in read(OUT / "observer-sources.json").items():
        assert row["parent_sha256"] == sha(Path(row["parent"]))
        assert row["derived_sha256"] == sha(
            ROOT / "rms_packed_observer/src" / (mode + ".rs")
        )
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage5.rms_vector_trial_data import tensor_sha

    values = torch.load(OUT / "observer-values.pt", weights_only=True)
    assert len(values) == 36
    for row in parity["rows"]:
        case = values[row["case"]]
        for field, key in (("operands", "operand_sha256"), ("sums", "sum_sha256")):
            assert (
                tensor_sha(case["packed"][field])
                == tensor_sha(case["wqc"][field])
                == row[key]
            )
        normal = next(r for r in gpu["rows"] if r["case"] == row["case"])
        assert row["normal_candidate_final_sha256"] == normal["output_sha256"]
    frozen = (ROOT / "rms_vector_probe/src/main.rs").read_text()
    packed = (ROOT / "rms_packed_probe/src/main.rs").read_text()
    component = lambda s: s[
        s.index("#[cube]\nfn rms_components") : s.index("\nfn bind(")
    ]
    assert component(frozen) == component(packed)
    records = ledger()
    packed_code = packing(records, gpu)
    counter = read(OUT / "counter-kernels.json")
    for row in counter["kernels"]:
        assert row["ptx_sha256"] == sha(Path(row["directory"]) / "kernel.ptx")
    graph = read(OUT / "graphs.json")
    assert graph["complete"] and len(graph["rows"]) == 9
    summaries, traces = [], []
    for row in graph["rows"]:
        direct = next(r for r in gpu["rows"] if r["case"] == row["case"])
        assert (
            row["output_sha256"]["packed"]
            == row["output_sha256"]["wqc"]
            == direct["output_sha256"]
        )
        assert row["input_sha256"] == direct["input_sha256"]
        dtype = read(Path(row["packed_directory"]) / "semantic.json")["buffers"][0][
            "dtype"
        ]
        original_wqc = Path(read(OLD / "gpu.json")["kernels"][dtype]["directory"])
        for name in ("kernel.cu", "kernel.ptx", "cubecl.ir", "manifest.json"):
            assert sha(Path(row["wqc_directory"]) / name) == sha(original_wqc / name)
        assert (
            row["packed_directory"]
            == gpu["kernels"][
                str(
                    read(Path(row["packed_directory"]) / "semantic.json")["buffers"][0][
                        "dtype"
                    ]
                )
            ]["directory"]
        )
        summary = {
            "case": row["case"],
            "shape": row["shape"],
            "ordinary_plan": row["ordinary_plan"],
        }
        for scope in ("rounds", "isolated"):
            if not row[scope]:
                continue
            assert len(row[scope]) == 18
            sequences = [
                tuple(c["name"] for c in row[scope] if c["round"] == i)
                for i in range(6)
            ]
            assert set(sequences) == set(
                itertools.permutations(("ordinary", "wqc", "packed"))
            )
            medians = {name: [] for name in ("ordinary", "wqc", "packed")}
            samples = {name: [] for name in medians}
            for capture in row[scope]:
                path = OUT / capture["trace"]
                assert sha(path) == capture["trace_sha256"]
                events = [
                    e for e in read(path)["traceEvents"] if e.get("cat") == "kernel"
                ]
                assert len(events) == 16 and all(
                    e["name"] == capture["kernel"] for e in events
                )
                assert len(capture["samples_us"]) == 16 and all(
                    x > 0 for x in capture["samples_us"]
                )
                assert all(
                    abs(e["dur"] - v) < 0.0011
                    for e, v in zip(events, capture["samples_us"])
                )
                medians[capture["name"]].append(
                    statistics.median(capture["samples_us"])
                )
                samples[capture["name"]].extend(capture["samples_us"])
                traces.append(capture)
            assert all(len(v) == 6 for v in medians.values())
            summary[scope] = {
                name: {
                    "median_us": statistics.median(samples[name]),
                    "round_medians_us": medians[name],
                }
                for name in medians
            }
            summary[scope]["packed_winning_rounds"] = {
                name: sum(a < b for a, b in zip(medians["packed"], medians[name]))
                for name in ("ordinary", "wqc")
            }
        summaries.append(summary)
    assert len(traces) == 216
    for name, digest in graph["wrappers"].items():
        assert sha(OUT / name) == digest
    snapshots = [c[k] for c in traces for k in ("activity_before", "activity_after")]
    clocks, processes = set(), []
    for snap in snapshots:
        assert snap["gpu"]["returncode"] == 0 and snap["processes"]["returncode"] == 0
        for line in snap["gpu"]["stdout"].splitlines():
            if "GPU-510ddc27-95ac-c62f-d49b-242cc5183c72" in line:
                fields = [v.strip() for v in line.split(",")]
                clocks.add((fields[5], fields[6]))
        processes.extend(snap["processes"]["stdout"].splitlines()[1:])
    result = {
        "complete": True,
        "cpu_cases": 36,
        "gpu_cases": 36,
        "output_elements": sum(r["elements"] for r in gpu["rows"]),
        "observer_cases": 36,
        "traces": len(traces),
        "preserved_files": len(preservation),
        "packed_code": packed_code,
        "cases": summaries,
        "coarse_clock_states": sorted(clocks),
        "compute_process_rows": processes,
        "independence": evidence.inventory(),
        "source_sha256": {p.name: sha(p) for p in ROOT.glob("rms_packed*.py")},
    }
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "complete",
                    "cpu_cases",
                    "gpu_cases",
                    "observer_cases",
                    "traces",
                    "preserved_files",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
