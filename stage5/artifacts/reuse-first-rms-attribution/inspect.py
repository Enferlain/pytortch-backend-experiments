"""Host-only reproducible policy/source/artifact checks for hs6."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "stage5/artifacts/reuse-first-rms"
SOURCES = [
    ".stage0-sources/cubek/crates/cubek-reduce/src/lib.rs",
    ".stage0-sources/cubek/crates/cubek-reduce/src/launch/strategy.rs",
    ".stage0-sources/cubek/crates/cubek-reduce/src/launch/base.rs",
    ".stage0-sources/cubek/crates/cubek-reduce/src/launch/utils.rs",
    ".stage0-sources/cubek/crates/cubek-reduce/src/routines/plane.rs",
    ".stage0-sources/cubek/crates/cubek-reduce/src/routines/cube.rs",
    ".stage0-sources/cubek/crates/cubek-reduce/src/eval/benchmarks/strategy.rs",
    ".stage0-sources/cubek/crates/cubek-reduce/src/components/readers/parallel.rs",
    ".stage0-sources/cubek/crates/cubek-reduce/src/components/readers/base.rs",
    ".stage0-sources/cubek/crates/cubek-reduce/src/components/global/plane.rs",
    ".stage0-sources/cubek/crates/cubek-reduce/src/components/global/cube.rs",
    ".stage0-sources/cubek/crates/cubek-reduce/src/components/instructions/sum.rs",
    ".stage0-sources/cubecl/crates/cubecl-std/src/tensor/virtual.rs",
    ".stage0-sources/cubecl/crates/cubecl-core/src/lib.rs",
    ".stage0-sources/cubecl/crates/cubecl-ir/src/properties.rs",
    ".stage0-sources/cubecl/crates/cubecl-cuda/src/runtime.rs",
    ".stage0-sources/cubecl/crates/cubecl-cpp/src/shared/vector.rs",
    "stage5/rms_probe/src/main.rs",
    "stage5/rms_selection_probe/src/main.rs",
    "stage5/rms_selection_probe/Cargo.toml",
    "stage5/rms_selection_probe/Cargo.lock",
]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ledger():
    records, name, columns = [], None, None
    for line in (OLD / "executed-sass.txt").read_text().splitlines():
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
                    "warp_instructions": int(line[c:d].strip().replace(",", "")),
                }
            )
    component = [r for r in records if "cubek" in r["kernel"]]
    first = next(
        i
        for i, r in enumerate(component)
        if r["instruction"].startswith("LOP3.LUT R0, R3, R7")
    )
    last = (
        next(
            i
            for i in range(first, len(component))
            if component[i]["instruction"] == "BSYNC B0"
        )
        + 1
    )
    region = component[first:last]
    total = sum(r["warp_instructions"] for r in component)
    count = sum(r["warp_instructions"] for r in region)
    return {
        "scope": "Retained primary actual SASS; attribution only, no new GPU execution",
        "records": records,
        "row_coordinate_region": {
            "start": region[0]["address"],
            "end": region[-1]["address"],
            "static_entries": len(region),
            "executed_warp_instructions": count,
            "kernel_total": total,
            "fraction_of_kernel_instructions": count / total,
            "region": region,
        },
    }


def main(capture):
    # Clear the preceding pass before any assertion or external probe can fail.
    (OUT / "validation.json").write_text(
        json.dumps({"complete": False, "state": "verification started"}, indent=2)
        + "\n"
    )
    binary = (
        ROOT / "stage5/cache/reuse-first-rms/target/release/stage5-rms-selection-probe"
    )
    policy = json.loads(subprocess.check_output([str(binary)], text=True))
    assert policy == read(OUT / "host-policy.json")
    assert len(policy["rows"]) == 9
    assert policy["controls"] == {
        "zero_work_planes": 1,
        "one_row_planes": 1,
        "odd_width_vector": 1,
        "unaligned_row_stride_vector": 1,
    }
    for row in policy["rows"]:
        b, n = row["shape"]
        assert row["plane_count"] == 8 and row["grid"] == b // 8
        assert row["input_vector_size"] == (4 if n == 1024 else 1)
        assert row["automatic_routine_winner"] is None

    # Negative controls call the same real helper through an independent source
    # compile; no fabricated stock routine winner is inferred from these values.
    locks = [
        tomllib.loads((ROOT / p).read_text())
        for p in (
            "stage5/rms_selection_probe/Cargo.lock",
            "stage5/rms_probe/Cargo.lock",
        )
    ]
    assert [p for p in locks[0]["package"] if "source" in p] == [
        p for p in locks[1]["package"] if "source" in p
    ]
    for name in SOURCES:
        if name.startswith(".stage0-sources/cubek/"):
            copy = ROOT / name.replace(
                ".stage0-sources/cubek/", "stage5/cache/codegen/cubek/", 1
            )
            assert sha(ROOT / name) == sha(copy), name

    graph = read(OLD / "graphs.json")["rows"][0]
    directory = Path(graph["component_directory"])
    cuda, ptx = [(directory / n).read_text() for n in ("kernel.cu", "kernel.ptx")]
    assert "v91 = v89 - v90" in cuda and "v93 = v91 + v90" in cuda
    assert "rem.u64" in ptx and "rem.u32" in ptx
    assert "mul.lo.s64 \t%rd7, %rd77, %rd1" in ptx
    assert "add.s64 \t%rd49, %rd48, %rd7" in ptx
    assert ptx.index("div.rn.f32") < ptx.index("$L__BB0_26:")
    observed = ledger()
    assert observed == read(OUT / "sass-ledger.json")
    region = observed["row_coordinate_region"]
    assert (
        region["executed_warp_instructions"] == 1536 and region["kernel_total"] == 47680
    )
    cold_call = [r for r in region["region"] if "CALL.REL" in r["instruction"]]
    assert len(cold_call) == 1 and cold_call[0]["warp_instructions"] == 0
    paths = [ROOT / p for p in SOURCES] + [
        binary,
        OLD / "graphs.json",
        OLD / "analysis.json",
        OLD / "executed-sass.txt",
        OUT / "host-policy.json",
        OUT / "host-policy-build.txt",
        OUT / "setup-order-failure.txt",
        OUT / "selection-references.txt",
        OUT / "preservation.json",
        OUT / "sass-ledger.json",
        Path(__file__),
    ]
    paths += [directory / n for n in ("cubecl.ir", "kernel.cu", "kernel.ptx")]
    paths += [
        Path(graph["ordinary_directory"]) / n
        for n in ("cubecl.ir", "kernel.cu", "kernel.ptx")
    ]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    evidence = {
        "files": hashes,
        "claims": {
            "stock_api_requires_explicit_routine": True,
            "conditional_plane_count_matches_trial": True,
            "conditional_primary_vector_width_is_four": True,
            "row_coordinate_warp_instructions": 1536,
            "row_coordinate_instruction_fraction": region[
                "fraction_of_kernel_instructions"
            ],
            "instruction_count_is_not_a_latency_decomposition": True,
            "new_gpu_compilation_or_execution": False,
        },
    }
    if capture:
        (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    else:
        assert evidence == read(OUT / "evidence.json"), "source/artifact/probe drift"

    preservation = read(OUT / "preservation.json")
    for name, h in preservation["files"].items():
        assert sha(ROOT / name) == h, name
    for name, state in preservation["upstream"].items():
        p = ROOT / ".stage0-sources" / name
        assert (
            subprocess.check_output(
                ["git", "-C", str(p), "rev-parse", "HEAD"], text=True
            ).strip()
            == state["head"]
        )
        assert (
            hashlib.sha256(
                subprocess.check_output(["git", "-C", str(p), "diff", "HEAD"])
            ).hexdigest()
            == state["diff_sha256"]
        )
    (OUT / "validation.json").write_text(
        json.dumps(
            {
                "complete": True,
                "preserved_files": len(preservation["files"]),
                "bound_sources_and_artifacts": len(hashes),
                "policy_rows": 9,
                "host_only": True,
                "evidence_sha256": sha(OUT / "evidence.json"),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "PASS: host policy, source/target attribution and",
        len(preservation["files"]),
        "preserved files",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture",
        action="store_true",
        help="freeze initial source identities; default verifies",
    )
    main(parser.parse_args().capture)
