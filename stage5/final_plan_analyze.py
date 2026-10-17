"""Offline checks and self-contained retention for the final promotion packet."""

import hashlib
import json
from pathlib import Path
import shutil
import statistics
import re

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/reduction-structure/final/promotion"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sass(path):
    functions, name, base, columns = {}, None, None, None
    for line in path.read_text().splitlines():
        if line.startswith("Kernel Name "):
            name, base, columns = line.split()[2], None, None
            functions[name] = {"instructions": [], "barriers": 0, "shuffles": 0}
        if line.startswith("Address "):
            start, end = line.index("Source"), line.index("Warp S")
            count, count_end = line.index("Instru"), line.index("Thread")
            if not (0 < start < end < count < count_end):
                raise ValueError("unexpected SourceCounters column layout")
            columns = (start, end, count, count_end)
        if name and re.match(r"^0x[0-9a-f]+ ", line):
            if columns is None:
                raise ValueError("instruction before SourceCounters column header")
            if base is None:
                base = int(line.split()[0], 16)
            inst = line[start:end].strip()
            executed = int(line[count:count_end].strip().replace(",", ""))
            if "BAR.SYNC" in inst:
                functions[name]["barriers"] += executed
            if "SHFL." in inst:
                functions[name]["shuffles"] += executed
            if "BRA " in inst or "BSSY " in inst:
                inst = re.sub(
                    r"0x[0-9a-f]{8,}", lambda m: f"+0x{int(m[0], 16) - base:x}", inst
                )
            functions[name]["instructions"].append(inst)
    return functions


def final_record(records):
    matches = [
        r
        for r in records
        if read(Path(r["directory"]) / "semantic.json")["reduction_extent"]
        == {"op": "constant", "value": 9}
    ]
    assert len(matches) == 1
    return matches[0]


def records_by_role(records):
    return {
        "partial": next(
            r for r in records if r["plan"]["strategy"] == "plane_shared_reference512"
        ),
        "final": final_record(records),
    }


def main():
    controls, graphs = (read(OUT / name) for name in ("controls.json", "graphs.json"))
    assert controls["complete"] and len(controls["corpus"]) == 148
    assert len(controls["direct"]) == 30 and graphs["complete"]
    assert (
        graphs["ordinary_static9_graph"]["selected"]["strategy"]
        == "independent_planes_final9"
    )
    assert all(all(i["matches_qualified_probe"].values()) for i in controls["identity"])
    assert read(OUT / "cache-reload.json")["compiler_invocations"] == 0
    assert read(OUT / "transitions.json")["complete"]
    current = sass(OUT / "executed-sass.txt")
    old = sass(OUT.parent / "graphs/executed-sass.txt")
    assert (
        current["cubecl5_88b13fadec6826fa"]["instructions"]
        == old["cubecl5_191aecc1359e95bb"]["instructions"]
    )
    assert current["cubecl5_b1f1603b0e5b0a49"]["barriers"] == 4608
    assert current["cubecl5_88b13fadec6826fa"]["barriers"] == 0
    assert current["cubecl5_88b13fadec6826fa"]["shuffles"] == 320
    (OUT / "sass-analysis.json").write_text(
        json.dumps(
            {
                "qualified_sass_instructions_match_after_relocation": True,
                "functions": current,
            },
            indent=2,
        )
    )
    directories, keys, cases = set(), set(), []
    post = OUT / "post-review"
    assert read(post / "controls.json")["complete"]
    assert len(read(post / "controls.json")["rejected"]) == 39
    assert read(post / "cache-reload.json")["compiler_invocations"] == 0
    assert read(post / "transitions.json")["complete"]
    ordinary = read(post / "ordinary.json")
    assert (
        ordinary["complete"]
        and ordinary["selected"]["strategy"] == "independent_planes_final9"
    )
    directories.add(Path(ordinary["directory"]))
    keys.update(ordinary["selection_keys"])
    assert len(ordinary["normal_producer_split9"]) == 2
    for row in ordinary["normal_producer_split9"]:
        assert (
            final_record(row["records"])["plan"]["strategy"]
            == "independent_planes_final9"
        )
        assert any(
            r["plan"]["strategy"] == "shared_tree_compensated" for r in row["records"]
        )
        directories.update(Path(r["directory"]) for r in row["records"])
        for r in row["records"]:
            keys.update(r["selection_keys"])
    for row in read(post / "cache-init.json")["rows"]:
        assert row["selected"]["strategy"] == "independent_planes_final9"
        keys.update(row["selection_keys"])
    post_storage = read(post / "graph-storage.json")
    assert post_storage["complete"] and len(post_storage["rows"]) == 6
    for row in post_storage["rows"]:
        assert (
            row["bitwise"]
            and final_record(row["records"]["promoted"])["plan"]["strategy"]
            == "independent_planes_final9"
        )
        for records in row["records"].values():
            directories.update(Path(r["directory"]) for r in records)
            for r in records:
                keys.update(r["selection_keys"])
    full = OUT / "full-feature"
    assert read(full / "controls.json")["complete"]
    assert read(full / "cache-reload.json")["compiler_invocations"] == 0
    for row in read(full / "cache-init.json")["rows"]:
        assert row["selected"]["strategy"] == "independent_planes_final9"
        keys.update(row["selection_keys"])
    for event in read(full / "cache-init.json")["events"]:
        if event["event"] == "artifact":
            directory = Path(event["directory"])
            if (
                read(directory / "descriptor.json")["physical_plan"]["strategy"]
                == "independent_planes_final9"
            ):
                directories.add(directory)
    previous_storage = read(OUT / "graph-storage.json")
    full_storage = read(full / "graph-storage.json")
    assert full_storage["complete"] and len(full_storage["rows"]) == 6
    for row in full_storage["rows"]:
        assert row["bitwise"]
        prior = next(
            r
            for r in previous_storage["rows"]
            if r["dtype"] == row["dtype"] and r["shape"] == row["shape"]
        )
        for tag in ("frozen", "promoted"):
            current_roles = records_by_role(row["records"][tag])
            old_roles = records_by_role(prior["records"][tag])
            for role, current in current_roles.items():
                old_record = old_roles[role]
                directories.add(Path(current["directory"]))
                keys.update(current.get("selection_keys", []))
                # A stable-winner guard can retain baseline on another build's
                # independent tuning capture. Only identical plans must emit
                # identical code; every producer is still the frozen plan.
                if current["plan"] != old_record["plan"]:
                    assert current["plan"]["strategy"] != "plane_shared_reference512"
                    continue
                for name in ("cubecl.ir", "kernel.cu", "kernel.ptx"):
                    assert (Path(current["directory"]) / name).read_bytes() == (
                        Path(old_record["directory"]) / name
                    ).read_bytes()
    compiler = read(ROOT / "cache/tuning" / next(iter(keys)) / "selection.json")[
        "key_data"
    ]["identity"]["compiler"]
    for path in (ROOT / "cache/tuning").glob("*/selection.json"):
        selection = read(path)
        if selection["key_data"]["identity"]["compiler"] == compiler and any(
            p["strategy"] == "independent_planes_final9"
            for p in selection["key_data"]["catalog"]
        ):
            keys.add(selection["key"])
    storage = read(OUT / "graph-storage.json")
    assert storage["complete"] and len(storage["rows"]) == 6
    for row in storage["rows"]:
        assert row["bitwise"]
        assert (
            final_record(row["records"]["promoted"])["plan"]["strategy"]
            == "independent_planes_final9"
        )
        for records in row["records"].values():
            directories.update(Path(r["directory"]) for r in records)
    for mode in ("init", "reload"):
        cache = read(OUT / f"cache-{mode}.json")
        assert cache["complete"]
        for row in cache["rows"]:
            assert row["selected"]["strategy"] == "independent_planes_final9"
            keys.update(row["selection_keys"])
        for event in cache["events"]:
            if event["event"] == "artifact":
                directory = Path(event["directory"])
                if (
                    read(directory / "descriptor.json")["physical_plan"]["strategy"]
                    == "independent_planes_final9"
                ):
                    directories.add(directory)
    traces = 0
    for case in graphs["cases"]:
        assert case["partial_byte_identity"] and case["frozen_pair_outputs_bitwise"]
        for records in case["records"].values():
            for record in records:
                directories.add(Path(record["directory"]))
                keys.update(record["selection_keys"])
        assert len(case["captures"]) == 30
        for capture in case["captures"]:
            path = Path(capture["trace"])
            assert sha(path) == capture["sha256"]
            events = sorted(
                (e for e in read(path)["traceEvents"] if e.get("cat") == "kernel"),
                key=lambda e: e["ts"],
            )
            tag = capture["variant"]
            if tag.startswith("isolated"):
                names = [
                    "cubecl5_b1f1603b0e5b0a49"
                    if tag == "isolated-tree"
                    else "cubecl5_88b13fadec6826fa"
                ]
            else:
                names = [
                    r["entrypoint"]
                    for r in case["records"][
                        "normal-float32" if tag == "normal" else tag
                    ]
                ]
            assert [e["name"] for e in events] == names * 16
            assert capture["durations_us"] == [
                [e["dur"] for e in events[i : i + len(names)]]
                for i in range(0, len(events), len(names))
            ]
            traces += 1
        summary = {}
        for tag in ("normal", "frozen", "promoted", "isolated-tree", "isolated-plane"):
            cells = [
                cell
                for c in case["captures"]
                if c["variant"] == tag
                for cell in c["durations_us"]
            ]
            samples = [sum(cell) for cell in cells]
            med = statistics.median(samples)
            summary[tag] = {
                "median_gpu_sum_us": med,
                "mad_gpu_sum_us": statistics.median(abs(x - med) for x in samples),
                "min_gpu_sum_us": min(samples),
                "max_gpu_sum_us": max(samples),
                "final_median_us": statistics.median(cell[-1] for cell in cells),
                "samples": len(samples),
            }
            assert med == case["summary"][tag]["median_gpu_sum_us"]
        frozen_partial = Path(case["records"]["frozen"][0]["directory"])
        promoted_partial = Path(case["records"]["promoted"][0]["directory"])
        old_graph = read(
            ROOT / "artifacts/reduction-structure/final/graphs/timing.json"
        )
        old_case = next(c for c in old_graph["cases"] if c["shape"] == case["shape"])
        assert old_case["input_sha256"] == case["input_sha256"]
        retained_partial = Path(old_case["artifacts"]["frozen"][0]["directory"])
        for name in ("cubecl.ir", "kernel.cu", "kernel.ptx"):
            assert (
                (frozen_partial / name).read_bytes()
                == (promoted_partial / name).read_bytes()
                == (retained_partial / name).read_bytes()
            )
        cases.append({"shape": case["shape"], "summary": summary})
    assert traces == 60
    directories.add(Path(graphs["ordinary_static9_graph"]["directory"]))
    retained = []
    for directory in sorted(directories):
        dest = OUT / "retained" / directory.name
        dest.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in (
            "descriptor.json",
            "semantic.json",
            "physical-plan.json",
            "manifest.json",
            "cubecl.ir",
            "kernel.cu",
            "kernel.ptx",
            "nvrtc.json",
            "resources.json",
            "integrity.json",
        ):
            if (directory / name).is_file():
                shutil.copyfile(directory / name, dest / name)
                files[name] = sha(dest / name)
        retained.append(
            {"source": str(directory), "retained": str(dest), "files": files}
        )
    for key in sorted(keys):
        dest = OUT / "selections" / key
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("selection.json", "timings.json"):
            source = ROOT / "cache/tuning" / key / name
            if source.is_file():
                shutil.copyfile(source, dest / name)
    frozen = read(ROOT / "artifacts/reduction-structure/final/graphs/frozen-files.json")
    assert all(sha(ROOT.parent / path) == h for path, h in frozen.items())
    assert all(
        sha(ROOT.parent / path) == h
        for path, h in read(OUT / "frozen-sources.json").items()
    )
    (OUT / "analysis.json").write_text(
        json.dumps(
            {
                "complete": True,
                "verified_traces": traces,
                "preserved_retained_files": len(frozen),
                "cases": cases,
                "retained": retained,
                "selection_keys": sorted(keys),
            },
            indent=2,
        )
    )
    print(
        "promotion analysis PASS",
        traces,
        "traces",
        len(retained),
        "artifact directories",
    )


if __name__ == "__main__":
    main()
