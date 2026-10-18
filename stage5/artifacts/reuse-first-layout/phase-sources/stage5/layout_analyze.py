"""Offline integrity, address, trace and executed-code checks for layout reuse."""

import csv
import gzip
import json
from pathlib import Path
import shutil
import statistics

from stage5.layout_trial import OUT, ROOT, check, sha, tensor_sha, write
from stage5.layout_model import verify


def read(p):
    return json.loads(Path(p).read_text())


def sass_ledger():
    records = []
    invocation = -1
    name = columns = None
    for line in (OUT / "executed-sass.txt").read_text().splitlines():
        if line.startswith("Kernel Name "):
            name = line.split()[2]
            invocation += 1
        elif line.startswith("Address "):
            columns = tuple(
                line.index(k) for k in ("Source", "Warp S", "Instru", "Thread")
            )
        elif name and line.startswith("0x"):
            a, b, c, d = columns
            records.append(
                dict(
                    kernel=name,
                    invocation=invocation,
                    address=line[:a].strip(),
                    instruction=line[a:b].strip(),
                    executed_warp_instructions=int(line[c:d].strip().replace(",", "")),
                )
            )
    assert records
    return records


def main():
    capture, cpu, neighbors, gpu, native, timing, plan = [
        read(OUT / n)
        for n in (
            "capture.json",
            "cpu.json",
            "cpu-neighbors.json",
            "gpu.json",
            "native-controls.json",
            "timing.json",
            "timing-plan.json",
        )
    ]
    assert all(d["complete"] for d in (capture, cpu, neighbors, gpu, native, timing))
    assert capture["source_sha256"] == sha(ROOT / "stage5/layout_trial.py")
    sources = read(OUT / "sources.json")
    for name, digest in sources["files"].items():
        assert sha(OUT / name) == digest, name
        prefix, relative = name.split("/", 1)
        current = (
            ROOT / relative
            if prefix == "phase-sources"
            else ROOT / ".stage0-sources/cubecl" / relative
        )
        assert sha(current) == digest, str(current)
    assert (
        len(capture["rows"]) == 33
        and len(cpu["rows"]) == 33
        and len(neighbors["rows"]) == 66
    )
    assert (
        cpu["source_sha256"]
        == neighbors["model_sha256"]
        == sha(ROOT / "stage5/layout_model.py")
    )
    assert not cpu["gpu_source_existed"] and neighbors["gpu_source_existed"]
    assert cpu["capture_sha256"] == sha(OUT / "capture.json")
    assert neighbors["plan_sha256"] == sha(OUT / "qualification-plan.json")
    identity_count = 0
    for row in read(OUT / "qualification-plan.json")["cases"]:
        cap = next(
            r
            for r in capture["rows"]
            if r["case"]["family"] == row["family"]
            and r["case"]["dtype"] == row["dtype"]
        )
        sem = read(OUT / cap["baseline"] / "semantic.json")
        identity_count += verify(sem, row["scalars"])["scalar_identities"]
    assert identity_count == 16195476
    assert (
        len(gpu["rows"]) == 78
        and len(gpu["live_controls"]) == 90
        and len(gpu["stream_controls"]) == 12
        and len(native["rows"]) == 8
    )
    assert gpu["source_sha256"] == sha(ROOT / "stage5/layout_qualify.py")
    from stage5.layout_runtime import BINARY

    assert gpu["binary_sha256"] == sha(BINARY)
    assert gpu["special_values_sha256"] == sha(OUT / "special-values.pt")
    import torch

    values = torch.load(OUT / "special-values.pt", weights_only=True)
    assert len(values) == 12
    for v in values.values():
        assert tensor_sha(v["actual"]) == tensor_sha(v["baseline"])
        check(torch, v["actual"], v["eager"])
    for row in gpu["rows"]:
        assert row["bitwise"] and row["policy"] and row["canaries"]
    for k in gpu["kernels"]:
        d = OUT / k["directory"]
        identity = read(d / "identity.json")
        assert identity["adapter"] == sha(ROOT / "stage5/layout_runtime.py")
        assert identity["diagnostic_binary"] == gpu["binary_sha256"]
        assert identity["semantic_emitter"] == sha(ROOT / "stage0/src/descriptor2.rs")
        assert identity["cpu_gate"] == sha(OUT / "cpu.json")
        assert identity["neighbors"] == sha(OUT / "cpu-neighbors.json")
        assert sha(d / "kernel.ptx") == k["ptx_sha256"]
        assert read(d / "manifest.json")["entrypoint"] == k["entrypoint"]
        assert read(d / "nvrtc.json")["options"].count("--fmad=false") == 1
    assert timing["source_sha256"] == sha(ROOT / "stage5/layout_measure.py")
    assert len(timing["rows"]) == 33 and plan["rounds"] == 6 and plan["calls"] == 16
    summary, traces, clocks = [], set(), set()
    count = 0
    for r in timing["rows"]:
        qual = next(q for q in gpu["rows"] if q["label"] == r["label"])
        assert (
            r["input_sha256"] == qual["input_sha256"]
            and r["output_sha256"] == qual["output_sha256"]
        )
        assert Path(r["artifacts"]["component"]["directory"]) == OUT / qual["candidate"]
        numel = 1
        for d in r["case"]["shape"]:
            numel *= d
        for name, a in r["artifacts"].items():
            d = Path(a["directory"])
            assert sha(d / "kernel.ptx") == a["ptx_sha256"]
            assert read(d / "manifest.json")["entrypoint"] == r["kernels"][name]
            if name != "component":
                dest = OUT / "timed-ordinary" / d.name
                dest.mkdir(parents=True, exist_ok=True)
                for p in d.iterdir():
                    if p.is_file() and p.suffix in (
                        ".json",
                        ".cu",
                        ".ptx",
                        ".ir",
                        ".log",
                    ):
                        shutil.copyfile(p, dest / p.name)
        assert len(r["rounds"]) == 6
        for rr in r["rounds"]:
            assert rr["order"] == plan["order"][rr["round"]]
            for state in ("activity_before", "activity_after"):
                for fields in list(csv.reader(rr[state]["stdout"].splitlines()))[1:]:
                    if "510ddc27" in ",".join(fields):
                        clocks.add((fields[4].strip(), fields[5].strip()))
            for name, p in rr["paths"].items():
                trace = OUT / p["trace"]
                assert str(trace) not in traces and sha(trace) == p["trace_sha256"]
                traces.add(str(trace))
                events = [
                    e
                    for e in json.loads(gzip.decompress(trace.read_bytes()))[
                        "traceEvents"
                    ]
                    if e.get("cat") == "kernel"
                ]
                assert len(events) == len(p["samples_us"]) == 16
                a = r["artifacts"][name]["plan"]
                coverage = (
                    a["workgroup"][0] * a["vector_width"] * a["elements_per_unit"]
                )
                grid = [(numel + coverage - 1) // coverage, 1, 1]
                assert all(
                    e["name"] == r["kernels"][name]
                    and e["args"]["grid"] == grid
                    and e["args"]["block"] == a["workgroup"]
                    for e in events
                )
                assert all(
                    abs(e["dur"] - v) < 0.002 for e, v in zip(events, p["samples_us"])
                )
                assert statistics.median(p["samples_us"]) == p["median_us"]
                count += len(events)
        medians = {
            n: statistics.median(
                [v for rr in r["rounds"] for v in rr["paths"][n]["samples_us"]]
            )
            for n in ("baseline", "component", "ordinary")
        }
        summary.append(
            dict(
                label=r["label"],
                median_us=medians,
                component_wins={
                    n: sum(
                        rr["paths"]["component"]["median_us"]
                        < rr["paths"][n]["median_us"]
                        for rr in r["rounds"]
                    )
                    for n in ("baseline", "ordinary")
                },
            )
        )
    assert count == 9504 and len(traces) == 594
    preservation = read(OUT / "preservation.json")
    for name, digest in preservation.items():
        assert sha(ROOT / name) == digest, name
    sass = sass_ledger()
    assert len({r["invocation"] for r in sass}) == 12
    raw_metrics = list(csv.DictReader((OUT / "ncu-import.csv").open()))
    ledgers = []
    for i in range(12):
        selected = [r for r in sass if r["invocation"] == i]
        metrics = {
            r["Metric Name"]: r["Metric Value"]
            for r in raw_metrics
            if r["ID"] == str(i)
        }
        instructions = sum(r["executed_warp_instructions"] for r in selected)
        assert instructions == int(metrics["smsp__inst_executed.sum"].replace(",", ""))
        ledgers.append(
            dict(
                invocation=i,
                kernel=selected[0]["kernel"],
                executed_warp_instructions=instructions,
                registers_per_thread=int(metrics["launch__registers_per_thread"]),
                instruction_groups={
                    op: sum(
                        r["executed_warp_instructions"]
                        for r in selected
                        if op in r["instruction"]
                    )
                    for op in ("LDG", "STG", "BSSY", "BSYNC", "MUFU.RCP", "LDL", "STL")
                },
                instructions=[
                    r
                    for r in selected
                    if r["executed_warp_instructions"]
                    and any(
                        s in r["instruction"]
                        for s in ("LDG", "STG", "BSSY", "BSYNC", "MUFU", "IMAD.HI")
                    )
                ],
            )
        )
    write("sass-ledger.json", ledgers)
    write(
        "analysis.json",
        dict(
            complete=True,
            scalar_address_identities=identity_count,
            gpu_cases=78,
            live_controls=90,
            stream_controls=12,
            native_controls=8,
            traces=len(traces),
            kernel_durations=count,
            preserved_files=len(preservation),
            coarse_sm_memory_clocks=sorted(clocks),
            timings=summary,
            ncu_invocations=12,
            source_sha256=sha(__file__),
        ),
    )
    print(
        "PASS",
        identity_count,
        "address identities;",
        count,
        "timed launches;",
        len(preservation),
        "frozen files",
    )


if __name__ == "__main__":
    main()
