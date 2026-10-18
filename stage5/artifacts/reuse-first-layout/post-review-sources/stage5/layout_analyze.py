"""Offline integrity, address, trace and executed-code checks for layout reuse."""

import csv
import ast
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
    hardened = {
        "stage5/layout_runtime.py",
        "stage5/layout_qualify.py",
        "stage5/layout_measure.py",
        "stage5/layout_analyze.py",
    }
    for name, digest in sources["files"].items():
        assert sha(OUT / name) == digest, name
        prefix, relative = name.split("/", 1)
        current = (
            ROOT / relative
            if prefix == "phase-sources"
            else ROOT / ".stage0-sources/cubecl" / relative
        )
        if relative not in hardened:
            assert sha(current) == digest, str(current)
    for name, digest in read(OUT / "post-review-sources.json")["files"].items():
        assert sha(OUT / "post-review-sources" / name) == sha(ROOT / name) == digest
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
    assert gpu["source_sha256"] == sha(OUT / "phase-sources/stage5/layout_qualify.py")
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
        assert identity["adapter"] == sha(
            OUT / "phase-sources/stage5/layout_runtime.py"
        )
        assert identity["diagnostic_binary"] == gpu["binary_sha256"]
        assert identity["semantic_emitter"] == sha(ROOT / "stage0/src/descriptor2.rs")
        assert identity["cpu_gate"] == sha(OUT / "cpu.json")
        assert identity["neighbors"] == sha(OUT / "cpu-neighbors.json")
        assert sha(d / "kernel.ptx") == k["ptx_sha256"]
        assert read(d / "manifest.json")["entrypoint"] == k["entrypoint"]
        assert read(d / "nvrtc.json")["options"].count("--fmad=false") == 1
    assert timing["source_sha256"] == sha(
        OUT / "phase-sources/stage5/layout_measure.py"
    )
    post = read(OUT / "post-review-gpu.json")
    assert post["complete"] and post["source_sha256"] == sha(
        ROOT / "stage5/layout_qualify.py"
    )
    assert post["binary_sha256"] == gpu["binary_sha256"]
    assert (
        len(post["rows"]) == 78
        and len(post["live_controls"]) == 90
        and len(post["stream_controls"]) == 12
    )
    for a, b in zip(gpu["rows"], post["rows"]):
        for field in (
            "label",
            "scalars",
            "input_sha256",
            "output_sha256",
            "bitwise",
            "policy",
            "canaries",
        ):
            assert a[field] == b[field]
    assert post["special_values_sha256"] == sha(OUT / "post-review-special-values.pt")
    for v in torch.load(
        OUT / "post-review-special-values.pt", weights_only=True
    ).values():
        assert tensor_sha(v["actual"]) == tensor_sha(v["baseline"])
        check(torch, v["actual"], v["eager"])
    assert len(post["kernels"]) == len(gpu["kernels"]) == 12
    for k in post["kernels"]:
        d = OUT / k["directory"]
        prior = next(p for p in gpu["kernels"] if p["semantic"] == k["semantic"])
        original = (
            (OUT / prior["directory"] / "kernel.ptx")
            .read_text()
            .replace(prior["entrypoint"], "ENTRY")
        )
        assert (d / "kernel.ptx").read_text().replace(
            k["entrypoint"], "ENTRY"
        ) == original
        assert sha(d / "kernel.ptx") == k["ptx_sha256"]
        identity = read(d / "identity.json")
        assert identity["adapter"] == sha(ROOT / "stage5/artifact.py")
        assert identity["layout_adapter"] == sha(ROOT / "stage5/layout_runtime.py")
        assert identity["diagnostic_binary"] == gpu["binary_sha256"]
        loads = [op["buffer"] for op in k["semantic"]["ops"] if op["op"] == "load"]
        assert len(loads) == len(set(loads))
    wrappers = read(OUT / "wrapper-inventory.json")
    assert wrappers["timing_sha256"] == sha(OUT / "timing.json")
    assert (
        len({r["path"] for r in wrappers["records"]}) == len(wrappers["records"]) == 50
    )
    for r in wrappers["records"]:
        path = OUT / r["path"]
        assert sha(path) == r["sha256"]
        calls = [
            c
            for c in ast.walk(ast.parse(path.read_text()))
            if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id in ("layout_compile", "cubecl_stage5_compile")
        ]
        assert len(calls) == 1 and ast.literal_eval(calls[0].args[0]) == r["semantic"]
    assert len(wrappers["bindings"]) == 33
    for r in timing["rows"]:
        binding = next(b for b in wrappers["bindings"] if b["label"] == r["label"])
        sem = read(Path(r["artifacts"]["component"]["directory"]) / "semantic.json")
        for name, paths in binding["matching_wrapper_paths"].items():
            assert paths
            assert all(
                next(w for w in wrappers["records"] if w["path"] == p)["semantic"]
                == sem
                and next(w for w in wrappers["records"] if w["path"] == p)["path_kind"]
                == name
                for p in paths
            )
    readonly = read(OUT / "counter-readonly.json")
    assert readonly["complete"] and readonly["unchanged"]
    for name, digest in readonly["files"].items():
        assert sha(OUT / name) == digest
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
                        if any(
                            token == op or token.startswith(op + ".")
                            for token in r["instruction"].split()
                            if not token.startswith("@")
                        )
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
            post_review_gpu_cases=78,
            normalized_ptx_identities=12,
            bound_wrappers=50,
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
