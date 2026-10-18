"""Offline numerical, provenance, trace and executed-code audit for l1m."""

import ast
import csv
import gzip
from pathlib import Path
import statistics
import re

import numpy as np

from stage5.column_layout_trial import (
    OUT,
    ROOT,
    FROZEN,
    read,
    write,
    sha,
    tensor_sha,
    check,
    initialize,
)
from stage5.layout_model import verify, divmod_fast, evaluate


def function_ast(path, name):
    return ast.dump(
        next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
    )


def sass_records():
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
                    instruction=line[a:b].strip(),
                    executed=int(line[c:d].strip().replace(",", "")),
                )
            )
    return records


def opcode_matches(instruction, op):
    token = next(t for t in instruction.split() if not t.startswith("@"))
    return token == op or token.startswith(op + ".")


def main():
    torch = initialize()
    from stage5.column_layout_runtime import verify_cpu, BINARY

    verify_cpu()
    cpu = read(OUT / "cpu.json")
    gpu = read(OUT / "gpu.json")
    timing = read(OUT / "timing.json")
    native = read(OUT / "native-controls.json")
    plan = read(OUT / "timing-plan.json")
    assert all(d["complete"] for d in (cpu, gpu, timing, native))
    for name, digest in read(OUT / "sources.json")["files"].items():
        assert sha(OUT / "sources" / name) == digest, name
        if name not in ("stage5/column_layout_runtime.py", "stage5/column_layout_analyze.py"):
            assert sha(ROOT / name) == digest, name
    for name, digest in read(OUT / "post-review-sources.json")["files"].items():
        assert sha(ROOT / name) == sha(OUT / "post-review-sources" / name) == digest, name
    from stage5.column_layout_review_checks import audit

    assert len(audit()) == 3
    controls = read(OUT / "review/native/controls.json")
    assert controls["complete"] and len(controls["rows"]) == 15
    assert all(r["caller_mutation_isolated"] and r["original_lookup_unpoisoned"]
               for r in controls["rows"])
    assert controls["source_sha256"] == sha(ROOT / "stage5/column_layout_review_checks.py")
    assert controls["runtime_sha256"] == sha(ROOT / "stage5/column_layout_runtime.py")
    identity_count = changed_addresses = 0
    for c, r in zip(read(OUT / "qualification-plan.json")["cases"], cpu["rows"]):
        cap = next(
            x
            for x in read(FROZEN / "capture.json")["rows"]
            if x["case"]["family"] == c["family"] and x["case"]["dtype"] == c["dtype"]
        )
        sem = read(FROZEN / cap["baseline"] / "semantic.json")
        verified = verify(sem, c["scalars"])
        assert all(verified[k] == r[k] for k in verified)
        identity_count += verified["scalar_identities"]
        if c["family"] == "bias_column":
            b, n = c["shape"]
            i = np.arange(b * n, dtype=np.uint64)
            q, _ = divmod_fast(i, n)
            op = next(o for o in sem["ops"] if o["op"] == "load" and o["buffer"] == 1)
            assert np.all(q < b) and np.array_equal(
                q, evaluate(op["index"], c["scalars"], i.astype(np.int64))
            )
            changed_addresses += len(i)
    assert identity_count == 16195476 and len(cpu["rows"]) == 66
    assert (
        len(gpu["rows"]) == 78
        and len(gpu["live_controls"]) == 90
        and len(gpu["stream_controls"]) == 12
        and len(native["rows"]) == 8
    )
    assert gpu["source_sha256"] == sha(ROOT / "stage5/column_layout_qualify.py")
    assert gpu["binary_sha256"] == sha(BINARY) and gpu["special_values_sha256"] == sha(
        OUT / "special-values.pt"
    )
    old = read(FROZEN / "post-review-gpu.json")
    for a, b in zip(gpu["rows"], old["rows"]):
        assert all(
            a[k] == b[k]
            for k in (
                "label",
                "scalars",
                "input_sha256",
                "output_sha256",
                "bitwise",
                "policy",
                "canaries",
            )
        )
    values = torch.load(OUT / "special-values.pt", weights_only=True)
    assert len(values) == 12
    for v in values.values():
        assert tensor_sha(v["actual"]) == tensor_sha(v["baseline"])
        check(torch, v["actual"], v["eager"])
    assert len(gpu["kernels"]) == 3
    lowering = []
    for k in gpu["kernels"]:
        d = Path(k["directory"])
        sem = read(d / "semantic.json")
        frozen = next(r for r in old["kernels"] if r["semantic"] == sem)
        fd = FROZEN / frozen["directory"]
        assert sha(d / "kernel.ptx") == k["ptx_sha256"]
        assert (
            read(d / "manifest.json")["metadata"]
            == read(fd / "manifest.json")["metadata"]
            == {"scalar_count": 7, "bytes": 80}
        )
        assert read(d / "physical-plan.json") == read(fd / "physical-plan.json")
        assert read(d / "nvrtc.json")["options"].count("--fmad=false") == 1
        ident = read(d / "identity.json")
        assert ident["adapter"] == sha(ROOT / "stage5/artifact.py") and ident[
            "column_adapter"
        ] == sha(OUT / "sources/stage5/column_layout_runtime.py")
        assert ident["diagnostic_binary"] == sha(BINARY) and ident[
            "semantic_emitter"
        ] == sha(ROOT / "stage0/src/descriptor2.rs")
        for name, digest in read(d / "integrity.json").items():
            assert sha(d / name) == digest
        cu = (d / "kernel.cu").read_text()
        fcu = (fd / "kernel.cu").read_text()
        assert cu.count("__umul64hi(") == 1 and fcu.count("__umul64hi(") == 2
        ir = (d / "cubecl.ir").read_text()
        fir = (fd / "cubecl.ir").read_text()
        assert ir.count("math.u_mul_hi") == 1 and fir.count("math.u_mul_hi") == 2
        assert ir.count("cube.read_scalar") == fir.count("cube.read_scalar") == 7
        slots = [int(v) for v in re.findall(r"info_v\d+\.scalars_i64\[(\d+)\]", cu)]
        frozen_slots = [
            int(v) for v in re.findall(r"info_v\d+\.scalars_i64\[(\d+)\]", fcu)
        ]
        assert slots == [0, 5, 6] and frozen_slots == [0, 2, 3, 4, 5, 6]
        lowering.append(
            dict(
                dtype=sem["buffers"][0]["dtype"],
                candidate=str(d),
                generic=str(fd),
                candidate_cuda_mulhi=1,
                generic_cuda_mulhi=2,
                candidate_ir_mulhi=1,
                generic_ir_mulhi=2,
                candidate_cuda_metadata_slots=slots,
                generic_cuda_metadata_slots=frozen_slots,
                candidate_attributes=read(d / "resources.json")["attributes"],
                generic_attributes=read(fd / "resources.json")["attributes"],
            )
        )
    frozen_source = (ROOT / "stage5/layout_probe/src/main.rs").read_text()
    new_source = (ROOT / "stage5/column_layout_probe/src/main.rs").read_text()
    for start, end in (
        ("    let builtin = |kind| {", "    let mut values = HashMap::new();"),
        ("            descriptor2::emit(", "    s.terminate_yield();"),
    ):
        assert (
            frozen_source[frozen_source.index(start) : frozen_source.index(end)]
            == new_source[new_source.index(start) : new_source.index(end)]
        )
    measured_source = OUT / "phase-sources/column_layout_measure.py"
    assert timing["source_sha256"] == sha(measured_source)
    assert function_ast(measured_source, "graphs") == function_ast(
        ROOT / "stage5/column_layout_measure.py", "graphs"
    )
    assert timing["backend_sha256"] == sha(
        ROOT / "stage5/column_layout_backend.py"
    ) and timing["plan_sha256"] == sha(OUT / "timing-plan.json")
    assert len(timing["rows"]) == len(plan["cases"]) == 12
    count = 0
    traces = set()
    clocks = set()
    summary = []
    for r in timing["rows"]:
        qual = next(q for q in gpu["rows"] if q["label"] == r["label"])
        assert (
            r["input_sha256"] == qual["input_sha256"]
            and r["output_sha256"] == qual["output_sha256"]
        )
        assert r["artifacts"]["component"]["directory"] == qual["candidate"]
        sem = read(Path(qual["candidate"]) / "semantic.json")
        for name, a in r["artifacts"].items():
            d = Path(a["directory"])
            if name == "ordinary":
                retained = OUT / "timed-ordinary" / d.name
                if d.exists():
                    assert (retained / "kernel.ptx").read_bytes() == (
                        d / "kernel.ptx"
                    ).read_bytes()
                d = retained
            assert sha(d / "kernel.ptx") == a["ptx_sha256"]
            assert read(d / "manifest.json")["entrypoint"] == r["kernels"][name]
            assert read(d / "semantic.json") == sem
            w = r["wrappers"][name]
            p = OUT / w["path"]
            assert sha(p) == w["sha256"]
            if Path(w["observed_cache_path"]).exists():
                assert p.read_bytes() == Path(w["observed_cache_path"]).read_bytes()
            calls = [
                n
                for n in ast.walk(ast.parse(p.read_text()))
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in ("column_compile", "cubecl_stage5_compile")
            ]
            assert len(calls) == 1 and ast.literal_eval(calls[0].args[0]) == sem
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
                    for e in read_gzip(trace)["traceEvents"]
                    if e.get("cat") == "kernel"
                ]
                assert len(events) == len(p["samples_us"]) == 16
                a = r["artifacts"][name]["plan"]
                coverage = (
                    a["workgroup"][0] * a["vector_width"] * a["elements_per_unit"]
                )
                numel = r["case"]["shape"][0] * r["case"]["shape"][1]
                grid = [(numel + coverage - 1) // coverage, 1, 1]
                assert all(
                    e["name"] == r["kernels"][name]
                    and e["args"]["grid"] == grid
                    and e["args"]["block"] == a["workgroup"]
                    for e in events
                )
                assert all(
                    abs(e["dur"] - s) < 0.002 for e, s in zip(events, p["samples_us"])
                )
                assert statistics.median(p["samples_us"]) == p["median_us"]
                count += len(events)
        med = {
            n: statistics.median(
                s for rr in r["rounds"] for s in rr["paths"][n]["samples_us"]
            )
            for n in ("generic", "component", "ordinary")
        }
        summary.append(
            dict(
                label=r["label"],
                median_us=med,
                wins={
                    n: sum(
                        rr["paths"]["component"]["median_us"]
                        < rr["paths"][n]["median_us"]
                        for rr in r["rounds"]
                    )
                    for n in ("generic", "ordinary")
                },
            )
        )
    assert count == 3456 and len(traces) == 216
    preservation = read(OUT / "preservation.json")
    for name, digest in preservation.items():
        assert sha(ROOT / name) == digest, name
    for name, digest in read(FROZEN / "integrity.json")["files"].items():
        assert sha(FROZEN / name) == digest, name
    records = sass_records()
    metrics = list(csv.DictReader((OUT / "ncu-import.csv").open()))
    ledger = []
    expected = [
        r["kernels"][name]
        for shape in plan["counter_shapes"]
        for r in timing["rows"]
        if r["case"]["shape"] == shape and r["case"]["dtype"] == "float32"
        for name in ("generic", "component", "ordinary")
    ]
    assert len(expected) == 9 and len({r["invocation"] for r in records}) == 9
    for i in range(9):
        selected = [r for r in records if r["invocation"] == i]
        assert all(r["kernel"] == expected[i] for r in selected)
        m = {r["Metric Name"]: r["Metric Value"] for r in metrics if r["ID"] == str(i)}
        total = sum(r["executed"] for r in selected)
        assert total == int(m["smsp__inst_executed.sum"].replace(",", ""))
        ledger.append(
            dict(
                invocation=i,
                kernel=expected[i],
                executed_warp_instructions=total,
                registers_per_thread=int(m["launch__registers_per_thread"]),
                groups={
                    op: sum(
                        r["executed"]
                        for r in selected
                        if opcode_matches(r["instruction"], op)
                    )
                    for op in ("LDG", "STG", "LDL", "STL", "BSSY", "BSYNC", "MUFU.RCP")
                },
            )
        )
    write("sass-ledger.json", ledger)
    write(
        "analysis.json",
        dict(
            complete=True,
            address_identities=identity_count,
            specialized_column_addresses=changed_addresses,
            gpu_cases=78,
            changed_gpu_cases=24,
            live_controls=90,
            stream_controls=12,
            native_controls=8,
            traces=len(traces),
            kernel_durations=count,
            coarse_sm_memory_clocks=sorted(clocks),
            timings=summary,
            wins={
                n: sum(r["wins"][n] for r in summary) for n in ("generic", "ordinary")
            },
            lowering=lowering,
            preserved_files=len(preservation),
            ncu_invocations=9,
            source_sha256=sha(__file__),
        ),
    )
    print(
        "PASS",
        identity_count,
        "addresses;",
        count,
        "timed launches;",
        len(preservation),
        "frozen files;9 executed-SASS invocations",
    )


def read_gzip(path):
    import json

    return json.loads(gzip.decompress(path.read_bytes()))


if __name__ == "__main__":
    main()
