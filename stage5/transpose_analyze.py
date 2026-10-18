"""Offline validation of the one-component transpose trial and retained traces."""

import ast
import csv
import gzip
import json
from pathlib import Path
import statistics

from stage5.transpose_trial import OUT, ROOT, sha, write
from stage5.transpose_model import SHAPES, verify, width


def read(p):
    return json.loads(Path(p).read_text())


def function_ast(path, name):
    return ast.dump(
        next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
    )


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
    capture = read(OUT / "capture.json")
    cpu = read(OUT / "cpu.json")
    gpu = read(OUT / "gpu.json")
    timing = read(OUT / "timing.json")
    native = read(OUT / "native-controls.json")
    plan = read(OUT / "timing-plan.json")
    assert all(d["complete"] for d in (capture, cpu, gpu, timing, native))
    assert (
        len(capture["records"]) == 27
        and len(cpu["rows"]) == 24
        and len(gpu["rows"]) == 54
    )
    assert (
        len(gpu["live_controls"]) == 42
        and len(gpu["stream_controls"]) == 6
        and len(gpu["view_controls"]) == 9
        and len(native["rows"]) == 7
    )
    assert capture["source_sha256"] == sha(OUT / "phase-sources/transpose_trial.py")
    assert (
        cpu["model_sha256"] == sha(OUT / "phase-sources/transpose_model.py")
        and not cpu["gpu_source_existed"]
    )
    assert cpu["capture_sha256"] == sha(OUT / "capture.json")
    assert ast.dump(
        ast.parse((OUT / "phase-sources/transpose_model.py").read_text())
    ) == ast.dump(ast.parse((ROOT / "stage5/transpose_model.py").read_text()))
    assert timing["source_sha256"] == sha(OUT / "phase-sources/transpose_measure.py")
    assert function_ast(
        OUT / "phase-sources/transpose_measure.py", "graphs"
    ) == function_ast(ROOT / "stage5/transpose_measure.py", "graphs")
    assert timing["runtime_sha256"] == sha(ROOT / "stage5/transpose_runtime.py")
    for r in cpu["rows"]:
        assert verify(r["m"], r["n"], r["width"]) == {
            k: v for k, v in r.items() if k != "dtype"
        }
    from stage5.transpose_runtime import BINARY, admit

    assert gpu["binary_sha256"] == sha(BINARY)
    assert gpu["special_values_sha256"] == sha(OUT / "special-values.pt")
    import torch

    values = torch.load(OUT / "special-values.pt", weights_only=True)
    from stage5.transpose_qualify import check

    for v in values.values():
        assert torch.equal(
            v["actual"].contiguous().view(torch.uint8),
            v["baseline"].contiguous().view(torch.uint8),
        )
        check(torch, v["actual"], v["eager"])
    for row in gpu["rows"]:
        assert row["bitwise"] and row["policy"] and row["canaries"]
        assert (row["shape"][1], row["shape"][0]) in SHAPES
        assert row["width"] == width(row["shape"][1], row["shape"][0], row["dtype"])
    for k in gpu["kernels"]:
        directory = OUT / k["directory"]
        sem = read(directory / "semantic.json")
        admit(sem)
        assert k["ptx_sha256"] == sha(directory / "kernel.ptx")
        assert read(directory / "manifest.json")["entrypoint"] == k["entrypoint"]
        assert read(directory / "nvrtc.json")["options"].count("--fmad=false") == 1
    assert len(timing["rows"]) == 18 and plan["rounds"] == 6 and plan["calls"] == 16
    raw_count = 0
    summary = []
    trace_paths = set()
    clocks = set()
    for r in timing["rows"]:
        assert len(r["rounds"]) == 6
        qualified = next(q for q in gpu["rows"] if q["label"] == r["label"])
        assert r["input_sha256"] == qualified["input_sha256"]
        assert r["output_sha256"] == qualified["output_sha256"]
        assert r["component"] == qualified["candidate"]
        candidate = OUT / r["component"]
        normal = OUT / "timed-ordinary" / Path(r["ordinary"]).name
        assert read(normal / "manifest.json")["entrypoint"] == r["kernels"]["ordinary"]
        assert (
            read(candidate / "manifest.json")["entrypoint"] == r["kernels"]["component"]
        )
        assert read(normal / "semantic.json") == read(candidate / "semantic.json")
        assert read(normal / "physical-plan.json") == r["ordinary_plan"]
        m, n = r["base_shape"]
        v = width(m, n, r["dtype"])
        for rr in r["rounds"]:
            for state in ("activity_before", "activity_after"):
                lines = rr[state]["gpu"]["stdout"].splitlines()
                records = list(csv.reader(lines))
                for fields in records[1:]:
                    if "510ddc27" in ",".join(fields):
                        clocks.add((fields[5].strip(), fields[6].strip()))
            assert set(rr["paths"]) == {"ordinary", "component"}
            for name, packet in rr["paths"].items():
                p = OUT / packet["trace"]
                assert str(p) not in trace_paths
                trace_paths.add(str(p))
                assert sha(p) == packet["trace_sha256"]
                trace = json.loads(gzip.decompress(p.read_bytes()))
                kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
                assert len(kernels) == 16 and all(
                    e["name"] == r["kernels"][name] for e in kernels
                )
                assert len(packet["samples_us"]) == 16
                assert all(
                    abs(e["dur"] - sample) < 0.002
                    for e, sample in zip(kernels, packet["samples_us"])
                )
                assert packet["median_us"] == statistics.median(packet["samples_us"])
                expected_grid = (
                    (n // v + 255) // 256
                    if name == "component"
                    else (
                        m * n
                        + r["ordinary_plan"]["workgroup"][0]
                        * r["ordinary_plan"]["vector_width"]
                        * r["ordinary_plan"]["elements_per_unit"]
                        - 1
                    )
                    // (
                        r["ordinary_plan"]["workgroup"][0]
                        * r["ordinary_plan"]["vector_width"]
                        * r["ordinary_plan"]["elements_per_unit"]
                    )
                )
                assert all(e["args"]["grid"] == [expected_grid, 1, 1] for e in kernels)
                assert all(
                    e["args"]["block"]
                    == (
                        [256, 1, 1]
                        if name == "component"
                        else r["ordinary_plan"]["workgroup"]
                    )
                    for e in kernels
                )
                raw_count += len(kernels)
        medians = {
            name: statistics.median(
                [s for rr in r["rounds"] for s in rr["paths"][name]["samples_us"]]
            )
            for name in ("ordinary", "component")
        }
        summary.append(
            dict(
                label=r["label"],
                **{k + "_us": val for k, val in medians.items()},
                component_wins=sum(
                    rr["paths"]["component"]["median_us"]
                    < rr["paths"]["ordinary"]["median_us"]
                    for rr in r["rounds"]
                ),
                component_grid=(n // v + 255) // 256,
                active_workers=n // v,
                vector=v,
                iterations_per_worker=m // v,
            )
        )
    assert raw_count == 3456 and len(trace_paths) == 216
    preserved = read(OUT / "preservation.json")
    for name, digest in preserved.items():
        if name == "stage5/.gitignore":
            before = OUT / "ignore.before"
            assert sha(before) == digest
            assert (
                (ROOT / name).read_text()
                == before.read_text()
                + "!artifacts/reuse-first-transpose/\n!artifacts/reuse-first-transpose/**/*.csv\n!artifacts/reuse-first-transpose/**/*.log\n!artifacts/reuse-first-transpose/*.ncu-repz\n"
            )
            continue
        assert sha(ROOT / name) == digest, name
    source = (
        ROOT
        / ".stage0-sources/cubecl/crates/cubecl-std/src/tensor/contiguous/perpendicular.rs"
    )
    assert source.read_bytes() == (OUT / "upstream-perpendicular.rs").read_bytes()
    s = source.read_text()
    body = s[s.index("#[cube(") : s.index("/// Launches")]
    for a, b in [
        ('#[cube(launch_unchecked, address_type = "dynamic")]', "#[cube]"),
        ("input: &Tensor<Vector<T, N>>", "input: &VirtualTensor<T, N>"),
        (
            "output: &mut Tensor<Vector<T, N>>",
            "output: &mut VirtualTensor<T, N, ReadWrite>",
        ),
        (
            "output.coordinate(offset_output * vector_size, axis)",
            "((offset_output * vector_size) / output.stride(axis)) % output.shape(axis)",
        ),
        ("let batched = input[index];", "let batched = input.read(index);"),
        ("output[index_out] = batched;", "output.write(index_out, batched);"),
    ]:
        assert body.count(a) == 1
        body = body.replace(a, b)
    assert body == (ROOT / "stage5/transpose_probe/src/component.rs").read_text()
    sass = sass_ledger()
    sass_rows = []
    counters = list(csv.DictReader((OUT / "ncu-import.csv").open()))
    assert {r["invocation"] for r in sass} == set(range(8))
    for invocation in range(8):
        rs = [r for r in sass if r["invocation"] == invocation]
        name = rs[0]["kernel"]
        metrics = [r for r in counters if int(r["ID"]) == invocation]
        totals = [r for r in metrics if r["Metric Name"] == "smsp__inst_executed.sum"]
        assert len(totals) == 1 and totals[0]["Kernel Name"] == name
        assert int(totals[0]["Metric Value"].replace(",", "")) == sum(
            r["executed_warp_instructions"] for r in rs
        )
        sass_rows.append(
            dict(
                kernel=name,
                invocation=invocation,
                executed_warp_instructions=sum(
                    r["executed_warp_instructions"] for r in rs
                ),
                bssy=sum(
                    r["executed_warp_instructions"]
                    for r in rs
                    if "BSSY" in r["instruction"]
                ),
                bsync=sum(
                    r["executed_warp_instructions"]
                    for r in rs
                    if "BSYNC" in r["instruction"]
                ),
                loads128=sum(
                    r["executed_warp_instructions"]
                    for r in rs
                    if "LDG.E.128." in r["instruction"]
                ),
                stores128=sum(
                    r["executed_warp_instructions"]
                    for r in rs
                    if "STG.E.128" in r["instruction"]
                ),
                local=sum(
                    r["executed_warp_instructions"]
                    for r in rs
                    if any(op in r["instruction"] for op in ("LDL", "STL"))
                ),
                shuffles=sum(
                    r["executed_warp_instructions"]
                    for r in rs
                    if "SHFL" in r["instruction"]
                ),
                barriers=sum(
                    r["executed_warp_instructions"]
                    for r in rs
                    if "BAR." in r["instruction"]
                ),
            )
        )
    write(
        "analysis.json",
        dict(
            complete=True,
            summary=summary,
            sass=sass_rows,
            clock_snapshots=sorted(clocks),
            raw_captures=len(trace_paths),
            raw_kernels=raw_count,
            preserved_files=len(preserved) - 1,
            declared_metadata_changes=["STAGE5_REPORT.md", "stage5/.gitignore"],
            cpu_scalar_identities=sum(r["scalar_identities"] for r in cpu["rows"]),
            component_pair_wins=sum(r["component_wins"] for r in summary),
            paired_comparisons=108,
            scope="Same-process paired complete graph device durations; coarse clocks vary. No stable cross-state intrinsic-cost, exclusive-GPU or stock-launcher claim.",
        ),
    )
    lines = [
        "| Case | Ordinary µs | Component µs | Component paired wins |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {r['label']} | {r['ordinary_us']:.3f} | {r['component_us']:.3f} | {r['component_wins']}/6 |"
        for r in summary
    ]
    (OUT / "timing-table.md").write_text("\n".join(lines) + "\n")
    print(
        "PASS",
        len(preserved) - 1,
        "preserved files;",
        raw_count,
        "raw kernels;",
        len(summary),
        "graphs; component wins",
        sum(r["component_wins"] for r in summary),
    )


if __name__ == "__main__":
    main()
