"""Qualification-only observers derived from frozen wqc and the one candidate."""

import copy
import json
from pathlib import Path
import subprocess

from stage5.rms_packed_trial import ROOT, OUT, OLD, sha, write


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def generate():
    crate = ROOT / "rms_packed_observer"
    (crate / "src").mkdir(parents=True, exist_ok=True)
    changes = {}
    for name in ("Cargo.toml", "Cargo.lock"):
        (crate / name).write_text(
            (ROOT / "rms_packed_probe" / name)
            .read_text()
            .replace("stage5-rms-packed-probe", "stage5-rms-packed-observer")
        )
    for mode, source in (
        ("wqc", ROOT / "rms_vector_probe/src/main.rs"),
        ("packed", ROOT / "rms_packed_probe/src/main.rs"),
    ):
        original = source.read_text()
        s = original
        s = (
            s.replace("memory::{IndexOp, LoadOp}", "memory::{IndexOp, LoadOp, StoreOp}")
            if mode == "packed"
            else s.replace(
                "math::{IAddOp, IMulOp, ISubOp},",
                "math::{IAddOp, IMulOp, ISubOp},\n        memory::{IndexOp, StoreOp},",
            )
        )
        s = replace_once(s, "fn main() ->", "pub fn generate() ->")
        start = s.index("    let arrays: Vec<_> = sem[")
        end = s.index("    let scalars: Vec<_>", start)
        s = (
            s[:start]
            + """    let x = b.buffer(descriptor2::elem(dtype).unwrap().to_type(b.ctx()));
    let trace = b.buffer(descriptor2::elem("float32").unwrap().to_type(b.ctx()));
    let y = b.buffer(descriptor2::elem(dtype).unwrap().to_type(b.ctx()));
    let arrays = vec![x, y];
"""
            + s[end:]
        )
        marker = """        assert_eq!(
            self.mode, 2,"""
        s = replace_once(
            s,
            marker,
            """        if self.mode == 3 {
            let ptr = s.register_with_result(&IndexOp::new(s.ctx_mut(), self.arrays[0], pos.value(s), None));
            s.register(&StoreOp::new(s.ctx_mut(), ptr, sum.read_value(s)));
            return;
        }
"""
            + marker,
        )
        s = replace_once(
            s,
            "    sink: &mut VirtualTensor<f32, One, ReadWrite>,\n",
            "    sink: &mut VirtualTensor<f32, One, ReadWrite>,\n    debug: &mut VirtualTensor<f32, One, ReadWrite>,\n",
        )
        s = replace_once(
            s,
            "            let item = reader.read(i);",
            """            let item = reader.read(i);
            #[unroll]
            for component in 0..4usize {
                let coordinate = row * (4usize * n + 32usize) + (32usize * i + UNIT_POS_X as usize) * 4usize + component;
                debug.write(coordinate, Vector::<f32, One>::new(item.elements.extract(comptime!(component))));
            }""",
        )
        s = replace_once(
            s,
            "        let sum = Vector::<f32, One>::new(sum.item());",
            """        debug.write(row * (4usize * n + 32usize) + 4usize * n + UNIT_POS_X as usize, Vector::<f32, One>::new(sum.item()));
        let sum = Vector::<f32, One>::new(sum.item());""",
        )
        marker = "    let input = VirtualTensor::<f32, Four>::__expand_new::<Operand>(&b, input);"
        s = replace_once(
            s,
            marker,
            """    let debug = ReplayExpand { mode: 3, arrays: vec![trace], ..input.clone() };
    let mut debug = VirtualTensor::<f32, One, ReadWrite>::__expand_new::<Replay>(&b, debug);
"""
            + marker,
        )
        s = replace_once(
            s, "        &mut sink,\n", "        &mut sink,\n        &mut debug,\n"
        )
        s = s.replace('"bytes":40', '"bytes":48')
        s = replace_once(
            s,
            '"arguments":[format!("{dtype} device pointer"),format!("{dtype} device pointer"),"info device pointer"]',
            '"arguments":[format!("{dtype} device pointer"),"float32 observation pointer",format!("{dtype} device pointer"),"info device pointer"]',
        )
        (crate / "src" / f"{mode}.rs").write_text(s)
        changes[mode] = {
            "parent": str(source),
            "parent_sha256": sha(source),
            "derived_sha256": sha(crate / "src" / f"{mode}.rs"),
        }
    (crate / "src/main.rs").write_text(
        'mod packed;\nmod wqc;\nfn main() -> Result<(), Box<dyn std::error::Error>> { match std::env::args().nth(3).as_deref() { Some("packed") => packed::generate(), Some("wqc") => wqc::generate(), _ => Err("observer mode".into()) } }\n'
    )
    write("observer-sources.json", changes)


def run():
    import torch
    from stage5.artifact import Kernel
    from stage5.rms_packed_trial_runtime import PLAN, templates, MECHANISM

    gpu = json.loads((OUT / "gpu.json").read_text())
    assert gpu["complete"] and len(gpu["rows"]) == 36
    binary = ROOT / "cache/reuse-first-rms/target/release/stage5-rms-packed-observer"
    kernels = {}
    for dtype, semantic in templates().items():
        for mode in ("wqc", "packed"):
            directory = OUT / "observers" / mode / dtype
            directory.mkdir(parents=True, exist_ok=True)
            entry = (
                (
                    "cubecl5_rms_cubek_vec4_"
                    if mode == "wqc"
                    else "cubecl5_rms_cubek_pack4_"
                )
                + "observe_"
                + dtype
            )
            packet = {
                "schema": 5,
                "semantic": semantic,
                "physical_plan": PLAN,
                "mechanism": "actual-cubek-rms-plane-vector4-v1"
                if mode == "wqc"
                else MECHANISM,
                "entrypoint": entry,
            }
            description = directory / "descriptor.json"
            description.write_text(json.dumps(packet, indent=2))
            p = subprocess.run(
                [str(binary), str(directory), str(description), mode],
                capture_output=True,
                text=True,
            )
            (directory / "compiler.txt").write_text(p.stdout + p.stderr)
            assert p.returncode == 0, p.stderr
            sem = copy.deepcopy(semantic)
            n = {"op": "scalar", "slot": 0}
            width = {
                "op": "add",
                "args": [
                    {"op": "mul", "args": [{"op": "constant", "value": 4}, n]},
                    {"op": "constant", "value": 32},
                ],
            }
            trace = {
                "access": "write",
                "dtype": "float32",
                "offset": {"op": "constant", "value": 0},
                "size": [{"op": "scalar", "slot": 1}, width],
                "stride": [width, {"op": "constant", "value": 1}],
            }
            sem["buffers"].insert(1, trace)
            for op in sem["ops"]:
                if op["op"] == "store":
                    op["buffer"] = 2
            (directory / "observer-semantic.json").write_text(json.dumps(sem, indent=2))
            kernels[dtype, mode] = Kernel(directory, sem, 0, PLAN, compile_ptx=True)
    rows, outputs = [], {}
    data = torch.load(OLD / "reference.pt", weights_only=True)
    for label, packet in data.items():
        original = packet["input"]
        rows_count, n = original.shape
        dtype = str(original.dtype).removeprefix("torch.")
        x = original.cuda()
        traces = {}
        results = {}
        for mode in ("wqc", "packed"):
            backing = torch.full(
                (rows_count * (4 * n + 32) + 8,),
                12345.0,
                device="cuda",
                dtype=torch.float32,
            )
            trace = backing[4:-4].view(rows_count, 4 * n + 32)
            y = torch.empty_like(x)
            kernels[dtype, mode]((n, rows_count), x, trace, y)
            cpu_trace = trace.cpu()
            padded = (n + 127) // 128 * 128
            assert torch.all(cpu_trace[:, padded : 4 * n] == 12345.0)
            assert torch.all(backing[[0, 1, 2, 3, -4, -3, -2, -1]].cpu() == 12345.0)
            traces[mode] = {
                "operands": cpu_trace[:, :padded].contiguous(),
                "sums": cpu_trace[:, 4 * n :].contiguous(),
            }
            results[mode] = y.cpu()
            assert tensor_sha(x) == tensor_sha(original)
        for field in ("operands", "sums"):
            assert tensor_sha(traces["wqc"][field]) == tensor_sha(
                traces["packed"][field]
            ), (label, field)
        assert tensor_sha(results["wqc"]) == tensor_sha(results["packed"])
        frozen = next(row for row in gpu["rows"] if row["case"] == label)
        assert tensor_sha(results["packed"]) == frozen["output_sha256"]
        rows.append(
            {
                "case": label,
                "operand_bitwise": True,
                "sum_bitwise": True,
                "final_bitwise": True,
                "normal_candidate_final_sha256": frozen["output_sha256"],
                "operand_sha256": tensor_sha(traces["packed"]["operands"]),
                "sum_sha256": tensor_sha(traces["packed"]["sums"]),
                "canaries": True,
            }
        )
        outputs[label] = traces
        write("parity.json", {"complete": False, "rows": rows})
    torch.save(outputs, OUT / "observer-values.pt")
    write(
        "parity.json",
        {
            "complete": True,
            "rows": rows,
            "values_sha256": sha(OUT / "observer-values.pt"),
            "source_sha256": sha(Path(__file__)),
            "observer_binary_sha256": sha(binary),
            "scope": "instrumented derived kernels only; per-lane FP32 operands feeding Sum, including neutral padding, and finalized all-lane sums; final bits also equal uninstrumented frozen wqc/candidate",
        },
    )


from stage5.rms_vector_trial_data import tensor_sha
