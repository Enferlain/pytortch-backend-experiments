"""Curate executed reduction artifacts; unchanged dependency source stays ignored."""

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
trace = json.loads((ROOT / "artifacts/schedule-replay.json").read_text())
events = json.loads((ROOT / "artifacts/runtime-events.json").read_text())
results = json.loads((ROOT / "artifacts/results.json").read_text())
selected = []
representatives = ROOT / "artifacts/representatives"
representatives.mkdir(exist_ok=True)
names = (
    "row_dynamic_torch.float32",
    "row_dynamic_torch.float16",
    "row_dynamic_torch.bfloat16",
    "column_strides_torch.float32",
    "fused_mean_torch.float32",
    "normalization_torch.float32",
    "normalization_torch.float16",
    "normalization_torch.bfloat16",
    "two_sums_normal_fusion_shared_reuse",
    "lazy_semantic_mask_in_reduction",
)
for check in results["checks"]:
    if check["name"] not in names:
        continue
    section = trace[slice(*check["trace_range"])]
    kernels = [v for v in section if v["event"] == "kernel"]
    # Row tail and dynamic reuse share an artifact; the column/layout fixture
    # retains every distinct strategy, including both natural split phases.
    for i, kernel in enumerate(kernels):
        semantic = kernel["descriptor"]
        entry = next(
            (
                e
                for e in events
                if e["event"] == "compile"
                and json.loads((Path(e["directory"]) / "semantic.json").read_text())
                == semantic
            ),
            None,
        )
        if entry is None:
            raise RuntimeError("no actual compiler evidence for semantic descriptor")
        source = Path(entry["directory"])
        label = check["name"].replace("torch.", "") + "-" + str(i)
        destination = representatives / label
        destination.mkdir(exist_ok=True)
        for name in (
            "descriptor.json",
            "semantic.json",
            "physical-plan.json",
            "cubecl.ir",
            "kernel.cu",
            "kernel.ptx",
            "manifest.json",
            "ptx-signature.txt",
            "nvrtc.json",
        ):
            shutil.copyfile(source / name, destination / name)
        (destination / "replay.json").write_text(json.dumps(kernel, indent=2))
        launches = [
            e
            for e in events
            if e["event"] == "launch" and e["entrypoint"] == entry["entrypoint"]
        ]
        (destination / "launches.json").write_text(json.dumps(launches, indent=2))
        (destination / "schedule.json").write_text(
            json.dumps(
                [
                    e
                    for e in section
                    if e["event"] in ("capability", "schedule", "fusion_delegate")
                ],
                indent=2,
            )
        )
        selected.append(
            {
                "label": label,
                "entrypoint": entry["entrypoint"],
                "source": str(source),
                "ptx_sha256": hashlib.sha256(
                    (source / "kernel.ptx").read_bytes()
                ).hexdigest(),
            }
        )

components = json.loads((ROOT / "artifacts/components.json").read_text())
component_representatives = []
for i, check in enumerate(components["masked_result_checks"]):
    source = Path(check["artifact"])
    label = "masked_enclosing_result_component-" + str(i)
    destination = representatives / label
    destination.mkdir(exist_ok=True)
    for name in (
        "descriptor.json",
        "semantic.json",
        "physical-plan.json",
        "cubecl.ir",
        "kernel.cu",
        "kernel.ptx",
        "manifest.json",
        "ptx-signature.txt",
        "nvrtc.json",
    ):
        shutil.copyfile(source / name, destination / name)
    (destination / "component.json").write_text(json.dumps(check, indent=2))
    component_representatives.append(
        {
            "label": label,
            "boundary": "validated descriptor; no synthetic Inductor graph",
            "ptx_sha256": hashlib.sha256(
                (source / "kernel.ptx").read_bytes()
            ).hexdigest(),
        }
    )

# Probe exactly the first executed row semantic/physical pair on the existing HIP compiler.
first = representatives / selected[0]["label"]
noncuda = ROOT / "artifacts/noncuda"
noncuda.mkdir(exist_ok=True)
for name in ("descriptor.json", "semantic.json", "physical-plan.json"):
    shutil.copyfile(first / name, noncuda / name)
binary = ROOT.parent / "stage0/target/debug/cubecl-stage0"
proc = subprocess.run(
    [str(binary), str(noncuda), "hip-probe", str(noncuda / "descriptor.json")],
    capture_output=True,
    text=True,
)
(noncuda / "compiler.log").write_text(proc.stdout + proc.stderr)
if proc.returncode:
    raise RuntimeError(proc.stderr)

# Full exact functions/source snapshots remain local; the eligible trace contains
# function identity, spans and hashes only. Graph edges do not prove runtime calls.
calls = json.loads((ROOT / "artifacts/compiler-path.json").read_text())
seen = set()
identities = []
snapshots = {}
for call in calls:
    key = (call["file"], call["line"], call["symbol"])
    if key in seen:
        continue
    seen.add(key)
    path = Path(call["file"])
    source = path.read_text()
    lines = source.splitlines()
    parsed = ast.parse(source)
    candidate = next(
        (
            node
            for node in ast.walk(parsed)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and min([node.lineno, *[x.lineno for x in node.decorator_list]])
            == call["line"]
        ),
        None,
    )
    text = (
        "\n".join(lines[candidate.lineno - 1 : candidate.end_lineno])
        if candidate
        else lines[call["line"] - 1]
    )
    identities.append(
        {
            **call,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "function_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "last_line": candidate.end_lineno if candidate else call["line"],
        }
    )
    snapshots[str(path)] = source
(ROOT / "artifacts/source-snapshots.json").write_text(json.dumps(snapshots, indent=2))
(ROOT / "artifacts/source-trace.json").write_text(json.dumps(identities, indent=2))
(ROOT / "artifacts/coverage.json").write_text(
    json.dumps(
        {
            "representatives": selected,
            "component_representatives": component_representatives,
            "hip_source_probe": "passed; source generation only",
        },
        indent=2,
    )
)
print("Stage 3 capture/HIP source probe PASS", len(selected))
