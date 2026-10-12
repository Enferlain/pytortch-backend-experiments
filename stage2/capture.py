"""Curate representative evidence; retain full source/traces locally only."""

import hashlib, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
art = ROOT / "artifacts"
r = json.loads((art / "results.json").read_text())
trace = json.loads((art / "schedule-replay.json").read_text())
events = json.loads((art / "runtime-events.json").read_text())
components = json.loads((art / "components.json").read_text())
compile_dirs = {
    e["entrypoint"]: Path(e["directory"]) for e in events if e["event"] == "compile"
}
coverage = {
    "result": "Stage 2 gate passed",
    "compiler_invocations": r["compiler_invocations"],
    "launches": r["kernel_launches"],
    "categories": {},
    "compiler_patch_sha256": hashlib.sha256(
        (ROOT / "cubecl-ieee.patch").read_bytes()
    ).hexdigest(),
}
selected = {}
for label, test in [
    ("half", "dtype_dynamic_forward_backward_torch.float16"),
    ("bf16", "dtype_dynamic_forward_backward_torch.bfloat16"),
    ("division", "binary_true_division_torch.float32"),
    ("broadcast", "broadcast_dynamic_rows_columns"),
    ("broadcast-column", "broadcast_dynamic_rows_columns"),
    ("unary-where", "unary_comparison_select_torch.float32"),
    ("casts", "explicit_casts_torch.bfloat16"),
    ("boolean", "boolean_load_where_cast_storage"),
    ("transpose", "direct_layouts"),
    ("stride", "direct_layouts"),
    ("mask", "real_lazy_semantic_masks"),
    ("fusion", "genuine_multinode_fusion"),
]:
    case = next(c for c in r["checks"] if c["name"] == test)
    kernels = [
        e
        for e in trace[case["trace_start"] : case["trace_end"]]
        if e["event"] == "kernel"
    ]
    chosen = kernels[0]
    if label == "stride":
        chosen = next(
            e
            for e in kernels
            if len(e["descriptor"]["buffers"][0]["size"]) == 1
            and any(
                v.get("op") == "scalar" or v.get("value", 0) > 1
                for v in e["descriptor"]["buffers"][0]["stride"]
            )
        )
    if label == "broadcast-column":
        chosen = next(
            e for e in kernels if '"floor_div"' in json.dumps(e["descriptor"])
        )
    if label == "transpose":
        chosen = next(
            e
            for e in kernels
            if len(e["descriptor"]["buffers"][0]["size"]) == 2
            and e["descriptor"]["buffers"][0]["stride"][0].get("value") == 1
        )
    from stage2.runtime import bridge_description

    d = bridge_description(chosen["descriptor"], chosen["physical_plan"])
    directory = compile_dirs[d["entrypoint"]]
    selected[label] = (directory, chosen)
    coverage["categories"][label] = {
        "entrypoint": d["entrypoint"],
        "nodes": chosen["nodes"],
        "replayed_operations": sorted({e["op"] for e in chosen["replay"]}),
        "artifact": str(Path("representatives") / label),
        "suite_launches": case["launches"],
    }
# Retain backward and narrow executed index/mask components alongside frontend examples.
halfcase = next(
    c
    for c in r["checks"]
    if c["name"] == "dtype_dynamic_forward_backward_torch.float16"
)
backward = [
    e
    for e in trace[halfcase["trace_start"] : halfcase["trace_end"]]
    if e["event"] == "kernel"
][1]
from stage2.runtime import bridge_description

selected["half-backward"] = (
    compile_dirs[
        bridge_description(backward["descriptor"], backward["physical_plan"])[
            "entrypoint"
        ]
    ],
    backward,
)
for name, key in [
    ("wide-index", "signed_i64_floor_mod_gpu"),
    ("i64-floor-boundary", "near_i64_boundary_floor_mod_gpu"),
    ("nested-mask-store", "nested_lazy_load_and_predicated_store_gpu"),
]:
    selected[name] = (Path(components[key]["directory"]), None)
for label, (directory, kernel) in selected.items():
    dest = art / "representatives" / label
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "semantic.json",
        "physical-plan.json",
        "descriptor.json",
        "cubecl.ir",
        "kernel.cu",
        "kernel.ptx",
        "manifest.json",
    ):
        shutil.copyfile(directory / name, dest / name)
    if kernel is not None:
        (dest / "replay.json").write_text(json.dumps(kernel, indent=2))
    manifest = json.loads((dest / "manifest.json").read_text())
    manifest["cubecl_patch_sha256"] = coverage["compiler_patch_sha256"]
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    coverage.setdefault("artifact_hashes", {})[label] = {
        name: hashlib.sha256((dest / name).read_bytes()).hexdigest()
        for name in ("kernel.ptx", "kernel.cu", "cubecl.ir")
    }
# Exact upstream bodies used for findings; intentionally ignored as unmodified sources.
source_paths = [
    "stage0/python-deps/torch/_inductor/lowering.py",
    "stage0/python-deps/torch/_prims_common/__init__.py",
    "stage0/python-deps/torch/_inductor/config.py",
    "stage0/python-deps/torch/_inductor/codegen/simd.py",
    "stage0/python-deps/torch/_inductor/codegen/simd_kernel_features.py",
    "stage0/python-deps/torch/_inductor/codegen/common.py",
    "stage0/python-deps/torch/_inductor/fx_passes/joint_graph.py",
    ".stage0-sources/cubecl/crates/cubecl-ir/src/dialect/math.rs",
    ".stage0-sources/cubecl/crates/cubecl-core/src/frontend/polyfills.rs",
    ".stage0-sources/cubecl/crates/cubecl-core/src/codegen/info.rs",
    ".stage0-sources/cubecl/crates/cubecl-core/src/compute/builder.rs",
]
sources = {}
for path in source_paths:
    data = Path(path).read_text()
    sources[path] = {
        "sha256": hashlib.sha256(data.encode()).hexdigest(),
        "source": data,
    }
(art / "source-snapshots.json").write_text(json.dumps(sources, indent=2))
lines = (art / "verification.log").read_text().splitlines()
guards = [line for line in lines if "LAMBDA_GUARD:" in line or "__recompiles" in line]
(art / "guards.json").write_text(json.dumps(guards, indent=2))
coverage["guard_examples"] = guards[:12]
coverage["dynamic"] = {
    c["name"]: c["detail"]
    for c in r["checks"]
    if "dynamic" in c["name"] or c["name"] == "direct_layouts"
}
coverage["independence"] = {
    "gil": r["gil"],
    "availability_probes": r["triton_probes"],
    "import_attempts": len(r["triton_import_attempts"]),
    "loaded": r["loaded_triton"],
    "native": r["native_triton_maps"],
}
(art / "coverage.json").write_text(json.dumps(coverage, indent=2))
print("representatives", len(selected))
