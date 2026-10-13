"""Curate executed artifacts and classify owners using scheduling and runtime evidence."""

import ast
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
trace = json.loads((ART / "schedule-replay.json").read_text())
events = json.loads((ART / "runtime-events.json").read_text())
results = json.loads((ART / "results.json").read_text())
raw_census = json.loads((ART / "fallback-census.json").read_text())
components = json.loads((ART / "components.json").read_text())
representatives = ART / "representatives"
representatives.mkdir(exist_ok=True)
files = (
    "descriptor.json",
    "semantic.json",
    "physical-plan.json",
    "cubecl.ir",
    "kernel.cu",
    "kernel.ptx",
    "manifest.json",
    "ptx-signature.txt",
    "nvrtc.json",
)
compiled = [e for e in events if e["event"] == "compile"]
by_semantic = {
    json.dumps(
        json.loads((Path(e["directory"]) / "semantic.json").read_text()), sort_keys=True
    ): e
    for e in compiled
}
coverage = []
graphs = []


def retain(source, label, evidence):
    target = representatives / label
    target.mkdir(exist_ok=True)
    for name in files:
        shutil.copyfile(source / name, target / name)
    (target / "evidence.json").write_text(json.dumps(evidence, indent=2))
    manifest = json.loads((source / "manifest.json").read_text())
    coverage.append(
        {
            "label": label,
            "entrypoint": manifest["entrypoint"],
            "ptx_sha256": hashlib.sha256(
                (source / "kernel.ptx").read_bytes()
            ).hexdigest(),
            "compiler_binary_sha256": manifest.get("compiler_binary_sha256"),
            "function_attributes": manifest.get("driver_function_attributes"),
        }
    )


def classification(node, name):
    if node["decision"] == "reject":
        return "unsupported", node["reason"]
    if node["execution_path"] == "cubecl_generated":
        return (
            "cubecl_generated",
            "Semantic capability accepted before normal SIMD scheduling; native/profiler launches cross-checked.",
        )
    if (
        node.get("target") == "aten.mm.out"
        and node.get("kernel") == "extern_kernels.mm"
    ):
        return (
            "vendor_library",
            "Normal Inductor mm extern selection; profiler records aten::mm and vendor GPU work.",
        )
    if name == "explicit_existing_rng_fallback":
        return (
            "explicit_fallback",
            "rng_policy=aten_extern selects existing fallback_random lowering before random replacement.",
        )
    return (
        "aten_extern",
        "Inductor-owned extern (including RandomSeeds), selected before generated scheduling.",
    )


def profile_owners(profile, graph_name):
    if profile is None:
        return []
    chrome = json.loads((ART / (profile["label"] + "-profiler.json")).read_text())[
        "traceEvents"
    ]
    cpu = {
        str(e.get("args", {}).get("External id")): e["name"]
        for e in chrome
        if e.get("cat") == "cpu_op"
    }
    owners = []
    for event in chrome:
        if event.get("cat") != "kernel":
            continue
        caller = cpu.get(str(event.get("args", {}).get("External id")))
        if event["name"] in profile["generated_entrypoints"]:
            path, reason = (
                "cubecl_generated",
                "Exact runtime entrypoint and native enqueue count match.",
            )
        elif caller == "aten::mm":
            path, reason = (
                "vendor_library",
                "GPU External id correlates to aten::mm; scheduler selects extern_kernels.mm/aten.mm.out.",
            )
        elif (
            caller == "aten::normal_" and graph_name == "explicit_existing_rng_fallback"
        ):
            path, reason = (
                "explicit_fallback",
                "GPU External id correlates to normal_ under explicit fallback_random policy.",
            )
        elif caller == "aten::random_":
            path, reason = (
                "aten_extern",
                "GPU External id correlates to random_; scheduler owns RandomSeeds/randint.low_out.",
            )
        elif (
            caller in ("aten::mul_", "aten::add_")
            and graph_name == "representative_training_optimizer_state"
        ):
            path, reason = (
                "eager_optimizer_outside_capture",
                "GPU External id correlates to the explicit non-foreach SGD update.",
            )
        elif (
            caller == "aten::fill_"
            and graph_name == "representative_training_optimizer_state"
        ):
            path, reason = (
                "autograd_runtime",
                "GPU External id correlates to fill_; CPU trace records ones_like creating the scalar loss upstream.",
            )
        else:
            raise RuntimeError(f"Unclassified GPU owner: {caller} / {event['name']}")
        owners.append(
            {
                "kernel": event["name"],
                "cpu_owner": caller,
                "execution_path": path,
                "reason": reason,
            }
        )
    return owners


for check in results["checks"]:
    section = trace[slice(*check["trace_range"])]
    section_events = events[slice(*check["event_range"])]
    profile = check.get("profile")
    current = None
    phases = []
    for item in section:
        if item["event"] == "capability_graph":
            current = {"is_backward": item["is_backward"], "nodes": [], "kernels": []}
            phases.append(current)
        elif item.get("boundary") == "pre_fusion":
            assert current is not None
            current["nodes"].append(item)
        elif item["event"] == "kernel":
            assert current is not None
            current["kernels"].append(item)
    retained = set()
    node_entries = []
    for phase_id, phase in enumerate(phases):
        for k in phase["kernels"]:
            entry = by_semantic[json.dumps(k["descriptor"], sort_keys=True)]
            source = Path(entry["directory"])
            if entry["entrypoint"] in retained:
                continue
            retained.add(entry["entrypoint"])
            label = check["name"].replace("torch.", "") + "-" + str(len(retained) - 1)
            launches = [
                e
                for e in section_events
                if e["event"] == "launch" and e["entrypoint"] == entry["entrypoint"]
            ]
            retain(
                source,
                label,
                {
                    "replay": k,
                    "schedule": phase["nodes"],
                    "launches": launches,
                    "profile": profile,
                    "check": check["name"],
                },
            )
        for node in phase["nodes"]:
            path, reason = classification(node, check["name"])
            groups = []
            for k in phase["kernels"]:
                if node["node"] not in k["nodes"]:
                    continue
                entry = by_semantic[json.dumps(k["descriptor"], sort_keys=True)]
                groups.append(
                    {
                        "members": k["nodes"],
                        "entrypoint": entry["entrypoint"],
                        "positive_grid_launches_in_category": sum(
                            e["event"] == "launch"
                            and e["entrypoint"] == entry["entrypoint"]
                            and e["grid"][0] > 0
                            for e in section_events
                        ),
                    }
                )
            node_entries.append(
                {
                    "identity": f"phase{phase_id}/{node['node']}",
                    "is_backward": phase["is_backward"],
                    "semantic_category": node.get("ir_type"),
                    "operations": node.get("semantic_ops"),
                    "selected_execution_path": path,
                    "selection_reason": reason,
                    "target": node.get("target"),
                    "wrapper_kernel": node.get("kernel"),
                    "static_wrapper_sites": 1,
                    "generated_groups": groups,
                    "triton_involvement": False,
                    "copies_materializations": "No CubeCL copy-in/out or layout conversion; ordinary PyTorch result/seed/mask/intermediate allocation and ABI metadata transfer remain. Vendor internal work is not claimed absent.",
                }
            )
    graphs.append(
        {
            "graph": check["name"],
            "nodes": node_entries,
            "positive_grid_cubecl_launches": sum(
                e["event"] == "launch" and e["grid"][0] > 0 for e in section_events
            ),
            "cubecl_compile_events": sum(
                e["event"] == "compile" for e in section_events
            ),
            "profile": profile,
            "profile_gpu_owners": profile_owners(profile, check["name"]),
            "cpu_call_counts": dict(Counter(profile["cpu_operator_names"]))
            if profile
            else None,
            "unsupported": check.get("cases", []),
            "graph_breaks": 0,
            "triton_kernels": 0,
            "count_scope": "Generated launch counts are category-wide, shared by fused members; do not sum per-node counts. CPU/profile counts cover one warmed representative call, not every specialization.",
            "optimizer_boundary": check.get("optimizer_boundary"),
        }
    )

for kind, rows in (
    ("counter_bits", components["counter_bit_checks"]),
    ("lazy_indirect", components["lazy_invalid_address_guards"]),
):
    for i, row in enumerate(rows):
        retain(
            Path(row["artifact"]),
            f"{kind}-{i}",
            {
                "component": row,
                "boundary": "Validated descriptor; not a fabricated Dynamo graph.",
            },
        )

runtime_checks = json.loads((ART / "runtime-checks.json").read_text())
for row in runtime_checks["checks"]:
    if row["case"].startswith("deterministic_"):
        graphs.append(
            {
                "graph": row["case"],
                "nodes": row.get("selection", row.get("reselected_extern")),
                "selected_execution_path": "aten_extern",
                "selection_reason": "Normal deterministic lowering reselects ScatterFallback before scheduling.",
                "no_generated_atomic": row["no_generated_atomic"],
                "graph_breaks": 0,
                "triton_kernels": 0,
            }
        )
(ART / "fallback-census.json").write_text(
    json.dumps(
        {
            "graphs": graphs,
            "execution_regions": raw_census["execution_regions"],
            "optimizer": {
                "path": "eager_aten_outside_capture",
                "reason": "Explicit non-fused/non-foreach SGD momentum; not an Inductor fallback or graph break.",
                "updates_compared": 4,
            },
            "policy": "Classifications combine scheduler IR owner/target, semantic replay, wrapper/native launch counts and warmed CPU/GPU profiler events. No Triton retry; no late IR-to-ATen reconstruction.",
        },
        indent=2,
    )
)

# Probe the same executed indirect semantic/physical pair, with the target's trap lowering.
chosen = next(
    x for x in coverage if x["label"].startswith("direct_storage_indirect_addresses")
)
source = representatives / chosen["label"]
noncuda = ART / "noncuda"
noncuda.mkdir(exist_ok=True)
for name in ("descriptor.json", "semantic.json", "physical-plan.json"):
    shutil.copyfile(source / name, noncuda / name)
proc = subprocess.run(
    [
        str(ROOT.parent / "stage0/target/debug/cubecl-stage0"),
        str(noncuda),
        "hip-probe",
        str(noncuda / "descriptor.json"),
    ],
    capture_output=True,
    text=True,
)
(noncuda / "compiler.log").write_text(proc.stdout + proc.stderr)
if proc.returncode:
    raise RuntimeError(proc.stderr)
hip_source = (noncuda / "kernel.hip").read_text()
assert "__builtin_trap" in hip_source and "__trap" not in hip_source

calls = json.loads((ART / "compiler-path.json").read_text())
seen = set()
sources = {}
parsed_sources = {}
identities = []
for call in calls:
    key = (call["file"], call["line"], call["symbol"])
    if key in seen:
        continue
    seen.add(key)
    path = Path(call["file"])
    if str(path) not in sources:
        sources[str(path)] = path.read_text()
        parsed_sources[str(path)] = ast.parse(sources[str(path)])
    source = sources[str(path)]
    candidate = next(
        (
            n
            for n in ast.walk(parsed_sources[str(path)])
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and min([n.lineno, *[d.lineno for d in n.decorator_list]]) == call["line"]
        ),
        None,
    )
    last = candidate.end_lineno if candidate else call["line"]
    function = "\n".join(source.splitlines()[call["line"] - 1 : last])
    identities.append(
        {
            **call,
            "last_line": last,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "function_sha256": hashlib.sha256(function.encode()).hexdigest(),
        }
    )
(ART / "source-snapshots.json").write_text(json.dumps(sources, indent=2))
(ART / "source-trace.json").write_text(json.dumps(identities, indent=2))
(ART / "coverage.json").write_text(
    json.dumps(
        {
            "representatives": coverage,
            "hip_probe": "passed source generation only; no HIP launch/portability claim",
            "source_stack_sha256": {
                str(p.relative_to(ROOT.parent)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in (
                    ROOT.parent / "fp_hardening/cubecl-fp.patch",
                    ROOT.parent / "stage0/target/debug/cubecl-stage0",
                    ROOT.parent / "stage0/target/debug/libcubecl_stage0.so",
                    ROOT.parent / "stage0/libstage0_launcher.so",
                    ROOT.parent / "stage0/launcher.c",
                    ROOT.parent / "stage0/src/descriptor4.rs",
                    ROOT.parent / "stage0/src/require.rs",
                )
            },
        },
        indent=2,
    )
)
print("Stage4 artifact capture / HIP source probe PASS", len(coverage))
