"""Small inspectable evidence set, without local caches or raw profiler arrays."""

import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


math = {
    name: json.loads((ARTIFACTS / ("math-" + name + ".json")).read_text())
    for name in ("cubecl", "triton_matched", "triton_default")
}
rows = []
for i, original in enumerate(math["cubecl"]["checks"]):
    row = {k: original[k] for k in ("op", "dtype")}
    row["implementations"] = {}
    for name, data in math.items():
        check = data["checks"][i]
        assert (check["op"], check["dtype"]) == (row["op"], row["dtype"])
        row["implementations"][name] = {
            k: v for k, v in check.items() if not k.endswith("bytes_hex")
        }
        row[name + "_bitwise_cubecl"] = (
            original["output_sha256"] == check["output_sha256"]
        )
    rows.append(row)
(ARTIFACTS / "policy.json").write_text(
    json.dumps(
        {
            "records": rows,
            "environment": {k: d["environment"] for k, d in math.items()},
            "raw_hashes": {
                k: digest(ARTIFACTS / ("math-" + k + ".json")) for k in math
            },
            "qualification": "Per-operation finite/special-value probes; not a proof of all graph numerical equivalence. Tensor-constant fp16/bf16 eager discrepancy shared by both configured compiled paths. RNG streams/reduction ordering not assumed identical.",
        },
        indent=2,
    )
)

profiling = json.loads((ARTIFACTS / "profiling.json").read_text())
sources = [(r["case"], Path(r["directory"])) for r in profiling["records"]]
matrix = json.loads((ARTIFACTS / "baseline-cubecl_stage5.json").read_text())
manifests = {
    json.loads(p.read_text())["entrypoint"]: p.parent
    for p in (ROOT / "cache/artifacts").glob("*/manifest.json")
}
for name in (
    "fused-float16-1048583",
    "rms_backward-float32-64x1024",
    "training-float32-5x9",
):
    r = next(r for r in matrix["records"] if r["name"] == name)
    for i, launch in enumerate(r["physical_launches"]):
        sources.append((name + "-" + str(i), manifests[launch["entrypoint"]]))
inventory = []
for name, source in sources:
    target = ARTIFACTS / "representatives" / name
    target.mkdir(parents=True, exist_ok=True)
    for file in (
        "descriptor.json",
        "semantic.json",
        "physical-plan.json",
        "cubecl.ir",
        "kernel.cu",
        "kernel.ptx",
        "manifest.json",
        "nvrtc.json",
        "resources.json",
        "integrity.json",
    ):
        if (source / file).is_file():
            shutil.copyfile(source / file, target / file)
    offline = ARTIFACTS / "offline" / source.name
    if (offline / "summary.json").is_file():
        for file in ("summary.json", "ptxas.txt", "nvdisasm.txt"):
            shutil.copyfile(offline / file, target / ("offline-" + file))
    inventory.append(
        {
            "name": name,
            "source": str(source),
            "files": {f.name: digest(f) for f in target.iterdir() if f.is_file()},
        }
    )
details = copy.deepcopy(json.loads((ARTIFACTS / "final-gates.json").read_text()))
(ARTIFACTS / "coverage.json").write_text(
    json.dumps(
        {
            "fixed_corpus_cases": len(matrix["records"]),
            "failures": matrix["failures"],
            "final_gates": details,
            "representatives": inventory,
            "independence": matrix["independence"],
            "native_launches": matrix["native_launches"],
            "compiler_invocations": matrix["compiler_invocations"],
        },
        indent=2,
    )
)
print(
    "Curated",
    len(inventory),
    "kernel artifacts and",
    len(rows),
    "per-operation policies",
    flush=True,
)
