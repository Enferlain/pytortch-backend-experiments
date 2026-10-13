"""Malformed physical packets fail before CubeCL IR construction."""

import copy
import json
from pathlib import Path
import subprocess
from stage0.bridge import native_path
from stage5.fixtures import captured
from stage5.plan import PhysicalCandidate

ROOT = Path(__file__).resolve().parent
base = {
    "schema": 5,
    "semantic": captured("add", "float32"),
    "physical_plan": PhysicalCandidate("vector_pointwise", (128, 1, 1), 4).to_dict(),
    "entrypoint": "cubecl5_validation",
}
cases = []
for label, change in [
    ("oversized_dimensions", {"workgroup": [4294967295, 4294967295, 2]}),
    (
        "uncatalogued_scalar256",
        {"strategy": "scalar_pointwise", "workgroup": [256, 1, 1], "vector_width": 1},
    ),
    ("invalid_vector_transaction", {"vector_width": 8}),
    ("invalid_plane_policy", {"plane_width": 64}),
    ("invalid_mapping", {"elements_per_unit": 7}),
]:
    d = copy.deepcopy(base)
    d["physical_plan"].update(change)
    cases.append((label, d))
d = copy.deepcopy(base)
d["semantic"] = None
cases.append(("invalid_envelope", d))
d = copy.deepcopy(base)
d["semantic"]["ops"][0]["op"] = "unknown"
cases.append(("unknown_semantic", d))
rows = []
for label, d in cases:
    out = ROOT / "artifacts/invalid" / label
    out.mkdir(parents=True, exist_ok=True)
    file = out / "descriptor.json"
    file.write_text(json.dumps(d))
    result = subprocess.run(
        [str(native_path("cubecl-stage0")), str(out), "cpp", str(file)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and not (out / "cubecl.ir").exists(), (
        result.stdout + result.stderr
    )
    rows.append(
        {
            "case": label,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "before_ir": True,
        }
    )
(ROOT / "artifacts/descriptor-controls.json").write_text(json.dumps(rows, indent=2))
print("Descriptor controls PASS", flush=True)
