"""Retain generated evidence, while keeping upstream source snapshots ignored."""

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
result = json.loads((ART / "results.json").read_text())
selected = [
    ("cpp", "float32", "rem_by_one"),
    ("cpp", "float32", "add_negative_zero_left"),
    ("cpp", "float32", "loaded_rem"),
    ("cpp", "float16", "div_negative_zero"),
    ("cpp", "bfloat16", "mul_negative_zero_left"),
    ("llvm", "float32", "rem_negative_zero"),
    ("llvm", "float32", "loaded_rem"),
    ("llvm", "float32", "contract_sub"),
    ("llvm", "int64", "srem_by_one"),
]
for route, dtype, case in selected:
    source = ART / "kernels" / route / dtype / case
    dest = ART / "representatives" / f"{route}-{dtype}-{case}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "cubecl.ir",
        "kernel.cu",
        "kernel.ll",
        "kernel.ptx",
        "manifest.json",
        "nvrtc.json",
    ):
        if (source / name).is_file():
            shutil.copyfile(source / name, dest / name)
    record = next(
        r
        for r in result["checks"]
        if (r["route"], r["dtype"], r["case"]) == (route, dtype, case)
    )
    (dest / "fixture.json").write_text(
        json.dumps(
            {
                **record,
                "policy": result["policy"],
                "compiler_patch_sha256": result["compiler_patch_sha256"],
            },
            indent=2,
        )
    )
paths = [
    "crates/cubecl-ir/src/dialect/math.rs",
    "crates/cubecl-ir/src/dialect/cmp.rs",
    "crates/cubecl-ir/src/scope.rs",
    "crates/cubecl-ir/src/properties.rs",
    "crates/cubecl-opt/src/passes/inst_combine.rs",
    "crates/cubecl-llvm/src/shared/to_llvm/math.rs",
    "crates/cubecl-cpp/src/shared/value.rs",
    "crates/cubecl-cpp/src/shared/base.rs",
    "crates/cubecl-llvm/src/shared/base.rs",
    "crates/cubecl-llvm/src/shared/math_library.rs",
    "crates/cubecl-llvm/src/nvptx/codegen.rs",
    "crates/cubecl-llvm/src/nvptx/libdevice.rs",
]
source_root = ROOT.parent / ".stage0-sources/cubecl"
(ART / "source-snapshots.json").write_text(
    json.dumps(
        {
            p: {
                "sha256": hashlib.sha256((source_root / p).read_bytes()).hexdigest(),
                "source": (source_root / p).read_text(),
            }
            for p in paths
        },
        indent=2,
    )
)
print("retained", len(selected), "representative kernels; upstream copies ignored")
