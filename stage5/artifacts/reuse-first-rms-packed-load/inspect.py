"""9e8 source and integer-address verification; never imports torch or runs GPU code."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "stage5/artifacts/reuse-first-rms-vector4"
CUBECL = ".stage0-sources/cubecl/crates/"
CUBEK = ".stage0-sources/cubek/crates/cubek-reduce/src/"
SOURCES = [
    "stage0/src/descriptor2.rs",
    "stage0/src/descriptor3.rs",
    "stage0/src/descriptor5.rs",
    "stage5/artifact.py",
    "stage5/rms_vector_probe/src/main.rs",
    "stage5/rms_vector_trial_runtime.py",
    "stage5/rms_vector_trial_data.py",
    CUBEK + "components/readers/parallel.rs",
    CUBEK + "components/readers/plane.rs",
    CUBEK + "components/readers/bound_checks.rs",
    CUBEK + "components/instructions/sum.rs",
    CUBECL + "cubecl-std/src/tensor/virtual.rs",
    CUBECL + "cubecl-std/src/tensor/view/base.rs",
    CUBECL + "cubecl-std/src/tensor/view/operations/virtual_tensor.rs",
    CUBECL + "cubecl-std/src/tensor/layout/plain.rs",
    CUBECL + "cubecl-std/src/tensor/layout/linear.rs",
    CUBECL + "cubecl-core/src/compute/builder.rs",
    CUBECL + "cubecl-ir/src/dialect/memory.rs",
    CUBECL + "cubecl-ir/src/types/mod.rs",
    CUBECL + "cubecl-cpp/src/shared/unary.rs",
    CUBECL + "cubecl-cpp/src/shared/signature.rs",
]
PRIMARY = (
    "stage5/artifacts/reuse-first-rms-vector4/candidate/"
    "5c853a16b2fca1e90b2af16494027cd0edfdb9f44514f4c93dc3ca490f0ec68c/"
)
ARTIFACTS = [
    PRIMARY + name for name in ("cubecl.ir", "kernel.cu", "kernel.ptx", "nvrtc.json")
] + [
    "stage5/artifacts/representatives/vector/" + name
    for name in ("cubecl.ir", "kernel.cu", "kernel.ptx", "nvrtc.json")
]
TEMPLATES = [
    "177394e178b0928b66264afbde55a1a3cc808be7a32e2890b0f8ee898e2074f4",
    "55f3919b96c6b5b78e23396ad9b7e33909e983423e950d64a544defdab7e45bd",
    "48767364e45478057a7a64181bf625c7f3f162b113c4b9cb347eb4343ddfaffb",
]
TEMPLATE_PATHS = [
    f"stage5/artifacts/reuse-first-selection/kernels/{key}/semantic.json"
    for key in TEMPLATES
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def read(path):
    return json.loads(path.read_text())


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def capture():
    assert not (OUT / "sources.json").exists(), "capture is one-shot"
    paths = SOURCES + ARTIFACTS + TEMPLATE_PATHS
    records = {}
    for name in paths:
        original = ROOT / name
        snapshot = OUT / "source" / name.removeprefix(".stage0-sources/")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(original.read_bytes())
        records[name] = {
            "sha256": sha(original),
            "snapshot": str(snapshot.relative_to(OUT)),
        }
    write("sources.json", records)
    tracked = command("git", "ls-files", "-z").split("\0")
    write(
        "preservation.json",
        {
            name: sha(ROOT / name)
            for name in tracked
            if name
            and (ROOT / name).is_file()
            and not name.startswith(".beads/")
            and name not in ("STAGE5_REPORT.md", "stage5/.gitignore")
        },
    )
    write(
        "provenance.json",
        {
            "head": command("git", "rev-parse", "HEAD"),
            "tracked_non_file_entries": [
                name for name in tracked if name and not (ROOT / name).is_file()
            ],
            "cubecl": command(
                "git", "-C", ".stage0-sources/cubecl", "rev-parse", "HEAD"
            ),
            "cubek": command("git", "-C", ".stage0-sources/cubek", "rev-parse", "HEAD"),
            "cubecl_diff_sha256": hashlib.sha256(
                command("git", "-C", ".stage0-sources/cubecl", "diff").encode()
            ).hexdigest(),
            "cubek_dirty": command(
                "git", "-C", ".stage0-sources/cubek", "status", "--porcelain"
            ),
            "mode": "source and host integer proof only; no compiler or GPU invocation",
        },
    )


def verify():
    write("validation.json", {"complete": False, "state": "verification started"})
    records = read(OUT / "sources.json")
    provenance = read(OUT / "provenance.json")
    for name in ("cubecl", "cubek"):
        assert (
            command("git", "-C", f".stage0-sources/{name}", "rev-parse", "HEAD")
            == provenance[name]
        )
    assert (
        hashlib.sha256(
            command("git", "-C", ".stage0-sources/cubecl", "diff").encode()
        ).hexdigest()
        == provenance["cubecl_diff_sha256"]
    )
    assert (
        command("git", "-C", ".stage0-sources/cubek", "status", "--porcelain")
        == provenance["cubek_dirty"]
    )
    for name, row in records.items():
        assert sha(ROOT / name) == row["sha256"], name
        assert sha(OUT / row["snapshot"]) == row["sha256"], name
    preserved = read(OUT / "preservation.json")
    for name, digest in preserved.items():
        assert sha(ROOT / name) == digest, name
    # All primary wqc evidence stays bound to the closed trial's own manifest.
    old_manifest = read(OLD / "evidence-integrity.json")
    for name, digest in old_manifest["artifacts"].items():
        assert sha(OLD / name) == digest, name
    for name in SOURCES:
        if name.startswith(CUBEK):
            copy = ROOT / name.replace(
                ".stage0-sources/cubek/", "stage5/cache/codegen/cubek/", 1
            )
            assert sha(copy) == sha(ROOT / name), name

    templates = []
    for name in TEMPLATE_PATHS:
        sem = read(ROOT / name)
        cut = next(i for i, op in enumerate(sem["ops"]) if op["op"] == "reduction")
        prefix = sem["ops"][:cut]
        dtype = sem["buffers"][0]["dtype"]
        assert prefix[0]["op"] == "load" and prefix[0]["buffer"] == 0
        assert prefix[0]["index"] == {
            "op": "add",
            "args": [
                {"op": "reduction"},
                {"op": "mul", "args": [{"op": "linear"}, {"op": "scalar", "slot": 0}]},
            ],
        }
        expected = (
            ["load", "mul"] if dtype == "float32" else ["load"] + ["cast"] * 4 + ["mul"]
        )
        assert [op["op"] for op in prefix] == expected
        if dtype != "float32":
            assert [op["dtype"] for op in prefix[1:-1]] == [
                "float32",
                dtype,
                "float32",
                "float32",
            ]
        assert prefix[-1]["dtype"] == "float32"
        assert prefix[-1]["args"] == [prefix[-2]["id"]] * 2
        assert sem["ops"][cut]["args"] == [prefix[-1]["id"]]
        templates.append(
            {
                "dtype": dtype,
                "storage_bytes_per_vector": 16 if dtype == "float32" else 8,
                "prefix": prefix,
            }
        )

    # Derive the actual ParallelReader/PlaneReader address equations for every
    # admitted B and N. This checks addresses only, not FP results or GPU code.
    vectors = inactive = row_skips = elements = 0
    for n in (32, 256, 1024, 4096):
        for rows in range(1, 66):
            for row in range((rows + 7) // 8 * 8):
                if row >= rows:
                    row_skips += 1
                    continue
                batch = ((row % rows) * n) // 4
                for k in range((n // 4 + 31) // 32):
                    for lane in range(32):
                        pos = 32 * k + lane
                        if pos >= n // 4:
                            inactive += 1
                            continue
                        start = 4 * (batch + pos)
                        assert start == row * n + 128 * k + 4 * lane
                        assert start % 4 == 0
                        assert (
                            row * n <= start and start + 3 < (row + 1) * n <= rows * n
                        )
                        for c in range(4):
                            scalar_pos = 4 * (batch + pos) + c
                            old = row * n + (scalar_pos - row * n)
                            assert old == start + c
                            assert 0 <= old < 2**63
                        vectors += 1
                        elements += 4
    alignments = []
    for size in (2, 4):
        for offset in range(8):
            address = 4096 + offset * size
            assert (address % (4 * size) == 0) == (offset % 4 == 0)
            alignments.append(
                {"storage_bytes": size, "offset": offset, "admit": offset % 4 == 0}
            )
    # Counterexamples explain why the current extent/layout restrictions stay.
    odd = []
    for n in (1, 3, 31, 33, 34, 255, 4099):
        assert 4 * (n // 4) < n  # floor-chunk reader drops a partial vector
        odd.append({"n": n, "uncovered_elements": n % 4})
    assert (33 * 4) % 16 != 0  # FP32 pitched row loses row-base alignment
    write(
        "validation.json",
        {
            "complete": True,
            "scope": "source identities, exact-template prefixes, integer address/bounds/alignment only",
            "gpu_candidate_implemented": False,
            "new_gpu_execution": False,
            "new_numerical_qualification": False,
            "sources": len(records),
            "preserved_files": len(preserved),
            "wqc_artifacts": len(old_manifest["artifacts"]),
            "templates": templates,
            "shapes": 260,
            "vector_reads_checked": vectors,
            "scalar_addresses_checked": elements,
            "inactive_vector_reads_skipped": inactive,
            "invalid_plane_rows_skipped": row_skips,
            "alignment_controls": alignments,
            "odd_width_counterexamples": odd,
            "inspector_sha256": sha(Path(__file__)),
        },
    )
    print(
        json.dumps(
            {
                "complete": True,
                "scalar_addresses_checked": elements,
                "preserved_files": len(preserved),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if args.capture:
        capture()
    verify()
