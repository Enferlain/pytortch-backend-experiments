"""Reproduce the isolated CubeK compatibility copy; upstream sources stay clean."""

import difflib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/reduction-structure/codegen"


def extract_fresh(archive, destination):
    """The generated compatibility tree contains only this pinned archive."""
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        files.extractall(destination, filter="data")


def main():
    original = ROOT.parent / ".stage0-sources/cubek"
    revision = subprocess.check_output(
        ["git", "-C", str(original), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == "4ca0235e7e4fddd82f67ec27bbc3d48c085be25a"
    destination = ROOT / "cache/codegen/cubek"
    archive = subprocess.check_output(["git", "-C", str(original), "archive", revision])
    extract_fresh(archive, destination)
    changed = []
    p = destination / "Cargo.toml"
    before = p.read_text()
    after = before
    for name in ("cubecl", "cubecl-common", "cubecl-environment", "cubecl-ir"):
        old = f'{name} = {{ git = "https://github.com/tracel-ai/cubecl", rev = "80bb36bf2eb1791f47e75dc4893da3046580ade6", default-features = false }}'
        new = f'{name} = {{ path = "{ROOT.parent / ".stage0-sources/cubecl/crates" / name}", default-features = false }}'
        assert old in after
        after = after.replace(old, new)
    p.write_text(after)
    changed.append((p, before, after))
    migrations = (
        (
            "cubek-tile/src/ops/softmax/planewise.rs",
            "EA::new(LOGIT_MASKED)",
            "EA::new(LOGIT_MASKED as f32)",
        ),
        (
            "cubek-tile/src/ops/softmax/rowwise.rs",
            "EA::new(LOGIT_MASKED)",
            "EA::new(LOGIT_MASKED as f32)",
        ),
        (
            "cubek-tile/src/ops/softmax/state.rs",
            "E::new(FULLY_MASKED_ROW_THRESHOLD)",
            "E::new(FULLY_MASKED_ROW_THRESHOLD as f32)",
        ),
        (
            "cubek-tile/src/tile/kind/procedural/phase.rs",
            "1.0 / divisor.unwrap() as f64",
            "1.0 / divisor.unwrap() as f32",
        ),
        (
            "cubek-tile/src/tile/packing/base.rs",
            "=> E2M1_F16_LIFT,",
            "=> E2M1_F16_LIFT as f64,",
        ),
        (
            "cubek-reduce/src/components/readers/mod.rs",
            "pub(crate) mod bound_checks;",
            "pub mod bound_checks;",
        ),
    )
    for name, old, new in migrations:
        p = destination / "crates" / name
        before = p.read_text()
        assert old in before
        after = before.replace(old, new)
        p.write_text(after)
        changed.append((p, before, after))
    OUT.mkdir(parents=True, exist_ok=True)
    patch = "".join(
        "".join(
            difflib.unified_diff(
                a.splitlines(True),
                b.splitlines(True),
                fromfile="a/" + str(p.relative_to(destination)),
                tofile="b/" + str(p.relative_to(destination)),
            )
        )
        for p, a, b in changed
    )
    (OUT / "cubek-compatibility.patch").write_text(patch)
    (OUT / "cubek-compatibility.json").write_text(
        json.dumps(
            {
                "cubek_revision": revision,
                "cubecl_revision": subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(ROOT.parent / ".stage0-sources/cubecl"),
                        "rev-parse",
                        "HEAD",
                    ],
                    text=True,
                ).strip(),
                "changes": [str(p.relative_to(destination)) for p, _, _ in changed],
                "scope": "five API type adaptations in unrelated tile routines, reader module visibility, local dependency paths; exercised reader and reduction bodies unchanged",
            },
            indent=2,
        )
    )
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(ROOT / "codegen_probe/Cargo.toml"),
        ],
        env={**os.environ, "CARGO_TARGET_DIR": str(ROOT.parent / "stage0/target")},
        check=True,
    )


if __name__ == "__main__":
    main()
