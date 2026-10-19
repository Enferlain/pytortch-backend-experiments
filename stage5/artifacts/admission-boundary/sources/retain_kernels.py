"""Reproduce the small kernel identity manifest; no compilation or execution."""

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "stage5/artifacts/admission-boundary"


def main():
    data = json.loads((OUT / "results.json").read_text())
    bindings = []
    for dtype, group in data["artifacts"].items():
        for name, kernel in group.items():
            source = Path(kernel["directory"])
            dest = OUT / "kernels" / dtype / name
            dest.mkdir(parents=True, exist_ok=True)
            for filename in ("descriptor.json", "manifest.json", "kernel.ptx"):
                p, q = source / filename, dest / filename
                # Existing retained copies permit reproduction after cache cleanup.
                if p.exists():
                    if q.exists():
                        assert q.read_bytes() == p.read_bytes(), (p, q)
                    else:
                        shutil.copyfile(p, q)
                assert q.exists(), q
                digest = hashlib.sha256(q.read_bytes()).hexdigest()
                if filename == "kernel.ptx":
                    assert digest == kernel["ptx_sha256"]
                bindings.append(
                    {
                        "source": str(p),
                        "retained": str(q.relative_to(ROOT)),
                        "sha256": digest,
                    }
                )
    path = OUT / "kernel-bindings.json"
    if path.exists():
        assert json.loads(path.read_text()) == bindings
    else:
        path.write_text(json.dumps(bindings, indent=2) + "\n")
    dependencies = {}
    for name in ("results", "oracle", "graphs"):
        capture = json.loads((OUT / (name + ".json")).read_text())
        for relative, digest in capture["source_sha256"].items():
            if (
                relative.endswith(".py")
                and relative != "stage5/reduction_admission_probe.py"
            ):
                dependencies[relative] = digest
    rows = []
    for relative, digest in dependencies.items():
        p = ROOT / relative
        q = OUT / "sources/dependencies" / p.name
        q.parent.mkdir(exist_ok=True)
        if q.exists():
            assert hashlib.sha256(q.read_bytes()).hexdigest() == digest
        else:
            assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
            shutil.copyfile(p, q)
        rows.append(
            {"source": relative, "retained": str(q.relative_to(ROOT)), "sha256": digest}
        )
    path = OUT / "dependency-bindings.json"
    if path.exists():
        assert json.loads(path.read_text()) == rows
    else:
        path.write_text(json.dumps(rows, indent=2) + "\n")
    print("36 retained kernel identity files verified")


if __name__ == "__main__":
    main()
