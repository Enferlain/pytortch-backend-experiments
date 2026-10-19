"""Retain current source and small executed-kernel identities, without GPU work."""

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "stage5/artifacts/reference512-admission"


def main():
    bindings = []

    def retain(source, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert source.read_bytes() == dest.read_bytes()
        else:
            shutil.copyfile(source, dest)
        bindings.append(
            {
                "source": str(source.relative_to(ROOT)),
                "retained": str(dest.relative_to(ROOT)),
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            }
        )

    for filename in (
        "backend.py",
        "runtime.py",
        "legality.py",
        "reference512_plan.py",
        "reference512_oracle.py",
        "reference512_templates.json",
        "reference512_checks.py",
        "reference512_gates.py",
        "reduction_topology_model.py",
        "reduction_rounding_probe.py",
        "reduction_policy.py",
    ):
        retain(ROOT / "stage5" / filename, OUT / "sources/current" / filename)
    executed = {}
    for row in json.loads((OUT / "graphs.json").read_text())["rows"]:
        for kernel in row["kernels"]:
            directory = Path(kernel["directory"])
            if str(directory) in executed:
                continue
            dest = OUT / "executed" / kernel["ptx_sha256"]
            for name in ("descriptor.json", "manifest.json", "kernel.ptx"):
                retain(directory / name, dest / name)
            assert (
                hashlib.sha256((dest / "kernel.ptx").read_bytes()).hexdigest()
                == kernel["ptx_sha256"]
            )
            executed[str(directory)] = str(dest.relative_to(ROOT))
    result = {"bindings": bindings, "executed": executed}
    path = OUT / "retention.json"
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        path.write_text(json.dumps(result, indent=2) + "\n")
    print("current source and", len(executed), "executed kernel identities retained")


if __name__ == "__main__":
    main()
