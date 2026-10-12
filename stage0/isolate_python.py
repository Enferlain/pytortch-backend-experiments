"""Expose an existing PyTorch installation in a separate venv without Triton.

Usage: uv venv stage0/.venv --python 3.14t
       python stage0/isolate_python.py /path/to/existing/site-packages
No packages in the source environment are changed. The symlinks are local setup.
"""
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
src = Path(sys.argv[1]).resolve()
dst = root / "python-deps"
dst.mkdir(exist_ok=True)
for p in src.iterdir():
    if p.name.startswith("triton") or p.name.endswith(".pth") or p.name == "__pycache__":
        continue
    target = dst / p.name
    if target.is_symlink():
        if target.resolve() != p:
            raise RuntimeError(f"existing link points at a different environment: {target}")
    elif target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")
    else:
        target.symlink_to(p, target_is_directory=p.is_dir())
sites = list((root / ".venv/lib").glob("python*/site-packages"))
if len(sites) != 1:
    raise RuntimeError("create the isolated venv first")
(sites[0] / "stage0-deps.pth").write_text(str(dst) + "\n")
