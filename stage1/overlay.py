"""Reproducible isolated live-wheel Inductor copy, never edit shared torch files."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch

ROOT = Path(__file__).resolve().parent
EXPECTED = '5c4886908584029761b579af026dcfb627c84070'

def install():
    if torch.version.git_version != EXPECTED:
        raise RuntimeError('Stage 1 source hooks require the recorded PyTorch wheel commit')
    if any(n == 'torch._inductor' or n.startswith('torch._inductor.') for n in sys.modules):
        raise RuntimeError('install Stage 1 overlay before importing Inductor')
    src = Path(torch.__file__).resolve().parent / '_inductor'
    dest = ROOT/'overlay/torch/_inductor'
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    edits = {'scheduler.py': [('        if not has_triton():\n',
        '        if get_current_backend(device.type) == "triton" and not has_triton():\n')]}
    patches = []
    hashes = {}
    for file, replacements in edits.items():
        path = dest/file
        before = path.read_text()
        after = before
        for old, new in replacements:
            if after.count(old) != 1:
                raise RuntimeError(f'source drift: {file}')
            after = after.replace(old, new)
        path.write_text(after)
        patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile='a/torch/_inductor/'+file, tofile='b/torch/_inductor/'+file))
        hashes[file] = {'original': hashlib.sha256(before.encode()).hexdigest(), 'patched': hashlib.sha256(after.encode()).hexdigest()}
    (ROOT/'artifacts/pytorch.patch').write_text(''.join(patches))
    (ROOT/'artifacts/overlay.json').write_text(json.dumps(hashes, indent=2))
    torch.__path__.insert(0, str(dest.parent))
