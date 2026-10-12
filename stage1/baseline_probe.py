"""Demonstrate the live unpatched Inductor availability gate."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stage1 import evidence

evidence.guard()
import torch

def f(x): return torch.sin(x)+1
try:
    torch.compile(f, backend='inductor', fullgraph=True)(torch.randn(257,device='cuda'))
except Exception as error:
    output = {'error_type':type(error).__name__, 'error':str(error), 'independence':evidence.inventory()}
    (Path(__file__).parent/'artifacts/unpatched-gate.json').write_text(json.dumps(output,indent=2))
    if 'TritonMissing' not in str(error): raise
else:
    raise AssertionError('expected the unpatched CUDA Triton availability gate')
