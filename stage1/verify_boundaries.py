"""Fresh-process tests of availability ownership and scheduler fusion vetoes."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stage1 import evidence
evidence.guard()
import torch
from stage1.overlay import install
install()
from torch._inductor import config
from torch._inductor.codegen.common import init_backend_registration
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.codegen.xpu.xpu_combined_scheduling import XPUCombinedScheduling
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.exc import TritonMissing
from stage1.backend import compile_backend, register, TRACE
from stage1.plan import PhysicalKernelPlan, bridge_description
from stage1.runtime import check_available, CubeCLBackendUnavailable

results = {}
check_available(torch.device('cuda',0))
with patch.dict(os.environ, {'STAGE0_CARGO_PROFILE':'_stage1_5_missing_profile_'}):
    try:
        check_available(torch.device('cuda',0))
    except CubeCLBackendUnavailable as error:
        assert 'cubecl-stage0' in str(error) and '--features llvm' not in str(error)
        results['missing_bridge'] = {'type':type(error).__name__, 'message':str(error)}
    else:
        raise AssertionError('missing compiler bridge must fail availability')
init_backend_registration()
register()
def f(x): return torch.sin(x)+1
try:
    torch.compile(f, backend='inductor', fullgraph=True)(torch.randn(257,device='cuda'))
except Exception as error:
    assert isinstance(error, TritonMissing) or 'TritonMissing' in str(error), str(error)
    results['default_cuda_preserved'] = str(error)
else:
    raise AssertionError('default CUDA must still require Triton after CubeCL registration')

for name, backend, device in [
    ('triton_cpu', TritonScheduling(None), torch.device('cpu')),
    ('generic_cpu', BaseScheduling(None), torch.device('cpu')),
    ('xpu_combined', XPUCombinedScheduling(None), torch.device('xpu',0)),
]:
    try:
        backend.check_backend_available(device)
    except Exception as error:
        assert name == 'xpu_combined' and type(error).__name__ == 'TritonMissing'
        results[name] = type(error).__name__
    else:
        assert name != 'xpu_combined'
        results[name] = 'unchanged CPU no-GPU gate'

# Unsupported physical schedules are rejected before descriptor/IR compilation.
bad_plan = PhysicalKernelPlan().to_dict()
bad_plan['elements_per_unit'] = 2
try:
    bridge_description({'schema':1, 'dtype':'float32', 'ops':[]}, bad_plan)
except NotImplementedError as error:
    results['unsupported_physical_schedule'] = str(error)
else:
    raise AssertionError('unsupported plan accepted')

# Exercise a genuine multi-node scheduler without expanding execution coverage.
# Two observable outputs require a fusion veto under the existing one-output ABI.
class FusionObserved(RuntimeError):
    pass
def stop_after_fusion(nodes):
    assert any(e.get('boundary') == 'fusion' and e['decision'] == 'reject' for e in TRACE)
    raise FusionObserved('CubeCL fusion veto observed before any codegen')
def branching(x):
    a = torch.sin(x)
    return a + 1, a * 2
try:
    with config.patch({'_post_fusion_custom_pass':stop_after_fusion}):
        torch.compile(branching, backend=compile_backend, fullgraph=True, dynamic=False)(torch.randn(257,device='cuda'))
except Exception as error:
    assert 'FusionObserved' in str(error), str(error)
    results['fusion_veto'] = [e for e in TRACE if e.get('boundary') == 'fusion']
    assert not any(e['event']=='kernel' for e in TRACE)
else:
    raise AssertionError('fixture must stop before codegen')
evidence.state('after_boundary_probes')
results['independence'] = evidence.inventory()
assert not results['independence']['loaded_modules']
assert not results['independence']['native_mappings']
(Path(__file__).parent/'artifacts/boundary-probes.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
