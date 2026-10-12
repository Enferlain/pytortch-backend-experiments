"""Exact original and overlay source evidence for the Stage 1.5 boundary audit."""
from pathlib import Path
import json
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stage1.evidence import guard
guard()
from stage1.source_trace import collect, TARGETS

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent/'stage1_5/artifacts'
OUT.mkdir(parents=True,exist_ok=True)
targets = {key:list(values) for key,values in TARGETS.items()}
targets['scheduler.py'] += ['Scheduler._init', 'BaseScheduling', 'SchedulerNode._compute_attrs', 'Scheduler.can_fuse']
targets['codegen/common.py'] += ['DeviceCodegen', 'register_backend_for_device', 'get_backend_features']
targets['codegen/simd.py'] += ['SIMDScheduling.can_fuse']
targets['codegen/triton.py'] = ['TritonScheduling.__init__', 'TritonScheduling.get_backend_features']
targets['codegen/xpu/xpu_combined_scheduling.py'] = ['XPUCombinedScheduling.__init__']
targets['codegen/cuda_combined_scheduling.py'] += ['CUDACombinedScheduling.__init__']
targets['codegen/halide.py'] = ['HalideScheduling.get_backend_features', 'HalideScheduling.define_kernel']
targets['codegen/pallas.py'] = ['PallasScheduling.can_fuse', 'PallasScheduling.define_kernel']
targets['shape_propagation.py'] = ['ShapePropagationOpsHandler.constant', 'ShapePropagationOpsHandler.dot']
targets['utils.py'] = ['upcast_compute_type', 'get_current_backend']
targets['dtype_propagation.py'] += ['DtypePropagationOpsHandler.constant', 'DtypePropagationOpsHandler.load', 'DtypePropagationOpsHandler.to_dtype']
original = collect(targets=targets, output=OUT/'original-source.json')
from stage1.overlay import install
install()
del targets['../_functorch/_aot_autograd/graph_compile.py']  # outside the unchanged Inductor overlay
targets['scheduler.py'] += ['BaseScheduling.check_backend_available']
for file,cls in [('codegen/triton.py','TritonScheduling'),
                 ('codegen/cuda_combined_scheduling.py','CUDACombinedScheduling'),
                 ('codegen/xpu/xpu_combined_scheduling.py','XPUCombinedScheduling')]:
    targets[file] += [cls+'.check_backend_available']
patched = collect(base=ROOT/'overlay/torch/_inductor', targets=targets, output=OUT/'patched-source.json')
for file,info in patched['files'].items():
    if file not in ('scheduler.py','codegen/triton.py','codegen/cuda_combined_scheduling.py',
                    'codegen/xpu/xpu_combined_scheduling.py'):
        assert info['sha256'] == original['files'][file]['sha256']
print(original['torch'], original['commit'], 'source snapshots captured')

for name in ['results.json', 'boundary-probes.json', 'config-triton-reads.json',
             'pytorch.patch', 'overlay.json', 'schedule-replay.json',
             'runtime-events.json', 'compiler-path.json', 'forward-profiler.json',
             'boundary-stage0-cpp.log', 'boundary-stage0-llvm.log']:
    source = ROOT/'artifacts'/name
    if not source.is_file():
        raise RuntimeError(f'missing regression evidence {source}; run the README verification commands first')
    shutil.copyfile(source, OUT/name)
for kernel in ['21d834bc279b6d28', '4c0f8eabbd050d4f']:
    shutil.copytree(ROOT/'artifacts/kernels'/kernel, OUT/'kernels'/kernel, dirs_exist_ok=True)
regressions = {route:json.loads((ROOT.parent/'stage0/artifacts'/route/'results.json').read_text())
               for route in ['cpp','llvm']}
(OUT/'regression.json').write_text(json.dumps(regressions,indent=2))
