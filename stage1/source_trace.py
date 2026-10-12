"""Exact live-wheel source evidence; run before backend implementation/use."""
import ast
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
TARGETS = {
    'compile_fx.py': ['compile_fx', 'compile_fx_inner', 'compile_fx_forward', 'compile_fx_backward',
        '_compile_fx_main.fw_compiler_base', '_compile_fx_main.bw_compiler'],
    '../_functorch/_aot_autograd/graph_compile.py': ['aot_stage2_autograd',
        '_aot_stage2b_compile_forward_or_inference'],
    'graph.py': ['GraphLowering.run', 'GraphLowering.codegen', 'GraphLowering._update_scheduler'],
    'scheduler.py': ['Scheduler.__init__', 'Scheduler.create_backend', 'Scheduler.codegen', 'SchedulerNode.codegen'],
    'codegen/simd.py': ['SIMDScheduling.codegen_node', 'SIMDScheduling._codegen_nodes',
        'SIMDScheduling.codegen_node_schedule', 'SIMDScheduling.codegen_node_schedule_with_kernel',
        'SIMDKernel.__init__', 'SIMDKernel.split_and_set_ranges'],
    'loop_body.py': ['LoopBody.__call__', 'LoopBodyBlock.__call__'],
    'codegen/common.py': ['CSEProxy._default', 'CSE.generate', 'Kernel.__enter__', 'init_backend_registration'],
    'codegen/cuda_combined_scheduling.py': ['CUDACombinedScheduling.codegen_node'],
    'dtype_propagation.py': ['promote_types'],
    'codegen/wrapper.py': ['PythonWrapperCodegen.generate_kernel_call'],
}

def collect():
    base = Path(torch.__file__).resolve().parent / '_inductor'
    result = {'torch': torch.__version__, 'commit': torch.version.git_version, 'files': {}}
    for path, targets in TARGETS.items():
        content = (base / path).read_text()
        tree = ast.parse(content)
        found = {}
        def walk(node, prefix=''):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = prefix + child.name
                    if name in targets:
                        found[name] = {'start': child.lineno, 'end': child.end_lineno,
                            'source': '\n'.join(content.splitlines()[child.lineno-1:child.end_lineno])}
                    walk(child, name+'.')
                else:
                    walk(child, prefix)
        walk(tree)
        missing = sorted(set(targets) - set(found))
        if missing:
            raise RuntimeError(f'missing source-trace targets in {path}: {missing}')
        result['files'][path] = {'path': str(base/path), 'sha256': hashlib.sha256(content.encode()).hexdigest(), 'symbols': found}
    (ROOT/'artifacts/source-trace.json').write_text(json.dumps(result, indent=2))
    return result

if __name__ == '__main__':
    r = collect()
    print(r['torch'], r['commit'])
    for f, evidence in r['files'].items():
        print(f, list(evidence['symbols']))
