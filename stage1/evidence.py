"""Fresh-process import guard and compiler-path instrumentation."""
import builtins
import importlib.abc
import importlib.util
import os
import sys
import threading

class TritonProbe(importlib.abc.MetaPathFinder):
    def __init__(self): self.probes = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'triton' or fullname.startswith('triton.'):
            self.probes.append(fullname)
        return None

PROBE = TritonProbe()
ATTEMPTS = []
GIL = {}
CALLS = []
_original_import = builtins.__import__

def guard():
    if 'PYTHON_GIL' in os.environ:
        raise RuntimeError('run without PYTHON_GIL forcing')
    sys.meta_path.insert(0, PROBE)
    if importlib.util.find_spec('triton') is not None:
        raise RuntimeError('run in the isolated Stage 0 environment without Triton')
    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and (name == 'triton' or name.startswith('triton.')):
            ATTEMPTS.append({'module':name,'caller':sys._getframe(1).f_code.co_filename,'line':sys._getframe(1).f_lineno})
            raise ImportError('Triton deliberately unavailable in Stage 1')
        return _original_import(name, globals, locals, fromlist, level)
    builtins.__import__ = guarded
    state('before_torch')

def state(phase):
    GIL[phase] = sys._is_gil_enabled()
    if GIL[phase]: raise RuntimeError('GIL enabled at '+phase)

SELECTED = {
    'GraphLowering.run','GraphLowering.codegen','GraphLowering._update_scheduler',
    'Scheduler.__init__','Scheduler.create_backend','Scheduler.codegen','SchedulerNode.codegen',
    'SIMDScheduling.codegen_node','SIMDScheduling._codegen_nodes',
    'SIMDScheduling.codegen_node_schedule','SIMDScheduling.codegen_node_schedule_with_kernel',
    'SIMDKernel.split_and_set_ranges','LoopBody.__call__','LoopBodyBlock.__call__',
    'InterpreterShim.run','CSEProxy._default','CSEProxy.load','CSEProxy.store',
    'CubeCLKernel.load','CubeCLKernel.store','CubeCLKernel.emit','CubeCLKernel.codegen_kernel',
    'CubeCLScheduling.define_kernel','CubeCLKernel.call_kernel',
    'CompiledKernel.__init__','CompiledKernel.__call__','Kernel.__call__',
    'compile_backend','compile_fx','compile_fx_inner','aot_module_simplified',
    'aot_dispatch_autograd','aot_dispatch_base','aot_dispatch_autograd_graph',
    'OutputGraph._call_user_compiler', 'compile_fx_forward', 'compile_fx_backward',
    'aot_stage2_autograd', '_aot_stage2b_compile_forward_or_inference',
    '_aot_stage2b_compile_backward',
}

def profile(frame, event, arg):
    if event == 'call' and (frame.f_code.co_qualname in SELECTED or frame.f_code.co_name in ('bw_compiler','fw_compiler_base')):
        CALLS.append({'symbol':frame.f_code.co_qualname, 'file':frame.f_code.co_filename,
                      'line':frame.f_code.co_firstlineno, 'thread':threading.get_ident()})

def inventory():
    loaded = [m for m in sys.modules if m == 'triton' or m.startswith('triton.')]
    native = [line for line in open('/proc/self/maps') if 'triton' in line.lower()]
    return {'availability_probes':PROBE.probes, 'import_attempts':ATTEMPTS,
            'loaded_modules':loaded,'native_mappings':native, 'gil_states':GIL,
            'python':sys.version, 'PYTHON_GIL':os.environ.get('PYTHON_GIL')}
