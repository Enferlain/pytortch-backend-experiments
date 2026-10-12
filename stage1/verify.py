"""Real Dynamo/AOTAutograd/Inductor/CubeCL acceptance; fresh process, no Triton."""
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(ROOT/'cache')
os.environ['TORCHINDUCTOR_AUTOGRAD_CACHE'] = '0'
from stage1 import evidence
evidence.guard()
import torch
evidence.state('after_torch')
from stage1.overlay import install
install()
from stage1.backend import compile_backend, TRACE
from stage1 import runtime
from torch.utils._config_module import SubConfigProxy
# Observe real config reads, not config serialization/hash inventories.
config_reads = {}
original_config_getattr = SubConfigProxy.__getattr__
def audit_config_read(self, name):
    value = original_config_getattr(self, name)
    frame = sys._getframe(1)
    if self._prefix == 'triton.' and '/_inductor/' in frame.f_code.co_filename:
        key = (name, frame.f_code.co_filename, frame.f_code.co_qualname, frame.f_lineno)
        config_reads[key] = str(value)
    return value
SubConfigProxy.__getattr__ = audit_config_read
# Static acceptance sizes plus deliberate invalid variants share one code object.
torch._dynamo.config.recompile_limit = 32
evidence.state('after_inductor_import')

torch.manual_seed(17)
def f(x): return torch.sin(x)+1
compiled = torch.compile(f, backend=compile_backend, fullgraph=True, dynamic=False)
RESULT = {'torch':torch.__version__,'torch_commit':torch.version.git_version,
          'device':torch.cuda.get_device_name(0),'checks':[], 'sizes':[],
          'float32_tolerance':{'rtol':2e-6,'atol':2e-6}}

def check(actual, expected):
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-6,equal_nan=True)

try:
    threading.setprofile_all_threads(evidence.profile)
    for n in [0,1,255,256,257,1_048_583]:
        x = torch.randn(n,device='cuda',requires_grad=True)
        before = runtime.KERNEL_LAUNCHES
        y = compiled(x)
        forward_launches = runtime.KERNEL_LAUNCHES-before
        if n:
            event = runtime.EVENTS[-1]
            assert event['pointers'] == [x.data_ptr(), y.data_ptr()]
            assert event['stream'] == torch.cuda.current_stream().cuda_stream
        assert forward_launches == int(n!=0), (n,forward_launches)
        check(y,f(x))
        before = runtime.KERNEL_LAUNCHES
        unit_tangent = torch.ones_like(y)
        g, = torch.autograd.grad(y,x,unit_tangent,retain_graph=True)
        if n:
            assert runtime.EVENTS[-1]['pointers'] == [unit_tangent.data_ptr(), x.data_ptr(), g.data_ptr()]
        backward_launches = runtime.KERNEL_LAUNCHES-before
        assert backward_launches == int(n!=0), (n,backward_launches)
        check(g,torch.cos(x))
        # Arbitrary upstream gradient verifies multiplication, not only cos(x).
        tangent = torch.randn_like(y)
        g, = torch.autograd.grad(y,x,tangent)
        if n:
            assert runtime.EVENTS[-1]['pointers'] == [tangent.data_ptr(), x.data_ptr(), g.data_ptr()]
        check(g,tangent*torch.cos(x))
        RESULT['sizes'].append({'n':n,'forward_launches':forward_launches,'backward_launches':backward_launches})
    evidence.state('after_shapes_and_aot_backward')
    x = torch.tensor([float('nan'),float('inf'),-float('inf'),-0.0,0.0,
                      1e-30,-1e-30,-3.14159265,3.14159265,-1000,1000,1e10,-1e10],device='cuda',requires_grad=True)
    y = compiled(x)
    check(y,f(x))
    g, = torch.autograd.grad(y,x,torch.ones_like(y))
    check(g,torch.cos(x))
    RESULT['checks'].append('NaN, infinities, signed zeros, ordinary/extreme forward/backward')
    threading.setprofile_all_threads(None)

    # The required forward also executes without requires_grad / saved-tensor state.
    for n in [0,1,255,256,257,1_048_583]:
        x = torch.randn(n,device='cuda')
        before = runtime.KERNEL_LAUNCHES
        y = compiled(x)
        assert runtime.KERNEL_LAUNCHES-before == int(n!=0)
        check(y,f(x))
    RESULT['checks'].append('plain inference forward at all required sizes')

    # Supplementary 2-input forward exercises the same existing ABI/add surface.
    def two_input(x,other): return torch.sin(x)+other
    other_compiled = torch.compile(two_input,backend=compile_backend,fullgraph=True,dynamic=False)
    x,other = torch.randn(257,device='cuda'),torch.randn(257,device='cuda')
    before = runtime.KERNEL_LAUNCHES
    y = other_compiled(x,other)
    assert runtime.KERNEL_LAUNCHES-before == 1
    assert runtime.EVENTS[-1]['pointers'] == [x.data_ptr(),other.data_ptr(),y.data_ptr()]
    check(y,two_input(x,other))
    RESULT['checks'].append('supplementary 2-input forward and pointer ABI')

    # Reuse supported static sizes without new native compiler invocations.
    count = runtime.COMPILER_INVOCATIONS
    for n in [257,255,1_048_583,1,256,0,257]:
        x = torch.randn(n,device='cuda',requires_grad=True)
        y = compiled(x)
        check(y,f(x))
        check(torch.autograd.grad(y,x,torch.ones_like(y))[0],torch.cos(x))
    assert runtime.COMPILER_INVOCATIONS == count
    RESULT['checks'].append({'repeat_sizes_no_native_recompile':runtime.COMPILER_INVOCATIONS})

    # Same delayed producer -> compiled kernel -> eager observation as Stage 0.
    streams = [torch.cuda.Stream(),torch.cuda.Stream()]
    ids = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(5_000_000)
            x = torch.randn(257,device='cuda',requires_grad=True)
            y = compiled(x)
            consumer = y*3-2
            g, = torch.autograd.grad(y,x,torch.ones_like(y))
            expected = (torch.sin(x)+1)*3-2
            expected_g = torch.cos(x)
            ids.append(stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(stream)
        check(consumer,expected)
        check(g,expected_g)
    assert len(set(ids)) == 2 and all(ids)
    RESULT['checks'].append({'non_default_stream_forward_backward':ids})

    # Reuse Stage 0 lifetime-pressure contract through the Inductor wrapper.
    n = 1_048_583
    pressure_evidence = []
    stream = streams[0]
    for trial in range(3):
        x = torch.randn(n,device='cuda',requires_grad=True)
        expected = torch.sin(x)+1
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(30_000_000)
            y = compiled(x)
            old_ptrs = {x.data_ptr(),y.data_ptr()}
            retained = y.detach().clone()
            complete = torch.cuda.Event()
            complete.record(stream)
        del x,y
        pressure = [torch.full((n,),23.0+i,device='cuda') for i in range(16)]
        reused = sum(p.data_ptr() in old_ptrs for p in pressure)
        early = reused if not complete.query() else 0
        torch.cuda.current_stream().wait_stream(stream)
        check(retained,expected)
        assert early == 0
        pressure_evidence.append({'recycled_addresses':reused,'reuses_observed_before_completion':early})
    RESULT['checks'].append({'allocator_pressure':pressure_evidence})

    # Profile a warmed accepted forward separately from eager reference and backward.
    x = torch.randn(257,device='cuda',requires_grad=True)
    compiled(x)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        y = compiled(x)
    prof.export_chrome_trace(str(ROOT/'artifacts/forward-profiler.json'))
    trace = json.loads((ROOT/'artifacts/forward-profiler.json').read_text())
    gpu = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel']
    names = [e['name'] for e in gpu]
    assert len(gpu)==1 and 'cubecl_' in names[0], names
    RESULT['profiler_forward_kernel_names'] = names
    RESULT['checks'].append('profiler: exactly one CubeCL forward GPU kernel')
    check(y,f(x))

    # Deliberate rejection instead of Dynamo suppression or Triton fallback.
    rejected = []
    def bad_op(x): return torch.tan(x)
    def reduction(x): return x.sum()
    def nonfinite_constant(x): return x+float('inf')
    def indirect(x,indices): return x[indices]
    for label,fn,x in [('operation',bad_op,torch.randn(257,device='cuda')),
                      ('reduction',reduction,torch.randn(257,device='cuda')),
                      ('dtype',f,torch.randn(257,device='cuda',dtype=torch.float64)),
                      ('stride',f,torch.randn(514,device='cuda')[::2]),
                      ('device',f,torch.randn(257)),
                      ('nonfinite_constant',nonfinite_constant,torch.randn(257,device='cuda'))]:
        trace_start = len(TRACE)
        compiles_before = runtime.COMPILER_INVOCATIONS
        try:
            torch.compile(fn,backend=compile_backend,fullgraph=True,dynamic=False)(x)
        except Exception as exc:
            text = str(exc)
            assert 'Stage 1' in text or 'CubeCL' in text, text
            checks = TRACE[trace_start:]
            expected_boundary = 'backend_selection' if label in ('dtype','stride','device') else 'pre_fusion'
            assert any(e.get('boundary') == expected_boundary and e.get('decision') == 'reject' for e in checks)
            assert not any(e['event'] in ('schedule','kernel') for e in checks)
            assert runtime.COMPILER_INVOCATIONS == compiles_before
            rejected.append({'case':label,'error':text,'boundary':expected_boundary,'trace':checks})
        else: raise AssertionError('unsupported case succeeded: '+label)
    try:
        torch.compile(indirect,backend=compile_backend,fullgraph=True,dynamic=False)(
            torch.randn(257,device='cuda'),torch.arange(257,device='cuda'))
    except Exception as exc:
        assert 'Stage 1' in str(exc) or 'CubeCL' in str(exc), str(exc)
        rejected.append({'case':'indirect','error':str(exc)})
    else:
        raise AssertionError('indirect case succeeded')
    RESULT['unsupported_cases'] = rejected
    evidence.state('after_execution_and_rejections')
    import ctypes
    native = next(iter(runtime.CACHE.values())).kernel.native
    native.stage0_launcher_calls.restype = ctypes.c_uint64
    native.stage0_kernel_launches.restype = ctypes.c_uint64
    RESULT['actual_native_launcher_calls'] = native.stage0_launcher_calls()
    RESULT['actual_native_kernel_launches'] = native.stage0_kernel_launches()
    assert RESULT['actual_native_launcher_calls'] == runtime.LAUNCHER_INVOCATIONS
    assert RESULT['actual_native_kernel_launches'] == runtime.KERNEL_LAUNCHES
    backward_stacks = [event['actual_replay_stack'] for event in TRACE if event['event']=='kernel'
        and any(op['op']=='cos' for op in event['descriptor']['ops'])]
    assert backward_stacks and any(frame['function']=='bw_compiler' for stack in backward_stacks for frame in stack)
    RESULT['checks'].append('backward replay stack reaches AOTAutograd bw_compiler and SIMD replay')
    RESULT['compiler_invocations'] = runtime.COMPILER_INVOCATIONS
    RESULT['native_launcher_invocations'] = runtime.LAUNCHER_INVOCATIONS
    RESULT['nonempty_kernel_launches'] = runtime.KERNEL_LAUNCHES
    RESULT['independence'] = evidence.inventory()
    assert not RESULT['independence']['loaded_modules']
    assert not RESULT['independence']['native_mappings']
    # Optional PyTorch imports attempt the absent package and already catch absence.
    # Package/native module loading, rather than rejected attempts, must stay zero.
    assert all('/torch/' in attempt['caller'] for attempt in evidence.ATTEMPTS)
    # Preserve inspectable representative emitted Inductor wrappers outside its cache.
    wrappers = ROOT/'artifacts/wrappers'
    wrappers.mkdir(exist_ok=True)
    for path in (ROOT/'cache').rglob('*.py'):
        text = path.read_text()
        if 'cubecl_compile_kernel' in text:
            target = wrappers/path.name
            if target.exists() and target.read_text() != text:
                raise RuntimeError(f'wrapper artifact filename collision: {path}')
            target.write_text(text)
    RESULT['result'] = 'Stage 1 works'
finally:
    SubConfigProxy.__getattr__ = original_config_getattr
    threading.setprofile_all_threads(None)
    (ROOT/'artifacts/results.json').write_text(json.dumps(RESULT,indent=2))
    (ROOT/'artifacts/schedule-replay.json').write_text(json.dumps(TRACE,indent=2))
    (ROOT/'artifacts/runtime-events.json').write_text(json.dumps(runtime.EVENTS,indent=2))
    (ROOT/'artifacts/compiler-path.json').write_text(json.dumps(evidence.CALLS,indent=2))
    (ROOT/'artifacts/config-triton-reads.json').write_text(json.dumps([
        {'setting': k[0], 'file': k[1], 'function': k[2], 'line': k[3], 'value': value}
        for k,value in sorted(config_reads.items())], indent=2))
print(json.dumps(RESULT,indent=2))
