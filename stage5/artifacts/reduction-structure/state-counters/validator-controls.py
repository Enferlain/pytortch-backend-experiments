import copy,ctypes as C,hashlib,json,tempfile
from pathlib import Path
from stage5.reduction_state_counter_analyze import analyze,capture,verify_primer_source
root=Path('stage5/artifacts/reduction-structure/state-counters').resolve();checks=[]
def cloned(source,dest):
 dest.mkdir()
 for child in source.iterdir():
  if child.name!='result.json':(dest/child.name).symlink_to(child.resolve(),target_is_directory=child.is_dir())
 (dest/'result.json').write_bytes((source/'result.json').read_bytes())
 return dest

def rejects(name,fn):
 try:fn()
 except (AssertionError,ValueError) as exc:checks.append({'name':name,'rejected':True,'reason':str(exc)[:500]})
 else:raise AssertionError(f'Failed to reject {name}')
base=root/'qualified-dram-00';second=root/'qualified-dram-01'
rejects('mixed-volatile-and-intrinsic-primer',lambda:analyze([base,root/'qualified-store-wb-00']))
rejects('mixed-wb-and-wt-primer',lambda:analyze([root/'qualified-store-wb-00',root/'qualified-store-wt-00']))
analyze([base,second]);checks.append({'name':'unmodified-positive-pair','passed':True})
with tempfile.TemporaryDirectory(prefix='stage5-counter-validator-') as temp:
 temp=Path(temp)
 for field,value in [('pm_metrics',['dram__bytes_read.sum']),('pm_interval_ns',2000),('sampler_sha256','wrong'),('clock_method','wrong'),('reset','gate'),('sampler_close_status',1)]:
  dest=cloned(second,temp/field);d=json.loads((dest/'result.json').read_text());d[field]=value;(dest/'result.json').write_text(json.dumps(d))
  rejects('mismatched-'+field,lambda:analyze([base,dest]))
 dest=cloned(second,temp/'harness');p=dest/'harness.py';original=p.read_bytes();p.unlink();p.write_bytes(original+b'\n# validator fixture\n');d=json.loads((dest/'result.json').read_text());d['harness_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();(dest/'result.json').write_text(json.dumps(d));rejects('different-self-consistent-harness-revision',lambda:analyze([base,dest]))
 dest=cloned(root/'qualified-store-wb-00',temp/'policy');p=dest/'store-control.json';d=json.loads(p.read_text());p.unlink();d['policy']='wt';d['only_source_substitution'][1]=d['only_source_substitution'][1].replace('__stwb','__stwt');p.write_text(json.dumps(d));rejects('wrong-store-policy-with-consistent-json',lambda:verify_primer_source(dest))
 d=json.loads((base/'result.json').read_text());row=copy.deepcopy(d['captures'][0]);row['mode']='same8';rejects('unsupported-selector-mode',lambda:capture(base,d,row));row=d['captures'][0];d=copy.deepcopy(d);d['reset']='gate';rejects('unsupported-selector-reset',lambda:capture(base,d,row))
lib=C.CDLL('/tmp/stage5-state-sampling.so');lib.state_clock_convert.argtypes=[C.c_uint64];lib.state_clock_convert.restype=C.c_uint64;lib.state_clock_anchor.argtypes=[C.POINTER(C.c_uint64)];lib.state_pm_init.argtypes=[C.c_char_p,C.c_uint64]
assert lib.state_clock_finish()==1
assert lib.state_clock_convert(1)==0
assert lib.state_clock_anchor(None)==1
assert lib.state_pm_init(None,1000)==1
assert lib.state_clock_begin()==0
assert lib.state_clock_begin()==1
assert lib.state_clock_finish()==0
assert lib.state_clock_begin()==0
assert lib.state_clock_finish()==0
checks.append({'name':'reject-active-clock-reentry-and-allow-next-capture','passed':True})
checks.append({'name':'four-invalid-ffi-states','passed':True})
(root/'validator-controls.json').write_text(json.dumps(checks,indent=2));print('Passed',len(checks),'validation controls, including four invalid FFI states.')
