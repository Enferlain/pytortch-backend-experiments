import json,os,pathlib,subprocess,sys
root=pathlib.Path('stage5/artifacts/reduction-structure/state-counters')
env=dict(os.environ,LD_LIBRARY_PATH='/usr/lib/wsl/lib:/home/imi/Projects/pytortch-backend-experiments/stage0/python-deps/nvidia/cu13/lib')
records=[]
# One controlled experiment: paired cache-policy interventions, order alternates
# across fresh processes. Eight independently initialized pairs per policy.
for i in range(8):
 for policy in (['wb','wt'] if i%2==0 else ['wt','wb']):
  out=root/f'store-{policy}-{i:02d}'
  cmd=['stage0/.venv/bin/python','-m','stage5.reduction_state_store_control','--store-policy',policy,'--output',str(out),'--rounds','1','--round-start',str(i),'--iterations','65536','--modes','read_write','write_read','--pm-metrics','dram__bytes_read.sum,dram__bytes_write.sum']
  with (root/f'store-{policy}-{i:02d}.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  records.append({'command':cmd,'returncode':r.returncode})
  (root/'store-control-commands.json').write_text(json.dumps(records,indent=2))
  if r.returncode:print('FAILED',policy,i,flush=True);sys.exit(r.returncode)
  d=json.loads((out/'result.json').read_text());print(policy,i,{k:v['median_us'] for k,v in d['summary'].items()},flush=True)
