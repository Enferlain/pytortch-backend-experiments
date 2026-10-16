import json,os,pathlib,subprocess
root=pathlib.Path('stage5/artifacts/reduction-structure/state-counters')
env=dict(os.environ,LD_LIBRARY_PATH='/usr/lib/wsl/lib:/home/imi/Projects/pytortch-backend-experiments/stage0/python-deps/nvidia/cu13/lib')
families=[('dram',16,'dram__bytes_read.sum,dram__bytes_write.sum',None),('clock',8,'sm__cycles_elapsed.avg.per_second,dram__cycles_elapsed.avg.per_second',None),('l2-volume',8,'lts__t_sectors.sum',None),('instructions',8,'smsp__inst_executed.sum',None),('store-wb',8,'dram__bytes_read.sum,dram__bytes_write.sum','wb'),('store-wt',8,'dram__bytes_read.sum,dram__bytes_write.sum','wt')]
records=[]
for family,rounds,metrics,policy in families:
 for i in range(rounds):
  out=root/f'qualified-{family}-{i:02d}'
  module='stage5.reduction_state_store_control' if policy else 'stage5.reduction_state_counters'
  cmd=['stage0/.venv/bin/python','-m',module,'--output',str(out),'--rounds','1','--round-start',str(i),'--iterations','65536','--modes','read_write','write_read','--pm-metrics',metrics]
  if policy:cmd+=['--store-policy',policy]
  with (root/f'qualified-{family}-{i:02d}.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  records.append({'command':cmd,'returncode':r.returncode})
  (root/'qualified-commands.json').write_text(json.dumps(records,indent=2))
  if r.returncode:print('FAILED',family,i,flush=True);break
  d=json.loads((out/'result.json').read_text());print(family,i,{k:v['median_us'] for k,v in d['summary'].items()},flush=True)
