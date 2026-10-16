import json,os,pathlib,subprocess,sys
root=pathlib.Path('stage5/artifacts/reduction-structure/state-counters')
env=dict(os.environ,LD_LIBRARY_PATH='/usr/lib/wsl/lib:/home/imi/Projects/pytortch-backend-experiments/stage0/python-deps/nvidia/cu13/lib')
families={
 'l2-hit':'lts__t_sector_hit_rate.pct',
 'l2-volume':'lts__t_sectors.sum',
 'clock':'sm__cycles_elapsed.avg.per_second,dram__cycles_elapsed.avg.per_second',
 'scoreboard':'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,sm__warps_active.avg.pct_of_peak_sustained_active',
 'dispatch':'smsp__inst_executed.sum,smsp__warps_eligible.avg.per_cycle_active',
}
records=[]
for family,metrics in families.items():
 for i in range(8):
  out=root/f'{family}-fresh-{i:02d}'
  cmd=['stage0/.venv/bin/python','-m','stage5.reduction_state_counters','--output',str(out),'--rounds','1','--round-start',str(i),'--iterations','65536','--modes','read_write','write_read','--pm-metrics',metrics]
  with (root/f'{family}-fresh-{i:02d}.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  records.append({'command':cmd,'returncode':r.returncode})
  (root/'families-isolated-commands.json').write_text(json.dumps(records,indent=2))
  if r.returncode: print('FAILED',family,i,flush=True);break
  d=json.loads((out/'result.json').read_text());print(family,i,{k:v['median_us'] for k,v in d['summary'].items()},flush=True)
