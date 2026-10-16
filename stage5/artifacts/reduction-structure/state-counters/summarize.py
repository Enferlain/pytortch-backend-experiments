import json,statistics
from pathlib import Path
from stage5.reduction_state_counter_analyze import analyze,stats
root=Path('stage5/artifacts/reduction-structure/state-counters')
families={'dram':16,'clock':8,'l2-volume':8,'store-wb':8,'store-wt':8}
summary={'complete':False,'families':{},'validated_counter_traces':0,'source_scope':'qualified-* with deferred C10 clock calibration only; exploratory captures retained separately'}
for family,n in families.items():
 dirs=[root/f'qualified-{family}-{i:02d}' for i in range(n)]
 full=analyze(dirs)
 (root/f'qualified-{family}-analysis.json').write_text(json.dumps(full,indent=2))
 rows=[r for rs in full['runs'].values() for r in rs]
 group={}
 for mode in ['write_read','read_write']:
  rr=[r for r in rows if r['mode']==mode]
  assert len(rr)==n
  metrics={}
  for metric in rr[0]['counter_window']['metrics']:
   metrics[metric]={'interior':stats([r['counter_window']['metrics'][metric]['value'] for r in rr])}
   if metric.endswith('.sum'):metrics[metric]['expanded_overlap_upper']=stats([r['counter_window']['metrics'][metric]['expanded_overlap_upper'] for r in rr])
  group[mode]={'target_us':stats([r['target_us'] for r in rr]),'metrics':metrics,'clock_guard_ns':stats([r['counter_window']['edge_guard_ns'] for r in rr]),'coverage_ns':stats([r['counter_window']['coverage_ns'] for r in rr])}
 assert group['write_read']['target_us']['max']<group['read_write']['target_us']['min']
 group['all_samples_separated']=True
 summary['families'][family]=group
 summary['validated_counter_traces']+=len(rows)
 print(family,{mode:{'target_us':v['target_us']['median'],'metrics':{metric:{k:stats['median'] for k,stats in fields.items()} for metric,fields in v['metrics'].items()}} for mode,v in group.items() if mode in ['write_read','read_write']},flush=True)
summary['complete']=True
(root/'summary.json').write_text(json.dumps(summary,indent=2))
