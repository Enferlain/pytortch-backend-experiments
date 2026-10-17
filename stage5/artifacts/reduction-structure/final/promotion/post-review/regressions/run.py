"""Repeat affected gates after review fixes, reusing historical preservation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import hashlib

project = Path.cwd()
packet = project / 'stage5/artifacts/reduction-structure/final/promotion'
out = packet / 'post-review/regressions'
out.mkdir(parents=True, exist_ok=True)
source = project / 'stage5/artifacts/reduction-structure/final/regressions-full/harness.py'
spec = importlib.util.spec_from_file_location('cwk9_preserve', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.OUT = out
tracked = subprocess.check_output(['git','ls-files','-z']).decode().split('\0')
roots = {project / x.partition('/artifacts/')[0] / 'artifacts' for x in tracked if '/artifacts/' in x} | {project / 'stage5/artifacts/reduction-structure'}
initial = m.artifact_files(roots)
snapshot = m.snapshot_history(initial)
names = {'components','controls','proof_controls','descriptor_controls','launch_controls','launch_hook_checks','cache_init','cache_reload','corpus','precision_matrix','precision_harness','frozen_order_qualification','accumulation_checks','accumulation_harness','long_row_checks'}
commands = [g for g in json.loads((project / 'stage5/artifacts/reduction-structure/gates.json').read_text()) if g['name'] in names]
results=[]
try:
    for gate in commands:
        command=gate['command']
        log=out / (gate['name']+'.log')
        with log.open('w') as f:
            run=subprocess.run(command,cwd=project,env={**os.environ,'STAGE0_CARGO_PROFILE':'release','CUDA_PATH':str(project/'stage0/cuda13-toolkit/nvidia/cu13')},stdout=f,stderr=subprocess.STDOUT)
        results.append({'name':gate['name'],'command':command,'exit_code':run.returncode,'log':str(log),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
        (out/'gates.json').write_text(json.dumps(results,indent=2))
        print(gate['name'],'PASS' if run.returncode==0 else 'FAIL',flush=True)
        if run.returncode:raise SystemExit(run.returncode)
finally:
    preservation=m.restore_history(project,snapshot,roots,initial)
    (out/'historical-preservation.json').write_text(json.dumps(preservation,indent=2))
    (out/'gate-status.json').write_text(json.dumps({'expected':len(commands),'finished':len(results),'complete':len(results)==len(commands) and all(r['exit_code']==0 for r in results) and preservation['all_hashes_match'] and preservation['no_new_historical_files']},indent=2))
