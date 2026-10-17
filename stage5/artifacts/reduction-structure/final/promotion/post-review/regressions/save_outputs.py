"""Retain explicit JSON outputs before the historical-preservation restore."""
from pathlib import Path
import hashlib
import json
import time
root = Path('stage5/artifacts/reduction-structure')
out = root / 'final/promotion/post-review/regressions'
paths = [root/'corpus.json',root/'precision-regression.json']
before = {p:p.read_bytes() if p.exists() else None for p in paths}
saved = {}
while not (out/'gate-status.json').exists():
    for path in paths:
        if path.exists():
            value = path.read_bytes()
            if value != before[path]:
                try: json.loads(value)
                except (ValueError,UnicodeDecodeError): continue
                (out/('captured-'+path.name)).write_bytes(value)
                saved[str(path)] = hashlib.sha256(value).hexdigest()
    time.sleep(.2)
(out/'captured-outputs.json').write_text(json.dumps(saved,indent=2))
