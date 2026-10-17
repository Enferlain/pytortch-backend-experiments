"""Reject a bad host-policy result and verify the previous pass is cleared."""

import importlib.util
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "inspect.py"
spec = importlib.util.spec_from_file_location("rms_attribution_inspect", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
validation = module.OUT / "validation.json"
assert json.loads(validation.read_text())["complete"]
original = module.subprocess.check_output
try:
    module.subprocess.check_output = lambda *args, **kwargs: "{}"
    try:
        module.main(False)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt policy accepted")
    assert not json.loads(validation.read_text())["complete"]
finally:
    module.subprocess.check_output = original

result = {
    "corrupt_policy_rejected": True,
    "previous_pass_cleared": True,
    "scope": "host-only mocked subprocess result; no frozen evidence modified",
}
(module.OUT / "review/failure-marker-check.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print("PASS: failed verification clears the previous successful marker")
