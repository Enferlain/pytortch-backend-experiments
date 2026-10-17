"""Focused host guard checks; no GPU dispatch or timing."""

import json
from unittest.mock import patch

from stage1 import evidence

evidence.guard()

from stage5 import rms_trial_runtime as runtime
from stage5 import rms_trial_analyze as analyze


def rejected(fn, error):
    try:
        fn()
    except error:
        return True
    raise AssertionError("guard accepted corrupted evidence or metadata")


runtime.cpu_gate()
original_sha = runtime.sha
producer = runtime.OUT / "phase-sources/cpu-rms_trial.py"
with patch.object(
    runtime, "sha", side_effect=lambda p: "0" * 64 if p == producer else original_sha(p)
):
    assert rejected(runtime.cpu_gate, AssertionError)

semantic = runtime.templates()["float32"]


def stub_init(self, *args, **kwargs):
    self.manifest = {
        "mechanism": runtime.MECHANISM,
        "metadata": {"scalar_count": 3, "bytes": 40},
    }


with patch.object(runtime.ExistingKernel, "__init__", stub_init):
    runtime.Kernel("unused", semantic, 0)


def bad_init(self, *args, **kwargs):
    stub_init(self)
    self.manifest["metadata"]["bytes"] = 32


with patch.object(runtime.ExistingKernel, "__init__", bad_init):
    assert rejected(lambda: runtime.Kernel("unused", semantic, 0), ValueError)

original_evidence_sha = analyze.sha
with patch.object(
    analyze,
    "sha",
    side_effect=lambda p: "0" * 64
    if p == runtime.OUT / "component-sources.json"
    else original_evidence_sha(p),
):
    assert rejected(analyze.main, AssertionError)

result = {
    "complete": True,
    "valid_cpu_gate": True,
    "modified_cpu_producer_rejected": True,
    "valid_metadata_accepted": True,
    "modified_metadata_rejected": True,
    "modified_provenance_rejected": True,
    "scope": "Host guard checks with mocked digest/base constructor; no GPU dispatch",
}
(runtime.OUT / "review/followup-checks.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print("PASS: CPU producer, metadata and provenance guard controls")
