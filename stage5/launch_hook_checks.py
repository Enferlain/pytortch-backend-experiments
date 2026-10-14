"""Failure injection for benchmark restoration and pinned-source routing."""

import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

from stage1 import evidence

evidence.guard()
import torch
from stage1.overlay import install

install()
from stage5 import launch_profile as probe, runtime

original = runtime.compile_kernel
argv = sys.argv
sys.argv = [
    "launch_profile",
    "--output",
    "stage5/cache/launch-hook-check/performance.json",
]
try:
    # Overlay is already installed in this test process. Inject a failure at
    # first graph construction, after main installs its routing factory.
    with (
        patch("stage1.overlay.install", return_value=None),
        patch.object(
            torch, "compile", side_effect=RuntimeError("controlled benchmark failure")
        ),
    ):
        try:
            probe.main()
        except RuntimeError as error:
            assert str(error) == "controlled benchmark failure"
        else:
            raise AssertionError("no injected failure")
    assert runtime.compile_kernel is original
finally:
    sys.argv = argv

real_output = probe.subprocess.check_output


def changed_import(command, **kwargs):
    result = real_output(command, **kwargs)
    if command[-1].endswith("stage5/artifact.py"):
        result = result.replace(
            b"from stage5.legality import reason", b"from stage5 import legality"
        )
    return result


try:
    with patch.object(probe.subprocess, "check_output", side_effect=changed_import):
        try:
            probe.reference_modules()
        except RuntimeError as error:
            assert "cannot be routed" in str(error)
        else:
            raise AssertionError("unverified reference import accepted")
finally:
    probe.reference_modules()
evidence.state("after_launch_hook_checks")
path = probe.ROOT / "artifacts/launch/hook-checks.json"
path.write_text(
    json.dumps(
        {
            "exception_restores_compiler_factory": True,
            "reference_import_change_fails_loud": True,
            "independence": evidence.inventory(),
            "source_sha256": hashlib.sha256(
                Path(probe.__file__).read_bytes()
            ).hexdigest(),
        },
        indent=2,
    )
)
print("Harness robustness checks PASS", flush=True)
