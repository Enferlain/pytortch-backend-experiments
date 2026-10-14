"""Failed gate commands retain evidence without replacing historical captures."""

import json
import hashlib
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from stage5 import precision_gates as runner

import torch
from stage1 import overlay
from stage5 import reduction_order_checks as order

# A complete legacy pair must not narrow the normal patch check silently.
normal_paths = [
    overlay.ROOT / "artifacts" / n for n in ("pytorch.patch", "overlay.json")
]
normal_bytes = [p.read_bytes() for p in normal_paths]
overlay.install(precision_casts=False)
assert [p.read_bytes() for p in normal_paths] == normal_bytes
legacy = json.loads(
    (overlay.ROOT / "artifacts/legacy-overlay/overlay.json").read_text()
)
assert "index_propagation.py" not in legacy and "lowering.py" not in legacy
overlay.install()

with TemporaryDirectory() as temporary:
    root = Path(temporary)
    canonical = root / "artifacts/precision/reduction-order.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("previous completed qualification\n")
    with (
        patch.object(order, "ROOT", root),
        patch.object(
            order,
            "qualify",
            side_effect=RuntimeError("controlled qualification failure"),
        ),
    ):
        try:
            order.main()
        except RuntimeError as error:
            assert str(error) == "controlled qualification failure"
        else:
            raise AssertionError("failed numerical diagnostic accepted")
    failed = json.loads(canonical.with_suffix(".failed.json").read_text())
    assert failed["accepted"] is False and failed["qualification_complete"] is False
    assert "controlled qualification failure" in failed["traceback"]
    assert canonical.read_text() == "previous completed qualification\n"

with TemporaryDirectory() as temporary:
    root = Path(temporary)
    original = Path(torch.__file__).parent / "_inductor"
    hashes = json.loads((runner.PROJECT / "stage1/artifacts/overlay.json").read_text())
    assert set(hashes) == {
        "scheduler.py",
        "codegen/triton.py",
        "codegen/cuda_combined_scheduling.py",
        "codegen/xpu/xpu_combined_scheduling.py",
        "index_propagation.py",
        "lowering.py",
    }, "normal precision overlay evidence required; legacy hashes are insufficient"
    for name in hashes:
        destination = root / "torch/_inductor" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, destination)
    subprocess.run(
        [
            "git",
            "apply",
            "--unidiff-zero",
            str(runner.PROJECT / "stage1/artifacts/pytorch.patch"),
        ],
        cwd=root,
        check=True,
    )
    assert all(
        hashlib.sha256((root / "torch/_inductor" / name).read_bytes()).hexdigest()
        == value["patched"]
        for name, value in hashes.items()
    )

with TemporaryDirectory() as temporary:
    project = Path(temporary)
    historical = [
        "stage1/artifacts/results.json",
        "stage5/artifacts/launch/controls.json",
    ]
    for path in historical:
        p = project / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("historical\n")
    source = project / "stage1/overlay.py"
    source.write_text("user source\n")

    def failure(*args, **kwargs):
        for path in historical:
            (project / path).write_text("failed current capture\n")
        return SimpleNamespace(returncode=7)

    root = project / "stage5"
    output = root / "artifacts/precision"
    with (
        patch.multiple(
            runner,
            ROOT=root,
            PROJECT=project,
            OUTPUT=output,
            CACHE=root / "cache/gates",
        ),
        patch.object(
            runner.subprocess,
            "check_output",
            return_value=("\0".join(historical) + "\0").encode(),
        ),
        patch.object(runner.subprocess, "run", side_effect=failure),
    ):
        try:
            runner.main()
        except SystemExit as error:
            assert error.code == 7
        else:
            raise AssertionError("failed subprocess accepted")
    assert all((project / p).read_text() == "historical\n" for p in historical)
    assert all(
        (output / "regressions" / p).read_text() == "failed current capture\n"
        for p in historical
    )
    assert source.read_text() == "user source\n"
    assert json.loads((output / "gates.json").read_text())[0]["exit_code"] == 7
    assert json.loads((output / "refreshed-artifacts.json").read_text()) == historical

path = Path(__file__).parent / "artifacts/precision/harness-controls.json"
path.write_text(
    json.dumps(
        {
            "failure_exit_propagates": True,
            "tracked_and_untracked_capture_bytes_restored": True,
            "failed_evidence_retained": True,
            "source_untouched": True,
            "serialized_patch_applies_to_pinned_wheel_exactly": True,
            "legacy_mode_preserves_normal_patch_and_hashes": True,
            "reduction_failure_recorded_without_replacing_completed_result": True,
        },
        indent=2,
    )
)
print("Precision harness failure controls PASS", flush=True)
