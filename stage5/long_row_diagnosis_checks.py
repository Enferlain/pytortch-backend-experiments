"""Evidence corruption controls; no GPU, capture or historical bytes modified."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

from stage5 import long_row_diagnose as diagnosis


def main():
    checks = []

    def rejects(name, callback):
        try:
            callback()
        except (AssertionError, KeyError, ValueError):
            checks.append(name)
        else:
            raise AssertionError("accepted corrupted evidence: " + name)

    parsed = diagnosis.instructions(
        "Kernel Name test\n"
        "0x10 BAR.SYNC.DEFER_BLOCKING 0x0 4 128\n"
        "0x20 @!P0 LDG.E.CONSTANT R1, [R2.64] 2 32\n"
    )
    assert parsed["test"] == {"BAR": 4, "LDG": 2}
    checks.append("dynamic_warp_counts_not_predicated_thread_counts")
    rejects("empty_SASS", lambda: diagnosis.instructions(""))
    rejects(
        "SASS_without_kernel_identity",
        lambda: diagnosis.instructions("0x10 EXIT 4 128"),
    )
    valid = diagnosis.profile("production-8")
    counts = {
        k["kernel"]: copy.deepcopy(k["dynamic_warp_instructions_by_opcode"])
        for k in valid["kernels"]
    }
    next(iter(counts.values()))["BAR"] += 1
    with patch.object(diagnosis, "instructions", return_value=counts):
        rejects(
            "SASS_counter_total_mismatch", lambda: diagnosis.profile("production-8")
        )
    original = Path.read_text
    path = diagnosis.OUTPUT / "production-8.json"
    capture = json.loads(original(path))
    for name, mutate in (
        ("incomplete_capture", lambda d: d.update(qualification_complete=False)),
        ("GIL_enabled_on_CubeCL", lambda d: d["environment"].update(gil=True)),
        (
            "loaded_native_Triton",
            lambda d: d["environment"]["independence"].update(
                native_mappings=["bad.so"]
            ),
        ),
        ("unmapped_kernel", lambda d: d.update(selected_kernels=[])),
    ):
        changed = copy.deepcopy(capture)
        mutate(changed)

        def substitute(p, *args, **kwargs):
            return json.dumps(changed) if p == path else original(p, *args, **kwargs)

        with patch.object(Path, "read_text", substitute):
            rejects(name, lambda: diagnosis.profile("production-8"))
    result = {"complete": True, "checks": checks}
    (diagnosis.OUTPUT / "harness-controls.json").write_text(
        json.dumps(result, indent=2)
    )
    print(len(checks), "evidence controls pass")


if __name__ == "__main__":
    main()
