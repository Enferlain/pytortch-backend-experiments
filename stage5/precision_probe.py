"""Counterfactual application of the generic patch to an isolated legacy overlay.

Install only into Stage1's regenerated overlay before importing Inductor.
Use Stage1's explicit legacy overlay mode before applying it. Ordinary overlay
installation now includes the correction. No wheel/checkouts are edited.
"""

import difflib
import hashlib
from pathlib import Path


def install(directory):
    from stage1.precision import source_edits

    edits = source_edits()
    patch, identities = [], {}
    for name, replacements in edits.items():
        path = Path(directory) / name
        before = path.read_text()
        after = before
        for old, new in replacements:
            if after.count(old) != 1:
                raise RuntimeError("precision probe source drift: " + name)
            after = after.replace(old, new, 1)
        path.write_text(after)
        identities[name] = {
            "original": hashlib.sha256(before.encode()).hexdigest(),
            "probe": hashlib.sha256(after.encode()).hexdigest(),
        }
        patch.extend(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/torch/_inductor/" + name,
                tofile="b/torch/_inductor/" + name,
            )
        )
    return {
        "experimental_only": True,
        "source_sha256": identities,
        "patch": "".join(patch),
    }
