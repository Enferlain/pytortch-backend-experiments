"""Overlay confinement (task 4.1)."""

import pytest
import torch

from cubecl_torch.testing.fixtures import REPO, run_python

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_lower_layers_do_not_install(tmp_path):
    code = """
import importlib, pathlib
for layer in ("runtime", "compiler"):
    for path in sorted(pathlib.Path("cubecl_torch", layer).glob("*.py")):
        importlib.import_module(".".join(path.with_suffix("").parts).removesuffix(".__init__"))
from cubecl_torch.inductor import overlay
print(json.dumps({
    "installed": overlay.installed(),
    "path": [p for p in torch.__path__ if "inductor-overlay" in p],
    "inductor": [m for m in sys.modules if m.startswith("torch._inductor")],
}))
"""
    prelude = "import json, sys\nimport torch\n"
    result = run_python(code, tmp_path, prelude=prelude)
    assert result == {"installed": False, "path": [], "inductor": []}


def test_unrecorded_pytorch_build_refused(tmp_path):
    code = """
import json, sys
import torch
torch.version.git_version = "0" * 40
from cubecl_torch.inductor import install
try:
    install()
    outcome = "installed"
except RuntimeError as error:
    outcome = str(error)
print(json.dumps({
    "outcome": outcome,
    "inductor": [m for m in sys.modules if m.startswith("torch._inductor")],
}))
"""
    result = run_python(code, tmp_path, prelude="")
    assert "recorded PyTorch wheel commit" in result["outcome"]
    assert result["inductor"] == []


# The one generic edit beyond Stage 1 (4z8.17, narrowed by multi-output-kernels
# group 5): graph casts to FP16/BF16 the user wrote round even without
# precision emulation; casts a decomposition inserted follow Triton.
EXPLICIT_CAST_HUNK = """@@ -1084,0 +1097,11 @@
+    # A graph cast to FP16/BF16 the user wrote is requested rounding: it
+    # materializes even without precision emulation, which only adds the
+    # per-operation barriers eager's storage implies. A cast a
+    # decomposition inserted (its original aten op is not _to_copy) follows
+    # Triton. Without metadata the cast counts as written. Compared by
+    # name: the overlay's op objects are not the tracer's.
+    origin = getattr(V.graph.current_node, 'meta', {}).get('original_aten')
+    written = origin is None or str(origin) in (
+        'aten._to_copy.default',
+        'prims.convert_element_type.default',
+    )
@@ -1086,2 +1109,2 @@
-        config.emulate_precision_casts
-        and (src_dtype in low_pr_fp or dtype in low_pr_fp)
+        (dtype in low_pr_fp and (written or config.emulate_precision_casts))
+        or (config.emulate_precision_casts and src_dtype in low_pr_fp)
"""


def test_installed_files_untouched_and_patch_extends_stage1(tmp_path):
    code = """
import hashlib, json, pathlib, sys
import torch
root = pathlib.Path(torch.__file__).resolve().parent / "_inductor"
def digests():
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
before = digests()
import cubecl_torch.inductor as ci
ci.install()
f = torch.compile(lambda x: torch.sin(x) + 1, backend=ci.compile_backend)
x = torch.randn(1000, device="cuda")
assert torch.allclose(f(x), torch.sin(x) + 1, rtol=0, atol=1e-6)
overlay = [p for p in torch.__path__ if "inductor-overlay" in p]
patch = (pathlib.Path(overlay[0]).parent / "pytorch.patch").read_text()
print(json.dumps({"unchanged": before == digests(), "count": len(before),
                  "overlay": overlay, "patch": patch}))
"""
    result = run_python(code, tmp_path, prelude="")
    assert result["unchanged"] and result["count"] > 100
    assert len(result["overlay"]) == 1 and result["overlay"][0].startswith(
        str(tmp_path)
    )
    stage1 = (REPO / "stage1/artifacts/pytorch.patch").read_text()
    assert EXPLICIT_CAST_HUNK in result["patch"]
    assert result["patch"].replace(EXPLICIT_CAST_HUNK, "") == stage1
