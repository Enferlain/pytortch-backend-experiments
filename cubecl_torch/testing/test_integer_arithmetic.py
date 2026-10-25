"""Integer and bool value arithmetic (OpenSpec integer-value-arithmetic, bead 4x9.1)."""

import re
from pathlib import Path

import pytest
import torch

from cubecl_torch.compiler import descriptor
from cubecl_torch.testing.fixtures import run_python

NATIVE = Path(__file__).resolve().parents[1] / "native" / "src"


def test_integer_dtype_table_matches_the_compiler():
    source = (NATIVE / "ir.rs").read_text()
    table = source[source.index("pub(crate) const INTEGERS") :]
    table = table[: table.index("];")]
    rust = {
        name: (signed == "true", int(width))
        for name, signed, width in re.findall(
            r'\("(\w+)", (true|false), (\d+)\)', table
        )
    }
    assert rust == descriptor.INTEGERS


# A compiled function's warm launches and its agreement with eager, per case.
LAUNCHES = """
from cubecl_torch.testing.fixtures import native_launches
def launches_and_result(fn, *args):
    torch._dynamo.reset()
    f = torch.compile(fn, backend=ci.compile_backend)
    f(*args)
    torch.cuda.synchronize()
    before = native_launches()
    y = f(*args)
    torch.cuda.synchronize()
    return native_launches() - before, y
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bool_predicates_fuse_into_pointwise_and_reduction_kernels(tmp_path):
    code = (
        LAUNCHES
        + """
a = torch.randn(64, 1000, device="cuda")
b = torch.randn(64, 1000, device="cuda")
a[0, :7] = float("nan"); b[1, :5] = float("nan")
out = {}
for name, fn in {
    "fmax_sum": lambda a, b: torch.fmax(a, b).sum(-1),
    "fmax_scaled": lambda a, b: torch.fmax(a, b) * 2,
}.items():
    n, y = launches_and_result(fn, a, b)
    expected = fn(a, b)
    out[name] = [n, bool(torch.allclose(y, expected, rtol=1e-5, atol=1e-4, equal_nan=True))]
print(json.dumps(out))
"""
    )
    result = run_python(code, tmp_path)
    assert result["fmax_sum"] == [1, True]
    assert result["fmax_scaled"] == [1, True]
