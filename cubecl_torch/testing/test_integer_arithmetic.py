"""Integer and bool value arithmetic (OpenSpec integer-value-arithmetic, bead 4x9.1)."""

import re
from pathlib import Path

from cubecl_torch.compiler import descriptor

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
