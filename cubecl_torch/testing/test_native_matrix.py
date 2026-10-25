"""The native compiler is byte-deterministic across processes over the reduction matrix."""

from cubecl_torch.compiler.build import compiler_path
from cubecl_torch.testing import native_matrix


def test_matrix_is_identical_across_compiler_processes(tmp_path):
    binary = compiler_path()
    report = native_matrix.compare(binary, binary, work=tmp_path)
    assert not report["problems"], report["problems"]
    # Rejections must be plan-legality refusals, not malformed matrix descriptors.
    compiled = set(report["identical"])
    for dtype in native_matrix.DTYPES:
        assert f"{dtype}-final9" in compiled
        for probe in (
            native_matrix.REFERENCE512_PROBES + native_matrix.FINAL_PLANE_PROBES
        ):
            assert f"{dtype}-probe-{probe}" in compiled
        for kind in native_matrix.KINDS:
            assert any(n.startswith(f"{dtype}-{kind}-") for n in compiled), kind
