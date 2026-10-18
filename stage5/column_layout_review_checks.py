"""Post-review descriptor ownership and unchanged GPU-body controls for l1m."""

import ast
import copy
from pathlib import Path

from stage5.column_layout_trial import OUT, ROOT, read, sha, write, initialize


def source_identity():
    before = ast.parse((OUT / "sources/stage5/column_layout_runtime.py").read_text())
    before.body.insert(2, ast.Import(names=[ast.alias(name="copy")]))
    for node in before.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("generic", "compile_kernel"):
            node.body.insert(0, ast.parse("semantic = copy.deepcopy(semantic)").body[0])
    current = ast.parse((ROOT / "stage5/column_layout_runtime.py").read_text())
    assert ast.dump(before) == ast.dump(current)


def audit():
    source_identity()
    old, new = (read(OUT / n) for n in ("gpu.json", "post-review-gpu.json"))
    assert old["complete"] and new["complete"]
    for key, count in (("rows", 78), ("live_controls", 90), ("stream_controls", 12)):
        assert len(new[key]) == len(old[key]) == count
    for a, b in zip(old["rows"], new["rows"]):
        assert all(a[k] == b[k] for k in (
            "label", "special", "scalars", "input_sha256", "output_sha256",
            "bitwise", "policy", "canaries",
        ))
    assert new["source_sha256"] == sha(ROOT / "stage5/column_layout_qualify.py")
    assert new["binary_sha256"] == old["binary_sha256"]
    assert new["special_values_sha256"] == sha(OUT / "post-review-special-values.pt")
    bodies = []
    assert len(old["kernels"]) == len(new["kernels"]) == 3
    for candidate in new["kernels"]:
        prior = next(k for k in old["kernels"] if k["semantic"] == candidate["semantic"])
        a, b = Path(prior["directory"]), Path(candidate["directory"])
        assert sha(b / "kernel.ptx") == candidate["ptx_sha256"]
        for directory in (a, b):
            for name, digest in read(directory / "integrity.json").items():
                assert sha(directory / name) == digest
        for filename in ("cubecl.ir", "kernel.cu", "kernel.ptx"):
            assert (a / filename).read_text().replace(prior["entrypoint"], "ENTRY") == (
                b / filename
            ).read_text().replace(candidate["entrypoint"], "ENTRY"), filename
        ident = read(b / "identity.json")
        assert ident["column_adapter"] == sha(ROOT / "stage5/column_layout_runtime.py")
        expected_identity = read(a / "identity.json")
        expected_identity["column_adapter"] = ident["column_adapter"]
        assert ident == expected_identity
        bodies.append(dict(dtype=candidate["semantic"]["buffers"][0]["dtype"],
                           directory=str(b), ir_cuda_ptx_identical=True))
    return bodies


def main():
    initialize()
    from stage5.column_layout_runtime import compile_kernel, generic
    from stage5.layout_runtime import templates

    assert not (OUT / "review/native/controls.json").exists()
    bodies = audit()
    controls = []
    for family, dtype, semantic in templates().values():
        for route in (("generic", generic), ("column", compile_kernel)):
            name, dispatch = route
            if name == "column" and family != "bias_column":
                continue
            original = copy.deepcopy(semantic)
            caller = copy.deepcopy(original)
            compiled = dispatch(caller, 0)
            caller["buffers"][0]["stride"][0] = {"op": "const", "value": 999}
            caller["scalars"].clear()
            caller["numel"] = {"op": "const", "value": 0}
            assert compiled.semantic == compiled.kernel.semantic == original
            fresh = dispatch(copy.deepcopy(original), 0)
            assert fresh.semantic == fresh.kernel.semantic == original
            if name == "column":
                assert fresh is compiled
            controls.append(dict(family=family, dtype=dtype, route=name,
                                 caller_mutation_isolated=True,
                                 original_lookup_unpoisoned=True))
    assert len(controls) == 15
    # Kernel loads above must not alter their retained artifacts.
    audit()
    write("review/native/controls.json", dict(
        complete=True, rows=controls, bodies=bodies, source_sha256=sha(__file__),
        runtime_sha256=sha(ROOT / "stage5/column_layout_runtime.py"),
    ))
    print("PASS15 descriptor-mutation controls;78/90/12 repeat;3 identical IR/CUDA/PTX bodies")


if __name__ == "__main__":
    main()
