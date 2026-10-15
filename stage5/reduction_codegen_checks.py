"""Bounded counters, ownership refusal, installed identity and frozen neighbors."""

import json
import os
from pathlib import Path
import subprocess

from stage5.reduction_codegen_mechanisms import ROOT, OUT, FROZEN, Component


def compile_packet(packet, route, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "descriptor.json"
    path.write_text(json.dumps(packet, indent=2))
    return subprocess.run(
        [
            str(ROOT.parent / "stage0/target/release/cubecl-stage0"),
            str(directory),
            route,
            str(path),
        ],
        text=True,
        capture_output=True,
    )


def main():
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch
    from stage0.bridge import nvrtc_compile
    from stage5.legality import numerical_reason
    from stage5.reduction_policy import qualified_for_runtime
    from stage5.reduction_topology_probe import candidate

    torch.set_num_threads(1)
    result = {"ownership_controls": [], "counter_boundaries": [], "code_identity": []}
    cache = ROOT / "cache/codegen/checks"
    packet = json.loads(
        (FROZEN / "kernels/float32/candidate/descriptor.json").read_text()
    )
    for name in ("normalization_float32-0", "two_sums_normal_fusion_shared_reuse-0"):
        sem = json.loads(
            (
                ROOT.parent / f"stage3/artifacts/representatives/{name}/descriptor.json"
            ).read_text()
        )
        p = {**packet, "semantic": sem, "entrypoint": "cubecl5_codegen_neighbor"}
        run = compile_packet(p, "cpp-probe-owner", cache / name)
        assert run.returncode and "owner-result probe requires" in run.stderr, (
            name,
            run.stderr,
        )
        normal = compile_packet(p, "cpp", cache / (name + "-normal"))
        assert normal.returncode == 0, normal.stderr
        source = (cache / (name + "-normal") / "kernel.cu").read_text()
        assert source.count("__syncthreads()") >= 2
        result["ownership_controls"].append(
            {
                "case": name,
                "owner_probe_rejected": True,
                "normal_broadcast_barriers": source.count("__syncthreads()"),
            }
        )
    for route in ("cpp-probe-unknown",):
        run = compile_packet(packet, route, cache / "unknown")
        assert run.returncode and "unknown codegen" in run.stderr
    for e in (1, 31, 32, 511, 512, 513, 2047, 2048, 2049, 1048576):
        torch.manual_seed(901)
        # Nonzero storage offset; every pointer still belongs to PyTorch.
        backing = torch.randn(9 * e + 1, device="cuda")
        x = backing[1:].view(1, 9 * e)
        kernels = {
            "frozen": Component(FROZEN / "kernels/float32/candidate"),
            **{
                m: Component(OUT / m / "float32")
                for m in ("uniform", "owner", "narrow", "uniform_narrow")
            },
        }
        values = {name: torch.empty((1, 9), device="cuda") for name in kernels}
        for name, k in kernels.items():
            k((9 * e, 1), x, values[name])
        assert all(
            torch.equal(values["frozen"].view(torch.int32), v.view(torch.int32))
            for v in values.values()
        )
        result["counter_boundaries"].append(
            {
                "extent": e,
                "four_probes_bitwise": True,
                "pytorch_input_storage_offset": 1,
            }
        )
    narrow = kernels["narrow"].kernel
    for n in (9 * 1048576 + 1, 1 << 40):
        before = narrow.launch_count
        try:
            narrow((n, 1), x, values["narrow"])
        except ValueError as error:
            assert "unsupported physical reduction extent" in str(error)
        else:
            raise AssertionError("missing counter proof bound")
        assert narrow.launch_count == before
    legacy = json.loads((FROZEN / "legacy-code-identity.json").read_text())["checks"]
    old = [Path(c["old"]) for c in legacy] + [
        FROZEN / "kernels" / d / "fixed_final"
        for d in ("float32", "float16", "bfloat16")
    ]
    includes = [str(Path(torch.__file__).parent.parent / "nvidia/cu13/include")]
    for i, directory in enumerate(old):
        p = json.loads((directory / "descriptor.json").read_text())
        dest = cache / f"legacy-{i}"
        run = compile_packet(p, "cpp", dest)
        assert run.returncode == 0, run.stderr
        assert (dest / "kernel.cu").read_bytes() == (
            directory / "kernel.cu"
        ).read_bytes()
        nvrtc_compile((dest / "kernel.cu").read_text(), dest, 0, includes)
        assert (dest / "kernel.ptx").read_bytes() == (
            directory / "kernel.ptx"
        ).read_bytes()
        result["code_identity"].append(
            {"old": str(directory), "new": str(dest), "cuda_and_ptx_identical": True}
        )
    for d in ("float32", "float16", "bfloat16"):
        p = json.loads(
            (FROZEN / "kernels" / d / "candidate/descriptor.json").read_text()
        )
        for mode in ("cpp", "cpp-probe-frozen"):
            dest = cache / (d + "-" + mode)
            run = compile_packet(p, mode, dest)
            assert run.returncode == 0, run.stderr
            expected = (
                (OUT / "uniform" / d)
                if mode == "cpp"
                else FROZEN / "kernels" / d / "candidate"
            )
            assert (dest / "kernel.cu").read_bytes() == (
                expected / "kernel.cu"
            ).read_bytes()
            result["code_identity"].append(
                {"case": d, "mode": mode, "cuda_identical": True}
            )
    for n in (9, 7283, 1048576):
        sem = {"schema": 3, "reduction_extent": {"op": "constant", "value": n}}
        assert numerical_reason(candidate(), sem, ())
        assert not qualified_for_runtime(sem, candidate())
    graph = json.loads((OUT / "graph-cubecl.json").read_text())
    frozen = json.loads((FROZEN / "topology-cubecl-r0.json").read_text())
    assert len(graph["rows"]) == 178
    for a, b in zip(graph["rows"], frozen["rows"], strict=True):
        for k in ("case", "shape", "dtype", "input_sha256", "values", "aten", "fp64"):
            assert a[k] == b[k], (a["case"], k)
    result["graph_rows_exact_recorded_values"] = 178
    result["eager_failures_preserved"] = (
        graph["numerical_failures"] == frozen["numerical_failures"]
    )
    assert result["eager_failures_preserved"]
    import numpy as np

    current = ROOT / "cache/codegen/norm/norm-cubecl.npz"
    old = ROOT / "cache/reduction-topology/norm/norm-cubecl.npz"
    with np.load(current) as a, np.load(old) as b:
        assert a.files == b.files
        for name in a.files:
            assert np.array_equal(a[name].view(np.uint8), b[name].view(np.uint8)), name
        result["normalization_arrays_bitwise"] = len(a.files)
    result["automatic_ineligibility_preserved"] = True
    result["complete"] = True
    (OUT / "checks.json").write_text(json.dumps(result, indent=2))
    print(
        "Codegen checks PASS",
        result["graph_rows_exact_recorded_values"],
        "graph rows;",
        len(result["counter_boundaries"]),
        "counter boundaries;",
        len(result["code_identity"]),
        "code identities",
        flush=True,
    )


if __name__ == "__main__":
    main()
