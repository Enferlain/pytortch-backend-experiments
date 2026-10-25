"""Byte comparison of two native compiler binaries over a reduction descriptor matrix.

The manifest corpus (Stage 5's 168 cases) is mostly sums. This matrix covers every
schema-5 reduction strategy and plan shape for sum, max and Welford over float32
and bfloat16, the final9 plan and every offline ``cpp-probe-*`` route, so emitter
refactors and CubeCL upgrades can be checked where the corpus does not reach.

    # one binary against itself: compilation is deterministic across processes
    python -m cubecl_torch.testing.native_matrix BIN BIN
    # before/after a change: build the old revision elsewhere, then
    python -m cubecl_torch.testing.native_matrix OLD_BIN NEW_BIN [--jobs 8] [--work DIR]

Each case is compiled by each binary in a fresh process. A case passes when both
reject it with the same error, or both produce the same ``kernel.cu``,
``cubecl.ir`` and ``info.txt`` bytes.
"""

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FILES = ("kernel.cu", "cubecl.ir", "info.txt")
DTYPES = ("float32", "bfloat16")
KINDS = ("sum", "max", "welford_reduce")
REFERENCE512_PROBES = ("frozen", "uniform", "owner", "narrow", "uniform_narrow")
FINAL_PLANE_PROBES = ("final-plane", "final-plane-uniform")


def constant(n):
    return {"op": "constant", "value": n}


def scalar(i):
    return {"op": "scalar", "slot": i}


def semantic(dtype, kind, extent=None):
    """A row reduction over a contiguous [outer, R] buffer, storing one statistic."""
    r = constant(extent) if extent is not None else scalar(1)
    scalars = [{"dtype": "int64", "id": 0, "positive": True}]
    if extent is None:
        scalars.append({"dtype": "int64", "id": 1, "positive": True})
    row = {"op": "mul", "args": [{"op": "linear"}, r]}
    index = {"op": "add", "args": [row, {"op": "reduction"}]}
    ops = [
        {"op": "load", "id": "x", "args": [], "buffer": 0, "dtype": dtype,
         "semantic_dtype": dtype, "index": index},
        {"op": "cast", "id": "w", "args": ["x"], "dtype": "float32",
         "semantic_dtype": dtype},
        {"op": "reduction", "id": "r", "args": ["w"], "dtype": "float32",
         "semantic_dtype": dtype, "source_dtype": dtype,
         "accumulator_dtype": "float32", "reduction_type": kind},
    ]  # fmt: skip
    result = "r"
    if kind.startswith("welford"):
        ops.append(
            {"op": "reduction_component", "id": "m2", "args": ["r"], "index": 1,
             "dtype": "float32", "semantic_dtype": "float32"}
        )  # fmt: skip
        result = "m2"
    ops += [
        {"op": "cast", "id": "o", "args": [result], "dtype": dtype,
         "semantic_dtype": dtype},
        {"op": "store", "args": ["o"], "buffer": 1, "dtype": dtype,
         "semantic_dtype": dtype, "index": {"op": "linear"},
         "iteration_scope": "outer"},
    ]  # fmt: skip
    return {
        "schema": 3,
        "index_dtype": "int64",
        "buffers": [
            {
                "access": "read",
                "dtype": dtype,
                "offset": constant(0),
                "size": [scalar(0), r],
                "stride": [r, constant(1)],
            },
            {
                "access": "write",
                "dtype": dtype,
                "offset": constant(0),
                "size": [scalar(0)],
                "stride": [constant(1)],
            },
        ],
        "numel": scalar(0),
        "outer": scalar(0),
        "reduction_extent": r,
        "scalars": scalars,
        "ops": ops,
    }


def final9(dtype):
    """stage5.final_plan's narrow contract: FP32 partials [outer, 9] summed per row."""
    d = semantic("float32", "sum", 9)
    d["buffers"][1]["dtype"] = dtype
    d["scalars"] = [{"dtype": "int64", "id": 0, "positive": True}]
    index = {
        "op": "add",
        "args": [
            {"op": "reduction"},
            {"op": "mul", "args": [constant(9), {"op": "linear"}]},
        ],
    }
    ops = [{"op": "load", "id": "x", "args": [], "buffer": 0, "dtype": "float32",
            "semantic_dtype": "float32", "index": index}]  # fmt: skip
    total = {"op": "reduction", "id": "r", "args": ["x"], "dtype": "float32",
             "semantic_dtype": dtype, "accumulator_dtype": "float32",
             "source_dtype": dtype, "reduction_type": "sum"}  # fmt: skip
    if dtype == "float32":
        ops.append(total)
        last = "r"
    else:
        ops.append({"op": "cast", "id": "w", "args": ["x"], "dtype": "float32",
                    "semantic_dtype": dtype})  # fmt: skip
        ops.append({**total, "args": ["w"]})
        last = "r"
        for name, ty in (("a", dtype), ("b", "float32"), ("c", dtype)):
            ops.append({"op": "cast", "id": name, "args": [last], "dtype": ty,
                        "semantic_dtype": dtype})  # fmt: skip
            last = name
    ops.append({"op": "store", "args": [last], "buffer": 1, "dtype": dtype,
                "semantic_dtype": dtype, "index": {"op": "linear"},
                "iteration_scope": "outer"})  # fmt: skip
    d["ops"] = ops
    return d


def plan(
    strategy, workgroup, outputs=1, width=1, plane=0, accumulators=None, merge=None
):
    p = {"version": 2, "strategy": strategy, "workgroup": workgroup,
         "vector_width": width, "elements_per_unit": 1,
         "outputs_per_workgroup": outputs, "plane_width": plane}  # fmt: skip
    if accumulators is not None:
        p["private_accumulators"] = accumulators
    if merge is not None:
        p["private_merge"] = merge
    return p


def plans():
    """Every schema-5 reduction plan shape plan.rs admits, plus illegal neighbours."""
    yield plan("shared_tree_reduction_baseline", [256, 1, 1])
    for a in (2, 4):
        yield plan("shared_tree_multi_accumulator", [256, 1, 1], accumulators=a)
    yield plan("shared_tree_compensated", [256, 1, 1])
    for w in (64, 128):
        yield plan("shared_tree", [w, 1, 1])
    for w in (64, 128, 256):
        yield plan("plane_shared", [w, 1, 1], plane=32)
    for w in (128, 256):
        for v in (2, 4, 8):
            yield plan("plane_shared", [w, 1, 1], width=v, plane=32)
    yield plan("plane_shared", [256, 1, 1], plane=32, accumulators=4)
    for m in ("halving", "adjacent"):
        yield plan(
            "plane_shared_load_batch", [256, 1, 1], plane=32, accumulators=4, merge=m
        )
    yield plan(
        "plane_shared_reference512",
        [512, 1, 1],
        plane=32,
        accumulators=4,
        merge="adjacent",
    )
    for y in (4, 8):
        yield plan("independent_planes", [32, y, 1], outputs=y, plane=32)
    for v in (2, 4, 8):
        for a in (1, 2, 4, 8, 16):
            yield plan(
                "independent_planes",
                [32, 8, 1],
                outputs=8,
                width=v,
                plane=32,
                accumulators=a,
            )
    for y in (4, 8):
        yield plan("coalesced_columns", [32, y, 1], outputs=32)


def envelope(sem, physical_plan):
    return {"schema": 5, "semantic": sem, "physical_plan": physical_plan,
            "entrypoint": "cubecl5_matrix"}  # fmt: skip


def cases():
    """``(name, descriptor, route)`` for the whole matrix."""
    for dtype in DTYPES:
        for kind in KINDS:
            for p in plans():
                shape = "x".join(map(str, p["workgroup"]))
                name = (
                    f"{dtype}-{kind}-{p['strategy']}-{shape}-v{p['vector_width']}"
                    f"-a{p.get('private_accumulators', 1)}-{p.get('private_merge', 'halving')}"
                )
                yield name, envelope(semantic(dtype, kind), p), "cpp"
        final = plan("independent_planes_final9", [32, 4, 1], outputs=4, plane=32)
        yield f"{dtype}-final9", envelope(final9(dtype), final), "cpp"
        reference = envelope(
            semantic(dtype, "sum"),
            plan(
                "plane_shared_reference512",
                [512, 1, 1],
                plane=32,
                accumulators=4,
                merge="adjacent",
            ),
        )
        for probe in REFERENCE512_PROBES:
            yield f"{dtype}-probe-{probe}", reference, f"cpp-probe-{probe}"
        small = envelope(
            semantic(dtype, "sum", 17),
            plan("independent_planes", [32, 4, 1], outputs=4, plane=32),
        )
        for probe in FINAL_PLANE_PROBES:
            yield f"{dtype}-probe-{probe}", small, f"cpp-probe-{probe}"


def _compile(binary, out, route, descriptor):
    proc = subprocess.run(
        [str(binary), str(out), route, str(descriptor)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        return {"error": (proc.stderr.strip().splitlines() or ["(no output)"])[-1]}
    return {"files": {name: (out / name).read_bytes() for name in FILES}}


def _case(old, new, work, name, d, route):
    case = work / name
    case.mkdir(parents=True)
    descriptor = case / "descriptor.json"
    descriptor.write_text(json.dumps(d))
    a = _compile(old, case / "old", route, descriptor)
    b = _compile(new, case / "new", route, descriptor)
    if "error" in a or "error" in b:
        if a == b:
            return name, "rejected", a["error"]
        return (
            name,
            "problem",
            f"old: {a.get('error', 'compiled')}; new: {b.get('error', 'compiled')}",
        )
    differ = [f for f in FILES if a["files"][f] != b["files"][f]]
    if differ:
        return name, "problem", f"differs in {', '.join(differ)}"
    return name, "identical", None


def compare(old, new, *, jobs=8, work=None):
    """``{"identical": [...], "rejected": {name: error}, "problems": {name: why}}``."""
    owned = work is None
    work = Path(work or tempfile.mkdtemp(prefix="cubecl-native-matrix."))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(lambda c: _case(old, new, work, *c), list(cases())))
    finally:
        if owned:
            shutil.rmtree(work, ignore_errors=True)
    report = {"identical": [], "rejected": {}, "problems": {}}
    for name, status, detail in results:
        if status == "identical":
            report["identical"].append(name)
        else:
            report["rejected" if status == "rejected" else "problems"][name] = detail
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--work", help="keep per-case outputs here")
    args = parser.parse_args(argv)
    report = compare(args.old, args.new, jobs=args.jobs, work=args.work)
    for name, why in sorted(report["problems"].items()):
        print("PROBLEM", name, why)
    total = sum(len(v) for v in report.values())
    print(
        f"{total} cases: {len(report['identical'])} byte-identical, "
        f"{len(report['rejected'])} rejected identically by both, "
        f"{len(report['problems'])} problems"
    )
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
