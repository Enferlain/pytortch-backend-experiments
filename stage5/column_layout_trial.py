"""CPU-first proof and common boundary for the single column layout trial."""

import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from stage5.layout_trial import (
    ROOT,
    OUT as FROZEN,
    sha,
    tensor_sha as tensor_sha,
    check as check,
)
from stage5.layout_model import verify, evaluate, divmod_fast, LIMIT

OUT = ROOT / "stage5/artifacts/column-layout"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "stage5/cache/column-layout")


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def initialize():
    from stage1 import evidence

    evidence.guard()
    from stage1.overlay import install

    install()
    import torch

    torch.set_num_threads(1)
    torch._dynamo.config.recompile_limit = 128
    return torch


def cpu():
    assert not (ROOT / "stage5/column_layout_probe/src/main.rs").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "cpu.json").exists()
    preserved = {}
    paths = (
        subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"])
        .decode()
        .split("\0")
    )
    for name in sorted(set(paths) - {""}):
        p = ROOT / name
        if (
            name.startswith(
                (".beads/", "stage5/column_layout", "stage5/artifacts/column-layout/")
            )
            or name in ("STAGE5_REPORT.md", "stage5/.gitignore")
            or not p.is_file()
        ):
            continue
        preserved[name] = sha(p)
    write("preservation.json", preserved)
    for name in ("STAGE5_REPORT.md", "stage5/.gitignore"):
        (OUT / (Path(name).name + ".before")).write_bytes((ROOT / name).read_bytes())
    frozen = read(FROZEN / "integrity.json")
    for name, digest in frozen["files"].items():
        assert sha(FROZEN / name) == digest, name
    plan = read(FROZEN / "qualification-plan.json")
    capture = read(FROZEN / "capture.json")
    rows = []
    for case in plan["cases"]:
        cap = next(
            r
            for r in capture["rows"]
            if r["case"]["family"] == case["family"]
            and r["case"]["dtype"] == case["dtype"]
        )
        sem = read(FROZEN / cap["baseline"] / "semantic.json")
        row = verify(sem, case["scalars"])
        row.update(label=case["label"], specialized=case["family"] == "bias_column")
        if row["specialized"]:
            b, n = row["shape"]
            assert b > 0 and n > 0 and b * n <= LIMIT
            linear = np.arange(b * n, dtype=np.uint64)
            quotient, _ = divmod_fast(linear, n)
            assert np.all(quotient < b)
            # Literal FixedDimLayout<usize>: (0 + row * stride1) / vector1.
            actual = (np.uint64(0) + quotient * np.uint64(1)) // np.uint64(1)
            y_load = next(
                o for o in sem["ops"] if o["op"] == "load" and o["buffer"] == 1
            )
            expected = evaluate(
                y_load["index"], case["scalars"], linear.astype(np.int64)
            )
            assert np.array_equal(actual, expected)
            assert np.array_equal(actual, quotient % np.uint64(b))
            assert np.all(actual < b)
            row["proof"] = (
                "Inside unchanged lazy guard: 0<=i<B*N, B,N>0; q=floor(i/N)<B, hence q%B=q. FixedDim scalar stride1/vector1 maps q to q."
            )
        rows.append(row)
    assert len(rows) == 66 and sum(r["specialized"] for r in rows) == 21
    write("qualification-plan.json", plan)
    originals = [
        r for r in plan["cases"] if r["family"] == "bias_column" and r["original"]
    ]
    write(
        "timing-plan.json",
        dict(
            cases=originals,
            rounds=6,
            calls=16,
            order=list(itertools.permutations(("generic", "component", "ordinary"))),
            counter_shapes=[[3, 255], [64, 257], [256, 4099]],
            measurement="Actual generated wrappers, CUPTI activity durations; no new warmup/primer/clock/cache control. NCU is separate code evidence only.",
        ),
    )
    write(
        "cpu.json",
        dict(
            complete=True,
            gpu_source_existed=False,
            rows=rows,
            address_identities=sum(r["scalar_identities"] for r in rows),
            source_sha256=sha(__file__),
            frozen_integrity_sha256=sha(FROZEN / "integrity.json"),
            frozen_model_sha256=sha(ROOT / "stage5/layout_model.py"),
            qualification_plan_sha256=sha(OUT / "qualification-plan.json"),
            timing_plan_sha256=sha(OUT / "timing-plan.json"),
            head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            dirty_start=subprocess.check_output(
                ["git", "status", "--short"], text=True
            ),
            gpu_semantics="Original scalar256, protected loads, arithmetic/casts, metadata ABI and ownership frozen; only column load coordinate changes.",
        ),
    )
    print(
        "PASS",
        len(rows),
        "CPU cases",
        sum(r["scalar_identities"] for r in rows),
        "address identities; GPU source absent",
        flush=True,
    )


if __name__ == "__main__":
    cpu()
