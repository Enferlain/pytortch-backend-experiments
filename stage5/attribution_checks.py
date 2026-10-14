"""Narrow analysis/harness controls; no GPU profiling or production monkeypatch."""

import json
from pathlib import Path
import sqlite3
import tempfile
from unittest.mock import patch

from stage5.system_analyze import analyze, union_ns
from stage5.precision_probe import install
from stage5.attribution import ROOT, sha


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.sqlite"
        with sqlite3.connect(path) as c:
            c.executescript("""
                CREATE TABLE StringIds(id INTEGER,value TEXT);
                CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER,eventType INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,demangledName INTEGER);
            """)
            c.executemany(
                "INSERT INTO StringIds VALUES (?,?)",
                [
                    (1, "cuLaunchKernel"),
                    (2, "cudaDeviceSynchronize"),
                    (3, "cubecl5_one"),
                    (4, "cubecl5_two"),
                    (5, "reset_kernel"),
                    (6, "vendor_kernel"),
                ],
            )
            ranges = [
                (1000, 10000, 1, "step:0", None, 59),
                (1000, 4000, 1, "generated_dispatch:test", None, 59),
                (1050, 3000, 1, "selection_dispatch", None, 59),
                (1100, 2900, 1, "compiled_dispatch", None, 59),
                (1150, 1160, 1, "metadata", None, 59),
                (1200, 1250, 1, "validation", None, 59),
                (1250, 1450, 1, "native_enqueue", None, 59),
                (2050, 2350, 1, "native_enqueue", None, 59),
            ]
            c.executemany("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?,?)", ranges)
            c.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
                [
                    (100, 900, 1, 1, 2),
                    (1100, 1400, 1, 2, 1),
                    (2100, 2300, 1, 3, 1),
                    (15000, 15100, 1, 4, 1),
                    (5200, 5300, 1, 5, 1),
                ],
            )
            c.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)",
                [
                    (1300, 1800, 2, 3),
                    (2200, 2600, 3, 4),
                    (16000, 16500, 4, 5),
                    (5400, 5900, 5, 6),
                ],
            )
        evidence = {
            "steps": 1,
            "deep": True,
            "implementation": "cubecl",
            "ordinary_native_launches": 2,
        }
        data = analyze(path, evidence)
        assert data["steps"][0]["kernel_count"] == 3
        assert data["steps"][0]["generated_count"] == 2
        assert data["native_launch_api"]["samples_us"] == [0.3, 0.2]
        assert "cudaDeviceSynchronize" not in data["api"]
        assert data["steps"][0]["wall_us"] == 9
        assert data["steps"][0]["generated_dispatch_union_us"] == 3
        assert data["exclusive_ranges"]["selection_dispatch"]["sum_us_per_step"] == 0.15
        assert data["exclusive_ranges"]["compiled_dispatch"]["sum_us_per_step"] == 1.24
        assert data["queue_after_api"]["samples_us"] == [0.0, 0.0, 0.1]
        assert union_ns([(1, 5), (2, 4), (5, 9), (12, 13)]) == 9
        try:
            analyze(path, dict(evidence, steps=2))
        except AssertionError:
            pass
        else:
            raise AssertionError("missing NVTX steps must fail")
        try:
            analyze(path, dict(evidence, ordinary_native_launches=3))
        except AssertionError:
            pass
        else:
            raise AssertionError("native census mismatch must fail")
        try:
            install(tmp)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing precision source must fail")
    # The fixed runner must reject application success without a fresh report,
    # even when a stale report already exists. No GPU process is launched here.
    from stage5 import attribution_run
    import sys
    import subprocess

    name = "systems-cubecl-deep-all-api"
    with tempfile.TemporaryDirectory() as tmp:
        fake_root = Path(tmp)
        cache = fake_root / "cache/attribution"
        cache.mkdir(parents=True)
        (cache / (name + ".nsys-rep")).write_bytes(b"stale test report")
        with (
            patch.object(attribution_run, "ROOT", fake_root),
            patch.object(attribution_run, "PROJECT", fake_root),
            patch.object(
                sys, "argv", ["attribution_run", "--only", "systems", "--all-api-probe"]
            ),
            patch.object(
                subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
            ),
        ):
            try:
                attribution_run.main()
            except RuntimeError as error:
                assert "no report" in str(error)
            else:
                raise AssertionError("stale Nsight report must fail")
        assert (cache / (name + ".nsys-rep")).is_file()
    result = {
        "measured_step_scope_excludes_reset_sync_and_kernels": True,
        "exclusive_intervals_subtract_nested_union_once": True,
        "queue_clamped_when_kernel_starts_before_API_returns": True,
        "missing_steps_and_native_mismatch_rejected": True,
        "stale_report_rejected": True,
        "vendor_driver_API_excluded_from_generated_population": True,
        "harness_sha256": sha(__file__),
    }
    (ROOT / "artifacts/attribution/analysis-controls.json").write_text(
        json.dumps(result, indent=2)
    )
    print("Attribution analysis controls PASS", flush=True)


if __name__ == "__main__":
    main()
