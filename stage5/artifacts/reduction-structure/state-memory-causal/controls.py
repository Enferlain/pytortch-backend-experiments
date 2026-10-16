"""CPU-only rejection controls; never modify retained captures or launch GPU work."""

import copy
import ctypes as C
import json
from pathlib import Path
import tempfile

from stage5.reduction_memory_analyze import analyze
from stage5.reduction_state_counter_analyze import sha


def main():
    root = Path(__file__).resolve().parent
    original = root / "baseline-before-00"
    evidence = []
    with tempfile.TemporaryDirectory(prefix="stage5-memory-admission-") as temporary:
        base = Path(temporary)
        fixture = base / root.name
        fixture.mkdir()
        (base / "state-counters").symlink_to(root.parent / "state-counters")
        for name in [
            "predeclared-plan.json",
            "predeclared-plan.sha256",
            "capture-source-sha256.json",
            "sampler.so",
            "idle.so",
        ]:
            (fixture / name).symlink_to(root / name)
        capture = fixture / original.name
        capture.mkdir()
        for file in original.iterdir():
            if file.name != "result.json":
                (capture / file.name).symlink_to(file)
        result_path = capture / "result.json"
        valid = json.loads((original / "result.json").read_text())
        result_path.write_text(json.dumps(valid))
        accepted = analyze(fixture)
        assert accepted["captures"] == 2 and not accepted["complete"]
        evidence.append(
            {"control": "valid two-capture fixture with pending schedule", "pass": True}
        )

        def reject(name, change):
            mutated = copy.deepcopy(valid)
            change(mutated)
            result_path.write_text(json.dumps(mutated))
            try:
                analyze(fixture)
            except (AssertionError, ValueError, KeyError):
                evidence.append({"control": name, "rejected": True})
            else:
                raise AssertionError(f"Control was admitted: {name}")

        reject("nonzero sampler cleanup", lambda d: d.update(sampler_close_status=1))
        reject(
            "incomplete plus explicit failure",
            lambda d: d.update(complete=False, error="fixture"),
        )
        reject("foreign output bits", lambda d: d.update(output_sha256="0" * 64))
        reject("foreign plan", lambda d: d.update(plan_sha256="0" * 64))
        reject(
            "unplanned conditions",
            lambda d: d["conditions"]["wr_same"]["steps"][1].__setitem__(1, "B"),
        )
        reject(
            "alias A and B",
            lambda d: d["pointer_spans"].__setitem__("B", d["pointer_spans"]["memory"]),
        )
        reject(
            "incorrect prepared count",
            lambda d: d["prepared_actions"][0].__setitem__("count", 1),
        )
        reject(
            "incorrect prepared address",
            lambda d: d["prepared_actions"][0].__setitem__("base_ptr", 0),
        )
        reject(
            "incorrect prepared geometry",
            lambda d: d["prepared_actions"][0].__setitem__("grid", [81, 1, 1]),
        )
        reject(
            "counter metric substitution",
            lambda d: d.update(pm_metrics=["lts__t_sectors.sum"]),
        )
        reject("counter interval substitution", lambda d: d.update(pm_interval_ns=2000))
        reject(
            "sampler binary substitution", lambda d: d.update(sampler_sha256="0" * 64)
        )
        reject(
            "callback binary substitution",
            lambda d: d.update(idle_library_sha256="0" * 64),
        )
        reject("counter-clock substitution", lambda d: d.update(clock_method="fixture"))
        reject(
            "recorded latency substitution",
            lambda d: d["captures"][0].__setitem__("target_us", 1),
        )
        reject(
            "prequeue lead substitution",
            lambda d: d["captures"][0].__setitem__("prequeue_lead_us", 0),
        )
        reject(
            "recorded gap substitution",
            lambda d: d["captures"][0].__setitem__("target_gap_us", 0),
        )

        mutated = copy.deepcopy(valid)
        mutated["captures"][0]["clock_anchors"][0]["wall_before_ns"] -= 1000000
        result_path.write_text(json.dumps(mutated))
        unqualified = analyze(fixture)
        assert len(unqualified["counter_window_failures"]) == 1
        assert unqualified["qualified_comparison_captures"] == 0
        assert unqualified["captures"] == 2 and not unqualified["complete"]
        assert original.name.rsplit("-", 1)[0] in unqualified["unqualified_contrasts"]
        evidence.append(
            {
                "control": "wide clock anchor parks complete contrast without hiding latency or shrinking guard",
                "pass": True,
            }
        )

        for kind in ["overflow", "extra GPU work"]:
            mutated = copy.deepcopy(valid)
            row = mutated["captures"][0]
            if kind == "overflow":
                name = f"r{row['round']}-{row['mode']}.pm.csv.decode.json"
                link = capture / name
                link.unlink()
                decode = json.loads((original / name).read_text())
                decode["overflow"] = 1
                link.write_text(json.dumps(decode))
            else:
                trace = json.loads(Path(row["trace"]).read_text())
                kernel = next(
                    e for e in trace["traceEvents"] if e.get("cat") == "kernel"
                )
                trace["traceEvents"].append(copy.deepcopy(kernel))
                fake = capture / "synthetic-extra-work.json"
                fake.write_text(json.dumps(trace))
                row["trace"] = str(fake)
                row["trace_sha256"] = sha(fake)
            result_path.write_text(json.dumps(mutated))
            try:
                analyze(fixture)
            except AssertionError:
                evidence.append({"control": kind, "rejected": True})
            else:
                raise AssertionError(kind)
            if kind == "overflow":
                link.unlink()
                link.symlink_to(original / name)

    library = C.CDLL(str(root / "idle.so"))
    library.state_idle_enqueue.argtypes = [
        C.c_void_p,
        C.c_uint64,
        C.POINTER(C.c_uint64),
    ]
    timestamps = (C.c_uint64 * 2)()
    assert library.state_idle_enqueue(None, 0, None) == 1
    assert library.state_idle_enqueue(None, 2500001, timestamps) == 1
    evidence.append(
        {"control": "FFI null and excessive delay rejected before CUDA", "pass": True}
    )
    (root / "controls.json").write_text(json.dumps(evidence, indent=2))
    print("Passed", len(evidence), "CPU-only admission controls")


if __name__ == "__main__":
    main()
