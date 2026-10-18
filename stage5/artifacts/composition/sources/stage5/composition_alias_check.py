"""Exact two-form extent admission controls before mixed-graph measurements."""

import copy

from stage5.composition_trial import ROOT, OUT, initialize, read, write, role, sha


def main():
    torch = initialize()
    from stage5.composition_runtime import compile_kernel
    from stage5.layout_qualify import fixture
    from stage5.layout_trial import tensor_sha, check

    assert read(OUT / "gpu.json")["complete"]
    assert not (OUT / "alias-gpu.json").exists()
    old = read(ROOT / "stage5/artifacts/column-layout/post-review-gpu.json")
    plan = read(ROOT / "stage5/artifacts/reuse-first-layout/qualification-plan.json")["cases"]
    rows, controls = [], []
    for k in old["kernels"]:
        incoming = copy.deepcopy(k["semantic"])
        incoming["numel"]["args"].reverse()
        assert role(incoming) == "column"
        dispatch = compile_kernel(incoming, 0, "both")
        assert dispatch.semantic == dispatch.kernel.semantic == incoming
        dtype = incoming["buffers"][0]["dtype"]
        cases = [c for c in plan if c["family"] == "bias_column" and c["dtype"] == dtype]
        for case, special in [(c, False) for c in cases] + [(cases[0], True)]:
            fn, args, stores = fixture(torch, case, special)
            expected = fn(*args)
            buf = torch.full((expected.numel() + 16,), 42, dtype=expected.dtype, device="cuda")
            y = buf[8:-8].view_as(expected)
            inputs = [tensor_sha(t) for t in args]
            dispatch(case["scalars"], *args, y)
            check(torch, y, expected)
            label = case["label"] + ("-special" if special else "")
            target = next(r for r in old["rows"] if r["label"] == label)
            assert tensor_sha(y) == target["output_sha256"]
            assert inputs == target["input_sha256"] == [tensor_sha(t) for t in args]
            assert all(bool((s[:8] == 42).all() & (s[-8:] == 42).all()) for s in stores + [buf])
            rows.append(dict(label=label, bitwise=True, canaries=True, original_descriptor_preserved=True))
        for field in ("numel", "ops", "buffers"):
            bad = copy.deepcopy(incoming)
            if field == "numel":
                bad["numel"]["op"] = "add"
            elif field == "ops":
                next(o for o in bad["ops"] if o["op"] == "add")["op"] = "mul"
            else:
                bad["buffers"][0]["offset"]["value"] = 1
            assert role(bad) == "unadmitted"
            try:
                compile_kernel(bad, 0, "both")
            except ValueError:
                pass
            else:
                raise AssertionError("new alias admits an unqualified field change")
            controls.append(dict(dtype=dtype, field=field, rejected=True))
        incoming["numel"]["args"].clear()
        assert len(dispatch.semantic["numel"]["args"]) == 2
    assert len(rows) == 24 and len(controls) == 9
    write("alias-gpu.json", dict(complete=True, rows=rows, controls=controls, source_sha256=sha(__file__)))
    print("PASS24 alternate-extent cases/9 exact-admission controls", flush=True)


if __name__ == "__main__":
    main()
