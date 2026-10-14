"""Qualify the long-row counterexample without changing policy or its gate.

The CPU model reproduces the actual FP32 per-unit chain and shared binary tree.
Alternative chain counts below are numerical components only, not new GPU plans.
A successful diagnostic explicitly retains the failing eager-comparison result.
"""

import ctypes as C
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def emulate(a, streams=1):
    import numpy as np

    padded = np.pad(a, ((0, 0), (0, (-a.shape[1]) % (256 * streams))))
    values = padded.reshape(a.shape[0], -1, streams, 256)
    partial = np.zeros((a.shape[0], streams, 256), dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        for i in range(values.shape[1]):
            partial = np.add(partial, values[:, i], dtype=np.float32)
        for stride in (2**k for k in reversed(range(streams.bit_length() - 1))):
            partial[:, :stride] = np.add(
                partial[:, :stride], partial[:, stride : 2 * stride], dtype=np.float32
            )
        shared = partial[:, 0].copy()
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            shared[:, :stride] = np.add(
                shared[:, :stride], shared[:, stride : 2 * stride], dtype=np.float32
            )
    return shared[:, 0]


def qualify():
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache/precision/order")
    from stage1 import evidence

    evidence.guard()
    import torch
    import numpy as np
    from stage1.overlay import install

    install()
    from stage5.backend import compile_backend, TRACE
    from stage5.attribution import artifacts, inventory, profile

    native = C.CDLL(str(PROJECT / "stage0/libstage0_launcher.so"))
    native.stage0_kernel_launches.restype = C.c_uint64
    graphs = []
    compiler = partial(compile_backend, tuning="baseline")

    def backend(gm, inputs):
        graphs.append(str(gm.graph))
        return compiler(gm, inputs)

    compiled = torch.compile(
        lambda x: x.sum(-1), backend=backend, fullgraph=True, dynamic=True
    )
    old = json.loads(
        (ROOT / "artifacts/attribution/cancellation-counterexample.json").read_text()
    )
    torch.manual_seed(old["seed"])
    for shape in old["rng_history"]:
        x = torch.randn(shape, device="cuda")
        out = compiled(x)
    eager = x.sum(-1)
    fp64 = x.double().sum(-1)
    assert out.tolist() == old["cubecl"], "counterexample topology/data drift"
    arrays = x.cpu().numpy()
    cpu = emulate(arrays)
    assert np.array_equal(cpu, out.cpu().numpy()), "CPU/GPU ordering differs"
    gate = bool(torch.all((out - eager).abs() <= 2e-5 + 2e-5 * eager.abs()))
    assert not gate, "known failure disappeared; requalify before updating evidence"
    alternatives = []
    for count in (2, 4, 8, 16):
        a = emulate(arrays, count).astype(np.float64)
        alternatives.append(
            {
                "private_chains": count,
                "gpu_implemented": False,
                "values": a.tolist(),
                "abs_error_fp64": abs(a - fp64.cpu().numpy()).tolist(),
                "passes_recorded_eager_gate": bool(
                    np.all(
                        abs(a - eager.cpu().numpy())
                        <= 2e-5 + 2e-5 * abs(eager.cpu().numpy())
                    )
                ),
            }
        )
    records = []
    for seed in (6142, 9241, 11, 101):
        torch.manual_seed(seed)
        a = torch.randn((8, 65539), device="cuda")
        got = compiled(a)
        model = emulate(a.cpu().numpy())
        assert np.array_equal(model, got.cpu().numpy())
        ref, truth = a.sum(-1), a.double().sum(-1)
        records.append(
            {
                "seed": seed,
                "shape": list(a.shape),
                "cpu_gpu_bit_exact": True,
                "eager_gate_passes": bool(
                    torch.all((got - ref).abs() <= 2e-5 + 2e-5 * ref.abs())
                ),
                "gpu_abs_error_fp64": (got.double() - truth).abs().tolist(),
                "eager_abs_error_fp64": (ref.double() - truth).abs().tolist(),
            }
        )
    # Special-value classifications retain ordinary addition, not Kahan's
    # unguarded inf-inf correction. This is still the unchanged tree.
    a = torch.zeros((5, 65539), device="cuda")
    a[0].fill_(-0.0)
    a[1, 0] = float("nan")
    a[2, 0] = float("inf")
    a[3, 0] = -float("inf")
    a[4, 0], a[4, 256] = float("inf"), -float("inf")
    got, ref = compiled(a), a.sum(-1)
    model = torch.from_numpy(emulate(a.cpu().numpy())).to("cuda")
    torch.testing.assert_close(got, ref, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(model, got, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(got[got == 0]), torch.signbit(ref[ref == 0]))
    output = ROOT / "artifacts/precision/reduction-order.json"
    measured = profile(torch, lambda: compiled(x), native, output)
    assert len(measured["kernels"]) == 1 and measured["kernels"][0]["name"].startswith(
        "cubecl5_"
    )
    result = {
        "qualification_complete": True,
        "existing_eager_gate_passes": gate,
        "classification": "FP32 accumulation-order accuracy limitation; no miscompile demonstrated",
        "gate_rtol": 2e-5,
        "gate_atol": 2e-5,
        "tolerance_unchanged": True,
        "cubecl": out.tolist(),
        "aten_float32": eager.tolist(),
        "fp64": fp64.tolist(),
        "cpu_gpu_bit_exact": True,
        "original_rng_history": old["rng_history"],
        "neighboring_seeds": records,
        "special_value_classes_and_zero_sign": True,
        "cpu_only_alternative_chains": alternatives,
        "profile": measured,
        "graphs": graphs,
        "schedule_replay": TRACE,
        "artifacts": artifacts(),
        "environment": inventory(torch, "cubecl"),
        "source_sha256": {
            p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()
            for p in (
                "stage0/src/descriptor3.rs",
                "stage0/src/descriptor5.rs",
                "stage5/reduction_order_checks.py",
            )
        },
    }
    return result


def main():
    output = ROOT / "artifacts/precision/reduction-order.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"accepted": False, "qualification_complete": False}
    try:
        result.update(qualify())
        result["accepted"] = True
    except Exception as error:
        result.update(
            failure=repr(error),
            traceback=traceback.format_exc(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        # A failed diagnostic must not overwrite the last completed qualification.
        output = output.with_suffix(".failed.json")
        raise
    finally:
        output.write_text(json.dumps(result, indent=2))
    print(
        "Ordering qualification PASS; original eager tolerance still FAILS", flush=True
    )


if __name__ == "__main__":
    main()
