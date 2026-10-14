"""Accuracy, physical routing and real Inductor regression for long sums."""

import ctypes as C
import hashlib
import math
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUTPUT = ROOT / "artifacts/accumulation"


def emulate_compensated(a):
    """Explicit FP32 model of private TwoSum and the compensated shared tree."""
    import numpy as np

    def add(a, c, b):
        t = np.add(a, b, dtype=np.float32)
        z = np.subtract(t, a, dtype=np.float32)
        error = np.add(
            np.subtract(a, np.subtract(t, z, dtype=np.float32), dtype=np.float32),
            np.subtract(b, z, dtype=np.float32),
            dtype=np.float32,
        )
        return t, np.where(
            np.isfinite(t), np.add(c, error, dtype=np.float32), np.float32(0)
        )

    padded = np.pad(a, ((0, 0), (0, (-a.shape[1]) % 256)))
    blocks = padded.reshape(a.shape[0], -1, 256)
    high = np.zeros((a.shape[0], 256), dtype=np.float32)
    low = np.zeros_like(high)
    with np.errstate(invalid="ignore", over="ignore"):
        for block in blocks.transpose(1, 0, 2):
            high, low = add(high, low, block)
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            h, c = add(high[:, :stride], low[:, :stride], high[:, stride : 2 * stride])
            low[:, :stride] = np.add(c, low[:, stride : 2 * stride], dtype=np.float32)
            high[:, :stride] = h
        return np.add(high[:, 0], low[:, 0], dtype=np.float32)


def run():
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "cache/accumulation/inductor")
    from stage1 import evidence

    evidence.guard()
    import numpy as np
    import torch
    from stage1.overlay import install

    install()
    from stage5 import artifact, runtime
    from stage5.attribution import inventory, profile
    from stage5.backend import compile_backend, TRACE
    from stage5.corpus import rms
    from stage5.fixtures import captured
    from stage5.plan import PhysicalCandidate
    from stage5.legality import candidates, signature
    from stage0.bridge import native_path

    native = C.CDLL(str(PROJECT / "stage0/libstage0_launcher.so"))
    native.stage0_kernel_launches.restype = C.c_uint64
    identity = artifact.compiler_identity(0)
    plan = PhysicalCandidate("shared_tree_compensated")
    records, representatives = [], []

    def retain(k, label):
        target = OUTPUT / "representatives" / label
        target.mkdir(parents=True, exist_ok=True)
        for name in (
            "descriptor.json",
            "semantic.json",
            "physical-plan.json",
            "cubecl.ir",
            "kernel.cu",
            "kernel.ptx",
            "manifest.json",
            "nvrtc.json",
            "resources.json",
        ):
            shutil.copyfile(k.directory / name, target / name)
        representatives.append(str(target.relative_to(PROJECT)))

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        sem = captured("row_sum", str(dtype).removeprefix("torch."))
        k = artifact.CompiledKernel(sem, 0, plan.to_dict(), identity)
        assert k.kernel.manifest["dynamic_shared_bytes"] == 2048
        for n in (
            0,
            1,
            31,
            255,
            256,
            257,
            1023,
            1024,
            1025,
            4095,
            4096,
            4097,
            65539,
            131071,
        ):
            torch.manual_seed(7301 + n)
            x = torch.randn((3, n), device="cuda", dtype=dtype)
            # Empty eager tensors choose stride (1,1); this captured component
            # explicitly declares (N,1). Preserve that contract without copies.
            x = x.as_strided((3, n), (n, 1))
            backing = torch.full((5,), 123, device="cuda", dtype=dtype)
            y = backing[1:4]
            k((n, 3), x, y)
            runtime.compare(y, x.sum(-1), sem)
            assert torch.equal(
                backing[[0, 4]], torch.full((2,), 123, device="cuda", dtype=dtype)
            )
            if dtype == torch.float32:
                assert np.array_equal(
                    emulate_compensated(x.cpu().numpy()), y.cpu().numpy()
                )
            records.append(
                {
                    "case": "component_tail",
                    "dtype": str(dtype),
                    "n": n,
                    "canaries": True,
                    "cpu_gpu_bit_exact": dtype == torch.float32,
                }
            )
        # Ordinary special-value addition behavior, including infinities with
        # finite correction state, opposite infinities, and both zero signs.
        x = torch.zeros((6, 65539), device="cuda", dtype=dtype)
        x[0].fill_(-0.0)
        x[1, 0] = float("nan")
        x[2, 0] = float("inf")
        x[3, 0] = -float("inf")
        x[4, 0], x[4, 256] = float("inf"), -float("inf")
        x[5, 0], x[5, 256] = 1.0, 0.001
        x[5, 512] = float("inf")
        y = torch.empty(6, device="cuda", dtype=dtype)
        k((65539, 6), x, y)
        ref = x.sum(-1)
        torch.testing.assert_close(y, ref, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(torch.signbit(y[y == 0]), torch.signbit(ref[ref == 0]))
        records.append({"case": "special_classes", "dtype": str(dtype), "exact": True})
        retain(k, "component-" + str(dtype).removeprefix("torch."))

    # Deliberate lost-unit and mixed-magnitude cases in one private lane.
    k = artifact.CompiledKernel(
        captured("row_sum", "float32"), 0, plan.to_dict(), identity
    )
    adversarial = torch.zeros(5, 65539, device="cuda")
    adversarial[0, [0, 256, 512]] = torch.tensor([1e8, 1.0, -1e8], device="cuda")
    adversarial[1] = -adversarial[0]
    adversarial[2, [0, 256, 512]] = torch.tensor([2**24, 0.25, -(2**24)], device="cuda")
    adversarial[3, :1024] = 2**-149
    adversarial[4, ::2] = 1024
    adversarial[4, 1::2] = -1024
    out = torch.empty(5, device="cuda")
    k((65539, 5), adversarial, out)
    truth = adversarial.double().sum(-1).float()
    torch.testing.assert_close(out, truth, rtol=0, atol=0)
    assert np.array_equal(
        emulate_compensated(adversarial.cpu().numpy()), out.cpu().numpy()
    )
    records.append(
        {
            "case": "adversarial_exact_fp64",
            "values": out.tolist(),
            "cpu_gpu_bit_exact": True,
        }
    )

    # Frozen counterexample is retained. Ordinary first_use policy must fix it,
    # including short-first graph reuse and repeated crossings of policy buckets.
    graphs = []

    def backend(gm, inputs):
        graphs.append(str(gm.graph))
        return compile_backend(gm, inputs)

    compiled = torch.compile(
        lambda x: x.sum(-1), backend=backend, fullgraph=True, dynamic=True
    )
    old = json.loads(
        (ROOT / "artifacts/attribution/cancellation-counterexample.json").read_text()
    )
    torch.manual_seed(old["seed"])
    for shape in old["rng_history"]:
        x = torch.randn(shape, device="cuda")
        got = compiled(x)
        torch.testing.assert_close(got, x.sum(-1), rtol=2e-5, atol=2e-5)
    model = emulate_compensated(x.cpu().numpy())
    assert np.array_equal(model, got.cpu().numpy())
    records.append(
        {
            "case": "original_short_first",
            "output": got.tolist(),
            "aten": x.sum(-1).tolist(),
            "fp64": x.double().sum(-1).tolist(),
            "old_output": old["cubecl"],
            "cpu_gpu_bit_exact": True,
        }
    )
    measured = profile(torch, lambda: compiled(x), native, OUTPUT / "original.json")
    assert len(measured["kernels"]) == 1 and measured["kernels"][0]["name"].startswith(
        "cubecl5_"
    )
    records.append({"case": "original_one_kernel", "profile": measured})
    # Warm both numerical-policy sides and the aligned vector proof class.
    # First entry into a new short-row alignment class may compile candidates;
    # compatible long rows themselves must reuse their existing artifact.
    for n in (4096, 4097):
        z = torch.randn(8, n, device="cuda")
        torch.testing.assert_close(compiled(z), z.sum(-1), rtol=2e-5, atol=2e-5)
    before_graphs = len(graphs)
    before_compiles = artifact.COMPILER_INVOCATIONS
    for seed, n in (
        (10017, 65539),
        (7301, 131071),
        (7301, 1048576),
        (11, 4096),
        (11, 4097),
        (101, 65539),
        (11, 257),
    ):
        torch.manual_seed(seed)
        x = torch.randn((8, n), device="cuda")
        graph_count = len(graphs)
        compile_count = artifact.COMPILER_INVOCATIONS
        got = compiled(x)
        torch.testing.assert_close(got, x.sum(-1), rtol=2e-5, atol=2e-5)
        records.append(
            {
                "case": "dynamic_policy_reuse",
                "seed": seed,
                "n": n,
                "new_graphs": len(graphs) - graph_count,
                "new_compiles": artifact.COMPILER_INVOCATIONS - compile_count,
            }
        )
    assert (
        len(graphs) == before_graphs
        and artifact.COMPILER_INVOCATIONS == before_compiles
    ), [r for r in records if r["case"] == "dynamic_policy_reuse"]
    records.append(
        {"case": "dynamic_artifact_reuse", "new_graphs": 0, "new_compiles": 0}
    )
    # Preserve a genuine reference limitation, not an acceptance waiver: ATen's
    # error alone exceeds the fixed eager-relative gate for this finite sum.
    torch.manual_seed(30003)
    z = torch.randn(8, 65539, device="cuda")
    precise = compiled(z)
    eager, truth = z.sum(-1), z.double().sum(-1)
    eager_gate = bool(torch.all((precise - eager).abs() <= 2e-5 + 2e-5 * eager.abs()))
    assert not eager_gate, "reference counterexample changed; requalify explicitly"
    torch.testing.assert_close(precise.double(), truth, rtol=2e-5, atol=2e-5)
    records.append(
        {
            "case": "reference_gate_counterexample",
            "seed": 30003,
            "eager_gate_passes": eager_gate,
            "cubecl": precise.tolist(),
            "aten": eager.tolist(),
            "fp64": truth.tolist(),
            "cpu_fsum_row1": math.fsum(z[1].cpu().tolist()),
            "classification": "ATen order error exceeds eager-relative tolerance; not waived",
        }
    )

    # Decision cache must distinguish both sides of the numerical boundary;
    # neither deterministic mode nor a hot timing cache can choose an old chain.
    sem = captured("row_sum", "float32")
    k = artifact.CompiledKernel(sem, 0, plan.to_dict(), identity)
    facts = []
    for n in (4096, 4097, 65539):
        x = torch.randn(8, n, device="cuda")
        y = torch.empty(8, device="cuda")
        accepted, rejected = candidates(sem, (n, 8), (x, y), k.kernel.resources)
        if n > 4096:
            assert accepted == [plan]
        else:
            assert plan not in accepted
        facts.append(signature(sem, (n, 8), (x, y)))
    assert facts[0] != facts[1]
    try:
        candidates(
            sem, (65539, 8), (x, y), {**k.kernel.resources, "device_max_threads": 128}
        )
    except ValueError as error:
        assert "numerical policy" in str(error) and "topology" in str(error)
    else:
        raise AssertionError("insufficient topology accepted for compensated plan")
    records.append(
        {"case": "numerical_policy_target_rejection", "before_compilation": True}
    )
    torch.use_deterministic_algorithms(True)
    try:
        x = torch.randn(8, 65539, device="cuda")
        got = compiled(x)
        torch.testing.assert_close(got, x.sum(-1), rtol=2e-5, atol=2e-5)
    finally:
        torch.use_deterministic_algorithms(False)
    records.append(
        {"case": "capability_precedes_timing", "long_only_compensated": True}
    )

    # Force one actual Rust/C++/NVRTC compile without deleting normal caches or
    # changing numerical options; the resulting PTX must match the warm artifact.
    before = artifact.COMPILER_INVOCATIONS
    fresh_identity = {**identity, "diagnostic_partition": uuid.uuid4().hex}
    cold = artifact.CompiledKernel(
        captured("row_sum", "float32"), 0, plan.to_dict(), fresh_identity
    )
    assert artifact.COMPILER_INVOCATIONS - before == 1 and not cold.cache_hit
    assert (cold.directory / "kernel.ptx").read_bytes() == (
        k.directory / "kernel.ptx"
    ).read_bytes()
    torch.manual_seed(10017)
    z = torch.randn(8, 65539, device="cuda")
    destination = torch.empty(8, device="cuda")
    before = native.stage0_kernel_launches()
    cold((65539, 8), z, destination)
    assert native.stage0_kernel_launches() - before == 1
    assert np.array_equal(
        emulate_compensated(z.cpu().numpy()), destination.cpu().numpy()
    )
    torch.testing.assert_close(destination, z.sum(-1), rtol=2e-5, atol=2e-5)
    retain(cold, "fresh-compiler")
    records.append(
        {
            "case": "actual_fresh_compiler",
            "compiler_invocations": 1,
            "native_launches": 1,
            "ptx_equal_warm": True,
        }
    )

    # Actual column strides, reduction->epilogue and lazy cat masking.
    for label, fn, warm, large in (
        ("column", lambda z: z.sum(0), (31, 8), (65539, 8)),
        ("mean_epilogue", lambda z: z.mean(-1) + 1, (8, 31), (8, 65539)),
        (
            "lazy_mask",
            lambda z: torch.cat([z, z[:, :1]], dim=1).sum(-1),
            (8, 31),
            (8, 65539),
        ),
        ("transpose", lambda z: z.sum(-1), (8, 31), (8, 65539)),
    ):
        c = torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)
        for shape in (warm, large):
            x = (
                torch.randn(shape[1], shape[0], device="cuda").T
                if label == "transpose"
                else torch.randn(shape, device="cuda")
            )
            got = c(x)
            torch.testing.assert_close(got, fn(x), rtol=2e-5, atol=2e-5)
        p = profile(torch, lambda: c(x), native, OUTPUT / (label + ".json"))
        assert all(v["name"].startswith("cubecl5_") for v in p["kernels"])
        if label in ("column", "mean_epilogue", "lazy_mask"):
            assert len(p["kernels"]) == 1
        records.append(
            {
                "case": label,
                "shape": list(x.shape),
                "stride": list(x.stride()),
                "profile": p,
            }
        )

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        c = torch.compile(rms, backend=backend, fullgraph=True, dynamic=True)
        for n in (31, 65539):
            x = torch.randn(3, n, device="cuda", dtype=dtype, requires_grad=True)
            refx = x.detach().clone().requires_grad_()
            upstream = torch.randn_like(x)
            got = c(x)
            ref = rms(refx)
            sem = {"schema": 3}
            runtime.compare(got, ref, sem)
            grad = torch.autograd.grad(got, x, upstream)[0]
            refgrad = torch.autograd.grad(ref, refx, upstream)[0]
            runtime.compare(grad, refgrad, sem)
        # New saved tensors for AOT donation; references stay outside capture.
        forward_profile = profile(
            torch, lambda: c(x), native, OUTPUT / (str(dtype) + "-forward.json")
        )
        assert len(forward_profile["kernels"]) == 2
        assert all(v["name"].startswith("cubecl5_") for v in forward_profile["kernels"])
        records.append(
            {
                "case": "normalization_forward",
                "dtype": str(dtype),
                "profile": forward_profile,
            }
        )
        y = c(x)
        p = profile(
            torch,
            lambda: torch.autograd.grad(y, x, upstream),
            native,
            OUTPUT / (str(dtype) + "-backward.json"),
        )
        assert len(p["kernels"]) == 1 and p["kernels"][0]["name"].startswith("cubecl5_")
        records.append(
            {
                "case": "normalization_backward",
                "dtype": str(dtype),
                "arbitrary_upstream": True,
                "profile": p,
            }
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(3000000)
        pressure_input = torch.randn(8, 65539, device="cuda")
        ref = pressure_input.sum(-1)
        from stage5.diagnostics import detailed_trace

        before = native.stage0_kernel_launches()
        with detailed_trace(True):
            got = compiled(pressure_input)
        launch = artifact.EVENTS[-1]
        assert launch["pointers"] == [pressure_input.data_ptr(), got.data_ptr()]
        assert launch["stream"] == stream.cuda_stream
        assert native.stage0_kernel_launches() - before == 1
        del pressure_input
        pressure = [torch.empty(8, 65539, device="cuda") for _ in range(16)]
        done = torch.cuda.Event()
        done.record()
    torch.cuda.current_stream().wait_event(done)
    torch.testing.assert_close(got, ref, rtol=2e-5, atol=2e-5)
    records.append(
        {
            "case": "nondefault_stream_pressure",
            "stream": stream.cuda_stream,
            "allocations": len(pressure),
            "original_pointers_and_current_stream": True,
        }
    )

    # Rust refuses a policy attached to the wrong topology/family before IR.
    for label, change in (
        ("bad_count", {"private_accumulators": 3}),
        ("bad_block", {"workgroup": [128, 1, 1]}),
        ("vector_compensation", {"vector_width": 2}),
    ):
        packet = {
            "schema": 5,
            "semantic": captured("row_sum", "float32"),
            "physical_plan": plan.to_dict(),
            "entrypoint": "cubecl5_invalid",
        }
        packet["physical_plan"].update(change)
        directory = ROOT / "cache/accumulation/invalid" / label
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "descriptor.json"
        path.write_text(json.dumps(packet))
        p = subprocess.run(
            [str(native_path("cubecl-stage0")), str(directory), "cpp", str(path)],
            capture_output=True,
            text=True,
        )
        assert p.returncode and not (directory / "cubecl.ir").exists()
        records.append({"case": label, "before_ir": True, "error": p.stderr})
    # Geometry rejection is native, before CUDA enqueue/counters, even when a
    # caller bypasses the Python plan validator. Exactly 2048 has been executed.
    before = native.stage0_kernel_launches()
    for dims, shared in (([256, 1, 1], 2049), ([256, 1, 1], 1536), ([128, 1, 1], 2048)):
        error = C.create_string_buffer(2048)
        rc = k.kernel.native.stage5_launch(
            k.kernel.handle,
            (C.c_uint64 * 3)(0, 0, 0),
            3,
            1,
            (C.c_uint * 3)(*dims),
            shared,
            1,
            torch.cuda.current_stream().cuda_stream,
            error,
            len(error),
        )
        assert rc == -1 and b"unsupported Stage5 launch geometry" in error.value
    assert native.stage0_kernel_launches() == before
    records.append(
        {"case": "native_compensated_scratch_boundary", "rejected_before_enqueue": True}
    )
    for value in runtime.CACHE.values():
        for compiled_plan in value.compiled.values():
            if compiled_plan.kernel.plan == plan:
                retain(compiled_plan, "inductor-" + compiled_plan.entrypoint)
    evidence.state("after_accumulation_checks")
    return {
        "accepted": True,
        "checks": records,
        "graphs": graphs,
        "schedule_replay": TRACE,
        "representatives": representatives,
        "environment": inventory(torch, "cubecl"),
        "compiler_invocations": artifact.COMPILER_INVOCATIONS,
        "native_launches": native.stage0_kernel_launches(),
        "tolerance_unchanged": True,
        "source_sha256": {
            p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()
            for p in (
                "stage0/src/descriptor3.rs",
                "stage0/src/descriptor5.rs",
                "stage0/launcher.c",
                "stage5/plan.py",
                "stage5/legality.py",
                "stage5/runtime.py",
                "stage5/accumulation_checks.py",
            )
        },
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        result = run()
    except Exception as error:
        (OUTPUT / "checks.failed.json").write_text(
            json.dumps(
                {
                    "accepted": False,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
        )
        raise
    (OUTPUT / "checks.json").write_text(json.dumps(result, indent=2))
    print("Long-reduction accuracy and Inductor checks PASS", flush=True)


if __name__ == "__main__":
    main()
