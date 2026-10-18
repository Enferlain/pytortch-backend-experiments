"""Bounded reuse-first transpose investigation; production code stays unchanged."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stage5/artifacts/reuse-first-transpose"
os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "stage5/cache/reuse-first-transpose")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(t):
    import torch

    return hashlib.sha256(
        t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def fused_contiguous(x):
    return (x.sin() + 1).contiguous()


def copy_contiguous(x):
    return x.contiguous()


def capture():
    from stage1 import evidence

    evidence.guard()
    from stage1.overlay import install

    install()
    import torch
    from torch._inductor import config
    from stage5 import artifact
    from stage5.backend import TRACE, compile_backend
    from stage5.corpus import sin
    from stage5.diagnostics import detailed_trace

    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "capture.json").exists(), "preserve original capture"
    torch.set_num_threads(1)
    rows = []
    for dtype in ("float32", "float16", "bfloat16"):
        for mode, fn in (
            ("view_fused", sin),
            ("contiguous_fused", fused_contiguous),
            ("contiguous_copy", copy_contiguous),
        ):
            torch._dynamo.reset()
            with config.patch({"fx_graph_cache": False}):
                compiled = torch.compile(
                    fn,
                    backend=lambda g, i: compile_backend(g, i, tuning="first_use"),
                    fullgraph=True,
                    dynamic=True,
                )
                for m, n in ((33, 257), (256, 513), (256, 512)):
                    label = f"{mode}-{dtype}-{m}x{n}"
                    torch.manual_seed(537 + m + n)
                    x = torch.randn(
                        (m, n), device="cuda", dtype=getattr(torch, dtype)
                    ).T
                    expected = fn(x)
                    start, ts = len(artifact.EVENTS), len(TRACE)
                    with detailed_trace():
                        actual = compiled(x)
                    torch.cuda.synchronize()
                    rtol, atol = {
                        "float32": (2e-6, 2e-6),
                        "float16": (0.002, 0.001),
                        "bfloat16": (0.016, 0.008),
                    }[dtype]
                    torch.testing.assert_close(
                        actual, expected, rtol=rtol, atol=atol, equal_nan=True
                    )
                    launches = [
                        e for e in artifact.EVENTS[start:] if e["event"] == "launch"
                    ]
                    assert len(launches) == 1, (label, launches)
                    event = next(
                        e
                        for e in reversed(artifact.EVENTS)
                        if e["event"] == "artifact"
                        and e["entrypoint"] == launches[0]["entrypoint"]
                    )
                    src = Path(event["directory"])
                    dst = OUT / "baseline" / src.name
                    dst.mkdir(parents=True, exist_ok=True)
                    for name in (
                        "descriptor.json",
                        "semantic.json",
                        "physical-plan.json",
                        "manifest.json",
                        "cubecl.ir",
                        "kernel.cu",
                        "kernel.ptx",
                        "resources.json",
                        "nvrtc.json",
                    ):
                        shutil.copyfile(src / name, dst / name)
                    rows.append(
                        dict(
                            label=label,
                            mode=mode,
                            dtype=dtype,
                            base_shape=[m, n],
                            input_shape=list(x.shape),
                            input_stride=list(x.stride()),
                            output_shape=list(actual.shape),
                            output_stride=list(actual.stride()),
                            eager_stride=list(expected.stride()),
                            correctness=True,
                            input_sha256=tensor_sha(x),
                            output_sha256=tensor_sha(actual),
                            artifact=str(dst.relative_to(OUT)),
                            launches=launches,
                            trace=TRACE[ts:],
                        )
                    )
                    print(label, "PASS", x.stride(), "->", actual.stride(), flush=True)
    write(
        "capture.json",
        dict(
            complete=True,
            records=rows,
            compiler_identity=artifact.compiler_identity(0),
            independence=evidence.inventory(),
            git_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            source_sha256=sha(__file__),
        ),
    )


if __name__ == "__main__":
    capture()
