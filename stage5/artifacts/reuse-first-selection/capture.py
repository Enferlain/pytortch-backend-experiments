"""Capture existing RMS inference descriptors; no candidate implementation/timing."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
sys.path.insert(0, str(PROJECT))
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(PROJECT / "stage5/cache/reuse-first")
os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from stage1 import evidence

    evidence.guard()
    from stage1.overlay import install

    install()
    import torch

    from stage5 import artifact
    from stage5.backend import TRACE, compile_backend
    from stage5.corpus import rms
    from stage5.diagnostics import detailed_trace
    from torch._inductor import config

    torch.set_num_threads(1)
    device = torch.cuda.current_device()
    records, copied = [], {}
    for dtype in ("float32", "float16", "bfloat16"):
        # Existing baseline only: this determines the semantic adapter boundary.
        # It supplies no candidate qualification or current selection/timing claim.
        def backend(gm, inputs):
            return compile_backend(gm, inputs, tuning="baseline")

        with config.patch({"fx_graph_cache": False}):
            compiled = torch.compile(rms, backend=backend, fullgraph=True, dynamic=True)
            for b, n in ((64, 1024), (64, 255), (16, 4099)):
                name = f"rms_inference-{dtype}-{b}x{n}"
                torch.manual_seed(8020 + b + n)
                x = torch.randn(
                    (b, n), dtype=getattr(torch, dtype), device=f"cuda:{device}"
                )
                expected = rms(x)
                trace_start, event_start = len(TRACE), len(artifact.EVENTS)
                with detailed_trace():
                    actual = compiled(x)
                torch.cuda.synchronize()
                rtol, atol = {
                    "float32": (2e-5, 2e-5),
                    "float16": (0.002, 0.001),
                    "bfloat16": (0.016, 0.008),
                }[dtype]
                torch.testing.assert_close(
                    actual, expected, rtol=rtol, atol=atol, equal_nan=True
                )
                events = artifact.EVENTS[event_start:]
                launches = [e for e in events if e["event"] == "launch"]
                assert len(launches) == 1, (name, launches)
                copies = []
                matches = [
                    e
                    for e in artifact.EVENTS
                    if e["event"] == "artifact"
                    and e["entrypoint"] == launches[0]["entrypoint"]
                ]
                assert matches, name
                for e in matches[-1:]:
                    source = Path(e["directory"])
                    target = OUT / "kernels" / source.name
                    if source.name not in copied:
                        target.mkdir(parents=True, exist_ok=True)
                        for filename in (
                            "descriptor.json",
                            "semantic.json",
                            "physical-plan.json",
                            "manifest.json",
                            "cubecl.ir",
                            "kernel.cu",
                            "kernel.ptx",
                            "nvrtc.json",
                            "resources.json",
                            "integrity.json",
                        ):
                            shutil.copyfile(source / filename, target / filename)
                        copied[source.name] = {
                            "directory": str(target.relative_to(OUT)),
                            "files": {
                                p.name: sha(p) for p in target.iterdir() if p.is_file()
                            },
                        }
                    copies.append(copied[source.name])
                raw = lambda t: (
                    t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                )
                records.append(
                    {
                        "case": name,
                        "rtol": rtol,
                        "atol": atol,
                        "equal_nan": True,
                        "correctness": True,
                        "policy": "baseline",
                        "shape": list(x.shape),
                        "dtype": dtype,
                        "seed": 8020 + b + n,
                        "input_sha256": hashlib.sha256(raw(x)).hexdigest(),
                        "output_sha256": hashlib.sha256(raw(actual)).hexdigest(),
                        "schedule_replay": TRACE[trace_start:],
                        "launches": launches,
                        "artifacts": copies,
                    }
                )
                print(name, "PASS", flush=True)
    evidence.state("after_reuse_selection")
    (OUT / "baseline-capture.json").write_text(
        json.dumps(
            {
                "scope": "Existing baseline semantic/codegen capture; no new mechanism or timings",
                "harness_sha256": sha(Path(__file__)),
                "records": records,
                "independence": evidence.inventory(),
                "compiler_identity": artifact.compiler_identity(device),
                "git_head": subprocess.check_output(
                    ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
                ).strip(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
