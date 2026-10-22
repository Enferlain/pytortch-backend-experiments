"""Structural, output and selection equivalence: Stage 5 backend vs this package.

Each corpus case runs in a fresh worker process per implementation and policy,
with separate empty artifact caches. A hook on each side's ``compile_kernel``
(the call every generated region makes) records regions in call order, so the
per-case list keeps multiplicity: two regions sharing one artifact stay two
records. Each record holds the region's role, descriptor digest, chosen plan,
the IR/CUDA/PTX digests of every compiled candidate, the candidate sets, and
the selections (key data without compiler provenance, which differs by design).

Usage (from the repository root, isolated venv):

    python -m cubecl_torch.testing.equivalence run --out results.json \
        [--names a,b] [--policies baseline,first_use] [--jobs 4]

``first_use`` runs replace both benchmarkers with the same deterministic timing
table, so selection is comparable; real first-use winners are not compared.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IMPLEMENTATIONS = ("stage5", "package")
POLICIES = ("baseline", "first_use")
FILES = ("cubecl.ir", "kernel.cu", "kernel.ptx")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def injected_ms(identity, perturb=None):
    """Deterministic per-plan kernel time; ``perturb`` makes one plan slowest."""
    if identity == perturb:
        return [1000.0] * 5
    return [1.0 + int(digest(identity.encode())[:8], 16) / 2**32] * 5


# ---------------------------------------------------------------------------
# Worker: runs inside a fresh interpreter.


def _stage5_setup(cache):
    os.environ.setdefault("STAGE0_CARGO_PROFILE", "release")
    from stage1 import evidence

    evidence.guard()
    import torch

    # The harness installed the Stage 1 overlay once; concurrent re-installs
    # would race on its fixed directory, so workers only put it on the path.
    overlay = REPO / "stage1/overlay/torch"
    if not (overlay / "_inductor").is_dir():
        raise RuntimeError(
            "Stage 1 overlay missing; run the harness, which installs it"
        )
    torch.__path__.insert(0, str(overlay))
    from stage5 import artifact, timing
    from stage5 import runtime as tuning
    from stage5.backend import compile_backend

    # Fixed cache root under stage5/: redirect it to an empty directory whose
    # source files are symlinks, so identity hashing still reads the originals.
    artifact.compiler_identity(0)
    root = Path(cache) / "stage5-root"
    root.mkdir(parents=True, exist_ok=True)
    for path in (REPO / "stage5").iterdir():
        if path.suffix in (".py", ".json"):
            try:
                (root / path.name).symlink_to(path)
            except FileExistsError:
                pass  # created by a concurrent worker
    artifact.ROOT = tuning.ROOT = root
    return torch, compile_backend, tuning, timing, lambda c: c.kernel.plan


def _package_setup(cache):
    os.environ["CUBECL_TORCH_CACHE_DIR"] = str(Path(cache) / "package-root")
    from cubecl_torch.runtime import guard

    guard.install()
    import torch

    import cubecl_torch.inductor as ci

    ci.install()
    from torch._inductor import config

    # Stage 5's numerics: eager precision emulation (which also disables FMA)
    # and correctly rounded division.
    config.emulate_precision_casts = True
    config.eager_numerics.division_rounding = True
    from cubecl_torch.inductor import timing, tuning

    return torch, ci.compile_backend, tuning, timing, lambda c: c.plan


def _artifact_digests(directory):
    return {name: digest((Path(directory) / name).read_bytes()) for name in FILES}


def _region(kernel, plan_of, selections):
    semantic = kernel.semantic
    compiled = {
        plan.identity: {"plan": plan.identity, **_artifact_digests(c.directory)}
        for plan, c in kernel.compiled.items()
    }
    last = kernel.last
    last_plan = plan_of(last).identity if last is not None else None
    accepted, rejected = set(), set()
    for plans, reasons in kernel._legality.values():
        accepted.update(p.identity for p in plans)
        rejected.update(canonical(r) for r in reasons)
    context = getattr(kernel, "reduction_context", None)
    return {
        "role": ("reduction" if semantic["schema"] == 3 else "pointwise")
        + (":" + context["role"] if context else ""),
        "descriptor": digest(canonical(semantic).encode()),
        "policy": kernel.policy,
        "plan": last_plan,
        "artifact": compiled.get(last_plan),
        "compiled": sorted(compiled.values(), key=canonical),
        "candidates": sorted(accepted),
        "rejected": sorted(rejected),
        "selections": sorted(
            (canonical(s) for s in selections.get(id(kernel), ())),
        ),
    }


def _tensor_digest(t):
    t = t.detach().contiguous()
    return {
        "dtype": str(t.dtype),
        "shape": list(t.shape),
        "sha256": digest(
            t.view(-1).view(__import__("torch").uint8).cpu().numpy().tobytes()
        )
        if t.numel()
        else "",
    }


def run_case(torch, backend, case_fixture, case):
    torch.manual_seed(6142)
    f = case_fixture(case)
    compiled = torch.compile(f.fn, backend=backend, fullgraph=True, dynamic=True)
    out = compiled(*f.args)
    if f.optimizer:
        out.backward()
        f.optimizer.step()
    elif f.upstream is not None:
        out.backward(f.upstream)
    torch.cuda.synchronize()
    outputs = {"out": out}
    for i, x in enumerate(f.args):
        if getattr(x, "grad", None) is not None:
            outputs[f"grad{i}"] = x.grad
    if f.model is not None:
        for name, p in f.model.named_parameters():
            outputs["param." + name] = p
    return {k: v.detach().clone() for k, v in outputs.items()}


def worker(impl, policy, names, cache, perturb=None, out=None):
    setup = _stage5_setup if impl == "stage5" else _package_setup
    torch, compile_backend, tuning, timing, plan_of = setup(cache)
    from functools import partial

    from stage5.corpus import cases, fixture

    torch._dynamo.config.recompile_limit = 128
    torch._dynamo.config.suppress_errors = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    calls, selections = [], {}
    original_compile = tuning.compile_kernel

    def recording_compile(*args, **kwargs):
        kernel = original_compile(*args, **kwargs)
        calls.append(kernel)
        return kernel

    tuning.compile_kernel = recording_compile
    original_select = tuning.AutotunedKernel.select

    def recording_select(self, key, key_data, plans, rejected, *args):
        winner = original_select(self, key, key_data, plans, rejected, *args)
        selections.setdefault(id(self), []).append(
            {
                "key_data": {
                    k: v for k, v in (key_data or {}).items() if k != "identity"
                },
                "winner": winner.identity,
            }
        )
        return winner

    tuning.AutotunedKernel.select = recording_select
    if policy == "first_use":

        def benchmark_kernel_gpu(self, kernel):
            return {
                "kernel_ms": injected_ms(plan_of_kernel(kernel), perturb),
                "enqueue_interval_ms": [],
                "profile_attempt_counts": [],
            }

        timing.KernelBenchmarker.benchmark_kernel_gpu = benchmark_kernel_gpu

    def plan_of_kernel(loaded):
        # Stage 5 kernels carry their plan; package kernels are found by object.
        if hasattr(loaded, "plan"):
            return loaded.plan.identity
        for kernel in calls:
            for plan, c in kernel.compiled.items():
                if c.kernel is loaded:
                    return plan.identity
        raise LookupError("benchmarked kernel not found")

    by_name = {c.name: c for c in cases()}
    results = []
    for name in names:
        calls.clear()
        selections.clear()
        torch._dynamo.reset()
        started = time.perf_counter()
        nondeterministic = {}
        try:
            backend = partial(compile_backend, tuning=policy)
            tensors = run_case(torch, backend, fixture, by_name[name])
            outputs = {k: _tensor_digest(v) for k, v in tensors.items()}
            if policy == "baseline":
                # Float atomics with colliding indices are order-nondeterministic
                # even within one implementation: detect, keep for tolerance check.
                again = run_case(torch, backend, fixture, by_name[name])
                for key, value in again.items():
                    if _tensor_digest(value) != outputs[key]:
                        path = Path(out).with_suffix(f".{key}.pt")
                        torch.save(tensors[key].cpu(), path)
                        nondeterministic[key] = str(path)
            error = None
        except Exception as exc:  # noqa: BLE001 - recorded and compared
            outputs, error = (
                None,
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}",
            )
        results.append(
            {
                "case": name,
                "impl": impl,
                "policy": policy,
                "regions": [_region(k, plan_of, selections) for k in calls],
                "outputs": outputs,
                "nondeterministic": nondeterministic,
                "error": error,
                "gil": sys._is_gil_enabled(),
                "seconds": time.perf_counter() - started,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Comparison.


def compare_case(a, b, *, outputs):
    """Differences between two case results (Stage 5 first)."""
    problems = []
    # The corpus holds no expected rejections: an error on either side fails,
    # so two equally broken runs never compare as identical.
    for side, r in (("stage5", a), ("package", b)):
        if r["error"] is not None:
            problems.append(f"{side} error: {r['error']}")
    ra, rb = a["regions"], b["regions"]
    if len(ra) != len(rb):
        problems.append(f"region count {len(ra)} vs {len(rb)}")
    for i, (x, y) in enumerate(zip(ra, rb)):
        for field in sorted(set(x) | set(y)):
            if x.get(field) != y.get(field):
                problems.append(f"region {i} {field} differs")
    if outputs and (a["outputs"] is None) != (b["outputs"] is None):
        problems.append("outputs missing")
    elif outputs and a["outputs"] is not None:
        # Tolerance only where Stage 5 itself is nondeterministic; a race the
        # package alone shows is a divergence.
        loose = set(a.get("nondeterministic", {}))
        for key in sorted(set(b.get("nondeterministic", {})) - loose):
            problems.append(f"output {key} nondeterministic in the package only")
        for key in sorted(set(a["outputs"]) | set(b["outputs"])):
            if key not in loose:
                if a["outputs"].get(key) != b["outputs"].get(key):
                    problems.append(f"output {key} differs bitwise")
            elif not _close(a, b, key):
                problems.append(
                    f"nondeterministic output {key} differs beyond tolerance"
                )
    if a["gil"] or b["gil"]:
        problems.append("GIL enabled")
    return problems


def _close(a, b, key):
    import torch

    paths = [r.get("nondeterministic", {}).get(key) for r in (a, b)]
    if not all(paths):
        # Repeated on one side by chance: only a bitwise match is checkable.
        return a["outputs"].get(key) == b["outputs"].get(key)
    x, y = (torch.load(p) for p in paths)
    try:
        torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-5, equal_nan=True)
    except AssertionError:
        return False
    return True


def compare(results):
    """Pair Stage 5 and package results per (case, policy); report per case."""
    index = {(r["impl"], r["case"], r["policy"]): r for r in results}
    report = []
    for (impl, case, policy), stage5 in sorted(index.items()):
        if impl != "stage5":
            continue
        package = index.get(("package", case, policy))
        problems = (
            ["package result missing"]
            if package is None
            else compare_case(stage5, package, outputs=policy == "baseline")
        )
        report.append(
            {
                "case": case,
                "policy": policy,
                "regions": len(stage5["regions"]),
                "nondeterministic_outputs": sorted(
                    set(stage5.get("nondeterministic", {}))
                    | set((package or {}).get("nondeterministic", {}))
                ),
                "identical": not problems,
                "problems": problems,
            }
        )
    return report


# ---------------------------------------------------------------------------
# Orchestration.


def _launch(impl, policy, names, cache, out, perturb):
    command = [
        sys.executable,
        "-m",
        "cubecl_torch.testing.equivalence",
        "worker",
        "--impl",
        impl,
        "--policy",
        policy,
        "--names",
        ",".join(names),
        "--cache",
        str(cache),
        "--out",
        str(out),
    ]
    if perturb:
        command += ["--perturb", perturb]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHON_GIL",)}
    env["TORCHINDUCTOR_CACHE_DIR"] = str(Path(cache) / f"inductor-{out.stem}")
    proc = subprocess.run(
        command, cwd=REPO, env=env, capture_output=True, text=True, check=False
    )
    if proc.returncode or not out.is_file():
        return [
            {
                "case": n,
                "impl": impl,
                "policy": policy,
                "regions": [],
                "outputs": None,
                "error": "worker failed: " + proc.stderr[-2000:],
                "gil": False,
                "seconds": 0,
            }
            for n in names
        ]
    return json.loads(out.read_text())


def install_stage1_overlay():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch\nfrom stage1.overlay import install\ninstall()",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
    )


def run(names, policies, *, jobs=2, work=None, perturb=None, impls=IMPLEMENTATIONS):
    if "stage5" in impls:
        install_stage1_overlay()
    work = Path(work or tempfile.mkdtemp(prefix="cubecl-equivalence."))
    tasks = []
    for policy in policies:
        for impl in impls:
            for name in names:
                # Separate empty caches per case: a shared tuning cache would let
                # one case reuse another's selection depending on worker order.
                cache = work / f"{impl}-{policy}" / name
                cache.mkdir(parents=True, exist_ok=True)
                out = work / "results" / f"{impl}-{policy}-{name}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                tasks.append((impl, policy, [name], cache, out, perturb))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for batch in pool.map(lambda t: _launch(*t), tasks):
            results.extend(batch)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--impl", choices=IMPLEMENTATIONS, required=True)
    w.add_argument("--policy", choices=POLICIES, required=True)
    w.add_argument("--names", required=True)
    w.add_argument("--cache", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--perturb")
    r = sub.add_parser("run")
    r.add_argument("--names", default="")
    r.add_argument("--policies", default=",".join(POLICIES))
    r.add_argument("--jobs", type=int, default=2)
    r.add_argument("--work")
    r.add_argument("--perturb")
    r.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(REPO))
    if args.command == "worker":
        results = worker(
            args.impl,
            args.policy,
            args.names.split(","),
            args.cache,
            args.perturb,
            args.out,
        )
        Path(args.out).write_text(json.dumps(results, indent=1))
        return 0
    from stage5.corpus import cases

    names = [n for n in args.names.split(",") if n] or [c.name for c in cases()]
    results = run(
        names,
        args.policies.split(","),
        jobs=args.jobs,
        work=args.work,
        perturb=args.perturb,
    )
    report = compare(results)
    Path(args.out).write_text(
        json.dumps({"report": report, "results": results}, indent=1)
    )
    identical = sum(r["identical"] for r in report)
    if not report:
        print("no (case, policy) pairs ran")
        return 1
    print(f"{identical}/{len(report)} (case, policy) pairs identical")
    for r in report:
        if not r["identical"]:
            print(r["case"], r["policy"], "; ".join(r["problems"][:5]))
    return 0 if identical == len(report) else 1


if __name__ == "__main__":
    sys.exit(main())
