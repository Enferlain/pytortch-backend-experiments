"""First-use experimental tuning; generic Inductor cache and event benchmark.

Selection state lives under ``<cache root>/tuning``. Its identity covers the
compiler provenance plus every selection-policy source in this package, so a
policy edit re-runs selection without invalidating compiled artifacts.
"""

import copy
import hashlib
import json
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch._inductor import config
from torch._inductor.codecache import PersistentCache

from cubecl_torch.compiler import build, descriptor
from cubecl_torch.compiler.nvrtc import DEFAULT
from cubecl_torch.compiler.plan import PhysicalCandidate, shape_bucket
from cubecl_torch.inductor import kernels
from cubecl_torch.inductor.diagnostics import tracing
from cubecl_torch.inductor.legality import (
    LONG_REDUCTION_THRESHOLD,
    candidates,
    exact_reduction,
    signature,
    welford_only,
    welford_reduction,
    welford_weight_output,
)
from cubecl_torch.inductor.reduction_policy import (
    SANITY_VERSION,
    qualified_for_runtime,
    sanity_limit,
)
from cubecl_torch.inductor.reduction_policy import (
    VERSION as NUMERICAL_POLICY,
)
from cubecl_torch.inductor.timing import KernelBenchmarker
from cubecl_torch.runtime import cache as artifact_cache

# Ordinary float sums are compared with a scale-aware runtime corruption guard
# (bead 4z8.29): the largest difference from the baseline may reach the
# elementwise tolerance at the output's largest magnitude. It separates summation
# reordering (about 1e-5 of that scale) from gross codegen failures (a dropped or
# doubled term, of the size of a term). It is not a bound on summation error: an
# output made only of cancelling sums keeps max|baseline| near 0. Numerical
# admissibility stays a separate, offline policy (reduction_policy, band tests).
SUM_GUARD_VERSION = "sum-scale-v1"
# A candidate failing its comparison is excluded and the selection continues
# with the rest; tests set CUBECL_TORCH_STRICT_CANDIDATES=1 to raise instead.
STRICT_ENV = "CUBECL_TORCH_STRICT_CANDIDATES"
EXCLUDED = "runtime_corruption_guard"
PACKAGE = Path(__file__).resolve().parent
EVENTS = kernels.EVENTS
# Selection policy: changing any of these re-runs selection, not compilation.
POLICY_SOURCES = (
    "tuning.py",
    "legality.py",
    "timing.py",
    "reduction_policy.py",
    "reference512_plan.py",
    "reference512_templates.json",
    "reference512_oracle.py",
    "reference512_model.py",
)
# A test hook: ``(plan_identity, round) -> list[float]`` replacing measured
# kernel times with deterministic ones. None measures on the GPU.
INJECTED_TIMINGS = None
CACHE = {}
LOCK = threading.RLock()


@dataclass(frozen=True)
class Choice:
    plan: PhysicalCandidate

    def hash_key(self):
        return self.plan.identity


class TimingCache(PersistentCache):
    def __init__(self, path, identity):
        # Own compiler/device provenance rather than CacheBase's Triton key.
        self.path = path
        self.system = identity

    def get_local_cache_path(self):
        return self.path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def frozen(semantic):
    if welford_reduction(semantic):
        # Welford's in-band plan that needs no vector alignment (welford-reductions 5),
        # also for a sum fused with it (bead 4z8.32).
        return PhysicalCandidate("plane_shared", plane_width=32, private_accumulators=4)
    return PhysicalCandidate(
        "shared_tree_reduction_baseline"
        if semantic["schema"] == 3
        else "scalar_pointwise_baseline"
    )


def welford_fallback(semantic, plans):
    """Welford keeps its widest legal vector plane plan (pair-tree lanes, in
    Triton's band also for 16-bit rows), else the four-accumulator plan."""
    if welford_only(semantic):
        vectors = [
            p for p in plans if p.strategy == "plane_shared" and p.vector_width > 1
        ]
        if vectors:
            return max(vectors, key=lambda p: (p.vector_width, p.workgroup[0]))
    return frozen(semantic)


BITS = {
    torch.float32: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
}


def references(node):
    """The values ``node`` reads: string operands in ``args`` lists, and
    ``value`` leaves of indirect index trees, through bodies and indices."""
    if isinstance(node, list):
        return [r for item in node for r in references(item)]
    if not isinstance(node, dict):
        return []
    found = [a for a in node.get("args", ()) if isinstance(a, str)]
    if node.get("op") == "value" and isinstance(node.get("id"), str):
        found.append(node["id"])
    return (
        found
        + [
            r
            for key, item in node.items()
            if key not in ("id", "args")
            for r in references(item)
        ]
        + references([a for a in node.get("args", ()) if not isinstance(a, str)])
    )


def output_semantic(semantic, buffer):
    """``semantic`` cut down to output ``buffer``'s store and what it depends
    on, so each output is compared by the kind of value it holds. A kernel
    with one output is its own projection. An output of a reduction kernel
    that depends on no reduction (a per-element copy) is compared as a
    pointwise value."""
    outputs = [b for b in semantic["buffers"] if b["access"] == "write"]
    if len(outputs) == 1:
        return semantic
    ops = semantic["ops"]
    defined = {op["id"]: op for op in ops if "id" in op}

    def stores_in(region):
        for op in region:
            if op["op"] == "store" and op["buffer"] == buffer:
                yield op
            yield from stores_in(op.get("body", ()))

    stores = list(stores_in(ops))
    needed, pending = set(), references(stores)
    while pending:
        name = pending.pop()
        if name in defined and name not in needed:
            needed.add(name)
            pending += references(defined[name])
    kept = [op for op in ops if op.get("id") in needed or op in stores]
    projected = {**semantic, "ops": kept}
    if semantic["schema"] == 3 and not any(op["op"] == "reduction" for op in kept):
        projected["schema"] = 4
    return projected


def compare(actual, expected, semantic, plan=None):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    dtype = actual.dtype
    if exact_reduction(semantic) or welford_weight_output(semantic):
        # Max/min and Welford weights (element counts) are order-independent:
        # a candidate that differs from the baseline beyond NaN payload is a
        # codegen bug, not reassociation.
        if dtype not in BITS:
            assert torch.equal(actual, expected)
            return
        nan = torch.isnan(actual)
        assert torch.equal(nan, torch.isnan(expected))
        assert torch.equal(
            actual[~nan].view(BITS[dtype]), expected[~nan].view(BITS[dtype])
        )
        return
    rtol, atol = tolerances(dtype, semantic)
    if welford_reduction(semantic):
        # Rows holding an infinity are non-finite in every plan, but whether
        # inf or NaN depends on where empty lanes meet it (as in Triton).
        finite = torch.isfinite(expected)
        assert torch.equal(torch.isfinite(actual), finite)
        # A coarse corruption guard, not an accuracy oracle: variance of rows
        # with a large mean is ill-conditioned, so correct plans legitimately
        # differ by far more than a sum tolerance (2e-5 relative already at
        # mean 1e3, std 1). Accuracy is qualified by the band test.
        if finite.any():
            delta = (actual[finite].double() - expected[finite].double()).abs()
            scale = expected[finite].double().abs().max().item()
            assert delta.max().item() <= 0.01 * scale + atol, (
                "Welford candidate exceeded the corruption guard"
            )
        return
    if plan is not None and qualified_for_runtime(semantic, plan):
        # Offline-qualified reassociation is not vetoed by the old tree's finite
        # result. These differences are diagnostics, never runtime input tuning.
        finite = torch.isfinite(actual) & torch.isfinite(expected)
        delta = (actual[finite] - expected[finite]).abs()
        largest = delta.max().item() if delta.numel() else 0.0
        magnitude = expected[finite].abs().max().item() if delta.numel() else 0.0
        limit = sanity_limit(str(dtype), magnitude)
        assert largest <= limit, "qualified reduction exceeded coarse corruption guard"
        EVENTS.append(
            {
                "event": "reduction_comparison",
                "policy": NUMERICAL_POLICY,
                "plan": plan.to_dict(),
                "finite_max_abs_difference": largest,
                "corruption_guard": SANITY_VERSION,
                "sanity_limit": limit,
            }
        )
    elif semantic["schema"] == 3 and dtype.is_floating_point:
        # Integer and bool outputs stay exact below: no order changes them.
        largest, guard = sum_guard(actual, expected, rtol, atol)
        assert largest <= guard, (
            f"sum candidate exceeded the corruption guard: {largest:.3g} > {guard:.3g}"
        )
        EVENTS.append(
            {
                "event": "reduction_comparison",
                "policy": NUMERICAL_POLICY,
                "plan": plan.to_dict() if plan is not None else None,
                "finite_max_abs_difference": largest,
                "corruption_guard": SUM_GUARD_VERSION,
                "sanity_limit": guard,
            }
        )
    else:
        torch.testing.assert_close(
            actual, expected, rtol=rtol, atol=atol, equal_nan=True
        )
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    infinite = torch.isinf(actual) & torch.isinf(expected)
    assert torch.equal(
        torch.signbit(actual[infinite]), torch.signbit(expected[infinite])
    )
    finite_zero = (actual == 0) & (expected == 0)
    if semantic["schema"] != 3:
        assert torch.equal(
            torch.signbit(actual[finite_zero]), torch.signbit(expected[finite_zero])
        )


def tolerances(dtype, semantic):
    """``(rtol, atol)`` of a candidate's comparison with the baseline."""
    if dtype == torch.bfloat16:
        return 0.016, 0.008
    if dtype == torch.float16:
        return 0.002, 0.001
    return (2e-5, 2e-5) if semantic["schema"] == 3 else (2e-6, 2e-6)


def sum_guard(actual, expected, rtol, atol):
    """``(largest finite difference, guard)``; the guard is ``atol + rtol *
    max|expected|``, never tighter than the elementwise tolerance it replaces."""
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    if not finite.any():
        return 0.0, atol
    a, e = actual[finite].double(), expected[finite].double()
    return (a - e).abs().max().item(), atol + rtol * e.abs().max().item()


def exclusion(actual, expected, semantic, error):
    """The selection record of a candidate that failed its comparison."""
    record = {"correctness": False, "excluded": EXCLUDED}
    if actual.dtype.is_floating_point:
        finite = torch.isfinite(actual) & torch.isfinite(expected)
        delta = (actual[finite].double() - expected[finite].double()).abs()
        record["max_abs_difference"] = delta.max().item() if delta.numel() else None
        if semantic["schema"] == 3:
            rtol, atol = tolerances(actual.dtype, semantic)
            record["guard"] = sum_guard(actual, expected, rtol, atol)[1]
            record["guard_version"] = SUM_GUARD_VERSION
    lines = [line for line in str(error).splitlines() if line.strip()]
    record["reason"] = lines[0] if lines else type(error).__name__
    return record


class AutotunedKernel:
    def __init__(
        self, semantic, device, policy, *, reduction_context=None, numerics=DEFAULT
    ):
        semantic = copy.deepcopy(semantic)
        from cubecl_torch.inductor.reference512_plan import validate

        self.reduction_context = validate(semantic, reduction_context)
        descriptor.check_buffers(
            semantic.get("buffers") or [],
            semantic.get("ops", []),
            reduction=semantic.get("schema") == 3,
        )
        self.semantic, self.device, self.policy = semantic, device, policy
        self.numerics = numerics
        from cubecl_torch.compiler.column_plan import semantic_reason as column_reason

        self._column_template = column_reason(semantic) is None
        self.identity = json.loads(
            canonical(build.identity(kernels.device_index(device)))
        )
        self.identity["tuning_source"] = hashlib.sha256(
            b"".join((PACKAGE / name).read_bytes() for name in POLICY_SOURCES)
            + (PACKAGE.parent / "compiler" / "plan.py").read_bytes()
            + (PACKAGE.parent / "compiler" / "final_plan.py").read_bytes()
            + (PACKAGE.parent / "compiler" / "column_plan.py").read_bytes()
            + (PACKAGE.parent / "compiler" / "column_templates.json").read_bytes()
        ).hexdigest()
        self.identity["numerical_policy"] = NUMERICAL_POLICY
        self.identity["math"] = build.math(numerics)
        self.compiled = {}
        self.selections = {}
        self.last = None
        self._local_keys = {}
        self._legality = {}
        self._bypass_recorded = False
        self.baseline_plan = frozen(semantic)
        effects = semantic.get("effects", {})
        self.effect_bypass = bool(
            effects.get("rng")
            or effects.get("mutation")
            or any(o["op"] == "indirect" for o in semantic["ops"])
        )
        self.selection_hits = 0

    @property
    def kernel(self):
        return self.last.kernel

    @property
    def directory(self):
        return self.last.directory

    def compile(self, plan):
        if plan not in self.compiled:
            self.compiled[plan] = kernels.CompiledKernel(
                self.semantic,
                kernels.device_index(self.device),
                plan.to_dict(),
                numerics=self.numerics,
            )
        return self.compiled[plan]

    def __call__(self, scalars, *tensors):
        with LOCK, torch.cuda.device(self.device):
            if self.policy.endswith("_diagnostic"):
                # Explicit measured alternatives never enter automatic selection.
                from cubecl_torch.compiler.indexing import evaluate

                plan = self.baseline_plan
                if (
                    self.semantic["schema"] == 3
                    and evaluate(self.semantic["reduction_extent"], scalars)
                    > LONG_REDUCTION_THRESHOLD
                ):
                    plan = (
                        PhysicalCandidate(
                            "plane_shared_reference512",
                            (512, 1, 1),
                            plane_width=32,
                            private_accumulators=4,
                            private_merge="adjacent",
                        )
                        if self.policy == "reference512_diagnostic"
                        else PhysicalCandidate(
                            "plane_shared_load_batch",
                            plane_width=32,
                            private_accumulators=4,
                            private_merge="adjacent"
                            if self.policy == "adjacent_merge_diagnostic"
                            else "halving",
                        )
                        if self.policy
                        in ("load_batch_diagnostic", "adjacent_merge_diagnostic")
                        else PhysicalCandidate(
                            "plane_shared", plane_width=32, private_accumulators=4
                        )
                        if self.policy == "combined_diagnostic"
                        else PhysicalCandidate("shared_tree_compensated")
                    )
                from cubecl_torch.inductor.legality import structural_reason

                target = self.compile(self.baseline_plan).kernel.resources
                why = structural_reason(plan, self.semantic, scalars, tensors, target)
                if why:
                    raise ValueError("unsupported diagnostic physical plan: " + why)
                self.last = self.compile(plan)
                return self.last(scalars, *tensors)
            baseline = self.compile(self.baseline_plan)
            bypass = (
                self.policy == "baseline"
                or baseline.kernel.geometry(scalars) == 0
                or self.effect_bypass
            )
            if bypass:
                self.last = baseline
                if not self._bypass_recorded:
                    EVENTS.append(
                        {
                            "event": "tuning_bypass",
                            "reason": "explicit baseline/effectful/checked-indirect",
                            "winner": baseline.plan.to_dict(),
                        }
                    )
                    self._bypass_recorded = True
                return baseline(scalars, *tensors)
            facts = signature(
                self.semantic, scalars, tensors, column_template=self._column_template
            )
            if self.reduction_context is not None:
                from cubecl_torch.inductor.reference512_plan import admits

                facts = (*facts, admits(self.reduction_context, scalars))
            if facts not in self._legality:
                self._legality[facts] = candidates(
                    self.semantic,
                    scalars,
                    tensors,
                    baseline.kernel.resources,
                    reduction_context=self.reduction_context,
                )
            plans, rejected = self._legality[facts]
            bucket = shape_bucket(self.semantic, scalars)
            # Catalog is immutable for this kernel and finite proof class;
            # rehashing every candidate on each warm call is redundant.
            local_key = (tuple(bucket), facts)
            key = self._local_keys.get(local_key)
            if key is None or key not in self.selections:
                key_data = {
                    "identity": self.identity,
                    "semantic": self.semantic,
                    "bucket": bucket,
                    "runtime_facts": json.loads(canonical(facts)),
                    "catalog": [p.to_dict() for p in plans],
                    "reduction_context": self.reduction_context,
                }
                key = hashlib.sha256(canonical(key_data).encode()).hexdigest()
                self._local_keys[local_key] = key
            else:
                key_data = None
            if key not in self.selections:
                if len(plans) == 1:
                    winner = plans[0]
                    EVENTS.append(
                        {
                            "event": "tuning_bypass",
                            "key": key,
                            "reason": "one legal physical candidate; numerical policy precedes timing",
                            "winner": winner.to_dict(),
                        }
                    )
                else:
                    winner = self.select(
                        key, key_data, plans, rejected, scalars, tensors, baseline
                    )
                self.selections[key] = winner
            else:
                self.selection_hits += 1
                if tracing():
                    EVENTS.append({"event": "selection_hit", "key": key})
            self.last = self.compile(self.selections[key])
            return self.last(scalars, *tensors)

    def select(self, key, key_data, plans, rejected, scalars, tensors, baseline):
        directory = artifact_cache.cache_root() / "tuning" / key
        directory.mkdir(parents=True, exist_ok=True)
        choices = [Choice(p) for p in plans]
        details = directory / "selection.json"
        cache = TimingCache(directory / "timings.json", self.identity)
        with artifact_cache.file_lock(directory / ".lock"):
            rows = None
            compensated = PhysicalCandidate("shared_tree_compensated")
            fallback = (
                compensated
                if any(p.strategy == "plane_shared_reference512" for p in plans)
                else welford_fallback(self.semantic, plans)
            )
            if fallback not in plans and compensated in plans:
                # Long-row sums admit only compensated plans (256 and 512
                # units, wide-reduction-workgroups 3.3): the frozen one.
                fallback = compensated
            winner = fallback
            if fallback not in plans:
                raise ValueError(
                    "physical selection has no qualified fallback candidate"
                )
            began = time.perf_counter()

            def benchmark(items):
                nonlocal rows, winner
                # Every output gets an isolated destination, compared on its own.
                outputs = [
                    i
                    for i, b in enumerate(self.semantic["buffers"])
                    if b["access"] == "write"
                ]
                scratch = {
                    i: torch.empty_strided(
                        tensors[i].shape,
                        tensors[i].stride(),
                        device=tensors[i].device,
                        dtype=tensors[i].dtype,
                    )
                    for i in outputs
                }
                args = tuple(scratch.get(i, t) for i, t in enumerate(tensors))
                baseline.kernel(scalars, *args)
                expected = {i: s.clone() for i, s in scratch.items()}
                semantics = {i: output_semantic(self.semantic, i) for i in outputs}
                compiled = [(choice, self.compile(choice.plan)) for choice in items]
                # Compile all before timing, then validate every candidate on
                # isolated destinations. Pure kernels neither mutate inputs nor
                # consume RNG; user output is not touched until final launch.
                before = torch.cuda.get_rng_state().clone()
                model_expected = None
                excluded = {}
                strict = os.environ.get(STRICT_ENV) == "1"
                for choice, c in compiled:
                    c.kernel(scalars, *args)
                    try:
                        if choice.plan.strategy == "plane_shared_reference512":
                            from cubecl_torch.inductor.reference512_oracle import (
                                check,
                                partials,
                            )

                            if model_expected is None:
                                model_expected = partials(tensors[0])
                            # reference512 kernels have one output.
                            check(scratch[outputs[-1]], model_expected)
                            EVENTS.append(
                                {
                                    "event": "reference512_model_check",
                                    "key": key,
                                    "policy": NUMERICAL_POLICY,
                                    "plan": choice.plan.to_dict(),
                                    "oracle": "frozen-reference512-cpu-v1",
                                }
                            )
                        else:
                            for failing in outputs:
                                compare(
                                    scratch[failing],
                                    expected[failing],
                                    semantics[failing],
                                    choice.plan,
                                )
                    except AssertionError as error:
                        # The fallback is the reference selection and reference512
                        # is checked against its qualification oracle: a mismatch
                        # there, or any under strict testing, stays fatal.
                        if (
                            strict
                            or choice.plan == fallback
                            or choice.plan.strategy == "plane_shared_reference512"
                        ):
                            raise AssertionError(
                                "physical candidate correctness: "
                                + choice.plan.identity
                                + "\n"
                                + str(error)
                            ) from error
                        excluded[choice] = exclusion(
                            scratch[failing],
                            expected[failing],
                            semantics[failing],
                            error,
                        )
                        if len(outputs) > 1:
                            excluded[choice]["output"] = failing
                        EVENTS.append(
                            {
                                "event": "candidate_excluded",
                                "key": key,
                                "plan": choice.plan.to_dict(),
                                **excluded[choice],
                            }
                        )
                assert torch.equal(before, torch.cuda.get_rng_state())
                compiled = [(ch, c) for ch, c in compiled if ch not in excluded]
                items = [ch for ch in items if ch not in excluded]
                timings = {choice: [] for choice in items}
                interval_timings = {choice: [] for choice in items}
                profile_counts = {choice: [] for choice in items}
                bm = KernelBenchmarker()
                for round_id in range(3):
                    rotation = compiled[round_id::] + compiled[:round_id]
                    for choice, c in rotation:
                        if INJECTED_TIMINGS is not None:
                            samples = INJECTED_TIMINGS(choice.plan.identity, round_id)
                            interval_timings[choice].append([])
                            profile_counts[choice].append([])
                            timings[choice].append(list(map(float, samples)))
                            continue
                        with c.kernel.prepared(scalars, *args):
                            with config.patch(
                                {"autotune_cudagraph_benchmarking": False}
                            ):
                                measurement = bm.benchmark_kernel_gpu(c.kernel)
                                samples = measurement["kernel_ms"]
                                interval_timings[choice].append(
                                    measurement["enqueue_interval_ms"]
                                )
                                profile_counts[choice].append(
                                    measurement["profile_attempt_counts"]
                                )
                        timings[choice].append(list(map(float, samples)))
                # Paired by round (bead 4z8.28): the SM clock drifts between
                # and within rounds (up to 2-3x in the first), moving every
                # candidate of a round together, so candidates are compared
                # with the baseline round by round, not by pooled spreads.
                round_medians = {
                    c: [statistics.median(r) for r in rounds]
                    for c, rounds in timings.items()
                }
                medians = {c: statistics.median(round_medians[c]) for c in items}
                best = min(items, key=medians.get)
                base = next(c for c in items if c.plan == fallback)

                def spread(c):
                    return statistics.median(
                        [abs(v - medians[c]) for rr in timings[c] for v in rr]
                    )

                # Near-equal plans split round wins among themselves; only the
                # comparison with the baseline decides.
                if all(
                    round_medians[best][r] < round_medians[base][r] for r in range(3)
                ):
                    winner = best.plan
                rows = [
                    {
                        "plan": c.plan.to_dict(),
                        "median_ms": medians[c],
                        "round_medians_ms": round_medians[c],
                        "mad_ms": spread(c),
                        "round_samples_ms": timings[c],
                        "enqueue_interval_samples_ms": interval_timings[c],
                        "profile_attempt_counts": profile_counts[c],
                        "correctness": True,
                        "directory": str(self.compile(c.plan).directory),
                    }
                    for c in items
                ] + [
                    {
                        "plan": c.plan.to_dict(),
                        **record,
                        "directory": str(self.compile(c.plan).directory),
                    }
                    for c, record in excluded.items()
                ]
                payload = {
                    "key": key,
                    "key_data": key_data,
                    "winner": winner.to_dict(),
                    "candidates": rows,
                    "rejected": rejected,
                    "rng_unchanged": True,
                    "output_isolated": True,
                    "measurement": "InductorBenchmarker warmup/event queue intervals plus exact CUPTI kernel durations; annotations/queue gaps excluded from selection; rotated three rounds",
                    "selection": "lowest median of round medians, if faster than the baseline in every round; otherwise frozen baseline",
                }
                tmp = details.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload, indent=2))
                tmp.replace(details)
                # The timing cache needs a value per choice; excluded never wins.
                return {**medians, **dict.fromkeys(excluded, float("inf"))}

            # Deterministic mode may consume an existing validated cache. Fresh
            # empirical selection is deliberately skipped; no effect semantics
            # or nondeterministic order is introduced by tuning.
            callback = (
                None if torch.are_deterministic_algorithms_enabled() else benchmark
            )
            scores = cache.lookup(
                choices, op="cubecl_physical", inputs=key, benchmark=callback
            )
            if scores and details.is_file():
                saved = json.loads(details.read_text())
                if canonical(saved["key_data"]) == canonical(key_data):
                    winner = PhysicalCandidate.from_dict(saved["winner"])
                    if winner not in plans:
                        raise ValueError("cached physical choice no longer legal")
            EVENTS.append(
                {
                    "event": "tune"
                    if rows is not None
                    else "tuning_cache_hit"
                    if scores
                    else "deterministic_baseline",
                    "key": key,
                    "winner": winner.to_dict(),
                    "candidate_count": len(plans),
                    "elapsed_ms": (time.perf_counter() - began) * 1000,
                    "details": str(details) if details.exists() else None,
                }
            )
        return winner


def compile_kernel(
    semantic,
    device,
    *,
    policy="first_use",
    reduction_context=None,
    numerics=DEFAULT,
):
    if policy not in (
        "first_use",
        "baseline",
        "combined_diagnostic",
        "compensated_diagnostic",
        "load_batch_diagnostic",
        "adjacent_merge_diagnostic",
        "reference512_diagnostic",
    ):
        raise ValueError("unsupported experimental tuning policy")
    key = (canonical(semantic), device, policy, numerics)
    if reduction_context is not None:
        key = (*key, canonical(reduction_context))
    with LOCK:
        if key not in CACHE:
            CACHE[key] = AutotunedKernel(
                semantic,
                device,
                policy,
                reduction_context=reduction_context,
                numerics=numerics,
            )
        return CACHE[key]
