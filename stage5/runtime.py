"""First-use experimental tuning; generic Inductor cache and event benchmark."""

from dataclasses import dataclass
import copy
import fcntl
import hashlib
import json
import statistics
import threading
import time

import torch
from torch._inductor import config
from torch._inductor.codecache import PersistentCache
from stage5.timing import KernelBenchmarker
from stage5 import artifact
from stage5.plan import PhysicalCandidate, shape_bucket
from stage5.legality import candidates, signature, LONG_REDUCTION_THRESHOLD
from stage5.diagnostics import tracing
from stage5.reduction_policy import (
    qualified_for_runtime,
    sanity_limit,
    SANITY_VERSION,
    VERSION as NUMERICAL_POLICY,
)

ROOT = artifact.ROOT
EVENTS = artifact.EVENTS
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
    return PhysicalCandidate(
        "shared_tree_reduction_baseline"
        if semantic["schema"] == 3
        else "scalar_pointwise_baseline"
    )


def compare(actual, expected, semantic, plan=None):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    dtype = actual.dtype
    rtol, atol = (
        (0.016, 0.008)
        if dtype == torch.bfloat16
        else (0.002, 0.001)
        if dtype == torch.float16
        else (2e-5, 2e-5)
        if semantic["schema"] == 3
        else (2e-6, 2e-6)
    )
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


class AutotunedKernel:
    def __init__(self, semantic, device, policy):
        semantic = copy.deepcopy(semantic)
        if not semantic.get("buffers") or any(
            b.get("access")
            != ("write" if i == len(semantic["buffers"]) - 1 else "read")
            for i, b in enumerate(semantic["buffers"])
        ):
            raise ValueError(
                "unsupported input contract: exactly one final output buffer"
            )
        self.semantic, self.device, self.policy = semantic, device, policy
        self.identity = json.loads(canonical(artifact.compiler_identity(device)))
        self.identity["tuning_source"] = hashlib.sha256(
            (ROOT / "runtime.py").read_bytes()
            + (ROOT / "legality.py").read_bytes()
            + (ROOT / "plan.py").read_bytes()
            + (ROOT / "timing.py").read_bytes()
            + (ROOT / "reduction_policy.py").read_bytes()
        ).hexdigest()
        self.identity["numerical_policy"] = NUMERICAL_POLICY
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
            self.compiled[plan] = artifact.CompiledKernel(
                self.semantic,
                self.device,
                plan.to_dict(),
                {k: v for k, v in self.identity.items() if k != "tuning_source"},
            )
        return self.compiled[plan]

    def __call__(self, scalars, *tensors):
        with LOCK, torch.cuda.device(self.device):
            if self.policy.endswith("_diagnostic"):
                # Explicit measured alternatives never enter automatic selection.
                from stage2.indexing import evaluate

                plan = self.baseline_plan
                if (
                    self.semantic["schema"] == 3
                    and evaluate(self.semantic["reduction_extent"], scalars)
                    > LONG_REDUCTION_THRESHOLD
                ):
                    plan = (
                        PhysicalCandidate(
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
                from stage5.legality import structural_reason

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
                            "winner": baseline.kernel.plan.to_dict(),
                        }
                    )
                    self._bypass_recorded = True
                return baseline(scalars, *tensors)
            facts = signature(self.semantic, scalars, tensors)
            if facts not in self._legality:
                self._legality[facts] = candidates(
                    self.semantic, scalars, tensors, baseline.kernel.resources
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
        directory = ROOT / "cache/tuning" / key
        directory.mkdir(parents=True, exist_ok=True)
        choices = [Choice(p) for p in plans]
        details = directory / "selection.json"
        cache = TimingCache(directory / "timings.json", self.identity)
        with (directory / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            rows = None
            winner = frozen(self.semantic)
            began = time.perf_counter()

            def benchmark(items):
                nonlocal rows, winner
                output = tensors[-1]
                scratch = torch.empty_strided(
                    output.shape,
                    output.stride(),
                    device=output.device,
                    dtype=output.dtype,
                )
                args = (*tensors[:-1], scratch)
                baseline.kernel(scalars, *args)
                expected = scratch.clone()
                compiled = [(choice, self.compile(choice.plan)) for choice in items]
                # Compile all before timing, then validate every candidate on
                # isolated destinations. Pure kernels neither mutate inputs nor
                # consume RNG; user output is not touched until final launch.
                before = torch.cuda.get_rng_state().clone()
                for choice, c in compiled:
                    c.kernel(scalars, *args)
                    try:
                        compare(scratch, expected, self.semantic, choice.plan)
                    except AssertionError as error:
                        raise AssertionError(
                            "physical candidate correctness: "
                            + choice.plan.identity
                            + "\n"
                            + str(error)
                        ) from error
                assert torch.equal(before, torch.cuda.get_rng_state())
                timings = {choice: [] for choice in items}
                interval_timings = {choice: [] for choice in items}
                profile_counts = {choice: [] for choice in items}
                bm = KernelBenchmarker()
                for round_id in range(3):
                    rotation = compiled[round_id::] + compiled[:round_id]
                    for choice, c in rotation:
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
                medians = {
                    c: statistics.median([v for r in rounds for v in r])
                    for c, rounds in timings.items()
                }
                best = min(items, key=medians.get)
                base = next(c for c in items if c.plan == frozen(self.semantic))
                wins = sum(
                    min(items, key=lambda c: statistics.median(timings[c][r])) == best
                    for r in range(3)
                )
                spread = lambda c: statistics.median(
                    [abs(v - medians[c]) for rr in timings[c] for v in rr]
                )
                if wins >= 2 and medians[best] + 3 * spread(best) < medians[
                    base
                ] - 3 * spread(base):
                    winner = best.plan
                rows = [
                    {
                        "plan": c.plan.to_dict(),
                        "median_ms": medians[c],
                        "mad_ms": spread(c),
                        "round_samples_ms": timings[c],
                        "enqueue_interval_samples_ms": interval_timings[c],
                        "profile_attempt_counts": profile_counts[c],
                        "correctness": True,
                        "directory": str(self.compile(c.plan).directory),
                    }
                    for c in items
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
                    "selection": "stable best and disjoint median +/- 3 MAD from baseline; otherwise frozen baseline",
                }
                tmp = details.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload, indent=2))
                tmp.replace(details)
                return medians

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


def compile_kernel(semantic, device, *, policy="first_use"):
    if policy not in (
        "first_use",
        "baseline",
        "combined_diagnostic",
        "compensated_diagnostic",
        "load_batch_diagnostic",
        "adjacent_merge_diagnostic",
    ):
        raise ValueError("unsupported experimental tuning policy")
    key = (canonical(semantic), device, policy)
    with LOCK:
        if key not in CACHE:
            CACHE[key] = AutotunedKernel(semantic, device, policy)
        return CACHE[key]
