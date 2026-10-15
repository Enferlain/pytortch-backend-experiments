# Fixed-topology secondary repeatability

Current decisions live in [STAGE5_REPORT.md](../../../../STAGE5_REPORT.md).
This packet investigates `64×65539` FP32 with the qualified reference512
arithmetic/work assignment frozen. Both backends use the exact retained partial
and final artifacts, the same input bytes, split9 and original ownership/FP
contracts. No production kernel, warmup, cache policy or eligibility changed.

## Result and remaining uncertainty

The approximately 21-versus-25 µs CubeCL partial difference can be reproduced
by execution history without changing its code or input. This is a device
execution-state effect, rather than a new topology or a different split.
The exact hardware/driver state responsible has **not** been identified.
`cwk.5` remains open; Stage 5 continues and Stage 6 stays deferred.

Eight completed capture groups retain 184 raw Torch/CUPTI traces. Median partial
microseconds below are separate controls, not measurements to pool together:

| RTX3090 control | CubeCL | Configured Triton |
| --- | ---: | ---: |
| Sparse prepared pair, initial monitored run | 25.185 | 23.781 |
| Same partial repeated in a burst, all 16 calls | 21.339 | 21.586 |
| Four priming pairs, then measured pair, telemetry disabled | 21.236 | 21.627 |
| Same priming, then unrelated 64 MiB fill | 25.668 | 24.413 |
| One input allocation, repeated driver-control calls | 21.320 | 21.632 |
| Sixteen identical input allocations, one visit per call | 21.373 | 21.593 |
| Sixteen module instances of the identical artifact | 21.313 | 21.648 |
| Entire burst queued behind existing activity kernel | 21.369 | 21.512 |

These are CUPTI kernel durations. All samples, including startup, recurring slow
samples and long outliers, remain in `result.json` and raw traces. In particular,
the gated Triton capture includes a 276.803 µs outlier. `analysis.json` records
counts, extrema, MADs, each round's median, first-call statistics and explicitly
labelled later-call statistics; it does not trim samples to qualify a default.
The final artifacts remain unchanged, generally around 2.3–2.5 versus 1.2–1.3 µs.

What the controls establish:

- The fast regime does not require reusing the same input allocation: rotating
  sixteen byte-identical 16 MiB allocations retains it. This contradicts a simple
  same-input L2-cache-reuse explanation.
- Rotating module instances likewise retains the fast regime. Resource attributes
  match and all instances load the same original artifact. This does not prove
  that the driver assigns distinct physical instruction storage to those modules.
- In the gated RTX3090 control, the median first CubeCL partial is 25.124 µs;
  later calls have median 21.341 µs. All 32 target launches were enqueued before
  the activity kernel ended in every capture. Host enqueue starvation therefore
  cannot be the sole cause of the first-call execution penalty. The raw trace
  records the driver API completion times; `prequeue_checks` verifies this.
- Inserting a synchronized 500 µs host delay produces a mixture of fast and slow
  samples, rather than a deterministic new latency. Similarly, ten-millisecond
  idle and 50M-cycle spin controls do not select a single state reliably.
- A memory sweep reliably restores slower samples, but it changes other device
  state too. That observation alone does not locate the cause in L2 or TLBs.
- Telemetry-disabled repetitions preserve the main effect. Coarse monitoring is
  not necessary to produce it; these samples do not establish every clock domain.

## Second profiler and clock sampling

The actual Nsight Systems command was:

```sh
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --gpu-metrics-devices=1 --gpu-metrics-frequency=100000 --capture-range=cudaProfilerApi --capture-range-end=stop --output=/home/imi/Projects/pytortch-backend-experiments/stage5/artifacts/reduction-structure/repeatability/nsys-state /home/imi/Projects/pytortch-backend-experiments/stage0/.venv/bin/python -m stage5.reduction_repeatability_probe --output /home/imi/Projects/pytortch-backend-experiments/stage5/artifacts/reduction-structure/repeatability/nsys-state-pass --ncu --ncu-control prime
```

`nsys stats --report cuda_gpu_kern_sum,cuda_kern_exec_sum --format csv` generated
the retained `nsys-stats_*.csv` files. Its five partial durations, extracted from
`nsys-state.sqlite`, are **25.317, 21.540, 21.220, 21.156, 21.317 µs**. Thus the
transition also appears in a second profiler client without an in-process Torch
profiler. Both clients use CUDA activity timestamps; this is not a claim of an
independent hardware clock source.

100 kHz GPC samples overlap: the first partial has raw values near
1.671e9–1.715e9 and the second near 1.650e9–1.715e9. There are only two samples per
partial. The export labels this metric `GPC Clock Frequency [MHz]` although its
raw values are near 1.7e9; we preserve the raw values and assume no unit conversion.
This capture does not establish a global clock ramp as the explanation, nor rule
out an unobserved internal clock/power or memory/dispatch transition. The RTX3090
has display activity enabled, but the recorded short kernel window has no sampled
pixel/vertex work. That cannot rule out Windows/display effects outside the window.

## Counter controls and failed collection attempts

For the single-kernel counter control, the actual primed command was:

```sh
ncu --target-processes all --profile-from-start off --replay-mode application --app-replay-mode strict --cache-control none --clock-control none --kernel-name 'regex:^cubecl5_42026f8357481503$' --launch-skip 4 --launch-count 1 --section-folder /home/imi/tools/nvidia/nsight-compute-2026.3.1/sections --metrics gpu__time_duration.sum,lts__t_sectors_op_read.sum,lts__t_sector_op_read_hit_rate.pct,dram__bytes_read.sum,sm__cycles_elapsed.avg --csv --export /home/imi/Projects/pytortch-backend-experiments/stage5/artifacts/reduction-structure/repeatability/ncu-prime /home/imi/Projects/pytortch-backend-experiments/stage0/.venv/bin/python -m stage5.reduction_repeatability_probe --output /home/imi/Projects/pytortch-backend-experiments/stage5/artifacts/reduction-structure/repeatability/ncu-prime-passes --ncu --ncu-control prime
```

The `prime_flush` command differs only in the recorded control/output paths;
`ncu-commands.json` contains both complete argument lists. `ncu-tool.json` records
the resolved executable, hash, version and explicit no-clock/no-cache-control
settings. Relevant raw rows from the retained `ncu-*-raw.csv` exports are:

| Counter control | GPU duration µs | DRAM read Mbyte | L2 read hit % | L2 read sectors |
| --- | ---: | ---: | ---: | ---: |
| prime | 26.304 | 16.812544 | 0.434713 | 527704 |
| prime_flush | 26.624 | 16.820352 | 0.432057 | 527939 |

Both replay measurements look cold, despite priming. They do not preserve the
21 µs warm CUPTI condition, so they cannot identify its missing warm-state cause.
Their durations are not substituted for the warm measurements above.

Application-range replay collects metrics across the whole selected range,
[as documented by NVIDIA](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#application-range-replay).
The corrected range commands/results are `ncu-app-range2-*`. They cover five
partial/final pairs; the second also includes the unrelated fill. Range DRAM
reads are 85.560064 versus 156.020992 Mbyte, and range durations 1.257824 versus
44.365120 ms. Those workloads include different control traffic and substantial
profiling/host effects; neither a per-partial latency difference nor an isolated
input-cache effect can be inferred by subtracting them.

The first two range CLI attempts were rejected for unsupported option
combinations, before running the probe. Commands and error logs remain intact.
The earlier `state-controls` run is incomplete because an extra-kernel assertion
expected a sleep name while ATen emits `spin_kernel(long)`. Its raw trace, source
and partial result are retained; corrected `state-controls-r1` passes. These
failures are explicitly excluded from summaries in `capture-corrections.json`.

## Display-disabled hardware control

`headless-3080` uses the byte-identical retained `.npy` input and the same SM86
artifacts on the unused, display-disabled RTX3080. This is a hardware control,
not a same-device performance comparison. Four-round medians are:

| Scope | CubeCL partial µs | Triton partial µs |
| --- | ---: | ---: |
| Sparse | 25.830 | 21.964 |
| Gated burst | 25.422 | 25.167 |

The RTX3090's specific transition is not reproduced: CubeCL changes little,
and Triton's direction reverses. This rules out treating the RTX3090 result as
a universal warmup rule; it does not prove a display cause, because SM count,
memory system, clocks and other hardware state also differ. All outputs and
input hashes still match the frozen comparison.

## Validation, ownership and scope

The harness reuses the existing `Component`, `Kernel.prepared`/`batch` and
`DriverKernel` mechanisms. PyTorch owns inputs, partials, final outputs, metadata
and streams. Driver controls retain tensor references, wait for readiness and
record current-stream use. Module/address controls preserve scalar metadata,
geometry and function resource attributes; their timing scope differs from the
normal validated graph's host path. No manual integer narrowing or FP rewrite
is introduced. CubeCL loads the retained PTX through the current driver; Triton
loads its retained cubins without reassembly.

Every completed group checks the exact input hash, bitwise partial/final equality
between backends and the frozen recorded output before measurement, then checks
input/artifact stability and output equality afterward. The eight completed
groups and eleven counter/timeline application passes verify successfully.
The complete numerical corpus was not rerun: there is no kernel change, and the
previous qualification evidence is preserved. This is not new default admission.

`historical-integrity.json` verifies all 689 previous manifest entries before
updating the canonical report. `STAGE5_REPORT.before.md` preserves that report's
pre-pass bytes. The checkout advanced externally from b74c15c to c94146c during
the pass; this agent performed no commit or reset. No production source changed.

Native review failed with a usage-limit error before returning findings; the user
then explicitly disabled native review. `review-status.json` records that this
new diagnostic pass has no completed independent review. Earlier uniform-loop
review approval does not cover it. Parent source/evidence checks are separate.

## Reproduction and next question

Use a **new** output directory for each capture. Existing directories are rejected.
From the repository root:

```sh
STAGE0_CARGO_PROFILE=release stage0/.venv/bin/python -m stage5.reduction_repeatability_probe \
  --output stage5/artifacts/reduction-structure/repeatability/repro-new \
  --rounds 4 --modes sparse gated_burst --monitor
stage0/.venv/bin/python -m stage5.reduction_repeatability_analyze
```

Additional explicit diagnostic flags are `--address-control`, `--module-control`
and `--gap-us 500`. The display-disabled capture used
`CUDA_VISIBLE_DEVICES=GPU-510ddc27-95ac-c62f-d49b-242cc5183c72` with
`--input-npy stage5/artifacts/reduction-structure/repeatability/input-float32-64-65539.npy`.
The analyzer verifies the fixed retained groups; a new capture is not silently
included in established summaries.

The remaining question is which GPU memory/dispatch, internal clock/power or
WSL/driver state distinguishes the RTX3090 startup/sparse and sustained regimes.
The evidence does not select a unique cause. Keep that question open rather
than change arithmetic or optimize a proxy. Final-kernel overhead can be measured
in explicitly separated regimes, but this packet installs no warmup workaround
and makes no broad performance/default claim.
