# Selector-preserving hardware counter investigation

Current Stage 5 decisions live in [STAGE5_REPORT.md](../../../../STAGE5_REPORT.md).
All production and retained reduction kernels remain frozen. This pass changes
only diagnostic measurement code and unrelated primer store-policy controls.
No reduction topology, split9, protected access, precision, final, storage/stream,
runtime warmup, eligibility or Stage 6 policy changes.

## Result and attribution boundary

The slow condition has substantially more **device-wide DRAM write traffic during
the target window**. A fresh-process, single-pass CUPTI PM protocol preserves the
selector in 96 fully validated counter captures. Across the 16 primary pairs,
the smallest slow write lower bound is **1,740,672 B**, above the largest fast
upper bound of **291,584 B**. This conclusion does not depend on estimating
missing boundary samples. Measured SM and memory clocks are nearly equal.

This identifies a hardware observation, not a completed root-cause proof.
Counters are device-wide; WSL lacks the PM context-switch trace. The windows
exclude primer execution for lower bounds but cannot exclude unrelated display
contexts. Dirty-cache eviction/writeback is consistent with the evidence; queued
write traffic, controller behavior and other memory-side interactions remain
unisolated. No display/WSL/driver/TLB cause is established.

The one cache-policy follow-up uses existing CUDA `__stwb` and `__stwt` on the
unrelated primer. Both retain the write-traffic difference and latency split.
The intervention therefore did not observably change the proposed state. It
neither proves nor refutes dirty-cache causality. No second cache intervention,
production workaround or kernel optimization is introduced.

## Reproduction of the unchanged selector

Every sample is retained; no numerical tolerance or qualification policy changes.
Each group below has 16 trials per order, using the original primer harness.
The 409.524-us slow tail is included, not trimmed.

| Uninstrumented by counters | Fast median, range us | Slow median, range us |
| --- | --- | --- |
| Initial fresh process | 21.8975, 21.640–22.460 | 25.9355, 25.732–26.380 |
| After the failed persistent-PM experiment | 21.796, 21.433–22.968 | 25.9535, 25.715–409.524 |
| Final fresh process | 21.926, 21.633–22.396 | 25.921, 25.685–26.242 |

All 96 baseline traces validate. All fast samples are below all slow samples in
each group. See `baselines-analysis.json`, the three `baseline-*` directories
and their original CUPTI activity JSON files.

The common gate/reset, allocation, scalar byte counts, read/write functions and
82x256 primer geometry are unchanged. Write→read selects fast; read→write selects
slow. Target remains `cubecl5_42026f8357481503`, grid576/block512/shared64B/40regs,
64x65539 FP32 and nine partials per row. Input hash is
`199358562e389b69090676348ef457ed0141f285d0609c9dd4ff6bfe3b44e218`.
Every accepted capture verifies complete partial/final output bit identity,
original artifact hashes and host launch completion before the common gate ends.
See `frozen-artifacts/`, `frozen-source-sha256.json` and per-capture results.

## Available mature mechanisms and chosen collection

Inventory/provenance is in `toolchain.json`, `mature-mechanisms.json`,
`cupti-capabilities.txt`, `cupti-pm-metrics.txt`, the losslessly compressed NCU
metric queries, and saved NCU/NSYS help/device queries.

- **CUPTI 13.0.1**, runtime CUDA13.0, RTX3090/GA102 on WSL2: native support queries
  report range profiling, PC sampling, SASS metrics and PM sampling supported.
  PM enable/disable succeeds. Early initialization/library-search failures are
  retained; they were not interpreted as platform limitations. The successful
  standalone probe uses WSL's driver library first in its search path.
- **NCU 2026.3.1** advertises application/range replay, no-cache-flush and
  no-clock-control modes, and PM sampling. Its metric query reports available
  profiling and PM metrics. That newer catalog is not interchangeable with
  CUPTI13.0.1's catalog. Earlier NCU replay failures remain in the parent
  repeatability packet; no new NCU replay is used for attribution here.
- **NSYS 2026.5.1** lists GPU-metric sampling and both devices. Its device IDs differ
  from CUDA's: CUDA0 is the RTX3090, NSYS/SMI1 is the RTX3090. Its collector was
  inventoried but not used to make counter claims.
- **Installed PyTorch PM machinery** was inspected: `PmSampler` and its activity
  monitor reuse CUPTI. Import fails because `cupti-python` is absent in the
  isolated environment (`pytorch-pm-probe.txt`); the full activity monitor also
  declares a newer CUPTI requirement. No environment/library upgrade is made.
  This justifies a small adapter to the existing CUDA13.0 CUPTI API rather than
  inventing a kernel or adopting another runtime.
- **NVML/SMI** supplies device/driver/clock/power snapshots. Those snapshots do not
  resolve an individual 22–26-us launch; no per-target power/throttle attribution
  is claimed. PC/range/SASS support queries are capability evidence only, not
  measurements of selector preservation.

The adapter uses the documented CUPTI host configuration, pass-count query,
PM enable/configure/start/stop/decode/evaluate APIs. It refuses multipass
configurations, incomplete decodes and overflow. Collection is 1000ns nominal
interval, 128MiB hardware buffer, KEEP_OLDEST, 200000 counter-image samples.
Each accepted process measures exactly one pair, with randomized order selected
by the retained round seed. The adapter requests no target replay, cache flush or explicit clock setting.
Internal profiler perturbation is assessed from the measured selector for each
configuration. Primer traffic is not aggregated into an exact target count.

[CUDA13.0 CUPTI PM documentation](https://docs.nvidia.com/cupti/13.0.1/main/main.html#cupti-pm-sampling-api)
and [NCU sampling/platform documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#pm-sampling)
explain the existing interfaces and device-wide/context-trace limitation.

## Clock conversion and counter bounds

PyTorch installs an approximate/TSC timestamp callback. Raw PM API columns named
`start_ns/end_ns` consequently contain that callback's clock values here, not
Unix timestamps. `*.pm-unix.csv` retains the raw columns alongside converted ones.

The final adapter uses installed `c10::ApproximateClockToUnixTimeConverter`:
construction before capture and `makeConverter()` afterward, matching Kineto's
lifecycle. The Python convenience wrapper makes the rate at construction and
was unsuitable for this calibration audit. Initial Python-conversion captures
and failed native-clock pilots remain exploratory, not the final counter proof.
A strict wall bracket initially rejected a 2ns calibration residual; that failed
pilot is retained. The final method records native raw/approx/realtime anchors,
verifies the callback clock domain, and propagates their measured uncertainty
into the counter window instead of silently treating either clock as exact.

[Pinned C10 implementation](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/c10/util/ApproximateClock.cpp)
and [pinned Kineto caller](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/csrc/autograd/profiler_kineto.cpp)
are the source references; local header/library hashes are retained.

An interior sample must fit wholly inside the target window after removing
1500ns plus the maximum native-anchor uncertainty from each edge. This is a
measurement boundary, not a numerical tolerance or a latency-sample filter.
Additive metrics additionally retain a conservative upper bound from every
sample overlapping the expanded window. Upper bounds may include edge primer
traffic and are never labeled exact target counts. Contiguous coverage,
nonnegative finite counts, raw/converted column identity and resource/provenance
checks are asserted. Complete samples/images, including all excluded boundary
intervals, remain available.

## Validated counter families

`summary.json` and `qualified-*-analysis.json` validate all planned repetitions
for the following five groups; none is assembled from a failed group's good
prefix. `qualified-commands.json` contains every exact invocation.

| Family | Fresh pairs | Fast median us | Slow median us |
| --- | ---: | ---: | ---: |
| DRAM reads/writes | 16 | 22.143 | 26.3105 |
| SM/memory clock counters | 8 | 22.368 | 26.354 |
| All-operation L2 sectors | 8 | 22.272 | 26.369 |
| Primer `__stwb` control | 8 | 21.967 | 26.450 |
| Primer `__stwt` control | 8 | 22.3555 | 26.3735 |

All fast/slow samples separate within each complete group. Compared with the
baseline, the approximately four-us split survives instrumentation.

| Primary DRAM metric | Fast median lower–upper B | Slow median lower–upper B |
| --- | ---: | ---: |
| `dram__bytes_read.sum` | 13,518,080–16,797,056 | 14,548,288–16,824,768 |
| `dram__bytes_write.sum` | 85,376–85,504 | 2,813,504–3,304,832 |

The bounds are interior and expanded-overlap sample sums, not an extrapolated
whole-kernel byte total. SM frequency (`sm__cycles_elapsed.avg.per_second`) has
interior time-weighted medians 1.6649 versus1.6777GHz; the slow state is slightly
higher. The memory-frequency metric (`dram__cycles_elapsed.avg.per_second`)
reports 9.49309 versus9.49334GHz in its native units. These measured clocks do
not explain the latency difference; unmeasured internal domains remain open.

`lts__t_sectors.sum` is **all-operation** activity, not read-only sectors or a hit
rate. Its bounded/interior values are retained, but do not isolate cache misses.
A direct one-pass hit/miss comparison was not obtained.

## Failed or unsuitable collection methods

All failures remain in their original directories/logs. Their counters do not
explain the ordinary fast/slow transition.

- `pm-dram-r0`: persistent PM session/repeated captures initially preserves the
  regimes, then both move to hundreds of us; decode eventually fills the image.
  The entire protocol is rejected, including its initially favorable prefix.
- `pm-dram-aligned`: includes a319.688-us fast-condition tail; not substituted
  for a complete final group. Earlier clock pilots lack the final calibration.
- `l2-fresh-00`: a newer NCU catalog name is rejected by CUPTI13.0.1 configuration.
- `l2-hit-fresh-00`: the hit-rate metric requires two passes and is refused.
- `scoreboard-fresh-00`, `scoreboard-raw-fresh-00`: normalized and raw requested
  scoreboard configurations require four/two passes and are refused.
- `dispatch-fresh-00`: requested instruction/eligible-warp pair requires two
  passes and is refused.
- Early single-counter/volume groups include clock-admission failures, retained
  without relaxing numerical or latency acceptance. Final L2-volume collection
  uses the corrected clock mechanism and completes all eight pairs.
- `qualified-instructions-00`: one-pass instruction counting initially measures
  fast22.338us, then the other condition reaches790.328us and decode reports
  `CUPTI_ERROR_OUT_OF_MEMORY`. Its failed activity trace is saved; this protocol
  fails selector preservation. No instruction/residency/stall values from it
  are used for attribution. Counter collection can perturb execution even
  without replay; a larger buffer would not validate that changed regime.

## Causal follow-up and remaining gap

The existing CUDA store intrinsics change only the unrelated writer's store
expression. Values stay1, and count, addresses, allocation, geometry, selector order
and target bytes stay fixed. The wb arm is a plumbing control: it checks that replacing the original
volatile store with an intrinsic still reproduces the split, without asserting
identical memory-ordering semantics. The generated PTX contains `st.global.wb.u32`
versus `st.global.wt.u32`; archived wrapper/source/PTX and commands prove the
substitution. This does not assume that a write-through hint cleans GPU device
memory's L2. [PTX cache operators](https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators)
describe the reference mechanism.

Median interior write bytes remain about2.77MB (`wb`) and2.81MB (`wt`) in the slow
condition, versus about85KB in fast. Neither eliminates the delay. Therefore:

- **Observed:** selector-preserving hardware sampling shows extra concurrent
  DRAM writes in slow execution; clock counters are nearly equal.
- **Supported cause:** no individual cache/controller structure has yet met the
  requested intervention standard. The read/write ordering remains a causal
  selector of the joint state, rather than an isolated explanation of its cost.
- **Hypothesis:** dirty-cache eviction/writeback adds memory work during target
  reads. Cache hit state, pending traffic/controller effects and internal
  clock/power behavior are not all separated by the available accepted metrics.

`cwk.5` remains open. A subsequent causal intervention must actually change the
proposed pending-write/cache state; the `__stwt` name is insufficient evidence.
No synthetic warmup is proposed for production.

## Reproduction, validation and review

From the repository root, build the diagnostic adapter with
`bash stage5/artifacts/reduction-structure/state-counters/build-sampler.sh`.
Use the `LD_LIBRARY_PATH` recorded in `commands.json` / the driver scripts:
WSL driver first, then the existing CUDA13.0 package library directory.
New capture directories must not exist. `run-qualified.py` records the final
predeclared counts; `PYTHONPATH=. python .../summarize.py` validates them.
The original baseline harness remains byte-identical.

The validator enforces matching metrics, interval, sampler binary/source, clock
method, reset, harness revision, primer PTX and store policy within each
compared group, and verifies the
store wrapper, sole source substitution and PTX operator. Fifteen admission/FFI
controls and four CLI controls pass; two CPU-only linked-stub decode failures
preserve a deliberately synthetic16-byte payload and status (not GPU counter
data). All96 saved counter captures still validate
after review fixes. A separate two-capture hardened-sampler smoke preserves
frozen bits and is validation only. Baselines use the unchanged original harness.

Static checks, CLI-admission controls, historical-manifest verification and
review disposition are retained in this packet. Raw CSV/counter images may be
losslessly gzipped; lossless-compression.json verifies original/decompressed
identities. No commit or push is made.

Sampler source and binary are independently hashed with build commands and
archived binaries retained. The binary does not embed a build-source hash, so
these are provenance records rather than cryptographic source-to-binary
attestation. This residual limitation is explicit; no stale-library equivalence
is assumed across different comparison groups.
