# Causal secondary-state primers

Current decisions live in [STAGE5_REPORT.md](../../../../STAGE5_REPORT.md).
This pass measures the frozen qualified uniform-loop partial at **64×65539
FP32**, split9, on RTX3090. No partial code, work assignment, arithmetic, loop,
protected-load semantics, precision, runtime warmup or default policy changes.

## Reproducible selector

Two deliberately ordered primers reproducibly select different costs for the
**same isolated target launch**:

1. Common gate and 64 MiB reset, then **read 64 MiB → write 64 MiB → target** selects
   the slow state.
2. Same gate/reset, then **write 64 MiB → read 64 MiB → target** selects the fast state.

Both sequences touch the same unrelated allocation with the same read/write
byte counts, kernel code, launch counts and geometry. The target input is never
written by a primer. Approximate primer durations are 355 µs in either order;
all launches are queued before the gate ends. This changes device work order
without introducing host starvation as the selector.

| Fresh process | Write→read: fast µs | Read→write: slow µs | Separation of every sample |
| --- | --- | --- | --- |
| read-write-r2, 16 trials each | median 21.892, range 21.466–22.683 | median 25.855, range 25.684–26.276 | 3.001 µs gap |
| selector-r3, 32 trials each | median 21.841, range 21.649–22.637 | median 25.895, range 25.594–26.665 | 2.957 µs gap |
| footprint-r4, 16 trials each | median 22.008, range 21.769–22.277 | median 25.950, range 25.822–26.180 | 3.545 µs gap |

No samples are removed. These are CUPTI activity kernel durations, not event
intervals or NCU replay. `analysis.json` verifies raw traces, source/artifact
hashes, all launch dimensions/resources, bitwise outputs and prequeue timestamps.
Five completed groups retain **744 raw traces**, including the additional
footprint controls collected after the regression run.

This meets the prerequisite for a subsequent counter comparison. It does not
by itself identify one hardware structure. Memory read/write history is causal
under this protocol; the exact cache/controller/translation or internal
power-state mechanism remains open in `cwk.5`.

## Matrix and competing causes

The first matrix has eight randomized repetitions per mode; the second uses
16 repetitions and lengthens the compute-only primer. The read/write matrix
also has 16 repetitions per mode. Every trial starts from the same gate and
64 MiB unrelated ATen fill. The target is a single partial; the original final
and bitwise validation execute after capture. Representative read-write-r2
medians are separate conditions:

| Primer after common reset | Target µs | What it tests |
| --- | ---: | --- |
| None | 25.792 | Slow reference condition |
| Eight identical target launches, same input | 21.523 | Same-code/read-history state |
| Eight identical target launches, different byte-identical input | 21.580 | Same-code state without target allocation reuse |
| Register compute, about 317 µs | 24.981 | Compute activity/duration without a large read working set |
| Dense read 64 MiB, about 276 µs | 21.858 | Unrelated read traffic, different instructions |
| Dense read 16 MiB, about 73 µs | 21.878 | Read working-set/traffic control |
| Dense read 4 MiB, about 20 µs | 24.656 | Smaller read footprint/traffic |
| Compact read 256 KiB | about 25 | Equal-count compact counterpart to sparse reads |
| Sparse reads over 256 MiB, one word every 4096 B | 23.732 | Large address span with only 256KiB requested values |
| Same sparse read repeated16 times | 21.974 | Repeated sparse read activity |
| Compact counterpart repeated16 times | 24.904 | Same count/dispatch pattern, smaller span |
| Touch target allocation once every 4096 B | about 25 | Target translation/address warming without a full input sweep |
| Dense write 64 MiB | 25.872 | Write traffic on the same unrelated allocation |
| Sparse writes over 256 MiB | 24.142 | Broad write address span with small requested payload |
| Eight target launches then write 64 MiB | 25.858 | Memory writes can reset an already selected fast state |
| Read64MiB then write 64 MiB | 25.855 | Equal-work slow selector |
| Write64MiB then read 64 MiB | 21.892 | Equal-work fast selector |

The initial matrix additionally tests retained Triton partial primers, priming
pairs with the fixed final, short spin after priming, and small reads after
priming. Their full distributions remain in the packet; no universal idle
threshold or display/clock explanation is inferred from them.

Interpretation boundaries:

- **Re-executing target code is unnecessary to select fast execution:** an
  unrelated dense-read kernel suffices, whereas many calls to the same read
  function on a small footprint do not. Different-allocation same-code primers
  also show that target data reuse is unnecessary. This rejects simple
  target-warmup/call-count explanations without claiming that every internal
  instruction/dispatch dependency is excluded.
- **Target address/TLB warming alone is insufficient:** sparse target touches
  do not select the fast regime. Reversing reads/writes over the same unrelated
  address span separates states despite equal total coverage. This does not
  rule out different internal translation/read/write paths.
- **Memory-side activity is a causal selector:** large reads select fast,
  large writes restore slow, and swapping their order reverses the target cost.
  Dirty-cache eviction, read/write controller behavior and other memory-side
  state remain hypotheses until counters distinguish them.
- **Generic compute/power activity alone is insufficient:** a register-compute
  primer as long as the memory primers leaves the target slow. This does not
  measure every internal memory clock or power domain, or establish a universal
  hardware rule.

The footprint follow-up further separates read duration from coverage. A
reused 256 KiB compact region read 128 times takes about 233µs yet leaves the target
at 24.813µs; target addresses touched every 4096 B for 64 repetitions take94µs and
leave it at 24.807µs. By comparison, a shorter16MiB read selects21.878µs. Small
reads after eight target primers preserve fast execution (about 21.4µs). Thus
long read activity or warming the target's sparse address set is insufficient;
the read working set/traffic pattern matters. A dirty-cache/writeback explanation
is a useful next hypothesis, not an established cache-hit or DRAM-byte result.
Target-only DRAM writes and cache eviction/writeback evidence would distinguish
it from other controller/internal-state explanations, provided collection keeps
the selected states intact.

## Exact controls and identity

The target is the retained `cubecl5_42026f8357481503` artifact: grid 576,
block 512, dynamic shared64B, executed allocation40 registers/thread. Target
PTX/input/partial/output hashes must agree across every capture group. Input
SHA-256 is `199358562e389b69090676348ef457ed0141f285d0609c9dd4ff6bfe3b44e218`.
The existing .npy fixture is loaded directly; RNG history is not regenerated.
The original final validates every target's complete partial buffer and output
bitwise, outside the measured capture. Artifacts and inputs are rehashed at end.

Read/write/compute probes use one 256 MiB PyTorch allocation, a separate sink, and the
same 82×256 launch geometry. `primer_read` and `primer_write` perform count-bounded
volatile scalar operations. The dense64MiB operation accesses 16,777,216 words at
stride 1; the sparse256MiB operation accesses 65,536 words at stride 1024 words.
Compact and sparse equal-count controls execute the same read function with
different scalar strides. The 4096B stride is a **virtual address spacing**;
actual GPU page size, TLB entries and cache hit rates are not assumed.
Writes store the allocation's existing value 1, so reversing read/write order
does not change its data bytes. Same-kernel primers retain the target's original
512-thread geometry and use a separate byte-identical input allocation where
specified.

The small diagnostic CUDA primers are necessary to isolate register arithmetic
with one terminal store and equal-count volatile reads at controlled spacing.
General ATen tensor/reduction kernels do not expose that narrow control. The
existing CUDA13.0 NVRTC and DriverKernel launcher are reused; no runtime,
allocator or stream ownership mechanism is invented. The partial itself is
launched through its existing validated native prepared path. Every source/PTX
and exact harness revision is retained per fresh capture directory.

All target and primer launches complete their host API calls before the gate
ends, verified from raw CPU/GPU timestamps; target gaps are recorded separately.
Long outliers and mixed modes remain in the summaries. No threshold is added
to numerical qualification, and no fast sample is substituted for slow history.

## Counters and reproduction

No new NCU replay is used to infer the secondary state. Earlier NCU controls
destroyed the fast regime and remain historical counterexamples. The final
reduction's separate NCU capture is only a code/resource trace for investigation
A; its durations are not used here.

For subsequent counters, use the equal-work read/write selectors, require the
target artifact and input/output identity, and first confirm that the collection
method still separates the CUPTI target distributions. If instrumentation makes
both states slow, it has failed this prerequisite; do not interpret counter
differences or aggregate primer traffic as a target-only explanation.

```sh
stage0/.venv/bin/python -m stage5.reduction_state_primers --output /tmp/new-selector --rounds 32 --iterations 65536 --modes read_write write_read compute same8_write64m
stage0/.venv/bin/python -m stage5.reduction_state_analyze
```

New output directories must not exist. Commands, archived sources, raw traces,
NVRTC options and identity manifests retain all completed conditions. Native
review is disabled by the user; this pass has no completed independent review.
