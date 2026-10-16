# Final bounded memory-state attribution pass

This packet is a causal intervention experiment for `cwk.5`, not a reduction
optimization or another counter inventory. All production and retained reduction
kernels, runtime warmup, eligibility, arithmetic, split9 and precision stay frozen.
The canonical current summary remains `STAGE5_REPORT.md`.

## Result and stop decision

All192 predeclared captures completed with unchanged input/partial/final bits and
frozen target artifacts. **176 captures qualify as complete coupled comparisons.**
One24MiB window fails the unchanged clock guard; that entire16-capture contrast
is unqualified for causal use. All192 raw latencies,191 individually admitted
counter windows and the explicit failure remain retained. There are no replacement
samples, retries, narrowed guards or favorable-prefix conclusions.

A smaller writer footprint causally changes the joint memory/latency state, but
no individual hardware/driver source is identified. **Park `cwk.5` as an environmental
benchmarking confound**, under the user's final-pass stop condition. Return Stage5's
focus to the already-qualified fixed-final component (`cwk.7`, generated-graph
validation). No production warmup, arithmetic, kernel, eligibility or Stage6 change.

### A: revisiting the same address is unnecessary

Eight samples/condition, medians; write bounds are decimal MB:

| Sequence after common reset | Target µs | Device write lower–upper MB |
| --- | ---: | ---: |
| writeA→readA |22.499|0.0854–0.0856|
| writeA→readB |22.144|0.0848–0.0856|
| readA→writeA |26.464|2.770–3.460|
| readA→writeB |26.207|2.644–3.691|

All read-last samples are21.923–23.012µs; all write-last samples are25.889–26.786µs.
The operation-order selector survives disjoint allocations, so revisiting the
writer's exact lines is not necessary. Small within-regime timing differences
remain; this is not a claim that same/disjoint addresses have identical cost.

### B: the smallest footprint changes writes and latency together

Each row has eight paired candidate/control observations. Write columns are
medians of conservative lower–upper byte bounds, expressed in decimal MB:

| Writer MiB | Target µs | Matched64MiB control µs | Writer-case writes MB | Control writes MB |
| --- | ---: | ---: | ---: | ---: |
|1.5|24.241|26.225|1.643–1.658|2.753–3.280|
|3|25.824|26.158|2.835–3.230|2.728–3.446|
|6|26.995|26.101|2.995–4.820|2.734–3.314|
|12|26.245|26.212|2.809–3.812|2.660–3.598|
|24|unqualified|retained|one clock-window failure|entire contrast excluded from causal conclusions|

For1.5MiB, **all eight pairs both improve latency and conclusively reduce writes**:
its largest write upper bound1,658,624B is below the smallest matched64MiB lower
bound2,500,224B. Median paired latency difference is−1.838µs (range−2.552 to−1.347).
This supports a causal contribution from write-footprint-sensitive memory state;
it does not prove the DRAM-write count is the sole mediator or assign a per-byte
cost. The write intervention also changes primer duration and can change cache/residency.

There is no monotonic capacity curve or demonstrated threshold at queried L2
size. At3MiB all targets are faster, but write bounds overlap the controls; at6MiB
seven of eight targets are slower; at12MiB timing is essentially tied. Wide upper
bounds can include primer-edge traffic and cannot identify a particular cache.
Do not fit a new footprint threshold or infer an L2 mechanism from these values.

### C: millisecond quiescence does not remove the slow write-associated state

The native CPU callback's requested wait is accurate, but CUDA host-function
scheduling adds substantial variable latency. Actual GPU gaps must be used:

| Requested CPU wait µs | Actual GPU gap range µs | Target median µs | Write lower–upper median MB |
| --- | ---: | ---: | ---: |
|0(callback)|198.4–1169.95|25.792|2.569–2.735|
|25|123.97–1105.38|25.859|2.408–2.667|
|250|491.83–1168.04|25.843|2.508–2.647|
|2500|2714.78–3679.90|25.681|2.482–2.649|

No-callback controls have approximately2µs gaps and26.11–26.22µs group medians.
Even2.7–3.7ms of no diagnostic GPU work leaves about2.5MB of interior target-window
writes and a target well above the read-last fast regime. Some latency improvement
exists, but the callback0 control already changes latency, short-gap distributions
overlap, and neither traffic nor target cost forms a clean monotonic drain curve.
The experiment does not resolve a25µs threshold. No shorter-delay implementation
or extra delay values are added after seeing this limitation.

The data do not establish a write-count-only explanatory law: the3MiB targets are faster
without conclusively lower write bounds; in one250µs-request pair the write upper
bound is below its control's lower bound but target latency increases0.291µs.
That individual observation is within measured dispersion, not proof of a second
mechanism, and must remain counterevidence to a forced one-variable explanation.

### Preserved failure, controls and review gap

`footprint_25165824-07`, mode`footprint_25165824`, has a34.582µs-wide wall anchor.
The unchanged1500ns+uncertainty rule produces a36,113ns edge guard for a26.535µs
target, leaving no wholly interior PM sample. `counter-window-failure.json` records
it. All16 raw latency observations in that contrast remain in `analysis.json`'s
`unqualified_contrasts`; its15 individually usable counter windows are retained
but not used as a favorable partial group. `complete=false` is intentional;
`execution_schedule_complete=true`, `qualified_comparison_captures=176`.
The analyzer exits2 to preserve the full-qualification failure.

Twenty-two CPU-only admission/FFI controls and four CLI controls pass, including
this wide-clock case: it must park the whole contrast while retaining timings.
Ruff/format, C++ format/build and frozen-hash checks pass. Graph Tier2 source
coverage reports no recorded gap for the new diagnostics; the reused sampler
has macro parse gaps and was read directly. This is not an exhaustive graph claim. All1393 tracked production
and retained source/artifact paths are unchanged; all2132 previous counter-packet
hashes match before the intentional current-report update. New packet manifests
bind the saved prior report. No numerical qualification tolerance changed.

`review-mcp` is not callable in this session; review workflow is unavailable.
Native review remains disabled by the user and was not substituted. This packet
has local verification, **not completed independent review**. `review-status.json`
records that gap. Nothing is committed or pushed.

## Predeclared protocol

`predeclared-plan.json` and its SHA-256 were written before any target capture in
this pass. `device-query.json` records the actual CUDA device-property query:
RTX3090, CUDA device0, 82 SMs, `L2_cache_size=6291456` bytes. This sets the footprint
values; no GA102 cache capacity was assumed.

There are 96 fresh-process pairs / 192 target captures, including four baseline
pairs before and four after. Each of the eleven intervention contrasts has eight
paired repetitions. The fixed schedule randomizes contrast order within each
repetition; the inherited harness randomizes the two conditions within each pair.
All samples and failures remain evidence; there is no latency filtering, trimming,
adaptive footprint selection, candidate sweep or counter replay.

- A: writeA→readA versus writeA→readB, and readA→writeA versus readA→writeB.
  A/B are separate, nonoverlapping 64MiB PyTorch allocations. Both operations
  request64MiB in every condition, using the same compiled scalar volatile
  reader/writer, 82×256 launch and sink. The allocations, not just region names,
  are different. Reset and target storage do not alias A/B.
- B: readA64MiB→writeA footprint→target, at L2/4, L2/2, L2, 2L2 and4L2:
  1.5/3/6/12/24MiB. Each is paired with the unchanged64MiB writer condition.
  Read size, launch geometry, allocation, addresses' starting point and selector
  structure are fixed. Only writer count changes. Footprint includes the main
  buffer; each reader/writer also writes the same83,968-byte sink, including in A/C.
- C: readA64MiB→writeA64MiB→idle→target. The existing CUDA
  `cuLaunchHostFunc` mechanism runs a native CPU callback with no CUDA calls or
  GPU work inside. Requested delays are0(callback control),25,250,2500µs,
  each paired with no callback. CPU monotonic callback timestamps and actual
  GPU write-end→target-start gaps are retained. The callback adds variable driver
  scheduling latency, so requested delays are not precise GPU gap durations.
  Short-delay overlaps must not be interpreted as a resolved25µs drain test.

Every pair uses the previously qualified, single-pass CUPTI13.0.1 PM protocol:
`dram__bytes_read.sum,dram__bytes_write.sum`,1000ns interval, fresh process/session,
128MiB hardware buffer, KEEP_OLDEST, at most200000 image samples, no replay.
The original native C10 capture-spanning timestamp converter and sampler binary
are reused. `capture-source-sha256.json` pins both helpers and capture source.
Source and binary hashes are independent provenance, not embedded build attestation.
An offline idle-helper rebuild reproduces machine `.text` exactly; ELF RUNPATH
differs(`/tmp` in the measured binary), so full-ELF byte reproducibility is not
claimed. The measurement environment puts the WSL driver directory first in
`LD_LIBRARY_PATH`. Both disassemblies and this limitation are retained in provenance.

All work is prequeued behind the same50M-cycle gate and64MiB reset. Raw CUPTI
activity must prove positive prequeue lead, exactly five GPU kernels in one
context/stream, the two declared primers and unchanged target resources. For C,
there is no intervening GPU kernel: the actual idle gap is measured directly.
External device contexts are not excluded, so “idle” means no diagnostic GPU work.

## Counter and identity admission

`reduction_memory_analyze.py` derives from the retained counter validator. It
preserves the1500ns plus measured clock-uncertainty edge guard, contiguous sample
coverage and additive lower/upper bounds. The lower bound sums samples wholly
inside the shrunken target window; the upper sums intervals touching the expanded
window and can include primer edges. Neither is an exact per-context target count.
PM is device-wide; display/context traffic is not separately excluded on WSL.

The analyzer checks unchanged original input/complete partial/final bits, actual
PTX/cubin hashes, target512threads/grid576/shared64/register40, CUDA/CUPTI/device
identity, primer source/PTX, actual prepared counts/pointers/strides/geometry,
nonaliasing, declared plan/conditions, matched sampler/configuration, decode
completion/overflow/cleanup and raw/converted counters. A numerical tolerance is
never involved. Both sampler and idle helper binaries are archived.

## Interpretation boundary

Changing memory history already causally selects the joint state. To strengthen
the proposed write-state explanation, an intervention must change target-window
write traffic and target latency correspondingly. A footprint trend or persistent
writes after an idle interval does not by itself identify L2 writeback, memory
controller queues, TLBs, display traffic, WSL or driver internals. In particular,
latency movement without write movement must remain counterevidence.

The user designated this as the last bounded hardware-state pass. If the precise
internal source remains unresolved, park `cwk.5` as an environmental benchmarking
confound, retain its established facts, and return Stage5's focus to the qualified
fixed-final component. Never install these primers or synthetic reads in production.

## Reproduction

Build the CPU callback helper with the installed CUDA13 headers and WSL driver:

```sh
g++ -std=c++17 -shared -fPIC -O2 stage5/reduction_memory_idle.cpp \
  -I stage0/python-deps/nvidia/cu13/include -L /usr/lib/wsl/lib \
  -l:libcuda.so.1 -o /tmp/stage5-memory-idle.so
```

The archived `sampler.so` reuses the existing diagnostic adapter; its build
command is retained in `../state-counters/build-sampler.sh`. Run `run.py` from the
repository root only into new capture directories. `commands.jsonl` records the
exact commands/return codes. Analyze with
`python -m stage5.reduction_memory_analyze`. Existing captures are never overwritten.

CUDA13 installed `cuda.h` documents `cuLaunchHostFunc`: it orders the host
function after preceding stream work, blocks subsequent work, considers the
stream idle during the function and forbids CUDA API calls inside it. This
existing facility is used only by diagnostics; PyTorch owns every GPU allocation
and the current stream, and the established readiness/record_stream protections
remain active. [NVIDIA API reference](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__EXEC.html).
