# torch-runtime-bridge: measurements

Machine: AMD Ryzen 9 3950X, WSL2. `cuda:1` RTX 3080, `cuda:0` RTX 3090. Bridge at CubeCL `fed6c0c0`
plus `fp.patch` and `bridge.patch`, CubeK `50cb94b2`. Timing against Triton runs in the Triton
reference venv: Python 3.14.5 free-threaded build, torch 2.14.1+cu130 (as `stage0/.venv`), Triton
3.8.0, whose import turns the GIL on; the runs below set `PYTHON_GIL=0`, so it stays off.

## Launch host cost (task 3.6)

Warm host time of one launch of `y = x + 1` over 1024 float32 values, the bridge's test kernel
against a Triton kernel with the same arguments, in one process
(`python -m cubecl_torch.testing.launch_cost`). Each sample is the mean of 500 back-to-back
launches; the kernels' samples are interleaved; median of 31 samples, with the interquartile
range as noise. `bridge` is the default launch, which, like PyTorch's operators and Inductor's
Triton kernels, does not record its tensors on the stream; `bridge recorded` is the opt-in that
does (design decision 3).

| GPU | Bridge median (IQR) | Bridge recorded median (IQR) | Triton raw launch median (IQR) |
| --- | --- | --- | --- |
| RTX 3080 | 18.5 µs (17.5–22.2) | 22.4 µs (21.3–27.9) | 19.9 µs (19.1–25.9) |
| RTX 3090 | 18.1 µs (17.6–18.7) | 21.1 µs (20.7–21.9) | 18.5 µs (18.2–19.0) |

Raw JSON: `launch-cost-3080.json`, `launch-cost-3090.json`. The 3080 run includes the bridge's
context guard (every call restores the thread's CUDA context); the 3090 run predates it, and the
3080 medians moved by under 0.2 µs with it.

**The spec's launch-cost requirement is met by the default launch**: its median is below Triton's
on both GPUs (by 1.4 µs on the 3080, by 0.4 µs, inside the noise, on the 3090). Recording costs
3–4 µs per launch and puts the opt-in 2.5–2.6 µs above Triton, about the noise of these runs.
An earlier run, with recording always on, measured the bridge 3–5 µs above Triton.

Time on this machine is bimodal between processes (the same configuration measures about 13 or
about 18 µs of CPU time per launch, pinned or not), so time is compared only within one process.
User-mode instruction counts per warm launch (`launch_cost.py --instructions`: `perf stat`
counting only the launch loop through perf's control descriptors, five processes each, RTX 3080,
`launch-cost-instructions.json`):

| Path | Instructions per launch |
| --- | --- |
| Bridge | 58,647–61,769 |
| Bridge recorded | 62,393–66,892 |
| Triton raw launch | 83,465–87,185 |
| `torch.add(out=)` | 34,279–41,762 |

Counts vary between processes too, by up to about 15% in other runs (no cause found: not Python's
garbage collector, not process start-up, which the loop-only count excludes), so a single process
is no measure. `test_bridge.py::test_launch_host_cost_against_triton` guards the fewest of three
bridge processes against the fewest of Triton's recorded ones; it does not guard time. The bridge
retires about 30% fewer instructions than Triton's launch yet takes about as long, which suggests
cache behaviour in CubeCL's launch path rather than more work (not profiled further).

What the launch costs, and what was removed to get here:

- Python wrapper: stream lookup through `torch._C._cuda_getCurrentStream` (0.2 µs, against
  3.4 µs for `torch.cuda.current_stream`) and cached `Stream` objects; recording both tensors on
  the stream (3–4 µs) only when asked.
- Bridge call about 13 µs: two server calls (the launch and the failure check, about 1,000
  instructions). Handles for external memory are cached by stream, address and size, which took
  the per-launch server calls from six to two.

## Concurrency (task 3.5)

Four threads, each on its own stream, launching at once with the GIL off:
`test_bridge.py::test_concurrent_threads` passes (correct and ordered results). Per-launch host time
with four threads is about 280–295 µs per thread (`bridge_4_threads` in the JSON, default launch),
against about 18 µs for one: worse than four launches one after another (about 75 µs), so more
than serialization.
PyTorch's own launches, `record_stream` and stream lookup scale across the same threads, which
points at the bridge's path through CubeCL, most likely its one lock per device (the calling-thread
transport's reentrant mutex), but that is not profiled. With the GIL on, the same run is about
500 µs. On a bead, with the design's escape, a lock per stream.

## Capture and allocation (tasks 3.3, 3.4)

`test_bridge.py`: ordering on the default and a side stream and across streams, an input freed while
its launch is queued under allocator pressure, the bridge's own allocations counted by PyTorch's
allocator statistics, capture and replay with a shape first seen inside the capture and after eager
use, and a second graph of the same shape. Three CubeCL changes were needed and are in
`bridge.patch`:

- **Launch metadata without host memory on bound streams.** The dynamic part of a launch's info
  (shapes, strides) was staged through pinned host memory: refused inside a capture
  (`cuMemAllocHost`), and a recorded copy would read host memory whose life only the owner's graph
  knows. On a bound stream it is now written by a kernel carrying the words in its parameters
  (`cubecl-cuda/src/compute/words.rs`), which captures as a node with the words baked in.
- **No metadata cache while the owner captures.** A buffer written by a node of the owner's graph
  holds its words only once that graph runs, and the graph needs it for its own life, so on a bound
  stream being captured (`cuStreamIsCapturing`) the cache is neither read nor filled. Without this,
  an eager launch of a shape first seen in a capture read zeroed metadata
  (`bound_streams.rs`, checked by disabling the bypass).
- **Failures on external memory.** A launch that fails records the failure on its outputs; external
  memory had no slice to hold it, so the failure was lost and the bridge returned success with
  garbage outputs. External memory now carries its failure claims, and the bridge checks its
  handles after every launch, raising the failure.

## Numerical band of the bridged reduce (task 4.2)

`python -m cubecl_torch.testing.cubek_band ROUTINE [--parallel]`: `welford_band.measure` over the
40 stored `row_sum` references with one bridged plan pinned wherever its structural rule admits the
kernel (RTX 3090). Twenty-four references are new (`welford_reference` `bridge_sum_cases`, `--new`;
stored cases unchanged byte for byte): float32, float16 and bfloat16 at 2048×4099, 300×4096, 7×2049
and 64×31, normal and offset data, eight inputs per case below 1024 rows. The sum consumer kernels
fuse their consumer into the reduction, which the structural rule refuses, so they have no verdict.
Raw results: `cubek-band-{unit,plane,cube}[-parallel].json`.

Parallel output vectorization changes no result (each pair of files is identical).

**16-bit row sums (16-bit in and out) are in Triton's band for every routine**: float16 and
bfloat16 at 2048×8192, 4096×2048, 2048×4099, 300×4096, 7×2049 and 64×31, 24 of 24 cases.

**Float32 row sums are outside**, by the sign test and the paired mean (p far below 0.01), on
almost every class. CubeK's mean row error against Triton's:

| Routine | 2048×8192 | 256×32768 | 4096×2048 | 2048×4099 | 300×4096 | 7×2049 | 64×31 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unit | 7.9–10.7× | 13.4–20.4× | 4.5–5.4× | 10.6–30.2× | 6.0–7.5× | 8.5–23.2× | 1.7–2.5× |
| plane | 1.23–1.67× | 1.26–2.46× | 1.18–1.19× | 1.37–2.22× | 1.19–1.37× | 1.17–1.90× | in band |
| cube | 1.11–1.29× | 1.24–1.27× | 1.06–1.26× | 1.17–1.48× | 1.08–1.26× | in band (normal), 1.54× (offset) | in band |

(ranges over the normal and offset data). The order follows the number of independent partial sums
per row: one chain per row for `unit`, one plane's lanes for `plane`, an inferred cube of at most 8
planes for `cube`, against Triton's up to 2048 lanes (16 warps) on long rows. On bead `zah.7`.

### Attribution: partial sums and merge order (bead `zah.7`)

Triton's launched configurations for every looped float32 row sum above (Triton venv, splitting
off): `R0_BLOCK` 2048, 16 warps, so 2048 partial sums per row merged by `tl.sum`'s tree; 64×31 runs
persistent with 2 warps. CubeK's inferred cube routine runs at most 8 planes of 32 units, each with a
4-lane vector accumulator: 1024 partial sums, then a plane sum and, with planes, an in-order scan
over the 8 plane results. Blueprints forced through the bridge (`cubek_band --planes`; diagnostic
plans, not in the catalog). Mean row error against Triton's, `*` outside the band:

| float32 case | cube (1024, in order) | cube_unit 8 (1024, tree) | cube_unit 16 (2048, tree) | cube 32 (4096, in order) | cube_unit 32 (4096, tree) |
| --- | --- | --- | --- | --- | --- |
| 2048×8192 normal | 1.105* | 1.048* | 1.013 | 1.275* | 0.968 |
| 2048×8192 offset | 1.288* | 1.181* | 1.163* | 1.694* | 1.158* |
| 256×32768 normal | 1.272* | 1.224* | 1.000 | 1.267* | 0.938 |
| 256×32768 offset | 1.245* | 1.127* | 1.123* | 1.626* | 1.135* |
| 4096×2048 normal | 1.059* | 1.025 | 1.021 | 1.170* | 1.021 |
| 4096×2048 offset | 1.256* | 1.185* | 1.184* | 1.437* | 1.184* |
| 2048×4096 normal | 1.059* | 1.005 | 0.997 | 1.310* | 1.019 |
| 2048×4096 offset | 1.249* | 1.167* | 1.165* | 1.689* | 1.184* |
| 2048×4099 normal | 1.170* | 1.144* | 1.042* | 1.280* | 0.984 |
| 2048×4099 offset | 1.481* | 1.020 | 1.033* | 2.728* | 1.041* |
| 300×4096 normal | 1.078* | 1.035 | 1.004 | 1.346* | 0.991 |
| 300×4096 offset | 1.259* | 1.172* | 1.178* | 1.677* | 1.154* |
| 7×2049 normal | 0.995 | 1.039 | 0.955 | 1.200 | 0.875 |
| 7×2049 offset | 1.535* | 0.937 | 0.961 | 2.447* | 0.973 |
| 64×31 (both) | in band | in band | in band | in band | in band |

16-bit rows stay in band for every variant. A forced blueprint with planes validates only at 32
planes: CubeK's forced-blueprint check compares the plane accumulator count with `cube_dim.x` (the
plane width) instead of `cube_dim.y` (the planes), so 2048 partial sums merged in order cannot be
forced; 4096 in order stands in for it.

- The merge order matters most: at the same 1024 partial sums a tree takes five more classes into
  band, and 4096 partial sums merged in order are worse than 1024 (up to 2.7×).
- With a tree, more partial sums close the gap on zero-mean data: at 16 and 32 planes every normal
  class is at Triton's error or below.
- Offset data stays 1.12–1.18× outside whatever the partial count. The generated CUDA adds each
  accumulator's 4 vector lanes last and in order (`((0 + l0) + l1) + l2) + l3`, from `vector_sum` in
  CubeK's `Sum::to_output_parallel`), after the tree; Triton sums a thread's elements first and its
  tree spans the threads. With a large mean every lane total is a quarter of the row's sum, so the
  final, largest additions run in order. Not yet confirmed by a forced change.

### With the CubeK patch stack (`lanes-first`, `plane-tree`, `forced-planes`)

Same harness and references, 3090, after the three CubeK patches (`native/patches/cubek`). The
files without `-pin` are these runs; the `-pin` files are the runs above, at the CubeK pin. 16-bit
classes stay in band for every variant. Float32, ratio of mean error to Triton's (`*` outside):

| Variant | Partials | In band (of 40) | Float32 outside |
| --- | --- | --- | --- |
| `cube` inferred (8 planes) | 1024 | 37 | 2048×8192 normal 1.03, 256×32768 normal 1.26, 2048×4099 normal 1.15 |
| `cube` forced 16 planes | 2048 | 38 | 2048×4099 normal 1.03, offset 1.03 |
| `cube` forced 32 planes | 4096 | 39 | 2048×4099 offset 1.05 |
| `cube_unit` forced 16 planes | 2048 | 38 | as `cube` forced 16 |
| `plane` | — | 27 | unchanged (all float32 but 64×31 and 7×2049 offset) |
| `unit` | — | 24 | unchanged (every float32 class) |

Offset data, 1.12–1.30× outside before, is in band everywhere the rows vectorize: the lanes were the
cause. Triton's choice for every shape here but 64×31 is `R0_BLOCK` 2048 with 16 warps, the same on
both GPUs (four elements per thread, folded in the thread first); `cube` with 16 planes is that
structure and passes every float32 class whose rows take 4-wide vectors. 2048×4099 rows cannot be
vectorized, so a CubeK unit holds one partial (512 in all) where a Triton thread holds four: the
remaining gap. The `plane` and `unit` routines sum each lane sequentially along the row, which no
merge order changes.

### The cube with Inductor's mapping (`unit-elements`, `mapping="inductor"`)

Inductor gives a looped INNER row sum one fixed config; `testing/data/inductor_row_configs.json`
records it (`testing.inductor_configs`, both GPUs): `R0_BLOCK` 2048 with 16 warps for every shape
here over 1024 elements, the same on the 3090 and the 3080, and a split reduction for 16×100000. The
`cube` plan with `mapping="inductor"` forces that blueprint per row (`bridged.inductor_blueprint`:
16 planes, 4 elements per unit through `unit-elements.patch`, whatever the load width).

`cubek-band-cube-inductor.json`, 3090: every class it takes is in band, 34 of 34 (float32 ratios
0.95–1.05, 2048×4099 0.97 and 1.02; 16-bit 1.00, or both errors zero). 64×31 has no verdict by
design: Inductor reduces rows up to 1024 elements persistently, with a config autotuned or chosen by
row count, so the mapping refuses them and the kernel keeps its other candidates
(`independent_planes` ran, in band).

## Correctness through the Inductor route (task 4.4)

`test_cubek_route.py`: each of the seven bridged plans pinned in a process with the runtime guard
installed, `x.sum(-1)` through `torch.compile(backend=ci.compile_backend)` at 2048×8192, 2048×4099,
300×4096, 7×2049 and 64×31, float32, float16 and bfloat16, normal and offset data (mean 1000;
mean 4 for float16, whose sums of mean 1000 overflow to inf, as Triton's do). On every class a plan
is admitted for, the pinned plan is what selection ran and every row is within
`64·eps32·Σ|x| + eps_out·|Σx|` of the float64 sum; the admission rule is checked against the band's
classes there too. The mapped plan skips 64×31, which it refuses (rows of at most 1024 elements).
All seven pass.

## Timing against Triton and our best plan (task 4.5)

`testing/cubek_timing.py`: `ncu` (fixed clocks), one warm call of each pinned plan in its own NVTX
range, three runs each, `x.sum(-1)`. "Native" is first-use selection with the bridged plans
withdrawn, pinned; the bridged plans are the ones admitted for the class. Medians in µs (all runs in
`cubek-timing-3080.json`, `cubek-timing-3090.json`; runs agree within 0.5 µs except Triton's first
small-shape run).

With the fixes below applied (`plane-dim` and `static-sizes` in the CubeCL stack,
`folded-plane-sum` in CubeK's), and the bridge compiling each static-shape sum for its sizes:

| Case | 3080 Triton | 3080 native | 3080 mapped cube | 3080 best of Burn's grid | 3090 Triton | 3090 native | 3090 mapped cube | 3090 best of Burn's grid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| float32 2048×8192 | 95.2 | 96.6 | 95.0 | — | 81.9 | 82.1 | 77.6 | — |
| float32 2048×4099 | 49.2 | 49.9 | 49.1 | — | 40.3 | 41.2 | 40.7 | — |
| float32 300×4096 | 9.1 | 9.3 | 9.2 | — | 7.8 | 8.1 | 7.9 | — |
| float32 7×2049 | 2.4 | 2.8 | 2.5 | — | 2.4 | 2.8 | 2.6 | — |
| bfloat16 2048×8192 | 48.7 | 48.9 | 48.8 | 48.7 plane | 39.9 | 40.2 | 40.3 | 40.4 cube |
| bfloat16 2048×4099 | 25.9 | 26.2 | 25.7 | 25.8 cube | 21.3 | 21.8 | 21.4 | 21.3 cube parallel |
| bfloat16 300×4096 | 5.7 | 5.9 | 5.8 | 5.9 cube | 5.1 | 5.4 | 5.5 | 5.2 cube |
| bfloat16 7×2049 | 2.3 | 2.6 | 2.4 | 2.4 cube | 2.4 | 2.7 | 2.4 | 2.4 cube parallel |
| bfloat16 64×31 | 2.0 | 2.1 | — | 2.1 plane | 2.1 | 2.1 | — | 2.1 plane |

On rows of 4096 or more the mapped cube is within 2% of Triton, or faster, except bfloat16
300×4096 on the 3090 (1.09×; Burn's cube is 1.04× there). On 7×2049 it is 0.1–0.2 µs over
(1.01–1.08×), and below our native plans everywhere but bfloat16 2048×8192 and 300×4096 on the 3090.
Triton's float32 2048×8192 on the 3090 measured 77.8 µs in the first run below and 81.9 µs in this
one; our native plan moved with it (79.4, 82.1), so the run's clocks differ, not the kernels.

Before the fixes, every bridged plan was slower than Triton on every class:

| Case | 3080 Triton | 3080 native | 3080 mapped cube | 3080 best of Burn's grid | 3090 Triton | 3090 native | 3090 mapped cube | 3090 best of Burn's grid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| float32 2048×8192 | 95.2 | 96.6 | 96.5 | — | 77.8 | 79.4 | 79.2 | — |
| float32 2048×4099 | 49.0 | 49.8 | 51.5 | — | 40.3 | 41.5 | 43.3 | — |
| float32 300×4096 | 9.1 | 9.4 | 11.3 | — | 7.8 | 8.2 | 10.1 | — |
| float32 7×2049 | 2.3 | 2.9 | 3.6 | — | 2.4 | 3.2 | 3.6 | — |
| bfloat16 2048×8192 | 48.7 | 49.1 | 64.4 | 50.1 plane | 40.0 | 40.5 | 55.0 | 41.7 plane |
| bfloat16 2048×4099 | 25.7 | 26.5 | 38.9 | 28.2 plane | 21.5 | 21.9 | 32.7 | 24.0 plane |
| bfloat16 300×4096 | 5.7 | 6.2 | 11.4 | 7.0 plane | 5.1 | 5.6 | 10.0 | 6.8 plane |
| bfloat16 7×2049 | 2.4 | 2.9 | 3.6 | 3.6 cube | 2.4 | 3.0 | 3.6 | 3.6 cube |
| bfloat16 64×31 | 2.0 | 2.2 | — | 3.0 plane | 2.1 | 2.2 | — | 3.0 plane |

There every bridged plan was slower than Triton and than our native plan. The cause, from `ncu` and the generated CUDA at
float32 300×4096 on the 3080 (bead `zah.9`): the same geometry as Triton (300 blocks of 512
threads, 85% achieved occupancy against 81%) but 5.6× the executed instructions (1,576,200 against
279,300: 328 per warp against 58):

- the row offset comes from a generic rank loop with runtime 32-bit division and modulo by
  shapes read from metadata;
- every load is masked with a per-lane select (the bridge forces `BoundChecks::Mask` and
  `IdleMode::Mask`); dropping both cut instructions by 5% only (1,489,800);
- the plane sums are CubeCL's polyfill, a `while (offset < min(PLANE_DIM, CUBE_DIM))` loop that
  is not unrolled (`warpSize` is a runtime value on CUDA), run twice (within a plane, then
  across planes in plane 0) on all four lanes, although `lanes-first` left three of them zero:
  40 shuffles a thread where Triton's tree has about 10.

The fixes, each at the level that produces the behaviour (bead `zah.9`; ncu at float32
300×4096 on the 3080, executed instructions and time of the mapped cube):

| Stage | Instructions | µs |
| --- | --- | --- |
| CubeK pin and our numerics patches | 1,576,200 | 11.6 |
| `plane-dim` (CubeCL): `PLANE_DIM` is the compiled warp size, so the plane polyfill unrolls and plane positions need no runtime division | 1,062,900 | 11.6 |
| `folded-plane-sum` (CubeK): only the folded lane takes part in the plane sums | 1,038,000 | 11.1 |
| `static-sizes` (CubeCL): the kernel is compiled for its sizes, as Inductor writes static sizes as literals: shapes, strides and index scalars (CubeK's reduce axis among them) are constants, so the rank loop and divisions fold and the row loop has a fixed trip count | 351,300 | 9.3 |
| Triton | 279,300 | 9.1 |

The bridge asks for `static-sizes` when the descriptor's rows and extent are constants (Inductor's
static shapes), so one kernel is compiled per shape, as Triton's. `plane-dim` also changes the
source of every native plane plan (64 corpus cases, `kernel.cu` and `kernel.ptx` only: the loop
bound `warpSize` becomes 32, the same shuffle offsets), so values are unchanged; the CubeK band
was rerun with all three and is unchanged for every plan (`cubek-band-*.json`).

First-call cost (`testing/cubek_first_call.py`, `cubek-first-call-3080.json`): wall time of the
first `torch.compile` call of `x.sum(-1)`, compile and first-use tuning included, in a fresh process
with empty caches (CUDA and Dynamo warmed on another graph), median of five, RTX 3080. "Today" is
the same route with the bridged plans withdrawn.

| Case | Today (ms) | Bridged offered (ms) | Bridged plans offered |
| --- | --- | --- | --- |
| float32 2048×8192 | 1324 | 1746 | mapped cube |
| float32 2048×4099 | 796 | 943 | mapped cube |
| float32 300×4096 | 2188 | 2261 | mapped cube |
| float32 7×2049 | 1282 | 1453 | mapped cube |
| float32 64×31 | 1193 | 1202 | none |
| bfloat16 2048×8192 | 3560 | 6570 | mapped cube and Burn's six |
| bfloat16 2048×4099 | 1723 | 3231 | mapped cube and Burn's six |
| bfloat16 300×4096 | 2419 | 3905 | mapped cube and Burn's six |
| bfloat16 7×2049 | 1496 | 2903 | mapped cube and Burn's six |
| bfloat16 64×31 | 1466 | 2478 | Burn's six |

Each bridged plan costs its in-process compile and its first-use timing; Burn's unit routines,
3–31× Triton's time on every class but 64×31, account for much of the 16-bit cost.

