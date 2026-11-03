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
