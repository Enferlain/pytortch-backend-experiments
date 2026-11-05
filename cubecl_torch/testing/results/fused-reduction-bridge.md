# Fused reduction bridge: results

OpenSpec change `fused-reduction-bridge` (bead zah.11, step 3). The reference computation is
RMSNorm's weight gradient, `grad_w[c] = Σ_r g[r,c]·x[r,c]·rstd[r]`.

## 1. Burn's fused reduce end to end (tasks 1.1 and 1.2)

**Setup.**
- Burn `.stage0-sources/burn` at 35589bf (2026-11-02), CubeCL fdba0a8, CubeK 06e1c8e; CUDA with
  fusion.
- The program is `fused-reduction-burn/` (`main.rs`). It computes the weight gradient in float32
  (bfloat16 storage is cast up) and checks it exactly against a float64 sum on small-integer inputs.
- At this pin Burn compiles through LLVM/NVPTX, not NVRTC. `exp` and its random generator need
  libdevice (`CUDA_PATH`), so timing inputs are generated on the host.

### Burn does not select its fused reduce

**Every graph tried runs unfused.** Results are exact, but each runs as an elementwise kernel and
then a plain `reduce_kernel<TensorArgs>`, with the product going through memory. Graphs tried:
`g·x·rstd`, `g·x` and `g·g` with row and column sums, and `exp(x).sum(1)`. CubeCL's autotune log
shows only the plain reduce being tuned; `fused_reduce_autotune` never runs.

**The cause is in Burn's planner.** It was traced with logging added to a scratch copy:
- `ReduceFuser` does build the fused reduce: the shapes match and it reports ready.
- Its score comes from its inner elementwise trace, which never registers the reduce op
  (`on_reduce` bypasses `Scoring::register`). So the fused reduce gets no credit for the
  intermediate it stops writing and reading.
- For the weight gradient it scores 210, equal to the plain elementwise fuser.
  `burn-fusion/src/search/block.rs` takes a strictly higher score, so the first fuser in registry
  order wins, and that is elementwise.
- For `exp(x).sum` both score 0. A score of 0 is never taken.
- No upstream issue was found. Our checkout is newer than every commit touching these files.

**With the reduce credited (scratch probe only, `PROBE_CREDIT=1`):**
- Float32 runs as one `reduce_kernel_fused` per call, exact at both shapes.
- Bfloat16 still runs unfused: the casts end the block.

So CubeK's `ReduceArgs` hooks do carry this computation in one kernel. Burn's planner just doesn't
choose it. We don't use Burn's planner; the bridge uses `ReduceArgs` and `reduce_kernel_virtual`
directly.

### Device time (RTX 3080, ncu fixed clocks, µs per call)

`fused-reduction-timing-3080.json`, from `fused_reduction_timing.py`.

| Case | Triton | Burn unfused | Burn fused (probe) |
| --- | --- | --- | --- |
| float32 64×1024 | 3.6 | 11.5 | 29.8 |
| float32 8192×1280 | 120.9 (118.3 + 2.6) | 798.0 | 3426.3 |
| bfloat16 64×1024 | 3.1 | 17.9 | (unfused) |
| bfloat16 8192×1280 | 64.4 (61.7 + 2.7) | 986.2 | (unfused) |

Triton at 8192×1280 is Inductor's split reduction: a looped first stage, then a persistent second
stage over the partials.

**Why Burn is slow: no split along a strided reduction axis.** ncu launch shapes for float32,
from `ncu --metrics gpu__time_duration.sum,launch__grid_size,launch__block_size,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active`
on the program's `time` mode (last launch of each kernel):

| Kernel | Grid × block | Time (µs) | DRAM % | Warps active % |
| --- | --- | --- | --- | --- |
| reduce, 8192×1280 | 2 × 256 | 617 | 9 | 13 |
| fused reduce, 8192×1280 | 2 × 256 | 3424 | 3 | 11 |
| elementwise, 8192×1280 | 10240 × 256 | 181 | 92 | 84 |
| reduce, 64×1024 | 1 × 256 | 7.9 | 5 | 16 |
| fused reduce, 64×1024 | 1 × 256 | 29.6 | 2 | 17 |

- **The reduce launches far too few units.** With the reduction axis strided, CubeK's inferred
  blueprint gives each unit a vector of 4 columns over the whole reduction: 320 units for 1280
  columns, two cubes on a 68-SM GPU. Nothing splits the reduction axis.
- **Burn's fused read costs about 5.5× its plain read on top** (3424 against 617 µs). That is its general
  `fuse_on_read`, which goes through a reference layout per element.

### What this means for the bridge

- **The mechanism works, unchanged.** Burn's planner scoring is not our concern.
- **Inductor's split is required for column reductions.** We receive the two split kernels (the
  first stage has 40960 outputs on the 3080) and lower each through CubeK. Without the split,
  CubeK's inferred blueprint has nothing to spread across the GPU here. The split is in the "grey
  zone" the architecture consumes by default.
- **Whether CubeK should split such reductions itself** is a CubeK question, recorded for zah.10's
  work (CubeK offers in-band structures). This change doesn't add one.
- **Our replay's per-element cost is a risk.** Group 5 measures it against Triton per kernel.
