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

## 2. Accuracy band (task 5.1)

**Setup.**
- The cases are `fused_reduction_reference.py`: the timing harness's `weight_gradient` at 64×1024
  and 8192×1280, in float32, float16 and bfloat16, normal inputs.
- Triton's references are recorded per GPU (`data/fused_reduction_triton-<gpu>.pt`). Inductor
  splits the 8192×1280 column sum 32 ways on the 3080 (stage 1 has 40960 outputs over 256 rows)
  and 64 ways on the 3090 (81920 outputs over 128 rows), so the comparison is made on the GPU the
  reference came from.
- Each run pins one fused routine on every kernel the coverage admits:
  `cubek_band ROUTINE --fused --gpu GPU`. Every case got its verdict from the pinned plan.
- Results: `fused-band-<routine>-<gpu>.json`. The verdicts are the same on both GPUs.

**Verdicts.** Every float16 and bfloat16 case is in band for every routine. Float32 mean error,
relative to Triton's:

| Routine | 64×1024 | 8192×1280 (3080 / 3090) |
| --- | --- | --- |
| `plane` | 0.994, in band | 0.81 / 0.90, in band |
| `cube_unit` | 1.033, **outside** | 0.78 / 0.91, in band |
| `unit` | 1.833, **outside** | 2.28 / 2.24, **outside** |

**Causes.** These were found from CubeK's inferred plans (`CUBECL_TORCH_FUSED_TRACE=1` prints
each launch's blueprint, cube count and vector sizes), Triton's TTGIR and PTX, and a CPU
emulation of both summation orders on the band's own inputs. The emulation reproduces each
measured mean error to three digits.

- **Triton, 64×1024:**
  - a `[32 × 64]` tile with layout `sizePerThread [4,1]`, `threadsPerWarp [8,4]`,
    `warpsPerCTA [1,8]`;
  - each thread holds 4 columns and rows `r` and `r+32`, and adds them in-thread with the
    second product contracted (`fma`, one rounding fewer);
  - a tree over 4 lanes and 8 warps finishes the sum.
- **`plane` (fixed).** The order is the same as Triton's: each lane holds rows `l` and `l+32`,
  then a plane tree. But CubeK guarded every read with a branch (`BoundChecks::Branch`) even
  though 64 rows divide evenly by 32 lanes. The branch's select sat between the read's last
  multiply and the accumulation, so NVRTC could not contract them. That cost 5% more error
  (emulated 4.82e-7 against 4.59e-7, measured 4.82e-7).
  - CubeK drops checks a shape doesn't need only at `AutotuneLevel::Full`, which its docs give
    for callers outside autotune. That fits the bridge: each launch is planned from its own static
    shape, and the blueprint is part of the kernel's identity.
  - The fused launch now passes `Full`, and `plane` runs `fma` like Triton.
- **`cube_unit`, 64×1024 (open).** The inferred blueprint gives each unit one element
  (`unit_elements: 1`, 256 shared accumulators for 64 rows), then a shared-memory tree. With no
  in-unit pair there is nothing to contract, so every add rounds the product separately.
- **`unit` (open).** Each unit sums its 4 columns over the whole reduction one row after another
  (64 rows, or 256 per split stage-1 output). Emulated 8.47e-7, measured 8.47e-7. Triton never
  sums more than 2 (64×1024) or 32 (stage 1, `R0_BLOCK` 8 over 256 rows) elements in sequence
  before a tree.

Both open cases are CubeK's summation order for these blueprints. Under the spec they are not
closed with a forced blueprint or an admission rule. They are recorded on bead zah.11.1.
