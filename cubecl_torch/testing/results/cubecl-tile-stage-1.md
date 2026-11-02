# cubecl-tile stage 1: results

OpenSpec change `cubecl-tile-stage-1`. Timing is `ncu` (fixed clocks), one warm call in an NVTX
range, plans pinned, three runs each. No second GPU job ran; the display GPU was on the 3090.

## Group 2: one real kernel through the production route

Float32 RMSNorm with a weight, `x * rsqrt(mean(x²) + 1e-6) * w`, 2048×8192, RTX 3080. The tile plan
is `tile_row_sum` at `[32,16,1]`, vector width 4 with scalar recipes, and upstream's summation
order. It is not yet in Triton's accuracy band (group 5), so this measures speed only.

| Kernel | Time (µs) | DRAM read (MB) |
|---|---|---|
| Tile route | 204.6, 206.7, 204.5 | 73.6, 73.8, 73.9 |
| `shared_tree_compensated` 512 | 224.2, 224.8, 224.5 | 85.5, 85.3, 85.6 |
| Triton | 231.9, 232.6, 222.0 | 85.3, 85.5, 79.7 |

The spike's `fused_probe` measured 216.8 µs. The route is faster because the per-element pass is
`Kernel::stores`, which stores vectors, where the probe stored scalars. Maximum relative error
against float64 is 2.69e-7 (spike: 2.6e-7).

Gate outcome (decision 2): met. The kernel is built with `KernelBuilder`, takes one pointer per
buffer plus one `info` pointer with `grid_constants: false`, uses the standard metadata (32 bytes
here) and 64 bytes of shared memory, and the compiler links no CUDA runtime. Bead `f05.9`'s
grid-constant ABI does not arise.

What the gate changed:

- The block is `[32,16,1]`, not `[512,1,1]`. The DSL reads a plane's index from the cube's y
  dimension, so on `[512,1,1]` every unit saw plane 0 and each row kept 1/16 of its sum.
  `launcher.c` allows the new shape; its calling convention is unchanged.
- `Kernel` steps a row by the whole block with the linear unit index, since its other plans
  cooperate along x alone.
- The route replaces only the reduction phase. All stores stay in `Kernel::stores`.

## Group 3: bounds and vector recipes

Upstream has two problems on a runtime row. On a dynamic axis `Procedural` sets `bound` to
`u32::MAX` and turns bounds checks off, so a partial step is not masked at all, which gives wrong
values as well as out-of-bounds reads. And a checked read evaluates the recipe, then selects. Our
copy bounds recipes by the runtime extents (`Procedural::within`) and evaluates at a redirected
coordinate.

The `validate_reads` option reports every load and store past its buffer, through the production
launcher.
At RMSNorm 2048×4099, width 1, before the redirect: 29 reports, indices 8,394,752 to 8,394,780,
the spike's range. After it: none, with the same output. No reports at 64×31, 256×100 and
16×65539, or at width 4 on whole steps. `validate_reads="unguarded"` drops the bounds and must
report; it finds the same 29. Reports print a length of 0 because CubeCL's message uses `%u` for two
64-bit arguments, so the second prints the index's high word; the comparison itself is correct.

Inductor splits rows past 65536 by default, so 65539 is tested as one row with `split_reductions`
off.

A zero-length row does not reach a tile kernel through Inductor: it fills an empty reduction, and
its positive shape symbols make the launch contract refuse a 0. Launched directly from a descriptor
without that declaration, the kernel overwrites its output with zeros and touches nothing else.

RMSNorm 2048×8192, RTX 3080:

| Plan | Time (µs) | DRAM read (MB) |
|---|---|---|
| Width 4, vector recipe | 206.8, 208.2, 208.0 | 74.1, 74.6, 75.1 |
| Width 4, scalar recipe (group 2) | 204.6, 206.7, 204.5 | 73.6, 73.8, 73.9 |
| Width 1, masked | 212.8, 214.8, 213.4 | 79.4, 79.3, 79.4 |

Sum consumer `x * x.sum(-1, keepdim=True)` 2048×8192, RTX 3080:

| Plan | Time (µs) | DRAM read (MB) |
|---|---|---|
| Width 4, vector recipe | 194.0, 194.3, 194.5 | 67.1, 67.1, 67.1 |
| Width 1, masked | 197.1, 196.4, 196.1 | 69.1, 69.0, 69.1 |
| Triton | 202.8, 201.9, 201.1 | 67.1, 67.1, 67.1 |

Vector recipes issue a quarter of the load instructions but gain no time on RMSNorm: the kernel is
memory-bound, and the difference is within the placement noise of memory `timing-placement-and-clock`.
A width 4 plan refuses a row that is not whole steps at every launch. All of this is speed only:
the plans keep upstream's summation order.

## Group 4: summation order

`reduce_row` does its final combine in Triton's order: a sum per vector lane along a plane's steps,
units xor 16→1, planes then lanes as trees. On the spike's inputs the kernel's output equals
`order_model.py`'s prediction bit for bit on every row, normal and offset. Mean row-sum error (the
middle column is the model's prediction for upstream's order, which sk8 also matched on the GPU):

| Data | Upstream order | This order | Triton |
|---|---|---|---|
| Normal | 8.25e-6 | 7.04e-6 | 6.96e-6 |
| Offset | 0.261 | 0.191 | 0.187 |

Band test on the RTX 3090, plan pinned, sign and mean-difference tests at p < 0.01. Every stored
`row_sum` and `sum_consumer` case the plan can run; `tile-band-width{1,4,8}.json` hold the results.

```bash
stage0/.venv/bin/python -m cubecl_torch.testing.tile_band 4 --out .../tile-band-width4.json <cases>
```

| Plan | Inputs | Verdict |
|---|---|---|
| Width 4 | float32 2048×8192: sum and consumer, normal and offset | In band |
| Width 4 | float32 256×32768: sum, and offset consumer | In band |
| Width 4 | float32 256×32768: consumer, normal | Outside, p = 1.4e-4 / 1.4e-5 |
| Width 8 | float16, bfloat16 2048×8192 and 300×4096 | In band |
| Width 1 | float16, bfloat16: consumer at 37×129 and 300×4096; sum and consumer at 2048×8192 | In band |
| Width 1 | float32, offset: sum and consumer at 2048×8192 and 256×32768 | In band |
| Width 1 | float32 2048×8192 and 256×32768, normal | Outside, p from 5e-8 to 6e-17 |

These misses are unexplained so far. Likely causes, not measured: at 32768 each lane runs 16 steps in
sequence, and width 1 has 512 accumulators against Triton's 2048. The second does not separate its
cases, since width 1 is in band on offset data with the same count. Both classes stay on today's
plans (bead cie).

RTX 3080, 2048×8192, with this order:

| Kernel | Time (µs) | DRAM read (MB) | Before (µs) | Triton (µs) |
|---|---|---|---|---|
| RMSNorm, width 4 | 195.6, 195.4, 197.4 | 67.2 | 206.8–208.2 | 222.0–232.6 |
| Sum consumer, width 4 | 194.2, 194.0, 193.8 | 67.1 | 194.0–194.5 | 201.1–202.8 |
| Sum consumer, width 1 | 199.0, 196.9, 198.1 | 69.3 | 196.1–197.1 | 201.1–202.8 |

The order costs no time at width 4, and RMSNorm gains: its DRAM reads fall from about 75 to 67 MB.
The kernel source grew from 308 to 655 lines, since the trees are unrolled; its compile time is
for task 7.2.

## Group 5: fused bodies

The route needed no change for this group. `sum`, `mean`, the sum consumer, a weighted sum times a
0-d tensor and RMSNorm each run as one fused tile kernel, at width 4 (2048×8192, 300×4096) and width
1 (2048×4099, 7×2049, 64×31) in float32, and at widths 8 (300×4096) and 1 (300×4099) in float16 and
bfloat16. The consumer and RMSNorm store per element in the reduction's own kernel. Every output is
within a bound derived from the kernel: two units of the output's rounding, one subnormal step, and
64 units of float32 rounding over the kernel of the magnitudes, tight enough that one dropped
element fails. Every access is in bounds.

Load hints come from the emitter. On the sum consumer the source has retained loads only before the
first barrier and streaming loads only after it. Its 67.1 MB of DRAM reads, from group 4's `ncu`
runs, is Triton's figure and one pass over `x` (2048 × 8192 × 4 bytes), below the 70 MB the task
quotes from the spike.

Shape symbols are the kernels' runtime scalars. One compiled kernel serves 2048×8192, 1024×8192 and
2048×4096 with no further compiler invocation, and three fresh compiler processes write
byte-identical IR, CUDA and PTX.

## Group 6: catalog and legality

Tile plans are in `catalog()` and offered only on the classes measured in Triton's band, keyed on what
is summed, since that and the width fix the order: 16-bit values (a 16-bit load, cast or not) at
widths 1 and 8 (width 8 only since group 7), and float32 data (float32 loads, or anything computed, such as `x * x`) at width 4
on whole steps, rows of 2048 to 8192. Every other kernel records each tile plan among its rejected
candidates with its reason. Below 2048 the 512-unit rule applies to them as well. Within that range
the measured row lengths are 2048, 4096 and 8192 (and 300×4096 for 16-bit consumers); lengths in
between are taken as the same class.

Inductor's ordinary split reductions carry no marker, so their partial kernels are row sums to the
legality and can be offered tile plans on their chunk length. Only the reference512 pairs, which
carry a reduction context, are refused.

The classes were measured before the rule was written. New Triton references (`welford_reference`,
`--new`; stored cases unchanged byte for byte): float32 4096×2048 and 2048×4096, float16 and
bfloat16 4096×2048. Results in `tile-band-width4-short.json`, `tile-band-width8-short.json` and
`tile-band-width1-short.json`: all in band.

Every family (`sum`, `mean`, the sum consumer, a weighted sum, RMSNorm) runs under index validation
at row lengths 31, 100, 4099 and 65539 (the last with splitting off) with no reports.

Six existing tests assumed no tile plan. Five keep their subject by excluding tile plans (`wide()`
covers the wide-reduction plans only; tile keeps the 2048 floor by name) or by expecting the storage
refusal that comes first for an eight-wide float32 line. One records the intended change: float32
sums over 8192 now admit the tile plan beside the compensated ones.

Manifest review (task 7.3), every change matched on content against the manifest from before tile
plans were offered: 590 tile entries added to rejected lists, none removed; 99 selections that differ
only in their key (the two new runtime facts, and in some the catalog) with the same winners; four
16-bit cases at 4099 (`row_sum`, `mean_epilogue`) where tile width 1 became a candidate and was compiled (withdrawn in group 7),
every other artifact byte-identical and the same plan chosen (`independent_planes`; tile does not
win at 16 or 64 rows). 16-bit `rms_inference` at 4099 is not offered one: it sums `x * x`, float32
data, which needs width 4 and whole steps. No IR, CUDA or PTX changed for any existing plan. The
manifest was re-recorded from those reviewed records.

## Group 7: acceptance

### Timing (task 7.1)

`ncu` as above, three runs each; "Existing" is the best plan before tile plans (pinned:
`shared_tree_compensated` 512 for float32, `plane_shared` 512 for bfloat16). RMSNorm is
`x * rsqrt(mean(x.float()²) + 1e-6) * w`; the consumer is `x * x.sum(-1, keepdim=True)`.

RTX 3080 (µs):

| Kernel | Tile plan | Tile | Existing | Triton |
|---|---|---|---|---|
| RMSNorm float32 2048×8192 | width 4 | 195.5–196.7 | 224.2–226.6 | 221.7–223.3 |
| RMSNorm bfloat16 2048×8192 | width 8 | 97.9–98.5 | 98.9–99.2 | 101.7–102.5 |
| RMSNorm bfloat16 2048×8192 | width 1 | 101.9–103.2 | 98.9–99.2 | 101.7–102.5 |
| RMSNorm float32 2048×4099 | width 1 | 102.7–103.2 | 101.6–102.2 | 105.4–105.6 |
| RMSNorm bfloat16 2048×4099 | width 1 | 58.8–61.1 | 52.7–53.2 | 59.2–59.4 |
| Consumer float32 2048×8192 | width 4 | 194.1–194.6 | 202.9–203.8 | 200.7–202.8 |
| Consumer bfloat16 2048×8192 | width 8 | 96.4–96.9 | 96.0–96.4 | 99.6–100.9 |
| Consumer bfloat16 2048×8192 | width 1 | 98.7–98.8 | 96.0–96.4 | 99.6–100.9 |
| Consumer float32 2048×4099 | width 1 | 100.9–101.0 | 100.1–100.7 | 100.7–101.3 |
| Consumer bfloat16 2048×4099 | width 1 | 56.4–56.8 | 51.4–52.0 | 49.5–50.2 |

RTX 3090 (µs, 2048×8192, the offered plans):

| Kernel | Tile | Existing | Triton |
|---|---|---|---|
| RMSNorm float32, width 4 | 159.0–159.6 | 182.5–184.6 | 187.0–204.6 |
| RMSNorm bfloat16, width 8 | 81.9–83.6 | 82.9–84.0 | 86.2–87.5 |
| Consumer float32, width 4 | 161.4–163.6 | 167.2–168.6 | 162.5–168.5 |
| Consumer bfloat16, width 8 | 78.8–91.0 | 81.4–89.8 | 84.6–96.6 |

The 3090 drives the display; its bfloat16 consumer rows each have one outlier run.

- Float32 data at width 4 is the clear gain: RMSNorm takes 12% less time than Triton on the 3080
  and 15% less on the 3090 (medians; 67 MB of DRAM reads against 80–87), the consumer 3–4% less
  on the 3080 and level to 3% on the 3090.
- 16-bit values at width 8 are level with the existing plan and 3–7% ahead of Triton.
- 16-bit values at width 1 were in Triton's band (4.2) but are slower than the existing plan:
  9–11% at 4099 (medians) and 2–4% at 8192, and 13% slower than Triton on the 4099 consumer. Tile plans for 16-bit values are
  therefore offered at width 8 only; width 1 is recorded as rejected with that reason.
- Float32 at width 1 (4099 rows) is level on time but outside Triton's band (4.2), so it is not
  offered (bead `cie`).

Workload shapes: the task asked for the census shapes that have a sum-only kernel. The workload
census (`scoreboard/results/census-inventory`) records kernel names and times, not shapes, and the
census with shapes (`wide-reduction-workgroups.md`, 6.3) has only LayerNorm, which is a Welford
kernel and outside this stage. The eager sum kernels in Anima, Qwen-Image and others
(`reduce_kernel` with `func_wrapper` or `MeanOps`) have no recorded shapes, so they are not timed
here (bead filed in 7.5).

### Cost (task 7.2)

RTX 3080, RMSNorm 2048×8192, fresh artifact cache per compile, median of three.

| | Tile | `plane_shared` 512 | `shared_tree_compensated` 512 |
|---|---|---|---|
| Compile, float32 (ms) | 46 | 28 | 30 |
| Compile, bfloat16 (ms) | 96 | 49 | 48 |

A tile kernel takes 1.5 to 2 times as long to compile: its trees are unrolled (655 lines of
source against 308 before group 4).

First call of the compiled function (compile plus first-use tuning), two runs each:

| | With tile plans | Without | Winner with |
|---|---|---|---|
| float32 | 4.46, 4.53 s | 3.97, 4.06 s | `tile_row_sum` width 4 |
| bfloat16 | 8.06, 8.05 s | 7.76, 7.70 s | `tile_row_sum` width 4 |

The first call costs 0.3 to 0.6 s more (4 to 12%), from the extra candidates compiled and
timed. The bfloat16 RMSNorm sums `x.float()²`, float32 data, so its winner is width 4.

### Manifest after the width 8 rule (task 7.3, again)

Against group 6's manifest, matched on content: 22 cases change, all 16-bit, all in rejected or
candidate lists. Tile width 1 is rejected as "offered at width 8 only" where it was rejected by the
512-unit rule (18 regions) or was a candidate (the four 4099 cases group 6 listed, which still
choose `independent_planes`, with the compiled tile kernel no longer among their artifacts); four
width 4 rejections reword their reason. No chosen plan, IR, CUDA or PTX changed. The manifest was
re-recorded from these records.
