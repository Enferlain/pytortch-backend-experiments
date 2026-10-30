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

Inductor splits rows past 65536 by default, and split reductions are not tile kernels, so 65539 is
tested with `split_reductions` off.

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
