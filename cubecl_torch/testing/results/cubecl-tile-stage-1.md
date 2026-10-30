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

Still open from reading the code, for group 3: on a dynamic axis `Procedural` sets `bound` to
`u32::MAX` and turns bounds checks off, so a partial step of a runtime row is neither masked nor
guarded. The gate's shapes are whole steps. This is wrong values as well as out-of-bounds reads.
