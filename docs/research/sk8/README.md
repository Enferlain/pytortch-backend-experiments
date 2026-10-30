# sk8: cubecl-tile prototype

Research for bead `sk8`: can CubeCL's tile DSL (`.stage0-sources/cubecl/crates/cubecl-tile`)
supply the kernel-to-thread mapping that `cubecl_torch` builds plan by plan? The bead's notes
hold the findings; this folder holds what produced them.

The case is the float32 sum consumer `x * x.sum(-1, keepdim=True)` at 2048×8192, against
Inductor+Triton. Triton's autotuner picks `R0_BLOCK` 8192 with 16 warps: each of 512 threads
holds 16 values of the row.

## Results (2026-10-30, ncu at base clocks)

Speed:

| Kernel (mode) | 3080 | 3080 DRAM read | 3090 | 3090 DRAM read |
|---|---|---|---|---|
| Triton | 203.9 µs | 67 MB | 167.0 µs | 67 MB |
| Tile, one row per cube, 16 planes × 32 units × vec4 (`nohint`) | 286 µs | 129 MB | 236.9 µs | 129 MB |
| Same, pass 1 `evict_last`, pass 2 `evict_first` (`hinted`, default) | 204.0 µs | 70 MB | 169.5 µs | 71 MB |
| Same with a sum per vector lane (`lanes`, hinted) | 204.7 µs | 67 MB | | |
| One row per plane, 16 rows per cube (`row_plane`) | 295 µs | 134 MB | | |

The tile DSL has no load cache hints. The kernels set them during expansion: a helper walks the
IR emitted so far and sets the CubeCL patch's `LoadOp` retained/streaming flags on the 16-byte
global loads (`hints` in `tile_proto/src/main.rs`). With them the tile kernel is level with
Triton. Without them the second pass re-reads the row from DRAM.

Accuracy (`order_model.py` models each order exactly in float32; every model below that has a GPU
kernel matches it bit for bit). "Closer/farther" counts rows whose sum is nearer to / further
from the float64 sum than Triton's:

| Order | normal: mean \|err\|, closer/farther | offset (+1000): same |
|---|---|---|
| Triton | 6.96e-6 | 0.187 |
| Tile `hinted` (per unit: each step's 4 lanes added, then into one sum) | 8.37e-6, 721/919 | 0.370, 274/887 |
| Tile `lanes` (a sum per lane) | 8.25e-6, 719/944 | 0.261, 290/659 |
| `lanes` + planes combined as a tree (model) | 7.09e-6, 789/825 | 0.221, 308/501 |
| `lanes` + planes and lanes as trees, units xor 16→1 (model) | 7.04e-6, 793/811 | 0.191, 314/353 |

Triton's `tl.sum` is a balanced tree: inside the thread over its 16 registers, across lanes xor
16 down to 1, then across warps. The tile DSL's order is fixed by its lowering: units combine xor
1 up to 16, planes in sequence through the cyclic rounds (plane 0, then 15 down to 1), the lanes
in sequence. The sequential plane combine costs the most. Matching Triton's bits would also need
its thread ownership (each thread's 4 repeats 2048 apart, not contiguous plane slices).

## Check: Inductor's loop body inside tile loops (RMSNorm)

The question: can Inductor's fused per-element code run inside tile loops, or does the tile DSL
only cover bare reductions? The case is RMSNorm with a weight, `x * rsqrt(mean(x²) + eps) * w`,
float32 at 2048×8192: work fused before the sum (`x²`) and after it (`* rsqrt`, `* w`).

`fused_probe/` takes the kernel's `semantic.json` as our pipeline produced it and runs our own
emitter (`cubecl_torch/native/src/{ir,semantic}.rs`, compiled in unchanged through `#[path]`)
inside two tile *recipes*: the value the reduction sums, and the value each output element
stores. A recipe is a hand-written expand type that holds the buffer pointers, binds the row and
column the way our reduction emitter does, and calls `semantic::emit`. The tile layout is the
Triton-shaped one above (one row per cube, 16 planes × 32 units); the summing pass's row load is
marked retained and the storing pass's streaming, through the descriptor ops' own flags.

| Kernel (3080, ncu) | Time | DRAM read |
|---|---|---|
| Triton | 228.6–231.0 µs | 85 MB |
| Ours today (`shared_tree_compensated` 512, first-use selection) | ~223 µs | 85.7 MB |
| Tile loops + our emitter (`fused_probe`, leaf width 4) | 229.8–231.3 µs | 85.6–86.7 MB |

Correct against float64 (max relative error 2.6e-7). Compile cost per kernel, same NVRTC:

| | Expansion | C++ | NVRTC | Total |
|---|---|---|---|---|
| Tile loops + our emitter | 7.6 ms (our `emit`: 0.1 ms) | 30 ms | 29 ms (1,188 lines) | ~67 ms |
| Ours today | 10–12 ms for the whole compiler process | | 20 ms (363 lines) | ~32 ms |

What the check found:

- It works with our emitter unchanged. A recipe's expansion runs in a `Scope` like ours, and a
  tensor argument's buffer (`as_slice` → `__extract_list`) is the pointer our loads index.
- Recipe values are scalar: the loads stay 4 bytes wide whatever the leaf width. It did not cost
  time here (the kernel is memory-bound), but vector loads need a recipe that returns a vector.
- Copying a recipe into an output needs a scalar destination (`fill_procedural`), so the output
  binds at width 1.
- Recipes are documented as memory-free; ours read memory. Overhangs and shared-memory staging of
  a recipe were not exercised.
- The row sum reaches the storing recipe through a second, writable binding of the sums buffer.
  A read-only second binding compiles to `const __restrict__` non-coherent loads that miss
  in-kernel writes (the `lanes` mode hit this).
- The probe covers plain ops and one sum: no mask regions, Welford, multi-output or splits.
- Compiling it inside `native/` behind an off-by-default feature still changed the default
  compiler binary's digest (the manifest change alters Cargo's crate hash), which would invalidate
  every artifact key; hence the separate crate.

## Checks before adoption: runtime extents, partial tiles, combines, vector recipes

**Runtime extents** (`tile_proto dynamic <rows>x<cols> ...`): the sum consumer with rows and
columns as runtime values, the cube's 16 planes taking the row's 128-wide steps in runs
(`Levels::planes_across`) and meeting in fixed slots, CubeK's `InSlots` shape.

- Correct on every shape tried (2048×8192, 1024×8192, 2048×4096, 300×4096, 2048×4099, 300×4099,
  256×65539, 300×1000, 512×4100, 64×100, 64×96, 64×31, 7×2049).
- One kernel per alignment class, not per shape: 4-wide when the row is whole 128-wide steps,
  scalar without masks when it is whole scalar steps, scalar with masks otherwise. A 4-wide line
  cannot be masked (`UncheckableVectorEdge`, and `AnyExtent` refuses it on a runtime axis), so a
  row of 1000 loads scalar where Triton would mask inside its vectors.
- No cost from runtime extents: 204.3 µs / 70.5 MB at 2048×8192 (the fixed-size kernel: 204.0);
  at 2048×4099, 101.2 µs / 34.5 MB against Triton's 101.6 µs / 33.8 MB.
- The DSL's elementwise ops (`product`, `copy_from`) plan their loops from the box's static
  extents and panic on a runtime axis. The scaling pass is therefore plain `#[cube]` code (units
  striding the row), which is what our emitter would write there anyway.

**Partial tiles with recipes** (`fused_probe`, RMSNorm at 2048×4099, now on the same
`planes_across` layout): correct (max relative error 2.9e-7), but a recipe is evaluated before the
bounds test: the masked read evaluates, then selects (`ProceduralExpand::__expand_read_checked_method`).
The last row's last step reads up to 28 floats past the tensor. Memory tiles redirect an
out-of-range index to a valid address before loading; recipes get the raw coordinate. Fix in our
copy: test bounds first and evaluate at a redirected coordinate (our emitter already has guarded
loads). Confirmed at run time with CubeCL's own `validate_index` in the summing recipe
(`fused_probe ... validate`), checking the row index against `x`'s logical length from the
binding's metadata: at 2048×4099 it reports 29 out-of-range indices, 8,394,752 to 8,394,780 (row
2047, columns 4099–4127, the last partial step); at 2048×8192, none. Compute Sanitizer's memcheck
(also with `--padding 4096`) reports nothing: CubeCL places tensors inside larger pooled blocks
(`ExclusivePages` rounds pages up to size buckets too), so the stray reads stay in memory it owns.
Under PyTorch's caching allocator the tensor can end at a neighbour or a segment boundary.
CubeCL's `Validate` execution mode would instrument only indices flagged `checked`, and the launch
macro generates only checked and unchecked launches; its `check_mode` setting is not read at this
pin.

**Combines (code read).** Every destination planes or cubes meet in (atomic, cyclic, relay, slots)
adds, and `Write::admits` refuses max, min or product into one: a max with planes splitting the
row needs a non-sum slot read, a short change. An accumulator cell is one number (`Acc: Numeric`;
`Monoid::combine_of` is lane-wise), so Welford (count, mean, M2) and argmax (value, index) need a
cell holding several values with its own combine through register blocks, unit shuffles and
drains: a real extension.

**Vector recipes (code read).** A procedural read is scalar by an explicit assert ("a procedural
read is scalar"). A vector-valued `evaluate`, falling back to one lane at a time, plus our
emitter's vector loads in our recipe, is a local change in the same function.

**The layout matters for code size.** On `planes_across` + slots the RMSNorm probe generates 308
lines instead of 1,188 (the cyclic combine unrolls 16 rounds): expansion 2.6 ms, C++ 7.6 ms,
NVRTC 20.9 ms, about 31 ms per kernel like ours, and the kernel runs in 216.8 µs / 78.8 MB
against Triton's 228.6–231.0 and our current ~223.

## Files

- `tile_proto/`: the tile kernels (Rust, against the pinned CubeCL checkout; CubeCL's CUDA
  runtime with the C++ backend, `cuda-cpp`, as production uses).
- `triton_ref.py`: generates the inputs and writes Triton's outputs, the chosen launch
  configs and the generated code, plus float64 references.
- `compare.py`: bitwise and accuracy comparison of a tile output against Triton and float64. Its
  closer/farther counts recover Triton's sum from its output, so they are slightly noisy;
  `order_model.py` compares exact sums.
- `order_model.py`: float32 models of Triton's and the tile kernels' summation orders, with
  variants.
- `triton_time.py`: runs the compiled Triton kernel, for timing under ncu.
- `fused_probe/`: the RMSNorm check (our emitter inside tile recipes, CubeCL's CUDA runtime).
- `rms_time.py`: runs RMSNorm compiled by `cubecl` (this repo's backend) or `triton`, for ncu.

## Running

Everything writes into a data directory of your choosing (about 1.1 GB for both inputs);
delete it afterwards. `D` below is that directory, `CU13` the venv's CUDA 13 wheel
(`stage0/python-deps/nvidia/cu13`): no system CUDA is installed, and CubeCL's C++ backend
needs its headers and NVRTC.

```bash
REF=/home/imi/Projects/sd-scripts/.venv-wsl/bin/python
TORCHINDUCTOR_CACHE_DIR=$D/ind TRITON_CACHE_DIR=$D/tri $REF docs/research/sk8/triton_ref.py $D 2048 8192

cargo build --release --manifest-path docs/research/sk8/tile_proto/Cargo.toml
export CUDA_PATH=$CU13 LD_LIBRARY_PATH=$CU13/lib CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
BIN=docs/research/sk8/tile_proto/target/release/tile_proto
# modes: hinted (default), nohint, lanes, row_plane
$BIN $D/normal.x.bin $D/normal.lanes.bin 2048 8192 1 lanes
$REF docs/research/sk8/compare.py $D normal $D/normal.lanes.bin lanes
$REF docs/research/sk8/order_model.py $D normal $D/normal.lanes.bin
```

`CUDA_VISIBLE_DEVICES=0` with PCI bus order is the 3080 (`1` the 3090). For timing, run the
binary with 4 repetitions under `ncu --launch-skip 2 --launch-count 1`; time Triton with
`triton_time.py <x.bin> 3080|3090` (all GPUs visible). Set `CUBECL_DEBUG_LOG=<file>` to log the
generated CUDA. The build writes several GB under `tile_proto/target`; remove it when done.

The RMSNorm check needs a `semantic.json`: compile RMSNorm once through this repo's backend with
`CUBECL_TORCH_CACHE_DIR=$D/cc` (`rms_time.py cubecl`, from the repository root with
`PYTHONPATH=$PWD` and `stage0/.venv/bin/python`) and take it from `$D/cc/artifacts/*/`.

```bash
cargo build --release --manifest-path docs/research/sk8/fused_probe/Cargo.toml
docs/research/sk8/fused_probe/target/release/fused_probe $D/out $SEMANTIC 2048 8192 3 4
```

It prints expansion, C++ and first-launch times and the error against float64, and writes the
generated CUDA to `$D/out/tile_kernel.cu`. For Triton under ncu, run `rms_time.py triton` once
without ncu first: a fresh Triton cache fails to build its launcher under ncu.
